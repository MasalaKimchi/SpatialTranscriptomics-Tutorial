"""Offline evidence that empty stage inputs fail before expensive work."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src import benchmark, data, eval as evaluation, foundation, labels, patches, train
from src import foundation_eval
from src.identity import IdentityValidationError
from src.validation import ConfigValidationError, StageValidationError, require_non_empty

pytestmark = pytest.mark.offline


def _forbidden(*_args, **_kwargs):
    raise AssertionError("expensive or output-producing seam was reached")


def _patch_cfg() -> dict:
    return {
        "patches": {
            "min_patch_px": 8,
            "output_size": 16,
            "context_scale": 1.0,
            "per_slide_stain_norm": False,
        }
    }


def _complete_cfg() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_data_stage_error_contract_exposes_stable_primitive_evidence() -> None:
    with pytest.raises(StageValidationError) as caught:
        require_non_empty(
            np.zeros((0, 3, 8, 8), dtype=np.float32),
            stage="prediction",
            subject="patch batch",
            guidance="Provide at least one patch.",
        )

    error = caught.value
    assert error.stage == "prediction"
    assert error.subject == "patch batch"
    assert error.observed == 0
    assert error.minimum == 1
    assert error.shape == (0, 3, 8, 8)
    assert str(error) == (
        "prediction: patch batch is empty (observed count=0, observed "
        "shape=(0, 3, 8, 8), expected >=1). Provide at least one patch."
    )


def test_data_preprocess_empty_sequence_fails_before_directory_or_loader(
    monkeypatch,
) -> None:
    monkeypatch.setattr(data, "pharma_processed_dir", _forbidden)
    monkeypatch.setattr(data.st, "load_visium_sample", _forbidden)

    with pytest.raises(StageValidationError, match="cohort_preprocessing") as caught:
        data.preprocess_cohort([], cfg={"unused": True})

    assert caught.value.observed == 0
    assert caught.value.minimum == 1


def test_data_summary_empty_sequence_fails_before_slide_loader(monkeypatch) -> None:
    monkeypatch.setattr(data, "load_slide", _forbidden)

    with pytest.raises(StageValidationError, match="cohort_summary"):
        data.cohort_summary([])


def test_label_cohort_empty_input_fails_before_output_directory(monkeypatch) -> None:
    monkeypatch.setattr(labels, "pharma_outputs_dir", _forbidden)

    with pytest.raises(StageValidationError, match="cohort_label_generation"):
        labels.build_labels_cohort([], cfg={"unused": True})


def test_label_empty_slide_frame_fails_before_writer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(labels, "pharma_outputs_dir", _forbidden)
    monkeypatch.setattr(
        labels,
        "build_labels_for_slide",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _forbidden)

    with pytest.raises(StageValidationError, match="label rows for slide slide_a"):
        labels.build_labels_cohort(["slide_a"], cfg={"unused": True})


def test_label_later_empty_slide_publishes_no_partial_cohort(monkeypatch) -> None:
    valid = pd.DataFrame(
        {
            "cluster": ["0"],
            "domain_name": ["immune_enriched"],
            "tme_class": ["immune_enriched"],
        }
    )

    def build(slide_id, _cfg):
        return valid if slide_id == "slide_a" else pd.DataFrame()

    monkeypatch.setattr(labels, "build_labels_for_slide", build)
    monkeypatch.setattr(labels, "pharma_outputs_dir", _forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _forbidden)

    with pytest.raises(StageValidationError, match="label rows for slide slide_b"):
        labels.build_labels_cohort(["slide_a", "slide_b"], cfg={"unused": True})


def test_label_slide_generation_rejects_zero_rows(monkeypatch) -> None:
    empty_adata = SimpleNamespace(
        obs_names=pd.Index([], dtype=object),
        obs=pd.DataFrame(
            {
                "clusters": pd.Series(dtype=object),
                "slide_id": pd.Series(dtype=object),
            }
        ),
    )
    fake_scanpy = SimpleNamespace(
        tl=SimpleNamespace(rank_genes_groups=lambda *_args, **_kwargs: None),
        get=SimpleNamespace(
            rank_genes_groups_df=lambda *_args, **_kwargs: pd.DataFrame(
                columns=["group", "scores", "names"]
            )
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(labels, "load_slide", lambda _sid: empty_adata)
    monkeypatch.setattr(labels, "tme_class_to_id", lambda _cfg: {"other": 0})
    monkeypatch.setattr(labels, "marker_genes_for_slide", lambda *_args: [])
    monkeypatch.setattr(labels, "compute_module_scores", lambda *_args: [])
    monkeypatch.setattr(labels.st, "genes_present", lambda *_args, **_kwargs: [])

    with pytest.raises(StageValidationError, match="slide_label_generation"):
        labels.build_labels_for_slide("slide_a", cfg={"seed": 0})


@pytest.mark.parametrize(
    ("mode", "columns"),
    [("modules", ["gene_A"]), ("genes", ["module_A"]), ("both", ["other"])],
)
def test_regression_empty_selection_is_actionable(mode, columns) -> None:
    frame = pd.DataFrame(columns=columns)

    with pytest.raises(StageValidationError, match=mode) as caught:
        labels.regression_columns(frame, cfg={"labels": {"regression_targets": mode}})

    assert caught.value.stage == "regression_target_selection"


def test_regression_valid_modes_preserve_column_order() -> None:
    frame = pd.DataFrame(columns=["module_b", "gene_a", "module_a", "other"])

    assert labels.regression_columns(
        frame, cfg={"labels": {"regression_targets": "modules"}}
    ) == ["module_b", "module_a"]
    assert labels.regression_columns(
        frame, cfg={"labels": {"regression_targets": "genes"}}
    ) == ["gene_a"]
    assert labels.regression_columns(
        frame, cfg={"labels": {"regression_targets": "both"}}
    ) == ["module_b", "module_a", "gene_a"]


def test_patch_empty_coordinates_fail_before_image_or_stack(monkeypatch) -> None:
    monkeypatch.setattr(patches, "coords_hires", lambda _adata: np.zeros((0, 2)))
    monkeypatch.setattr(patches.st, "get_image", _forbidden)
    monkeypatch.setattr(np, "stack", _forbidden)

    with pytest.raises(StageValidationError, match="patch_extraction") as caught:
        patches._extract_spot_patches(
            SimpleNamespace(
                obs_names=pd.Index([], dtype=object),
                obs=pd.DataFrame({"slide_id": pd.Series(dtype=object)}),
            ),
            "slide_a",
            np.eye(2, 3),
            _patch_cfg(),
        )

    assert caught.value.shape == (0, 2)


def test_stain_empty_input_fails_before_slide_scan(monkeypatch) -> None:
    monkeypatch.setattr(data, "load_slide", _forbidden)

    with pytest.raises(StageValidationError, match="stain_reference"):
        patches.fit_reference_stain([], cfg={"unused": True})


def test_patch_cohort_empty_input_fails_before_stain_or_cache(monkeypatch) -> None:
    monkeypatch.setattr(patches, "fit_reference_stain", _forbidden)
    monkeypatch.setattr(patches, "save_patch_arrays", _forbidden)

    with pytest.raises(StageValidationError, match="patch_cohort"):
        patches.build_patch_cohort([], cfg={"unused": True})


@pytest.mark.skipif(patches.SpotPatchDataset is None, reason="PyTorch unavailable")
def test_dataset_rejects_empty_and_mismatched_rows_explicitly() -> None:
    empty_patches = np.zeros((0, 3, 8, 8), dtype=np.float32)
    empty_labels = pd.DataFrame(columns=["tme_class_id", "module_signal"])
    with pytest.raises(StageValidationError, match="patch rows"):
        patches.SpotPatchDataset(empty_patches, empty_labels)

    valid_patches = np.zeros((2, 3, 8, 8), dtype=np.float32)
    one_label = pd.DataFrame({"tme_class_id": [0], "module_signal": [1.0]})
    with pytest.raises(StageValidationError, match="row counts differ") as caught:
        patches.SpotPatchDataset(valid_patches, one_label)
    assert caught.value.shape == (2, 1)


@pytest.mark.skipif(patches.SpotPatchDataset is None, reason="PyTorch unavailable")
def test_dataset_valid_rows_preserve_shape_and_order() -> None:
    patch_array = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    frame = pd.DataFrame(
        {"tme_class_id": [1, 0], "module_signal": [2.5, 3.5]},
        index=[7, 3],
    )
    dataset = patches.SpotPatchDataset(
        patch_array, frame, reg_cols=["module_signal"]
    )

    first_patch, first_class, first_regression = dataset[0]
    assert len(dataset) == 2
    np.testing.assert_array_equal(first_patch.numpy(), patch_array[0])
    assert first_class == 1
    np.testing.assert_allclose(first_regression.numpy(), [2.5])


def test_patch_valid_extraction_preserves_shape_and_spot_order(
    synthetic_anndata_factory,
) -> None:
    adata = synthetic_anndata_factory(slide_id="slide_valid", n_spots=3)
    reference = np.asarray(
        [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=np.float64
    )

    patch_array, metadata = patches.extract_all_patches_for_slide(
        adata, "slide_valid", reference, cfg=_patch_cfg()
    )

    assert patch_array.shape == (3, 3, 16, 16)
    assert metadata["spot_id"].tolist() == adata.obs_names.tolist()


@pytest.mark.parametrize("slide_ids", [[], ["slide_a"], ["slide_a", "slide_a"]])
def test_loso_fold_admission_rejects_fewer_than_two_unique_slides(slide_ids) -> None:
    with pytest.raises(StageValidationError, match="loso_fold_admission") as caught:
        train.loso_folds(slide_ids)

    assert caught.value.minimum == 2


def test_loso_training_empty_inputs_fail_before_config_or_fold(monkeypatch) -> None:
    monkeypatch.setattr(train, "load_config", _forbidden)
    monkeypatch.setattr(train, "train_one_fold", _forbidden)

    with pytest.raises(StageValidationError, match="loso_training_admission"):
        train.train_loso([], pd.DataFrame())

    with pytest.raises(StageValidationError, match="cohort label rows"):
        train.train_loso(["slide_a", "slide_b"], pd.DataFrame())


def test_loso_benchmark_empty_inputs_fail_before_training_or_cache(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "load_config", _forbidden)
    monkeypatch.setattr(benchmark, "train_loso", _forbidden)

    with pytest.raises(StageValidationError, match="loso_benchmark_admission"):
        benchmark.run_loso_benchmark([], pd.DataFrame())


def test_align_zero_rows_fails_before_patch_indexing(monkeypatch) -> None:
    patch_array = np.ones((2, 3, 4, 4), dtype=np.float32)
    metadata = pd.DataFrame(
        {"slide_id": ["slide_a", "slide_a"], "spot_id": ["p0", "p1"]}
    )
    label_frame = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["label_only"], "tme_class_id": [0]}
    )
    monkeypatch.setattr(
        train,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patch_array, metadata),
    )

    with pytest.raises(IdentityValidationError, match="patch_label_alignment") as caught:
        train.load_slide_patches("slide_a", label_frame, cfg={})

    assert {issue.code for issue in caught.value.issues} == {
        "label_only",
        "metadata_only",
    }


def _fold_cfg() -> dict:
    return {
        "training": {
            "device": "cpu",
            "model": "resnet18",
            "pretrained": False,
            "augment": False,
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "cls_weight": 1.0,
            "reg_weight": 1.0,
            "patience": 1,
        },
        "labels": {
            "classification_col": "tme_class_id",
            "regression_targets": "modules",
            "tme_classes": ["a", "b"],
        },
    }


def _fold_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slide_id": ["train", "held"],
            "spot_id": ["train_0", "held_0"],
            "tme_class_id": [0, 1],
            "module_signal": [0.2, 0.8],
        }
    )


def test_train_fold_empty_slide_list_fails_before_device_or_model(monkeypatch) -> None:
    monkeypatch.setattr(train, "resolve_device", _forbidden)
    monkeypatch.setattr(train, "build_model", _forbidden)

    with pytest.raises(StageValidationError, match="training slide IDs"):
        train.train_one_fold([], "held", _fold_labels(), cfg=_fold_cfg())


@pytest.mark.parametrize("empty_subject", ["train", "held"])
def test_train_fold_empty_member_fails_before_device_or_model(
    monkeypatch, empty_subject
) -> None:
    non_empty_patches = np.ones((1, 3, 4, 4), dtype=np.float32)
    non_empty_labels = _fold_labels().iloc[[0]].reset_index(drop=True)

    def load_stub(slide_id, *_args, **_kwargs):
        if slide_id == empty_subject:
            return np.zeros((0, 3, 4, 4), dtype=np.float32), non_empty_labels.iloc[:0]
        return non_empty_patches, non_empty_labels

    monkeypatch.setattr(train, "load_slide_patches", load_stub)
    monkeypatch.setattr(train, "resolve_device", _forbidden)
    monkeypatch.setattr(train, "build_model", _forbidden)

    with pytest.raises(StageValidationError, match="cnn_fold_training"):
        train.train_one_fold(["train"], "held", _fold_labels(), cfg=_fold_cfg())


def test_train_fold_empty_regression_selection_fails_before_device_or_model(
    monkeypatch,
) -> None:
    patches = np.ones((1, 3, 4, 4), dtype=np.float32)
    no_targets = _fold_labels().drop(columns=["module_signal"])

    def load_stub(slide_id, *_args, **_kwargs):
        row = 0 if slide_id == "train" else 1
        return patches, no_targets.iloc[[row]].reset_index(drop=True)

    monkeypatch.setattr(train, "load_slide_patches", load_stub)
    monkeypatch.setattr(train, "resolve_device", _forbidden)
    monkeypatch.setattr(train, "build_model", _forbidden)

    with pytest.raises(StageValidationError, match="regression_target_selection"):
        train.train_one_fold(["train"], "held", no_targets, cfg=_fold_cfg())


@pytest.mark.parametrize("empty_subject", ["train", "held"])
def test_rf_fold_empty_member_fails_before_estimator(
    monkeypatch, empty_subject
) -> None:
    non_empty_patches = np.ones((1, 3, 4, 4), dtype=np.float32)
    non_empty_labels = _fold_labels().iloc[[0]].reset_index(drop=True)

    def load_stub(slide_id, *_args, **_kwargs):
        if slide_id == empty_subject:
            return np.zeros((0, 3, 4, 4), dtype=np.float32), non_empty_labels.iloc[:0]
        return non_empty_patches, non_empty_labels

    monkeypatch.setattr(benchmark, "load_slide_patches", load_stub)
    monkeypatch.setattr(benchmark, "train_eval_rf_baseline", _forbidden)

    with pytest.raises(StageValidationError, match="rf_fold_training"):
        benchmark.run_rf_loso_fold(
            ["train"], "held", _fold_labels(), fold=0, cfg=_fold_cfg()
        )


class _ForbiddenPredictionModel:
    def eval(self):
        return _forbidden()


@pytest.mark.parametrize(
    ("patch_array", "batch_size", "subject"),
    [
        (np.zeros((0, 3, 4, 4), dtype=np.float32), 2, "NCHW patch batch"),
        (np.ones((1, 3, 4, 4), dtype=np.float32), 0, "batch size"),
    ],
)
def test_predict_rejects_empty_batch_before_device_or_model(
    monkeypatch, patch_array, batch_size, subject
) -> None:
    monkeypatch.setattr(evaluation, "resolve_device", _forbidden)

    with pytest.raises(StageValidationError, match=subject):
        evaluation.predict_cnn(
            _ForbiddenPredictionModel(), patch_array, batch_size=batch_size
        )


def test_rf_public_training_rejects_empty_before_radiomics_or_estimator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluation, "radiomics_from_patches", _forbidden)
    monkeypatch.setattr(evaluation, "RandomForestClassifier", _forbidden)

    with pytest.raises(StageValidationError, match="rf_training"):
        evaluation.train_eval_rf_baseline(
            np.zeros((0, 3, 4, 4), dtype=np.float32),
            _fold_labels().iloc[:0],
            np.ones((1, 3, 4, 4), dtype=np.float32),
            _fold_labels().iloc[[0]],
            cfg=_fold_cfg(),
        )


class _ForbiddenEncoder:
    def __call__(self, *_args, **_kwargs):
        return _forbidden()


def test_foundation_embedding_empty_batch_fails_before_model_forward() -> None:
    with pytest.raises(StageValidationError, match="foundation_embedding") as caught:
        foundation.extract_frozen_embeddings(
            np.zeros((0, 3, 8, 8), dtype=np.float32),
            _ForbiddenEncoder(),
            foundation.FOUNDATION_MODELS["kaiko_vits16"],
            device="cpu",
        )

    assert caught.value.shape == (0, 3, 8, 8)


def test_foundation_loso_empty_inputs_fail_before_encoder_or_cache(monkeypatch) -> None:
    monkeypatch.setattr(foundation, "load_config", _forbidden)
    monkeypatch.setattr(foundation, "load_frozen_encoder", _forbidden)
    monkeypatch.setattr(foundation, "load_or_extract_slide_embeddings", _forbidden)

    with pytest.raises(StageValidationError, match="foundation_loso_admission"):
        foundation.run_foundation_loso([], pd.DataFrame())

    with pytest.raises(StageValidationError, match="cohort label rows"):
        foundation.run_foundation_loso(
            ["slide_a", "slide_b"], pd.DataFrame(), cfg={}
        )


@pytest.mark.parametrize("cfg", [{}, {"foundation": {}}])
@pytest.mark.parametrize(
    "helper",
    [
        "foundation_config",
        "load_frozen_encoder",
        "load_or_extract_slide_embeddings",
        "save_benchmark_report",
    ],
)
def test_explicit_invalid_config_fails_before_foundation_or_report_seams(
    cfg, helper, monkeypatch, tmp_path
) -> None:
    labels_frame = pd.DataFrame()
    monkeypatch.setattr(foundation, "load_config", _forbidden)
    monkeypatch.setattr(foundation, "_foundation_model_spec_resolved", _forbidden)
    monkeypatch.setattr(foundation, "_embedding_cache_path", _forbidden)
    monkeypatch.setattr(foundation, "pharma_processed_dir", _forbidden)
    monkeypatch.setattr(foundation, "resolve_device", _forbidden)
    monkeypatch.setattr(foundation, "load_slide_patches", _forbidden)
    monkeypatch.setattr(foundation, "extract_frozen_embeddings", _forbidden)
    monkeypatch.setattr(foundation.np, "load", _forbidden)
    monkeypatch.setattr(foundation.np, "savez_compressed", _forbidden)
    monkeypatch.setattr(evaluation, "load_config", _forbidden)
    monkeypatch.setattr(evaluation, "pharma_outputs_dir", _forbidden)
    monkeypatch.setattr(evaluation.pd, "DataFrame", _forbidden)

    with pytest.raises(ConfigValidationError):
        if helper == "foundation_config":
            foundation.foundation_config(cfg)
        elif helper == "load_frozen_encoder":
            foundation.load_frozen_encoder(cfg)
        elif helper == "load_or_extract_slide_embeddings":
            foundation.load_or_extract_slide_embeddings(
                "slide_a", labels_frame, cfg=cfg
            )
        else:
            evaluation.save_benchmark_report(
                [], path=tmp_path / "must-not-exist.csv", cfg=cfg
            )

    assert not (tmp_path / "must-not-exist.csv").exists()


def test_foundation_and_report_valid_configs_preserve_existing_behavior(
    monkeypatch, tmp_path
) -> None:
    cfg = _complete_cfg()
    default_calls = 0

    def tracked_default() -> dict:
        nonlocal default_calls
        default_calls += 1
        return copy.deepcopy(cfg)

    monkeypatch.setattr(foundation, "load_config", tracked_default)
    assert foundation.foundation_config() == cfg["foundation"]
    assert default_calls == 1

    monkeypatch.setattr(foundation, "load_config", _forbidden)
    assert foundation.foundation_config(cfg) == cfg["foundation"]

    created: list[tuple[tuple, dict]] = []

    def create_model(*args, **kwargs):
        created.append((args, kwargs))
        return foundation.torch.nn.Identity()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))
    model, device, spec = foundation.load_frozen_encoder(cfg, device="cpu")
    assert isinstance(model, foundation.torch.nn.Identity)
    assert str(device) == "cpu"
    assert spec is foundation.FOUNDATION_MODELS["kaiko_vits16"]
    assert created[0][0][0].startswith("hf_hub:")

    no_cache_cfg = copy.deepcopy(cfg)
    no_cache_cfg["foundation"]["cache"] = False
    labels_frame = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["spot_a"], "tme_class_id": [0]}
    )
    monkeypatch.setattr(
        foundation,
        "_embedding_cache_path",
        lambda *_args, **_kwargs: tmp_path / "unused-cache.npz",
    )
    monkeypatch.setattr(
        foundation,
        "load_slide_patches",
        lambda *_args, **_kwargs: (
            np.ones((1, 3, 8, 8), dtype=np.float32),
            labels_frame,
        ),
    )
    monkeypatch.setattr(
        foundation,
        "extract_frozen_embeddings",
        lambda *_args, **_kwargs: np.ones((1, spec.embedding_dim), dtype=np.float32),
    )
    embeddings, aligned = foundation.load_or_extract_slide_embeddings(
        "slide_a",
        labels_frame,
        cfg=no_cache_cfg,
        encoder_bundle=(model, device, spec),
    )
    assert embeddings.shape == (1, spec.embedding_dim)
    assert aligned.equals(labels_frame)
    assert not (tmp_path / "unused-cache.npz").exists()

    report_cfg = copy.deepcopy(cfg)
    report_cfg["experiment"] = "explicit_valid"
    monkeypatch.setattr(evaluation, "load_config", _forbidden)
    monkeypatch.setattr(evaluation, "pharma_outputs_dir", lambda: tmp_path)
    report = evaluation.save_benchmark_report(
        [
            {
                "model": "cnn",
                "fold": 0,
                "val_slide": "slide_a",
                "balanced_accuracy": 0.5,
                "macro_f1": 0.4,
                "mean_pearson_r": 0.3,
                "mean_r2": 0.2,
            }
        ],
        cfg=report_cfg,
        upstream_lineage={"test_parent": "current"},
    )
    assert report.name == "benchmark_report_explicit_valid.csv"
    assert pd.read_csv(report)["experiment"].tolist() == ["explicit_valid"]


def test_foundation_empty_slide_embeddings_fail_before_probe(monkeypatch) -> None:
    non_empty_labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_b"],
            "spot_id": ["a0", "b0"],
            "tme_class_id": [0, 1],
            "module_signal": [0.1, 0.9],
        }
    )
    monkeypatch.setattr(foundation, "load_frozen_encoder", lambda _cfg: object())
    monkeypatch.setattr(
        foundation,
        "load_or_extract_slide_embeddings",
        lambda *_args, **_kwargs: (
            np.zeros((0, 4), dtype=np.float32),
            non_empty_labels.iloc[:0],
        ),
    )
    monkeypatch.setattr(foundation, "train_eval_linear_probe", _forbidden)

    with pytest.raises(StageValidationError, match="foundation_loso_embedding"):
        foundation.run_foundation_loso(
            ["slide_a", "slide_b"], non_empty_labels, cfg={"seed": 0}
        )


def test_foundation_probe_empty_inputs_fail_before_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(foundation, "make_pipeline", _forbidden)

    with pytest.raises(StageValidationError, match="foundation_probe_training"):
        foundation.train_eval_linear_probe(
            np.zeros((0, 4), dtype=np.float32),
            _fold_labels().iloc[:0],
            np.ones((1, 4), dtype=np.float32),
            _fold_labels().iloc[[0]],
            cfg=_fold_cfg(),
        )


def test_foundation_task_filter_rejects_zero_retained_rows() -> None:
    embeddings = np.ones((2, 4), dtype=np.float32)
    task_labels = pd.DataFrame({"tme_class": ["out_of_scope", "unknown"]})

    with pytest.raises(StageValidationError, match="foundation_task_filter") as caught:
        foundation_eval.prepare_classification_task(
            embeddings, task_labels, "confident_3class"
        )

    assert caught.value.observed == 0
    with pytest.raises(ValueError, match="Unknown task"):
        foundation_eval.prepare_classification_task(
            embeddings, task_labels, "not_a_task"
        )


def test_nested_loso_empty_cohort_fails_before_probe(monkeypatch) -> None:
    monkeypatch.setattr(foundation_eval, "_fit_probe", _forbidden)

    with pytest.raises(StageValidationError, match="nested_loso_admission"):
        foundation_eval.nested_loso_classification(
            [], {}, task="confident_3class"
        )


def test_nested_loso_requires_three_slides_before_task_preprocessing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(foundation_eval, "prepare_classification_task", _forbidden)
    monkeypatch.setattr(foundation_eval, "_fit_probe", _forbidden)

    with pytest.raises(StageValidationError, match="nested_loso_admission") as caught:
        foundation_eval.nested_loso_classification(
            ["slide_a", "slide_b"],
            {"slide_a": object(), "slide_b": object()},  # type: ignore[arg-type]
            task="confident_3class",
        )
    assert caught.value.minimum == 3
