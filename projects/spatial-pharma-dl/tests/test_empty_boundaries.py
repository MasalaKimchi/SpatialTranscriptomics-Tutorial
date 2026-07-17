"""Offline evidence that empty stage inputs fail before expensive work."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import benchmark, data, eval as evaluation, labels, patches, train
from src.validation import StageValidationError, require_non_empty

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
    monkeypatch.setattr(labels, "pharma_outputs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        labels,
        "build_labels_for_slide",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _forbidden)

    with pytest.raises(StageValidationError, match="label rows for slide slide_a"):
        labels.build_labels_cohort(["slide_a"], cfg={"unused": True})


def test_label_slide_generation_rejects_zero_rows(monkeypatch) -> None:
    empty_adata = SimpleNamespace(
        obs_names=pd.Index([], dtype=object),
        obs=pd.DataFrame({"clusters": pd.Series(dtype=object)}),
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
            SimpleNamespace(),
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

    with pytest.raises(StageValidationError, match="patch_label_alignment") as caught:
        train.load_slide_patches("slide_a", label_frame, cfg={})

    assert caught.value.observed == 0


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
