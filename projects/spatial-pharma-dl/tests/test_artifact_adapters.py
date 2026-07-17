"""Offline production-reader evidence for durable scientific artifacts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import data, eval as evaluation, foundation, labels, models, patches
from src.validation import (
    PreprocessingManifest,
    admit_run,
    finalize_preprocessing_resolution,
    resolve_post_qc_preprocessing,
)
from utils import st_helpers as st
from utils.artifacts import ArtifactValidationError, manifest_path

pytestmark = pytest.mark.offline


def _make_h5ad_writable(adata):
    for frame in (adata.obs, adata.var):
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.StringDtype) or isinstance(
                frame[column].dtype, pd.CategoricalDtype
            ):
                frame[column] = pd.Series(
                    pd.array(frame[column].tolist(), dtype=object),
                    index=frame.index,
                    dtype=object,
                )
    adata.obs_names = pd.Index(
        pd.array(adata.obs_names.tolist(), dtype=object), dtype=object
    )
    adata.var_names = pd.Index(
        pd.array(adata.var_names.tolist(), dtype=object), dtype=object
    )
    return adata


def _processed_adata(factory, slide_id: str = "slide_a"):
    adata = _make_h5ad_writable(factory(n_spots=4, n_genes=4))
    adata.obs["slide_id"] = pd.Series(
        pd.array([slide_id] * 4, dtype=object), index=adata.obs.index, dtype=object
    )
    adata.obs["clusters"] = pd.Series(
        pd.array(["0"] * 4, dtype=object), index=adata.obs.index, dtype=object
    )
    adata.obsm["X_pca"] = np.zeros((4, 2), dtype=np.float32)
    record = finalize_preprocessing_resolution(
        resolve_post_qc_preprocessing(
            slide_id=slide_id,
            input_spots=4,
            input_genes=4,
            after_filter_cells_spots=4,
            after_filter_cells_genes=4,
            after_filter_genes_spots=4,
            after_filter_genes_genes=4,
            post_qc_spots=4,
            post_qc_genes=4,
            requested_hvg=3,
            requested_pcs=2,
            requested_neighbors=3,
            requested_graph_pcs=2,
        ),
        actual_hvgs=3,
    ).to_dict()
    import json

    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    adata.uns["spatial_pharma_preprocessing"] = record
    adata.uns["spatial_pharma_preprocessing_canonical_json"] = canonical
    return adata


def test_real_nineteen_artifact_pipeline_uses_production_adapters(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    """Round-trip every retained production artifact through its real adapter."""
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    lineage = {"admitted_run": "current"}
    paths = []

    root = _make_h5ad_writable(synthetic_anndata_factory(n_spots=4, n_genes=4))
    root.obs["total_counts"] = np.asarray([10, 11, 12, 13], dtype=np.int64)
    root.obs["n_genes_by_counts"] = np.asarray([3, 3, 4, 4], dtype=np.int64)
    root.obs["pct_counts_mt"] = np.asarray([1.0, 2.0, 3.0, 4.0])
    root.obs["clusters"] = pd.Series(
        pd.array(["0"] * 4, dtype=object), index=root.obs.index, dtype=object
    )
    root.obsm["X_pca"] = np.zeros((4, 2), dtype=np.float32)
    for filename in (
        "adata_raw.h5ad",
        "adata_qc.h5ad",
        "adata_clustered.h5ad",
        "adata_features.h5ad",
    ):
        paths.append(st.save_adata(root, filename))
        assert st.load_adata(filename).shape == (4, 4)

    processed = _processed_adata(synthetic_anndata_factory)
    paths.append(data.save_slide(processed, "slide_a", cfg=cfg))
    assert data.load_slide("slide_a", cfg=cfg).n_obs == 4

    label_frame = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["spot_0", "spot_1"],
            "cluster": ["0", "0"],
            "cluster_id": [0, 0],
            "domain_name": ["domain_0", "domain_0"],
            "tme_class": ["other", "other"],
            "tme_class_id": [0, 0],
        }
    )
    paths.append(labels.save_label_table(label_frame, "slide_a", cfg=cfg))
    pd.testing.assert_frame_equal(labels.load_label_table("slide_a", cfg=cfg), label_frame)
    domain_frame = label_frame[
        ["slide_id", "cluster", "domain_name", "tme_class"]
    ].drop_duplicates(ignore_index=True)
    paths.append(labels.save_domain_table(domain_frame, ["slide_a"], cfg=cfg))
    pd.testing.assert_frame_equal(
        labels.load_domain_table(["slide_a"], cfg=cfg), domain_frame
    )

    patch_meta = label_frame[["slide_id", "spot_id"]].assign(
        x=[1.0, 2.0], y=[3.0, 4.0], native_patch_px=[8, 8]
    )
    patch_values = np.zeros((2, 3, 4, 4), dtype=np.float32)
    paths.append(
        patches.save_patch_arrays("slide_a", patch_values, patch_meta, cfg=cfg)
    )
    np.testing.assert_array_equal(
        patches.load_patch_arrays("slide_a", cfg=cfg)[0], patch_values
    )
    paths.append(patches.save_patch_index(label_frame, cfg=cfg))
    pd.testing.assert_frame_equal(
        patches.load_patch_index(cfg=cfg, sample_ids=["slide_a"]), label_frame
    )

    spec = foundation.FoundationModelSpec(
        repo_id="local/test",
        backend="test",
        license="test",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        embedding_dim=2,
    )
    monkeypatch.setitem(foundation.FOUNDATION_MODELS, "kaiko_vits16", spec)
    monkeypatch.setattr(
        foundation,
        "load_slide_patches",
        lambda *_args, **_kwargs: (
            patch_values,
            label_frame[["slide_id", "spot_id"]].assign(_patch_source_row=[0, 1]),
        ),
    )
    expected_embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    monkeypatch.setattr(
        foundation,
        "extract_frozen_embeddings",
        lambda *_args, **_kwargs: expected_embeddings.copy(),
    )
    bundle = (object(), "cpu", spec)
    foundation.load_or_extract_slide_embeddings(
        "slide_a", label_frame, cfg=cfg, encoder_bundle=bundle
    )
    embedding_path = foundation._embedding_cache_path("slide_a", cfg)
    paths.append(embedding_path)
    cached, _ = foundation.load_or_extract_slide_embeddings(
        "slide_a", label_frame, cfg=cfg, encoder_bundle=bundle
    )
    np.testing.assert_array_equal(cached, expected_embeddings)

    checkpoint_path = tmp_path / "outputs" / "model.pt"
    checkpoint_lineage = {"patch": "current", "labels": "current"}
    models.save_model_checkpoint(
        checkpoint_path,
        model=torch.nn.Sequential(torch.nn.Linear(3, 2)),
        metadata={
            "model_name": "resnet18",
            "experiment": "v2",
            "pretrained": False,
            "n_classes": 2,
            "n_reg_targets": 1,
            "cls_col": "tme_class_id",
            "reg_cols": ["gene_A"],
            "val_slide": "slide_b",
            "train_slides": ["slide_a"],
            "fold": 0,
        },
        cfg=cfg,
        upstream_lineage=checkpoint_lineage,
    )
    paths.append(checkpoint_path)
    assert models.load_local_checkpoint_payload(
        checkpoint_path, cfg=cfg, upstream_lineage=checkpoint_lineage
    )["fold"] == 0

    report_path = tmp_path / "outputs" / "benchmark.csv"
    report_rows = [
        {
            "model": "cnn",
            "fold": 0,
            "val_slide": "slide_a",
            "balanced_accuracy": 0.5,
            "macro_f1": 0.4,
            "mean_pearson_r": 0.2,
            "mean_r2": 0.1,
        }
    ]
    paths.append(
        evaluation.save_benchmark_report(
            report_rows, report_path, cfg=cfg, upstream_lineage=lineage
        )
    )
    assert len(
        evaluation.load_benchmark_report(
            report_path,
            cfg=cfg,
            upstream_lineage=lineage,
            expected_row_identity=[
                [cfg.get("experiment", "v2"), "cnn", 0, "slide_a"]
            ],
        )
    ) == 1

    configured_ids = [
        slide_id
        for cohort in ("oncology", "external", "benchmark")
        for slide_id in cfg["cohorts"][cohort]
    ]
    cohort_value = admit_run(cfg, available_slide_ids=configured_ids).manifest.to_dict()
    preprocessing_value = PreprocessingManifest(
        slide_ids=["slide_a"],
        records=[processed.uns["spatial_pharma_preprocessing"]],
    ).to_dict()
    json_specs = (
        ("cohort_manifest", "cohort_manifest", cohort_value),
        ("preprocessing_manifest", "preprocessing_manifest", preprocessing_value),
        (
            "experiment_summary",
            "summary",
            {
                "experiment": "v2",
                "classification_col": "tme_class_id",
                "regression_targets": ["gene_A"],
                "context_scale": 2.0,
                "patch_version": "v1",
                "cnn_mean_balanced_accuracy": 0.5,
                "cnn_mean_pearson_r": 0.2,
                "rf_mean_balanced_accuracy": 0.4,
                "rf_mean_pearson_r": 0.1,
            },
        ),
    )
    for result_name, artifact_kind, value in json_specs:
        path = tmp_path / "outputs" / f"{result_name}.json"
        paths.append(
            evaluation.save_json_result(
                value,
                path,
                result_name=result_name,
                artifact_kind=artifact_kind,
                cfg=cfg,
                upstream_lineage=lineage,
            )
        )
        assert evaluation.load_json_result(
            path,
            result_name=result_name,
            artifact_kind=artifact_kind,
            cfg=cfg,
            upstream_lineage=lineage,
            expected_value=value,
        ) == value

    table_specs = {
        "cohort_summary": pd.DataFrame(
            {
                "slide_id": ["slide_a"],
                "n_spots": [4],
                "n_genes": [4],
                "n_clusters": [1],
                "median_genes": [3.5],
                "median_counts": [11.5],
                "median_mito_pct": [2.5],
            }
        ),
        "training_history": pd.DataFrame(
            {
                "fold": [0],
                "val_slide": ["slide_a"],
                "epoch": [0],
                "train_loss": [1.0],
                "val_loss": [1.1],
            }
        ),
        "nested_loso_results": pd.DataFrame(
            {
                "model": ["cnn"],
                "fold": [0],
                "held_out_slide": ["slide_a"],
                "task": ["classification"],
                "n_train": [2],
                "n_test": [1],
                "coverage": [1.0],
                "selected_candidate": ["base"],
                "inner_macro_f1": [0.5],
                "accuracy": [0.5],
                "macro_f1": [0.5],
                "balanced_accuracy": [0.5],
                "majority_macro_f1": [0.3],
                "majority_balanced_accuracy": [0.5],
            }
        ),
        "model_task_summary": pd.DataFrame(
            {
                "task": ["classification"],
                "model": ["cnn"],
                "mean_macro_f1": [0.5],
                "min_macro_f1": [0.5],
                "mean_balanced_accuracy": [0.5],
                "majority_macro_f1": [0.3],
                "mean_coverage": [1.0],
                "f1_lift_vs_majority": [0.2],
            }
        ),
    }
    for table_name, frame in table_specs.items():
        path = tmp_path / "outputs" / f"{table_name}.csv"
        paths.append(
            evaluation.save_result_table(
                frame,
                path,
                table_name=table_name,
                cfg=cfg,
                upstream_lineage=lineage,
            )
        )
        assert len(
            evaluation.load_result_table(
                path,
                table_name=table_name,
                cfg=cfg,
                upstream_lineage=lineage,
                expected_rows=1,
            )
        ) == 1

    assert len(paths) == 19
    assert len(set(paths)) == 19
    assert all(path.is_file() and manifest_path(path).is_file() for path in paths)


def test_root_h5ad_round_trip_uses_additive_manifest(tmp_path, monkeypatch, synthetic_anndata_factory):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    adata = _make_h5ad_writable(synthetic_anndata_factory(n_spots=3, n_genes=3))
    path = st.save_adata(adata, "adata_raw.h5ad")

    restored = st.load_adata("adata_raw.h5ad")

    assert restored.shape == adata.shape
    assert path.name == "adata_raw.h5ad"
    assert manifest_path(path).is_file()


def test_processed_h5ad_round_trip_and_wrong_identity_rejection(tmp_path, monkeypatch, synthetic_anndata_factory):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    adata = _processed_adata(synthetic_anndata_factory)
    path = data.save_slide(adata, "slide_a", cfg=cfg)
    assert data.load_slide("slide_a", cfg=cfg).obs["slide_id"].tolist() == ["slide_a"] * 4

    adata.obs["slide_id"] = "slide_b"
    with pytest.raises(ArtifactValidationError):
        data.save_slide(adata, "slide_a", cfg=cfg)
    assert manifest_path(path).is_file()


def test_label_and_domain_tables_round_trip_with_processed_lineage(tmp_path, monkeypatch, synthetic_anndata_factory):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    adata = _processed_adata(synthetic_anndata_factory)
    data.save_slide(adata, "slide_a", cfg=cfg)
    frame = pd.DataFrame(
        {
            "slide_id": ["slide_a"],
            "spot_id": ["spot_0"],
            "cluster": ["0"],
            "cluster_id": [0],
            "domain_name": ["domain_0"],
            "tme_class": ["other"],
            "tme_class_id": [0],
        }
    )
    label_path = labels.save_label_table(frame, "slide_a", cfg=cfg)
    assert labels.load_label_table("slide_a", cfg=cfg).equals(frame)
    domain = frame[["slide_id", "cluster", "domain_name", "tme_class"]]
    domain_path = labels.save_domain_table(domain, ["slide_a"], cfg=cfg)
    assert labels.load_domain_table(["slide_a"], cfg=cfg).equals(domain)
    assert manifest_path(label_path).is_file() and manifest_path(domain_path).is_file()


def test_trusted_local_patch_and_index_round_trip(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    data.save_slide(_processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg)
    values = np.zeros((2, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["spot_0", "spot_1"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "native_patch_px": [8, 8],
        }
    )
    path = patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)
    restored, restored_meta = patches.load_patch_arrays("slide_a", cfg=cfg)
    np.testing.assert_array_equal(restored, values)
    pd.testing.assert_frame_equal(restored_meta, meta)
    assert manifest_path(path).is_file()

    original_decoder = patches._read_trusted_local_patch_npz

    def aba_decoder(snapshot, slide_id):
        held = path.with_name("held-patch.npz")
        path.rename(held)
        path.write_bytes(b"unadmitted-patch-bytes")
        path.unlink()
        held.rename(path)
        return original_decoder(snapshot, slide_id)

    monkeypatch.setattr(patches, "_read_trusted_local_patch_npz", aba_decoder)
    np.testing.assert_array_equal(
        patches.load_patch_arrays("slide_a", cfg=cfg)[0], values
    )

    index = meta[["slide_id", "spot_id"]].assign(
        cluster=["0", "0"],
        cluster_id=[0, 0],
        domain_name=["domain_0", "domain_0"],
        tme_class=["other", "other"],
        tme_class_id=[0, 0],
    )
    labels.save_label_table(index, "slide_a", cfg=cfg)
    index_path = patches.save_patch_index(index, cfg=cfg)
    pd.testing.assert_frame_equal(
        patches.load_patch_index(cfg=cfg, sample_ids=["slide_a"]), index
    )
    assert manifest_path(index_path).is_file()


def test_patch_path_resolution_is_pure(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    path = patches.patch_cache_path("slide_a", data.load_config())
    assert not path.parent.exists()


def test_embedding_cache_round_trip_is_safe_primitive_npz(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    data.save_slide(_processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg)
    spec = foundation.FoundationModelSpec(
        repo_id="local/test",
        backend="test",
        license="test",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        embedding_dim=2,
    )
    monkeypatch.setitem(foundation.FOUNDATION_MODELS, "kaiko_vits16", spec)
    label_rows = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["spot_0", "spot_1"],
        }
    )
    patch_rows = label_rows.assign(x=[0.0, 1.0], y=[0.0, 1.0], native_patch_px=[8, 8])
    patch_values = np.zeros((2, 3, 4, 4), dtype=np.float32)
    patches.save_patch_arrays("slide_a", patch_values, patch_rows, cfg=cfg)
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    monkeypatch.setattr(
        foundation,
        "load_slide_patches",
        lambda *_args, **_kwargs: (patch_values, label_rows.assign(_patch_source_row=[0, 1])),
    )
    monkeypatch.setattr(
        foundation,
        "extract_frozen_embeddings",
        lambda *_args, **_kwargs: expected.copy(),
    )
    bundle = (object(), "cpu", spec)
    miss, _ = foundation.load_or_extract_slide_embeddings(
        "slide_a", label_rows, cfg=cfg, encoder_bundle=bundle
    )
    hit, _ = foundation.load_or_extract_slide_embeddings(
        "slide_a", label_rows, cfg=cfg, encoder_bundle=bundle
    )
    np.testing.assert_array_equal(miss, expected)
    np.testing.assert_array_equal(hit, expected)
    assert manifest_path(foundation._embedding_cache_path("slide_a", cfg)).is_file()


def test_benchmark_report_round_trip_and_schema_rejection(tmp_path):
    cfg = data.load_config()
    rows = [{
        "model": "cnn", "fold": 0, "val_slide": "slide_a",
        "balanced_accuracy": 0.5, "macro_f1": 0.4,
        "mean_pearson_r": 0.3, "mean_r2": 0.2,
    }]
    path = tmp_path / "benchmark_report_v2.csv"
    evaluation.save_benchmark_report(
        rows, path=path, cfg=cfg, upstream_lineage={"checkpoints": ["abc"]}
    )
    restored = evaluation.load_benchmark_report(
        path,
        cfg=cfg,
        upstream_lineage={"checkpoints": ["abc"]},
        expected_row_identity=[["v2_remediation", "cnn", 0, "slide_a"]],
    )
    assert restored.columns.tolist() == [
        "model", "fold", "val_slide", "balanced_accuracy", "macro_f1",
        "mean_pearson_r", "mean_r2", "experiment",
    ]
    assert restored.iloc[0]["model"] == "cnn"
    assert manifest_path(path).is_file()


def test_named_summary_table_and_json_round_trip(tmp_path):
    cfg = data.load_config()
    table = pd.DataFrame(
        {
            "fold": [0],
            "val_slide": ["slide_a"],
            "epoch": [0],
            "train_loss": [1.0],
            "val_loss": [2.0],
        }
    )
    table_path = tmp_path / "training_history.csv"
    evaluation.save_result_table(
        table, table_path, table_name="training_history", cfg=cfg,
        upstream_lineage={"checkpoints": ["abc"]},
    )
    pd.testing.assert_frame_equal(
        evaluation.load_result_table(
            table_path,
            table_name="training_history",
            cfg=cfg,
            upstream_lineage={"checkpoints": ["abc"]},
            expected_rows=1,
        ),
        table,
    )
    json_path = tmp_path / "experiment_v2_summary.json"
    payload = {
        "experiment": "v2",
        "classification_col": "tme_class_id",
        "regression_targets": ["gene_A"],
        "context_scale": 1.0,
        "patch_version": "v1",
        "cnn_mean_balanced_accuracy": 0.5,
        "cnn_mean_pearson_r": 0.4,
        "rf_mean_balanced_accuracy": 0.3,
        "rf_mean_pearson_r": 0.2,
    }
    evaluation.save_json_result(
        payload, json_path, result_name="experiment_summary", cfg=cfg,
        upstream_lineage={"report": "abc"},
    )
    assert evaluation.load_json_result(
        json_path,
        result_name="experiment_summary",
        cfg=cfg,
        upstream_lineage={"report": "abc"},
        expected_value=payload,
    ) == payload


def test_report_and_result_readers_require_current_independent_lineage(tmp_path):
    cfg = data.load_config()
    report_path = tmp_path / "benchmark.csv"
    rows = [{
        "model": "cnn", "fold": 0, "val_slide": "slide_a",
        "balanced_accuracy": 0.5, "macro_f1": 0.4,
        "mean_pearson_r": 0.3, "mean_r2": 0.2,
    }]
    evaluation.save_benchmark_report(
        rows, path=report_path, cfg=cfg, upstream_lineage={"checkpoint": "current"}
    )
    identity = [["v2_remediation", "cnn", 0, "slide_a"]]
    with pytest.raises(ArtifactValidationError, match="missing_expected_lineage"):
        evaluation.load_benchmark_report(
            report_path, cfg=cfg, expected_row_identity=identity
        )
    with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
        evaluation.load_benchmark_report(
            report_path,
            cfg=cfg,
            upstream_lineage={"checkpoint": "stale"},
            expected_row_identity=identity,
        )


def test_named_schema_registry_rejects_unknown_and_malformed_payloads(tmp_path):
    cfg = data.load_config()
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        evaluation.save_result_table(
            pd.DataFrame({"totally_wrong": [1]}),
            tmp_path / "training_history.csv",
            table_name="training_history",
            cfg=cfg,
            upstream_lineage={"checkpoint": "current"},
        )
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        evaluation.save_result_table(
            pd.DataFrame({"value": [1]}),
            tmp_path / "unknown.csv",
            table_name="unknown_table",
            cfg=cfg,
            upstream_lineage={"parent": "current"},
        )
    with pytest.raises(ArtifactValidationError):
        evaluation.save_json_result(
            {"nonsense": True},
            tmp_path / "cohort_manifest.json",
            result_name="cohort_manifest",
            cfg=cfg,
            upstream_lineage={"admission": "current"},
            artifact_kind="cohort_manifest",
        )


def test_patch_lineage_uses_actual_partial_cohort_reference_and_index_parents(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    cfg["cohorts"]["oncology"] = ["missing_slide", "slide_a"]
    cfg["patches"]["per_slide_stain_norm"] = False
    data.save_slide(_processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg)
    context = patches._patch_artifact_context("slide_a", cfg)
    assert context["stain_reference"]["slide_id"] == "slide_a"

    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "slide_id": ["slide_a"],
            "spot_id": ["spot_0"],
            "x": [1.0],
            "y": [2.0],
            "native_patch_px": [8],
        }
    )
    patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)
    index = meta[["slide_id", "spot_id"]].assign(
        cluster=["0"], cluster_id=[0], domain_name=["domain_0"],
        tme_class=["other"], tme_class_id=[0],
    )
    with pytest.raises(ArtifactValidationError, match="missing_payload"):
        patches.save_patch_index(index, cfg=cfg)
    labels.save_label_table(index, "slide_a", cfg=cfg)
    path = patches.save_patch_index(index, cfg=cfg)
    declared = patches.read_artifact_manifest(path).fingerprint.to_dict()["inputs"]
    assert list(declared["source"]["label_manifests"]) == ["slide_a"]
    assert list(declared["upstream"]["patch_manifests"]) == ["slide_a"]


def test_public_result_writers_reject_missing_lineage_before_filesystem_side_effects(
    tmp_path,
):
    cfg = data.load_config()
    table = pd.DataFrame(
        {
            "fold": [0], "val_slide": ["slide_a"], "epoch": [0],
            "train_loss": [1.0], "val_loss": [1.0],
        }
    )
    targets = [
        lambda path: evaluation.save_benchmark_report(
            [{
                "model": "cnn", "fold": 0, "val_slide": "slide_a",
                "balanced_accuracy": 0.5, "macro_f1": 0.5,
                "mean_pearson_r": 0.0, "mean_r2": 0.0,
            }],
            path=path,
            cfg=cfg,
        ),
        lambda path: evaluation.save_result_table(
            table, path, table_name="training_history", cfg=cfg
        ),
        lambda path: evaluation.save_json_result(
            {
                "experiment": "v2", "classification_col": "tme_class_id",
                "regression_targets": ["gene_A"], "context_scale": 1.0,
                "patch_version": "v1", "cnn_mean_balanced_accuracy": 0.5,
                "cnn_mean_pearson_r": 0.0, "rf_mean_balanced_accuracy": 0.5,
                "rf_mean_pearson_r": 0.0,
            },
            path,
            result_name="experiment_summary",
            cfg=cfg,
        ),
    ]
    for index, writer in enumerate(targets):
        path = tmp_path / f"missing-{index}" / "result"
        with pytest.raises(ArtifactValidationError, match="missing_expected_lineage"):
            writer(path)
        assert not path.parent.exists()


def _rebind_manifest_to_payload(path):
    sidecar = manifest_path(path)
    tree = json.loads(sidecar.read_text(encoding="utf-8"))
    raw = path.read_bytes()
    tree["payload"]["byte_count"] = len(raw)
    tree["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
    sidecar.write_text(
        json.dumps(tree, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_patch_requires_fully_admitted_processed_parent(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    processed = data.save_slide(
        _processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg
    )
    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["spot_0"], "x": [1.0],
         "y": [2.0], "native_patch_px": [8]}
    )

    original_payload = processed.read_bytes()
    processed.write_bytes(b"corrupt")
    with pytest.raises(ArtifactValidationError, match="byte_count_mismatch|checksum_mismatch"):
        patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)

    processed.write_bytes(original_payload)
    processed.unlink()
    with pytest.raises(FileNotFoundError):
        patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)

    data.save_slide(_processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg)
    processed.write_bytes(b"not-an-h5ad")
    _rebind_manifest_to_payload(processed)
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)


def test_patch_index_requires_admitted_label_and_patch_parents_and_invalidates_mixed_generation(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    data.save_slide(_processed_adata(synthetic_anndata_factory), "slide_a", cfg=cfg)
    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["spot_0"], "x": [1.0],
         "y": [2.0], "native_patch_px": [8]}
    )
    patch_path = patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)
    index = meta[["slide_id", "spot_id"]].assign(
        cluster=["0"], cluster_id=[0], domain_name=["domain_0"],
        tme_class=["other"], tme_class_id=[0],
    )
    label_path = labels.save_label_table(index, "slide_a", cfg=cfg)
    index_path = patches.save_patch_index(index, cfg=cfg)

    label_payload = label_path.read_bytes()
    label_path.unlink()
    with pytest.raises(ArtifactValidationError, match="missing_payload"):
        patches.save_patch_index(index, cfg=cfg)
    label_path.write_bytes(label_payload)

    patch_payload = patch_path.read_bytes()
    patch_path.write_bytes(b"corrupt-patch")
    with pytest.raises(ArtifactValidationError, match="byte_count_mismatch|checksum_mismatch"):
        patches.save_patch_index(index, cfg=cfg)
    patch_path.write_bytes(patch_payload)

    label_path.write_bytes(b"not-parquet")
    _rebind_manifest_to_payload(label_path)
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        patches.save_patch_index(index, cfg=cfg)

    labels.save_label_table(index, "slide_a", cfg=cfg)
    labels.save_label_table(index.assign(gene_A=[1.0]), "slide_a", cfg=cfg)
    with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
        patches.load_patch_index(index_path, cfg=cfg, sample_ids=["slide_a"])


def test_shared_patch_lineage_records_actual_supplied_reference(
    tmp_path, monkeypatch, synthetic_anndata_factory
):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    cfg["cohorts"]["oncology"] = ["slide_a", "slide_b"]
    cfg["patches"]["per_slide_stain_norm"] = False
    for slide_id in ("slide_a", "slide_b"):
        data.save_slide(
            _processed_adata(synthetic_anndata_factory, slide_id), slide_id, cfg=cfg
        )
    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["spot_0"], "x": [1.0],
         "y": [2.0], "native_patch_px": [8]}
    )
    path = patches.save_patch_arrays(
        "slide_a", values, meta, cfg=cfg, reference_slide_id="slide_b"
    )
    inputs = patches.read_artifact_manifest(path).fingerprint.to_dict()["inputs"]
    assert inputs["source"]["stain_reference"]["slide_id"] == "slide_b"


def test_named_production_schemas_reject_overlapping_partitions_and_metric_ranges():
    overlap = {
        "schema_version": "cohort-manifest-v1",
        "allow_partial": True,
        "configured": [{
            "slide_id": "a", "cohort": "oncology", "status": "configured",
            "reason_code": None, "reason": None,
        }],
        "included": [{
            "slide_id": "a", "cohort": "oncology", "status": "included",
            "reason_code": None, "reason": None,
        }],
        "skipped": [{
            "slide_id": "a", "cohort": "oncology", "status": "skipped",
            "reason_code": "missing_processed_slide", "reason": "missing",
        }],
        "failed": [],
    }
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        evaluation._json_payload(overlap, "cohort_manifest")

    invalid_metrics = pd.DataFrame(
        [{
            "model": "kaiko_vits16", "fold": 0, "held_out_slide": "slide_a",
            "task": "all_4class", "n_train": 10, "n_test": 5,
            "coverage": 1.0, "selected_candidate": "ridge",
            "inner_macro_f1": 999.0, "accuracy": 999.0, "macro_f1": 999.0,
            "balanced_accuracy": 999.0, "majority_macro_f1": 999.0,
            "majority_balanced_accuracy": 999.0,
        }]
    )
    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        evaluation._result_table_schema(invalid_metrics, "nested_loso_results")
