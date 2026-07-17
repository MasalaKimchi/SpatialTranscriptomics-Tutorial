"""Offline production-reader evidence for durable scientific artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data, eval as evaluation, foundation, labels, patches
from src.validation import finalize_preprocessing_resolution, resolve_post_qc_preprocessing
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
