"""Offline production-reader evidence for durable scientific artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data, labels
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
