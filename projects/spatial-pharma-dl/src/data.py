"""Data loading, preprocessing, and cohort management for Spatial Pharma DL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pandas as pd
import yaml
from requests.exceptions import RequestException

from . import bootstrap  # noqa: F401 — ensures repo root is on sys.path
from .identity import validate_anndata_spot_identity
from .validation import (
    ConfigValidationError,
    ValidationIssue,
    finalize_preprocessing_resolution,
    require_non_empty,
    resolve_config,
    resolve_post_qc_preprocessing,
)
from utils import st_helpers as st


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class SourceAcquisitionError(RuntimeError):
    """A documented remote dataset acquisition failure safe for admission policy."""


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and strictly resolve YAML config from configs/default.yaml."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(
                (
                    ValidationIssue(
                        "config",
                        str(exc),
                        "valid YAML whose root is a mapping",
                        "Correct the YAML syntax before starting the pipeline.",
                    ),
                )
            ) from exc
    return resolve_config(raw).to_dict()


def cohort_slide_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return all slide ids across oncology, external, and benchmark cohorts."""
    if cfg is None:
        cfg = load_config()
    else:
        cfg = resolve_config(cfg).to_dict()
    cohorts = cfg["cohorts"]
    sample_ids = cohorts["oncology"] + cohorts["external"] + cohorts["benchmark"]
    require_non_empty(
        sample_ids,
        stage="cohort_configuration",
        subject="configured slide sequence",
        guidance="Configure at least one slide in cohorts before starting the pipeline.",
    )
    return sample_ids


def pharma_processed_dir() -> Path:
    """Return data/processed/pharma/ (created if missing)."""
    d = st.processed_dir() / "pharma"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pharma_outputs_dir() -> Path:
    """Return outputs/pharma/ (created if missing)."""
    d = st.outputs_dir() / "pharma"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_filename(sample_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", sample_id)


def available_processed_slide_ids(sample_ids: list[str]) -> set[str]:
    """Return cached slide IDs without creating or opening any path."""
    base = st.project_root() / "data" / "processed" / "pharma"
    return {
        sample_id
        for sample_id in sample_ids
        if (base / f"{safe_filename(sample_id)}_clustered.h5ad").is_file()
    }


def _mito_prefix(adata) -> str:
    """Return 'mt-' for mouse or 'MT-' for human based on gene names."""
    if any(g.startswith("MT-") for g in adata.var_names[:500]):
        return "MT-"
    return "mt-"


def preprocess_slide(
    adata,
    sample_id: str,
    cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    """Run tutorial-equivalent QC + clustering pipeline on one slide."""
    if cfg is None:
        cfg = load_config()
    else:
        cfg = resolve_config(cfg).to_dict()
    validate_anndata_spot_identity(
        adata,
        sample_id,
        stage="preprocess_slide_source_identity",
    )

    import scanpy as sc

    prep = cfg["preprocessing"]
    seed = seed if seed is not None else cfg.get("seed", st.SEED)
    st.set_seeds(seed)

    adata = adata.copy()
    input_spots, input_genes = int(adata.n_obs), int(adata.n_vars)
    prefix = _mito_prefix(adata)
    adata.var["mt"] = adata.var_names.str.startswith(prefix)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )
    sc.pp.filter_cells(adata, min_counts=prep["min_counts"])
    after_filter_cells_spots = int(adata.n_obs)
    after_filter_cells_genes = int(adata.n_vars)
    sc.pp.filter_genes(adata, min_cells=prep["min_cells"])
    after_filter_genes_spots = int(adata.n_obs)
    after_filter_genes_genes = int(adata.n_vars)
    adata = adata[adata.obs["pct_counts_mt"] < prep["max_pct_mito"]].copy()
    post_qc_spots, post_qc_genes = int(adata.n_obs), int(adata.n_vars)

    resolution = resolve_post_qc_preprocessing(
        slide_id=sample_id,
        input_spots=input_spots,
        input_genes=input_genes,
        after_filter_cells_spots=after_filter_cells_spots,
        after_filter_cells_genes=after_filter_cells_genes,
        after_filter_genes_spots=after_filter_genes_spots,
        after_filter_genes_genes=after_filter_genes_genes,
        post_qc_spots=post_qc_spots,
        post_qc_genes=post_qc_genes,
        requested_hvg=prep["n_top_genes_hvg"],
        requested_pcs=prep["n_pcs"],
        requested_neighbors=prep["n_neighbors"],
        requested_graph_pcs=prep["n_pcs_neighbors"],
    )

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=resolution.to_dict()["resolved"]["hvg_call"],
        flavor="seurat",
    )
    actual_hvgs = int(adata.var["highly_variable"].sum())
    resolution = finalize_preprocessing_resolution(
        resolution,
        actual_hvgs=actual_hvgs,
    )
    resolved = resolution.to_dict()["resolved"]
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.pp.pca(adata_hvg, n_comps=resolved["pca"], random_state=seed)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]

    sc.pp.neighbors(
        adata,
        n_neighbors=resolved["neighbors"],
        n_pcs=resolved["graph_pcs"],
        random_state=seed,
    )
    sc.tl.umap(adata, random_state=seed)
    st.run_leiden(
        adata,
        resolution=prep["leiden_resolution"],
        key_added="clusters",
        seed=seed,
    )
    adata.obs["slide_id"] = sample_id
    adata.uns["spatial_pharma_preprocessing"] = resolution.to_dict()
    return adata


def save_slide(adata, sample_id: str) -> Path:
    """Save preprocessed AnnData to data/processed/pharma/."""
    path = pharma_processed_dir() / f"{safe_filename(sample_id)}_clustered.h5ad"
    adata.write_h5ad(path)
    return path


def load_slide(sample_id: str):
    """Load cached preprocessed slide."""
    import anndata as ad

    path = pharma_processed_dir() / f"{safe_filename(sample_id)}_clustered.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run notebook 01 or preprocess_cohort() first."
        )
    return ad.read_h5ad(path)


def preprocess_cohort(
    sample_ids: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download and preprocess all slides in sample_ids."""
    if cfg is None:
        cfg = load_config()
    if sample_ids is None:
        sample_ids = cohort_slide_ids(cfg)
    require_non_empty(
        sample_ids,
        stage="cohort_preprocessing",
        subject="admitted slide sequence",
        guidance="Admit at least one available slide before preprocessing.",
    )
    paths = {}
    for sid in sample_ids:
        out = pharma_processed_dir() / f"{safe_filename(sid)}_clustered.h5ad"
        if out.exists() and not force:
            paths[sid] = out
            continue
        print(f"Loading and preprocessing: {sid}")
        try:
            adata = st.load_visium_sample(sid)
        except (ConnectionError, TimeoutError, URLError, RequestException) as exc:
            raise SourceAcquisitionError(
                f"Could not acquire the configured public slide {sid}."
            ) from exc
        adata = preprocess_slide(adata, sid, cfg=cfg)
        paths[sid] = save_slide(adata, sid)
        print(
            f"  -> {out} ({adata.n_obs} spots, "
            f"{adata.obs['clusters'].nunique()} clusters)"
        )
    return paths


def cohort_summary(sample_ids: list[str] | None = None) -> pd.DataFrame:
    """Build summary table for processed slides."""
    if sample_ids is None:
        sample_ids = cohort_slide_ids()
    require_non_empty(
        sample_ids,
        stage="cohort_summary",
        subject="admitted slide sequence",
        guidance="Admit at least one processed slide before building the summary.",
    )
    rows = []
    for sid in sample_ids:
        adata = load_slide(sid)
        rows.append(
            {
                "slide_id": sid,
                "n_spots": adata.n_obs,
                "n_genes": adata.n_vars,
                "n_clusters": adata.obs["clusters"].nunique(),
                "median_genes": float(adata.obs["n_genes_by_counts"].median()),
                "median_counts": float(adata.obs["total_counts"].median()),
                "median_mito_pct": float(adata.obs["pct_counts_mt"].median()),
            }
        )
    return pd.DataFrame(rows)


def tumor_type_for_slide(sample_id: str) -> str:
    """Map sample_id to tumor type for gene panel selection."""
    s = sample_id.lower()
    if "breast" in s or "visium_hne" in s:
        if "mouse" in s or "visium_hne" in s:
            return "mouse_brain"
        return "breast"
    if "colorectal" in s:
        return "colorectal"
    if "ovarian" in s:
        return "ovarian"
    if "glioblastoma" in s:
        return "glioblastoma"
    return "breast"
