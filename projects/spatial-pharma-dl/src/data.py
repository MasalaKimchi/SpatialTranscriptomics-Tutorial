"""Data loading, preprocessing, and cohort management for Spatial Pharma DL."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pandas as pd
import yaml
from requests.exceptions import RequestException

from . import bootstrap  # noqa: F401 — ensures repo root is on sys.path
from .identity import IdentityValidationError, validate_anndata_spot_identity
from .validation import (
    ConfigValidationError,
    ValidationIssue,
    finalize_preprocessing_resolution,
    require_non_empty,
    resolve_config,
    resolve_post_qc_preprocessing,
)
from utils import st_helpers as st
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactValidationError,
    admit_artifact,
    artifact_reuse_status,
    build_fingerprint,
    publish_artifact,
)


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


def pharma_processed_path() -> Path:
    """Resolve the processed-artifact directory without creating it."""
    return st.project_root() / "data" / "processed" / "pharma"


def pharma_outputs_dir() -> Path:
    """Return outputs/pharma/ (created if missing)."""
    d = st.outputs_dir() / "pharma"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_filename(sample_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", sample_id)


def _processed_slide_path(sample_id: str) -> Path:
    return pharma_processed_path() / f"{safe_filename(sample_id)}_clustered.h5ad"


def _processed_fingerprint(
    sample_id: str,
    cfg: dict[str, Any],
    *,
    source_content_digest: str | None = None,
):
    resolved = resolve_config(cfg).to_dict()
    source: dict[str, object] = {
        "provider": "squidpy-public-visium",
        "sample_id": sample_id,
    }
    if source_content_digest is not None:
        if type(source_content_digest) is not str or len(source_content_digest) != 64:
            raise ArtifactValidationError(
                "invalid_fingerprint_inputs",
                artifact_kind="processed_slide",
                basename=_processed_slide_path(sample_id).name,
            )
        source["observed_content_sha256"] = source_content_digest
    return build_fingerprint(
        "processed_slide",
        {
            "configuration": resolved,
            "source": source,
            "upstream": {},
            "identity": {"sample_id": sample_id, "seed": resolved["seed"]},
        },
    )


def available_processed_slide_ids(
    sample_ids: list[str],
    cfg: dict[str, Any] | None = None,
) -> set[str]:
    """Return cached slide IDs without creating or opening any path."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    return {
        sample_id
        for sample_id in sample_ids
        if artifact_reuse_status(
            _processed_slide_path(sample_id),
            expected_kind="processed_slide",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["processed_slide"],
            expected_fingerprint=_processed_fingerprint(sample_id, resolved),
            reader=lambda candidate, sid=sample_id: _read_processed_slide(candidate, sid),
        ).reusable
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
    adata.uns["spatial_pharma_preprocessing_canonical_json"] = resolution.canonical_json
    adata.uns["spatial_pharma_seed"] = seed
    return adata


def _processed_schema(adata, sample_id: str) -> dict[str, object]:
    try:
        validate_anndata_spot_identity(
            adata,
            sample_id,
            stage="processed_slide_artifact",
            require_slide_id=True,
        )
    except IdentityValidationError:
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        ) from None
    if "spatial" not in adata.obsm or "X_pca" not in adata.obsm:
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        )
    if "spatial" not in adata.uns or "clusters" not in adata.obs:
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        )
    canonical = adata.uns.get("spatial_pharma_preprocessing_canonical_json")
    if type(canonical) is not str or len(canonical.encode("utf-8")) > 65_536:
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        )
    try:
        record = json.loads(canonical)
        if type(record) is not dict:
            raise ValueError
        from .validation import PreprocessingManifest

        PreprocessingManifest(slide_ids=[sample_id], records=[record])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        ) from None
    if int(record["counts"]["post_qc_spots"]) != int(adata.n_obs) or int(
        record["counts"]["post_qc_genes"]
    ) != int(adata.n_vars):
        raise ArtifactValidationError(
            "reader_validation_failed",
            artifact_kind="processed_slide",
            basename=_processed_slide_path(sample_id).name,
        )
    adata.uns["spatial_pharma_preprocessing"] = record
    keys = [(sample_id, spot_id) for spot_id in adata.obs_names.tolist()]
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "identity_sha256": hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "x_pca_shape": [int(value) for value in adata.obsm["X_pca"].shape],
        "spatial_shape": [int(value) for value in adata.obsm["spatial"].shape],
    }


def _read_processed_slide(path: Path, sample_id: str):
    import anndata as ad

    value = ad.read_h5ad(path)
    schema = _processed_schema(value, sample_id)
    return value, schema


def save_slide(
    adata,
    sample_id: str,
    *,
    cfg: dict[str, Any] | None = None,
    source_content_digest: str | None = None,
) -> Path:
    """Save preprocessed AnnData to data/processed/pharma/."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = _processed_slide_path(sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _processed_schema(adata, sample_id)
    publish_artifact(
        path,
        artifact_kind="processed_slide",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["processed_slide"],
        fingerprint=_processed_fingerprint(
            sample_id, resolved, source_content_digest=source_content_digest
        ),
        payload_format="h5ad",
        payload_schema=schema,
        write_payload=lambda temporary: adata.write_h5ad(temporary),
        reader=lambda temporary: _read_processed_slide(temporary, sample_id),
    )
    return path


def load_slide(
    sample_id: str,
    *,
    cfg: dict[str, Any] | None = None,
    source_content_digest: str | None = None,
):
    """Load cached preprocessed slide."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = _processed_slide_path(sample_id)
    try:
        admission = admit_artifact(
            path,
            expected_kind="processed_slide",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["processed_slide"],
            expected_fingerprint=_processed_fingerprint(
                sample_id, resolved, source_content_digest=source_content_digest
            ),
            reader=lambda candidate: _read_processed_slide(candidate, sample_id),
        )
    except ArtifactValidationError as exc:
        if exc.reason_code == "missing_payload":
            raise FileNotFoundError(
                f"Missing {path}. Run notebook 01 or preprocess_cohort() first."
            ) from exc
        raise
    adata, observed_schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != observed_schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch",
            artifact_kind="processed_slide",
            basename=path.name,
        )
    return adata


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
        out = _processed_slide_path(sid)
        if not force:
            try:
                load_slide(sid, cfg=cfg)
            except FileNotFoundError:
                pass
            except ArtifactValidationError as exc:
                if exc.reason_code not in {"legacy_artifact", "stale_fingerprint"}:
                    raise
            else:
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
        paths[sid] = save_slide(adata, sid, cfg=cfg)
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
