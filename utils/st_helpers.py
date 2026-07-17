"""Shared helpers for the Spatial Transcriptomics tutorial series.

These utilities are imported by every notebook so that paths, seeds, dataset
loading, and a few repetitive chores stay consistent and reproducible.

Design notes
------------
- No hard-coded absolute paths. All paths are derived from this file's location
  via :func:`project_root`, so the tutorial works after a ``git clone`` anywhere.
- Functions raise clear, actionable errors when files or genes are missing,
  rather than failing deep inside a plotting call.
"""

from __future__ import annotations

import os
import random
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Paths (pathlib only, no hard-coded absolutes)
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Return the repository root (the parent of the ``utils`` folder)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Return ``<root>/data`` (created if missing)."""
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    """Return ``<root>/data/raw`` (created if missing)."""
    d = data_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir() -> Path:
    """Return ``<root>/data/processed`` (created if missing). Holds .h5ad caches."""
    d = data_dir() / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir() -> Path:
    """Return ``<root>/outputs`` (created if missing). Holds CSVs and figures."""
    d = project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_pharma_paths(start: Path | None = None) -> tuple[Path, Path]:
    """Locate repo + pharma subproject and add both to ``sys.path``.

    Used by pharma notebooks and scripts so imports resolve as
    ``from utils import st_helpers`` and ``from src.data import ...``.
    """
    import sys

    root = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (root / "utils").is_dir():
            break
        if root.parent == root:
            raise FileNotFoundError(
                "Could not find repository root (no utils/ directory while walking up)."
            )
        root = root.parent

    pharma = root / "projects" / "spatial-pharma-dl"
    for path in (root, pharma):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root, pharma


def figures_dir() -> Path:
    """Return ``<root>/outputs/figures`` (created if missing). Holds gallery PNGs."""
    d = outputs_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, filename: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to ``outputs/figures/<filename>`` and return the path."""
    path = figures_dir() / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Default seed used across the whole tutorial.
SEED = 0


def set_seeds(seed: int = SEED) -> int:
    """Seed Python, NumPy (and PYTHONHASHSEED) for reproducible runs.

    Scanpy/Squidpy functions that are stochastic (PCA, UMAP, Leiden) also take a
    ``random_state`` argument; pass ``SEED`` there too for full determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Dataset loading (squidpy built-in, with a clear fallback message)
# ---------------------------------------------------------------------------


def load_dataset(name: str = "visium_hne"):
    """Load the tutorial dataset as an AnnData object.

    Parameters
    ----------
    name
        ``"visium_hne"`` (default) -> Squidpy's bundled 10x mouse-brain Visium
        H&E sample (:func:`squidpy.datasets.visium_hne_adata`). This single call
        fetches the count matrix, spatial coordinates, the H&E image, and the
        scale factors, caching them under squidpy's data directory.

    Notes
    -----
    Requires network access on first call; afterwards squidpy serves the cached
    copy. If the download fails, check connectivity or see the troubleshooting
    section in the project README.
    """
    import squidpy as sq

    if name != "visium_hne":
        raise ValueError(
            f"Unknown dataset {name!r}. This tutorial ships with 'visium_hne'."
        )
    return sq.datasets.visium_hne_adata()


def load_visium_sample(sample_id: str):
    """Load any public 10x Visium slide by sample_id via squidpy.

    Parameters
    ----------
    sample_id
        e.g. ``'V1_Breast_Cancer_Block_A_Section_1'`` or ``'visium_hne'`` for the
        tutorial mouse-brain sample.

    Returns
    -------
    AnnData
        Visium object with spatial image and coordinates attached.
    """
    import squidpy as sq

    if sample_id == "visium_hne":
        return load_dataset("visium_hne")
    adata = sq.datasets.visium(sample_id)
    adata.var_names_make_unique()
    return adata


def get_library_id(adata) -> str:
    """Return the (single) Visium library id stored in ``adata.uns['spatial']``."""
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        raise KeyError(
            "adata.uns['spatial'] is missing. Did you load a Visium dataset with "
            "an attached image? (see notebook 02)."
        )
    return list(adata.uns["spatial"].keys())[0]


def get_scalefactors(adata) -> dict:
    """Return the Visium scale-factor dict for the dataset's library."""
    lib = get_library_id(adata)
    return adata.uns["spatial"][lib]["scalefactors"]


def get_image(adata, res: str = "hires") -> np.ndarray:
    """Return the H&E image array (``res`` is ``'hires'`` or ``'lowres'``)."""
    lib = get_library_id(adata)
    images = adata.uns["spatial"][lib]["images"]
    if res not in images:
        raise KeyError(
            f"Resolution {res!r} not available; have {list(images)}."
        )
    return images[res]


# ---------------------------------------------------------------------------
# AnnData caching
# ---------------------------------------------------------------------------

_ROOT_H5AD_CHAIN = (
    "adata_raw.h5ad",
    "adata_qc.h5ad",
    "adata_clustered.h5ad",
    "adata_features.h5ad",
)


def _root_h5ad_path(filename: str) -> Path:
    """Resolve a tutorial cache path without changing the filesystem."""
    if type(filename) is not str or Path(filename).name != filename:
        raise ValueError("filename must be a plain cache basename")
    return project_root() / "data" / "processed" / filename


def _root_h5ad_fingerprint(
    filename: str,
    *,
    source: dict[str, object] | None = None,
    upstream: dict[str, object] | None = None,
):
    from utils.artifacts import build_fingerprint

    if filename in _ROOT_H5AD_CHAIN:
        index = _ROOT_H5AD_CHAIN.index(filename)
        source_identity: dict[str, object] = (
            {"provider": "squidpy", "dataset": "visium_hne"}
            if index == 0
            else {}
        )
        upstream_identity = (
            {}
            if index == 0
            else {
                "artifact": _ROOT_H5AD_CHAIN[index - 1],
                "fingerprint": _root_h5ad_fingerprint(
                    _ROOT_H5AD_CHAIN[index - 1]
                ).digest,
            }
        )
    else:
        if type(source) is not dict or type(upstream) is not dict:
            from utils.artifacts import ArtifactValidationError

            raise ArtifactValidationError(
                "invalid_fingerprint_inputs",
                artifact_kind="root_h5ad",
                basename=filename,
            )
        source_identity = source
        upstream_identity = upstream
    return build_fingerprint(
        "root_h5ad",
        {
            "configuration": {},
            "source": source_identity,
            "upstream": upstream_identity,
            "identity": {"filename": filename, "stage": filename.removesuffix(".h5ad")},
        },
    )


def _root_h5ad_schema(value, filename: str):
    from utils.artifacts import ArtifactValidationError

    obs_names = value.obs_names.tolist()
    var_names = value.var_names.tolist()
    if (
        any(type(item) is not str or not item for item in (*obs_names, *var_names))
        or len(set(obs_names)) != len(obs_names)
        or len(set(var_names)) != len(var_names)
    ):
        raise ArtifactValidationError(
            "reader_validation_failed", artifact_kind="root_h5ad", basename=filename
        )
    required_obs: tuple[str, ...] = ()
    required_obsm: tuple[str, ...] = ()
    if filename in {"adata_qc.h5ad", "adata_clustered.h5ad", "adata_features.h5ad"}:
        required_obs = ("total_counts", "n_genes_by_counts", "pct_counts_mt")
    if filename in {"adata_clustered.h5ad", "adata_features.h5ad"}:
        required_obs += ("clusters",)
        required_obsm = ("X_pca",)
    if any(column not in value.obs for column in required_obs) or any(
        key not in value.obsm for key in required_obsm
    ):
        raise ArtifactValidationError(
            "reader_validation_failed", artifact_kind="root_h5ad", basename=filename
        )
    if filename in _ROOT_H5AD_CHAIN and (
        "spatial" not in value.obsm
        or "spatial" not in value.uns
        or np.asarray(value.obsm["spatial"]).shape != (value.n_obs, 2)
        or not np.isfinite(np.asarray(value.obsm["spatial"], dtype=np.float64)).all()
    ):
        raise ArtifactValidationError(
            "reader_validation_failed", artifact_kind="root_h5ad", basename=filename
        )
    schema = {
        "n_obs": int(value.n_obs),
        "n_vars": int(value.n_vars),
        "obs_names_sha256": __import__("hashlib").sha256(
            json.dumps(obs_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "var_names_sha256": __import__("hashlib").sha256(
            json.dumps(var_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "required_obs": list(required_obs),
        "required_obsm": list(required_obsm),
    }
    return schema


def _root_h5ad_reader(path: Path, filename: str):
    import anndata as ad

    value = ad.read_h5ad(path)
    return value, _root_h5ad_schema(value, filename)


def save_adata(
    adata,
    filename: str,
    *,
    source: dict[str, object] | None = None,
    upstream: dict[str, object] | None = None,
) -> Path:
    """Write ``adata`` to ``data/processed/<filename>`` and return the path."""
    from utils.artifacts import ARTIFACT_CONTRACT_VERSIONS, publish_artifact

    path = _root_h5ad_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _root_h5ad_fingerprint(filename, source=source, upstream=upstream)
    schema = _root_h5ad_schema(adata, filename)
    publish_artifact(
        path,
        artifact_kind="root_h5ad",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["root_h5ad"],
        fingerprint=fingerprint,
        payload_format="h5ad",
        payload_schema=schema,
        write_payload=lambda temporary: adata.write_h5ad(temporary),
        reader=lambda temporary: _root_h5ad_reader(temporary, filename),
        observed_schema=lambda decoded: decoded[1],
    )
    return path


def load_adata(
    filename: str,
    *,
    source: dict[str, object] | None = None,
    upstream: dict[str, object] | None = None,
):
    """Read an AnnData from ``data/processed/<filename>``.

    Raises a helpful error (naming the notebook that creates it) if absent.
    """
    from utils.artifacts import (
        ARTIFACT_CONTRACT_VERSIONS,
        ArtifactValidationError,
        admit_artifact,
    )

    path = _root_h5ad_path(filename)
    fingerprint = _root_h5ad_fingerprint(filename, source=source, upstream=upstream)
    try:
        admission = admit_artifact(
            path,
            expected_kind="root_h5ad",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["root_h5ad"],
            expected_fingerprint=fingerprint,
            reader=lambda candidate: _root_h5ad_reader(candidate, filename),
        )
    except ArtifactValidationError as exc:
        if exc.reason_code == "missing_payload":
            raise FileNotFoundError(
                f"Cached AnnData not found: {path}\n"
                "Run the earlier notebook that produces it before this one."
            ) from exc
        raise
    value, observed_schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != observed_schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch",
            artifact_kind="root_h5ad",
            basename=filename,
        )
    return value


_ROOT_RESULT_COLUMNS = {
    "qc_summary": ("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    "cluster_markers": ("group", "names", "scores", "logfoldchanges", "pvals", "pvals_adj"),
    "image_features": (
        "mean_r", "std_r", "mean_g", "std_g", "mean_b", "std_b",
        "hematoxylin_mean", "eosin_mean", "glcm_contrast", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "entropy_mean", "edge_density",
        "tissue_fraction",
    ),
    "integration_metrics": ("task", "target", "metric", "value"),
}


def root_artifact_lineage(filename: str) -> dict[str, str]:
    """Return bounded current lineage for a root tutorial H5AD."""
    from utils.artifacts import read_artifact_manifest

    manifest = read_artifact_manifest(_root_h5ad_path(filename))
    if manifest.artifact_kind != "root_h5ad":
        raise ValueError("Root tutorial lineage requires a root_h5ad artifact.")
    return {
        "fingerprint": manifest.fingerprint.digest,
        "manifest_sha256": __import__("hashlib").sha256(
            manifest.canonical_json.encode("utf-8")
        ).hexdigest(),
        "payload_sha256": manifest.payload_sha256,
    }


def _root_result_schema(frame, result_name: str, *, include_index: bool):
    from utils.artifacts import ArtifactValidationError

    expected = _ROOT_RESULT_COLUMNS.get(result_name)
    if expected is None or tuple(frame.columns) != expected or frame.empty:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    if frame.isnull().any().any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    return {
        "result_name": result_name,
        "columns": list(expected),
        "rows": int(len(frame)),
        "include_index": include_index,
        "index": [str(value) for value in frame.index] if include_index else [],
    }


def save_root_result_table(
    frame,
    path: Path,
    *,
    result_name: str,
    upstream_lineage: dict[str, object],
    include_index: bool = False,
) -> Path:
    """Publish one named root-notebook table through the artifact contract."""
    from utils.artifacts import (
        ARTIFACT_CONTRACT_VERSIONS,
        build_fingerprint,
        publish_artifact,
    )

    if type(upstream_lineage) is not dict or not upstream_lineage:
        raise ValueError("Root result publication requires current upstream lineage.")
    schema = _root_result_schema(frame, result_name, include_index=include_index)
    fingerprint = build_fingerprint(
        "summary",
        {
            "configuration": {},
            "source": {"result_name": result_name, "columns": schema["columns"]},
            "upstream": upstream_lineage,
            "identity": {"rows": schema["rows"], "index": schema["index"]},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    def reader(candidate: Path):
        import pandas as pd

        restored = pd.read_csv(candidate, index_col=0 if include_index else None)
        return restored, _root_result_schema(
            restored, result_name, include_index=include_index
        )

    publish_artifact(
        path,
        artifact_kind="summary",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["summary"],
        fingerprint=fingerprint,
        payload_format="csv",
        payload_schema=schema,
        write_payload=lambda temporary: frame.to_csv(
            temporary, index=include_index
        ),
        reader=reader,
        observed_schema=lambda decoded: decoded[1],
    )
    return path


def adata_reuse_status(filename: str):
    """Return contract-aware optional-cache status without creating directories."""
    from utils.artifacts import ARTIFACT_CONTRACT_VERSIONS, artifact_reuse_status

    return artifact_reuse_status(
        _root_h5ad_path(filename),
        expected_kind="root_h5ad",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["root_h5ad"],
        expected_fingerprint=_root_h5ad_fingerprint(filename),
        reader=lambda candidate: _root_h5ad_reader(candidate, filename),
    )


# ---------------------------------------------------------------------------
# Gene-existence guard (datasets differ; always check before plotting/scoring)
# ---------------------------------------------------------------------------


def genes_present(adata, candidates: Iterable[str], verbose: bool = True) -> list:
    """Return the subset of ``candidates`` that exist in ``adata.var_names``.

    Spatial datasets vary in which genes they contain (and in gene-name casing:
    mouse uses ``Snap25``, human uses ``SNAP25``). Always filter a wishlist of
    marker genes through this helper before passing them to a plotting or scoring
    function, so a missing gene never crashes the notebook.
    """
    present = [g for g in candidates if g in adata.var_names]
    if verbose:
        missing = [g for g in candidates if g not in adata.var_names]
        if missing:
            print(f"[genes_present] skipping {len(missing)} absent gene(s): {missing}")
        if not present:
            print("[genes_present] WARNING: none of the requested genes are present.")
    return present


def first_present_gene(adata, candidates: Sequence[str]) -> str:
    """Return the first gene from ``candidates`` that exists, else raise."""
    for g in candidates:
        if g in adata.var_names:
            return g
    raise KeyError(
        f"None of these genes are in the dataset: {list(candidates)}. "
        "Inspect adata.var_names to pick alternatives."
    )


# ---------------------------------------------------------------------------
# Leiden clustering compatibility wrapper
# ---------------------------------------------------------------------------


def run_leiden(adata, resolution: float = 1.0, key_added: str = "clusters",
               seed: int = SEED):
    """Run Leiden clustering across scanpy versions.

    Newer scanpy prefers the fast igraph flavor; older builds use leidenalg.
    This wrapper tries the modern call and falls back gracefully.
    """
    import scanpy as sc

    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except TypeError:
        # Older scanpy without the `flavor` kwarg.
        sc.tl.leiden(
            adata, resolution=resolution, key_added=key_added, random_state=seed
        )
    return adata
