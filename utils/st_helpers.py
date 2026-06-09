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


def save_adata(adata, filename: str) -> Path:
    """Write ``adata`` to ``data/processed/<filename>`` and return the path."""
    path = processed_dir() / filename
    adata.write_h5ad(path)
    return path


def load_adata(filename: str):
    """Read an AnnData from ``data/processed/<filename>``.

    Raises a helpful error (naming the notebook that creates it) if absent.
    """
    import anndata as ad

    path = processed_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cached AnnData not found: {path}\n"
            "Run the earlier notebook that produces it before this one "
            "(each notebook saves its result into data/processed/)."
        )
    return ad.read_h5ad(path)


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
