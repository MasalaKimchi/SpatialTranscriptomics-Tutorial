"""Per-spot H&E patch extraction, stain normalization, and PyTorch Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.color import rgb2hed, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import st_helpers as st  # noqa: E402

from .data import load_config, pharma_processed_dir, safe_filename  # noqa: E402


def patch_size_px(adata, min_patch: int = 8) -> tuple[int, int]:
    """Return (patch_size, half) in hires pixels from Visium scale factors."""
    sf = st.get_scalefactors(adata)
    scalef = sf["tissue_hires_scalef"]
    patch = int(round(sf["spot_diameter_fullres"] * scalef))
    patch = max(patch, min_patch)
    return patch, patch // 2


def coords_hires(adata) -> np.ndarray:
    """Spot centers in hires image pixel coordinates."""
    sf = st.get_scalefactors(adata)
    return adata.obsm["spatial"] * sf["tissue_hires_scalef"]


def extract_patch(img: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    """Crop square patch centered on (x, y), clipped to image bounds."""
    H, W = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - half), min(W, xi + half)
    y0, y1 = max(0, yi - half), min(H, yi + half)
    return img[y0:y1, x0:x1].copy()


def _normalize_od(od: np.ndarray) -> np.ndarray:
    return np.maximum(od, 1e-6)


def stain_matrix_macenko(img: np.ndarray) -> np.ndarray:
    """Estimate 2x3 H&E stain matrix from RGB uint8 image via Macenko method."""
    img_f = img.reshape(-1, 3).astype(np.float64) / 255.0
    img_f = img_f[(img_f > 0.01).all(axis=1)]
    if len(img_f) < 100:
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    od = -np.log(_normalize_od(img_f))
    cov = np.cov(od.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v1 = eigvecs[:, np.argsort(eigvals)[-1]]
    v2 = eigvecs[:, np.argsort(eigvals)[-2]]
    proj = od @ np.stack([v1, v2], axis=1)
    angle = np.arctan2(proj[:, 1], proj[:, 0])
    min_a = np.percentile(angle, 1)
    max_a = np.percentile(angle, 99)
    hem = od[(angle >= min_a) & (angle <= max_a)].mean(axis=0)
    eos = od[(angle < min_a) | (angle > max_a)].mean(axis=0)
    stain = np.stack([hem, eos], axis=0)
    stain = stain / np.linalg.norm(stain, axis=1, keepdims=True)
    return stain


def macenko_normalize(
    patch: np.ndarray, ref_stain: np.ndarray, target_stain: np.ndarray | None = None
) -> np.ndarray:
    """Normalize patch colors to target stain matrix (default: ref_stain)."""
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return patch
    target = target_stain if target_stain is not None else ref_stain
    flat = patch.reshape(-1, 3).astype(np.float64) / 255.0
    od = -np.log(_normalize_od(flat))
    try:
        conc, _, _, _ = np.linalg.lstsq(ref_stain.T, od.T, rcond=None)
        recon = (target.T @ conc).T
        rgb = np.exp(-recon) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb.reshape(patch.shape)
    except np.linalg.LinAlgError:
        return patch


def resize_patch(patch: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize patch to (size, size, 3) uint8."""
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return np.zeros((size, size, 3), dtype=np.uint8)
    out = resize(
        patch, (size, size), order=1, preserve_range=True, anti_aliasing=True
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def patch_to_tensor(patch: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC to float32 CHW in [0, 1] for PyTorch."""
    return patch.transpose(2, 0, 1).astype(np.float32) / 255.0


def patch_features(patch: np.ndarray) -> dict[str, float]:
    """Handcrafted radiomics features (notebook 09 parity) for RF baseline."""
    p = patch
    if p.shape[0] < 4 or p.shape[1] < 4:
        return {}
    feats = {}
    for i, c in enumerate("rgb"):
        feats[f"mean_{c}"] = float(p[..., i].mean())
        feats[f"std_{c}"] = float(p[..., i].std())
    hed = rgb2hed(p / 255.0)
    feats["hematoxylin_mean"] = float(hed[..., 0].mean())
    feats["eosin_mean"] = float(hed[..., 1].mean())
    gray = img_as_ubyte(rgb2gray(p))
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi / 2], levels=256, symmetric=True, normed=True
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    feats["entropy_mean"] = float(rank_entropy(gray, disk(3)).mean())
    feats["edge_density"] = float(sobel(rgb2gray(p)).mean())
    feats["tissue_fraction"] = float((gray < 220).mean())
    return feats


def build_patch_index(
    adata,
    slide_id: str,
    ref_stain: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract all spot patches and build index DataFrame."""
    cfg = cfg or load_config()
    min_patch = cfg["patches"]["min_patch_px"]
    out_size = cfg["patches"]["output_size"]

    img = st.get_image(adata, "hires")
    patch_px, half = patch_size_px(adata, min_patch)
    coords = coords_hires(adata)

    if ref_stain is None:
        ref_stain = stain_matrix_macenko(img)

    rows = []
    for i, (x, y) in enumerate(tqdm(coords, desc=f"patches {slide_id[:20]}")):
        raw = extract_patch(img, x, y, half)
        norm = macenko_normalize(raw, ref_stain)
        resized = resize_patch(norm, out_size)
        spot_id = adata.obs_names[i]
        rows.append(
            {
                "slide_id": slide_id,
                "spot_id": spot_id,
                "x": float(x),
                "y": float(y),
                "patch_px": patch_px,
            }
        )
        # Store resized patch as bytes path alternative: save to npz later in dataset
        adata.obsm.setdefault("_patch_cache", {})
    return pd.DataFrame(rows)


def extract_all_patches_for_slide(
    adata,
    slide_id: str,
    ref_stain: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (N, 3, H, W) float32 tensor array and metadata DataFrame."""
    cfg = cfg or load_config()
    min_patch = cfg["patches"]["min_patch_px"]
    out_size = cfg["patches"]["output_size"]

    img = st.get_image(adata, "hires")
    _, half = patch_size_px(adata, min_patch)
    coords = coords_hires(adata)

    patches = []
    meta_rows = []
    for i, (x, y) in enumerate(coords):
        raw = extract_patch(img, x, y, half)
        norm = macenko_normalize(raw, ref_stain)
        resized = resize_patch(norm, out_size)
        patches.append(patch_to_tensor(resized))
        meta_rows.append(
            {
                "slide_id": slide_id,
                "spot_id": adata.obs_names[i],
                "x": float(x),
                "y": float(y),
            }
        )
    return np.stack(patches, axis=0), pd.DataFrame(meta_rows)


def fit_reference_stain(sample_ids: list[str], cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Fit Macenko reference stain matrix from first available slide."""
    from .data import load_slide

    cfg = cfg or load_config()
    for sid in sample_ids:
        try:
            adata = load_slide(sid)
            img = st.get_image(adata, "hires")
            return stain_matrix_macenko(img)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No processed slides found for stain reference.")


def save_patch_arrays(
    slide_id: str, patches: np.ndarray, meta: pd.DataFrame
) -> Path:
    """Save patch tensors to data/processed/pharma/<slide>_patches.npz."""
    path = pharma_processed_dir() / f"{safe_filename(slide_id)}_patches.npz"
    np.savez_compressed(path, patches=patches, meta=meta.to_dict("list"))
    return path


def load_patch_arrays(slide_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached patch arrays for a slide."""
    path = pharma_processed_dir() / f"{safe_filename(slide_id)}_patches.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build_patch_cohort() first.")
    data = np.load(path, allow_pickle=True)
    meta = pd.DataFrame(data["meta"].item())
    return data["patches"], meta


def build_patch_cohort(
    sample_ids: list[str],
    ref_stain: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build patches for all slides; return reference stain matrix."""
    from .data import load_slide

    cfg = cfg or load_config()
    if ref_stain is None:
        ref_stain = fit_reference_stain(sample_ids, cfg)
    for sid in sample_ids:
        try:
            adata = load_slide(sid)
        except FileNotFoundError:
            print(f"Skipping {sid} (not preprocessed)")
            continue
        patches, meta = extract_all_patches_for_slide(adata, sid, ref_stain, cfg)
        save_patch_arrays(sid, patches, meta)
        print(f"Saved {len(patches)} patches for {sid}")
    return ref_stain


def save_patch_index(labels: pd.DataFrame, path: Path | None = None) -> Path:
    """Save combined patch+label index to data/processed/pharma/patch_index.parquet."""
    path = path or pharma_processed_dir() / "patch_index.parquet"
    labels.to_parquet(path, index=False)
    return path


try:
    import torch
    from torch.utils.data import Dataset

    class SpotPatchDataset(Dataset):
        """PyTorch Dataset over cached patches + label DataFrame."""

        def __init__(
            self,
            patches: np.ndarray,
            labels: pd.DataFrame,
            cluster_col: str = "cluster_id",
            gene_cols: list[str] | None = None,
        ):
            self.patches = patches
            self.labels = labels.reset_index(drop=True)
            self.cluster_col = cluster_col
            self.gene_cols = gene_cols or [
                c for c in labels.columns if c.startswith("gene_")
            ]
            assert len(self.patches) == len(self.labels)

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx: int):
            x = torch.from_numpy(self.patches[idx])
            row = self.labels.iloc[idx]
            y_cls = int(row[self.cluster_col])
            y_reg = torch.tensor(
                [float(row[c]) for c in self.gene_cols], dtype=torch.float32
            )
            return x, y_cls, y_reg

except ImportError:
    SpotPatchDataset = None  # type: ignore
