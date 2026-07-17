"""Per-spot H&E patch extraction, stain normalization, and PyTorch Dataset."""

from __future__ import annotations

import hashlib
import json
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

from . import bootstrap  # noqa: F401
from utils import st_helpers as st

from .data import (
    _processed_fingerprint,
    _processed_slide_path,
    load_config,
    pharma_processed_path,
    safe_filename,
)
from .identity import validate_anndata_spot_identity
from .validation import StageValidationError, require_non_empty, resolve_config
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactReuseStatus,
    ArtifactValidationError,
    admit_artifact,
    artifact_reuse_status,
    build_fingerprint,
    publish_artifact,
    read_artifact_manifest,
)


def patch_size_px(
    adata, min_patch: int = 8, context_scale: float = 1.0
) -> tuple[int, int]:
    """Return (patch_size, half) in hires pixels from Visium scale factors."""
    sf = st.get_scalefactors(adata)
    scalef = sf["tissue_hires_scalef"]
    patch = int(round(sf["spot_diameter_fullres"] * scalef * context_scale))
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
    out = resize(patch, (size, size), order=1, preserve_range=True, anti_aliasing=True)
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
        gray,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True,
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    feats["entropy_mean"] = float(rank_entropy(gray, disk(3)).mean())
    feats["edge_density"] = float(sobel(rgb2gray(p)).mean())
    feats["tissue_fraction"] = float((gray < 220).mean())
    return feats


def patch_cache_path(slide_id: str, cfg: dict[str, Any] | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    version = cfg["patches"].get("version", "v1")
    return pharma_processed_path() / f"{safe_filename(slide_id)}_patches_{version}.npz"


def _parent_manifest_context(
    path: Path,
    *,
    expected_kind: str,
    expected_fingerprint=None,
) -> dict[str, str]:
    manifest = read_artifact_manifest(path)
    if manifest.artifact_kind != expected_kind or (
        expected_fingerprint is not None
        and manifest.fingerprint.digest != expected_fingerprint.digest
    ):
        raise ArtifactValidationError(
            "stale_parent_artifact", artifact_kind=expected_kind, basename=path.name
        )
    return {
        "fingerprint": manifest.fingerprint.digest,
        "manifest_sha256": hashlib.sha256(
            manifest.canonical_json.encode("utf-8")
        ).hexdigest(),
        "payload_sha256": manifest.payload_sha256,
    }


def _reference_slide_id(slide_id: str, cfg: dict[str, Any]) -> str | None:
    if cfg["patches"].get("per_slide_stain_norm", False):
        return None
    for candidate in cfg["cohorts"]["oncology"]:
        try:
            _parent_manifest_context(
                _processed_slide_path(candidate),
                expected_kind="processed_slide",
                expected_fingerprint=_processed_fingerprint(candidate, cfg),
            )
        except (ArtifactValidationError, FileNotFoundError):
            continue
        return candidate
    return slide_id


def _patch_artifact_context(slide_id: str, cfg: dict[str, Any]) -> dict[str, object]:
    processed = _parent_manifest_context(
        _processed_slide_path(slide_id),
        expected_kind="processed_slide",
        expected_fingerprint=_processed_fingerprint(slide_id, cfg),
    )
    reference_id = _reference_slide_id(slide_id, cfg)
    reference = None
    if reference_id is not None:
        reference = {
            "slide_id": reference_id,
            **_parent_manifest_context(
                _processed_slide_path(reference_id),
                expected_kind="processed_slide",
                expected_fingerprint=_processed_fingerprint(reference_id, cfg),
            ),
        }
    return {"processed_slide": processed, "stain_reference": reference}


def _patch_fingerprint(slide_id: str, cfg: dict[str, Any]):
    resolved = resolve_config(cfg).to_dict()
    context = _patch_artifact_context(slide_id, resolved)
    reference = context["stain_reference"]
    return build_fingerprint(
        "patch",
        {
            "configuration": resolved,
            "source": {
                "stain_reference_policy": (
                    "per-slide" if reference is None else "first-admitted-oncology-slide"
                ),
                "stain_reference": reference,
            },
            "upstream": context,
            "identity": {"slide_id": slide_id},
        },
    )


def _patch_schema(
    patches: np.ndarray, meta: pd.DataFrame, *, slide_id: str
) -> dict[str, object]:
    required = ["slide_id", "spot_id", "x", "y", "native_patch_px"]
    if (
        type(patches) is not np.ndarray
        or patches.dtype != np.float32
        or patches.ndim != 4
        or patches.shape[1] != 3
        or patches.shape[0] == 0
        or not np.isfinite(patches).all()
        or np.any((patches < 0.0) | (patches > 1.0))
        or meta.columns.tolist() != required
        or len(meta) != len(patches)
        or meta.empty
    ):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="patch")
    if any(type(value) is not str or value != slide_id for value in meta["slide_id"]):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="patch")
    if any(type(value) is not str or not value.strip() for value in meta["spot_id"]):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="patch")
    if meta.duplicated(["slide_id", "spot_id"]).any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="patch")
    numeric = meta[["x", "y", "native_patch_px"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (meta["native_patch_px"] <= 0).any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="patch")
    keys = meta[["slide_id", "spot_id"]].to_dict("split")["data"]
    return {
        "keys": ["meta", "patches"],
        "patch_shape": [int(value) for value in patches.shape],
        "patch_dtype": "float32",
        "metadata_columns": required,
        "identity_sha256": hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "legacy_decode_policy": "trusted-local-writer-only",
    }


def _read_trusted_local_patch_npz(path: Path, slide_id: str):
    """Decode the legacy object field only after generic sidecar admission.

    A checksum is integrity, not authenticity. This adapter supports only archives
    emitted in-process by :func:`save_patch_arrays`; Phase 5 owns safe migration.
    """
    with np.load(path, allow_pickle=True) as data:
        if set(data.files) != {"patches", "meta"}:
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind="patch", basename=path.name
            )
        patch_values = data["patches"]
        if type(patch_values) is not np.ndarray or patch_values.dtype.hasobject:
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind="patch", basename=path.name
            )
        metadata_value = data["meta"]
        if metadata_value.shape != () or metadata_value.dtype != object:
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind="patch", basename=path.name
            )
        metadata_tree = metadata_value.item()
    if type(metadata_tree) is not dict or any(
        type(key) is not str or type(value) is not list
        for key, value in metadata_tree.items()
    ):
        raise ArtifactValidationError(
            "reader_validation_failed", artifact_kind="patch", basename=path.name
        )
    meta = pd.DataFrame(metadata_tree)
    schema = _patch_schema(patch_values, meta, slide_id=slide_id)
    return (patch_values, meta), schema


def _extract_spot_patches(
    adata,
    slide_id: str,
    ref_stain: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract normalized, resized patches for all spots on a slide."""
    validate_anndata_spot_identity(
        adata,
        slide_id,
        stage="patch_extraction",
        require_slide_id=True,
    )
    patch_cfg = cfg["patches"]
    min_patch = patch_cfg["min_patch_px"]
    out_size = patch_cfg["output_size"]
    context_scale = float(patch_cfg.get("context_scale", 1.0))
    per_slide = bool(patch_cfg.get("per_slide_stain_norm", False))

    coords = coords_hires(adata)
    require_non_empty(
        coords,
        stage="patch_extraction",
        subject=f"spot coordinates for slide {slide_id}",
        guidance="Retain at least one spatial spot before extracting patches.",
    )
    img = st.get_image(adata, "hires")
    stain = stain_matrix_macenko(img) if per_slide else ref_stain
    _, half = patch_size_px(adata, min_patch, context_scale)

    patches, meta_rows = [], []
    for i, (x, y) in enumerate(coords):
        raw = extract_patch(img, x, y, half)
        norm = macenko_normalize(raw, stain)
        resized = resize_patch(norm, out_size)
        patches.append(patch_to_tensor(resized))
        meta_rows.append(
            {
                "slide_id": slide_id,
                "spot_id": adata.obs_names[i],
                "x": float(x),
                "y": float(y),
                "native_patch_px": half * 2,
            }
        )
    return np.stack(patches, axis=0), pd.DataFrame(meta_rows)


def extract_all_patches_for_slide(
    adata,
    slide_id: str,
    ref_stain: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (N, 3, H, W) float32 tensor array and metadata DataFrame."""
    if cfg is None:
        cfg = load_config()
    return _extract_spot_patches(adata, slide_id, ref_stain, cfg)


def fit_reference_stain(
    sample_ids: list[str], cfg: dict[str, Any] | None = None
) -> np.ndarray:
    """Fit Macenko reference stain matrix from first available slide."""
    from .data import load_slide

    if cfg is None:
        cfg = load_config()
    require_non_empty(
        sample_ids,
        stage="stain_reference",
        subject="admitted slide sequence",
        guidance="Admit at least one processed slide before fitting a stain reference.",
    )
    for sid in sample_ids:
        adata = load_slide(sid, cfg=cfg)
        img = st.get_image(adata, "hires")
        return stain_matrix_macenko(img)
    raise FileNotFoundError("No processed slides found for stain reference.")


def save_patch_arrays(
    slide_id: str,
    patches: np.ndarray,
    meta: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> Path:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = patch_cache_path(slide_id, resolved)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _patch_schema(patches, meta, slide_id=slide_id)

    def write_payload(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, patches=patches, meta=meta.to_dict("list"))

    publish_artifact(
        path,
        artifact_kind="patch",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        fingerprint=_patch_fingerprint(slide_id, resolved),
        payload_format="npz-legacy-local-object",
        payload_schema=schema,
        write_payload=write_payload,
        reader=lambda temporary: _read_trusted_local_patch_npz(temporary, slide_id),
        observed_schema=lambda decoded: decoded[1],
    )
    return path


def load_patch_arrays(
    slide_id: str, cfg: dict[str, Any] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = patch_cache_path(slide_id, resolved)
    read_artifact_manifest(path)
    admission = admit_artifact(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=_patch_fingerprint(slide_id, resolved),
        reader=lambda candidate: _read_trusted_local_patch_npz(candidate, slide_id),
    )
    value, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="patch", basename=path.name
        )
    return value


def patch_reuse_status(slide_id: str, cfg: dict[str, Any] | None = None):
    """Return a contract-aware cache decision without creating directories."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = patch_cache_path(slide_id, resolved)
    try:
        read_artifact_manifest(path)
    except ArtifactValidationError as exc:
        return ArtifactReuseStatus(False, exc.reason_code)
    return artifact_reuse_status(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=_patch_fingerprint(slide_id, resolved),
        reader=lambda candidate: _read_trusted_local_patch_npz(candidate, slide_id),
    )


def build_patch_cohort(
    sample_ids: list[str],
    ref_stain: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build patches for all slides; return reference stain matrix."""
    from .data import load_slide

    if cfg is None:
        cfg = load_config()
    require_non_empty(
        sample_ids,
        stage="patch_cohort",
        subject="admitted slide sequence",
        guidance="Admit at least one slide before building patch caches.",
    )
    if ref_stain is None:
        ref_stain = fit_reference_stain(sample_ids, cfg)
    for sid in sample_ids:
        adata = load_slide(sid, cfg=cfg)
        patches, meta = extract_all_patches_for_slide(adata, sid, ref_stain, cfg)
        save_patch_arrays(sid, patches, meta, cfg=cfg)
        print(f"Saved {len(patches)} patches for {sid}")
    return ref_stain


def _index_fingerprint_for_ids(sample_ids: list[str], cfg: dict[str, Any]):
    from .labels import _label_path, _table_fingerprint

    return build_fingerprint(
        "patch_index",
        {
            "configuration": cfg,
            "source": {
                "label_manifests": {
                    sid: _parent_manifest_context(
                        _label_path(sid),
                        expected_kind="label_table",
                        expected_fingerprint=_table_fingerprint(
                            "label_table", [sid], cfg
                        ),
                    )
                    for sid in sample_ids
                },
            },
            "upstream": {
                "patch_manifests": {
                    sid: _parent_manifest_context(
                        patch_cache_path(sid, cfg),
                        expected_kind="patch",
                        expected_fingerprint=_patch_fingerprint(sid, cfg),
                    )
                    for sid in sample_ids
                },
            },
            "identity": {"slide_ids": sample_ids},
        },
    )


def _read_patch_index(path: Path, sample_ids: list[str]):
    from .labels import _table_schema

    frame = pd.read_parquet(path)
    schema = _table_schema(frame, kind="label_table", sample_ids=sample_ids)
    return frame, schema


def save_patch_index(
    labels: pd.DataFrame,
    path: Path | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    artifact_context: dict[str, object] | None = None,
) -> Path:
    """Save combined patch+label index to data/processed/pharma/patch_index.parquet."""
    if artifact_context is not None:
        raise ArtifactValidationError(
            "invalid_fingerprint_inputs", artifact_kind="patch_index"
        )
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = path or pharma_processed_path() / "patch_index.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = list(dict.fromkeys(labels["slide_id"].tolist()))
    _frame, schema = _read_patch_index_from_frame(labels)
    publish_artifact(
        path,
        artifact_kind="patch_index",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["patch_index"],
        fingerprint=_index_fingerprint_for_ids(sample_ids, resolved),
        payload_format="parquet",
        payload_schema=schema,
        write_payload=lambda temporary: labels.to_parquet(temporary, index=False),
        reader=lambda temporary: _read_patch_index(temporary, sample_ids),
        observed_schema=lambda decoded: decoded[1],
    )
    return path


def _read_patch_index_from_frame(frame: pd.DataFrame):
    from .labels import _table_schema

    sample_ids = list(dict.fromkeys(frame["slide_id"].tolist()))
    return frame, _table_schema(frame, kind="label_table", sample_ids=sample_ids)


def load_patch_index(
    path: Path | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    path = path or pharma_processed_path() / "patch_index.parquet"
    expected_ids = (
        sample_ids
        if sample_ids is not None
        else [
            *resolved["cohorts"]["oncology"],
            *resolved["cohorts"]["external"],
            *resolved["cohorts"]["benchmark"],
        ]
    )
    admission = admit_artifact(
        path,
        expected_kind="patch_index",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch_index"],
        expected_fingerprint=_index_fingerprint_for_ids(expected_ids, resolved),
        reader=lambda candidate: _read_patch_index(candidate, expected_ids),
    )
    frame, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="patch_index", basename=path.name
        )
    return frame


try:
    import torch
    from torch.utils.data import Dataset

    class SpotPatchDataset(Dataset):
        """PyTorch Dataset over cached patches + label DataFrame."""

        def __init__(
            self,
            patches: np.ndarray,
            labels: pd.DataFrame,
            cls_col: str = "tme_class_id",
            reg_cols: list[str] | None = None,
        ):
            require_non_empty(
                patches,
                stage="patch_dataset",
                subject="patch rows",
                guidance="Provide at least one patch and aligned label row.",
            )
            require_non_empty(
                labels,
                stage="patch_dataset",
                subject="label rows",
                guidance="Provide at least one patch and aligned label row.",
            )
            if len(patches) != len(labels):
                raise StageValidationError(
                    stage="patch_dataset",
                    subject="aligned patch and label rows",
                    observed=min(len(patches), len(labels)),
                    minimum=max(len(patches), len(labels)),
                    shape=(len(patches), len(labels)),
                    guidance="Align one label row to every patch before dataset creation.",
                    message=(
                        "patch_dataset: patch and label row counts differ "
                        f"(observed shape={(len(patches), len(labels))}, expected "
                        "equal first dimensions). Align one label row to every patch "
                        "before dataset creation."
                    ),
                )
            self.patches = patches
            self.labels = labels.reset_index(drop=True)
            self.cls_col = cls_col
            self.reg_cols = reg_cols or [
                c for c in labels.columns if c.startswith("module_")
            ]

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx: int):
            x = torch.from_numpy(self.patches[idx])
            row = self.labels.iloc[idx]
            y_cls = int(row[self.cls_col])
            y_reg = torch.tensor(
                [float(row[c]) for c in self.reg_cols], dtype=torch.float32
            )
            return x, y_cls, y_reg

except ImportError:
    SpotPatchDataset = None  # type: ignore
