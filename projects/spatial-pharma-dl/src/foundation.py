"""Frozen pathology-foundation embeddings and lightweight LOSO probes.

The foundation encoder is always run in evaluation/inference mode. Only the
scikit-learn classifier and regressors are fitted on the training slides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_config, pharma_processed_dir, safe_filename
from .device import device_label, resolve_device
from .eval import classification_metrics, regression_metrics
from .identity import (
    IdentityIssue,
    IdentityValidationError,
    align_labels_with_metadata,
)
from .labels import classification_column, regression_columns
from .train import _maybe_subsample, load_slide_patches, loso_folds
from .validation import StageValidationError, require_non_empty, resolve_config


@dataclass(frozen=True)
class FoundationModelSpec:
    """Load and preprocessing contract for a frozen patch encoder."""

    repo_id: str
    backend: str
    license: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    embedding_dim: int


FOUNDATION_MODELS = {
    # Compact (22M parameters), ungated, and suitable for a laptop tutorial.
    # Its weights are research/non-commercial; do not use for pharma decisions.
    "kaiko_vits16": FoundationModelSpec(
        repo_id=("1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms"),
        backend="timm",
        license="kaiko-non-commercial",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        embedding_dim=384,
    ),
    # Larger independent pathology encoder trained with iBOT on TCGA tiles.
    "phikon": FoundationModelSpec(
        repo_id="owkin/phikon",
        backend="transformers_cls",
        license="owkin-non-commercial",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        embedding_dim=768,
    ),
}


class _TransformersCLSEncoder(torch.nn.Module):
    """Expose a Hugging Face ViT CLS token as a tensor-returning encoder."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]


def _resolved_config_arg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve supplied config while reserving default loading for ``None``."""
    if cfg is None:
        return load_config()
    return resolve_config(cfg).to_dict()


def _foundation_config_resolved(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["foundation"]


def foundation_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolved_config_arg(cfg)
    return _foundation_config_resolved(resolved)


def _foundation_model_spec_resolved(
    cfg: dict[str, Any],
) -> tuple[str, FoundationModelSpec]:
    fm_cfg = _foundation_config_resolved(cfg)
    name = fm_cfg.get("model", "kaiko_vits16")
    if name not in FOUNDATION_MODELS:
        raise ValueError(
            f"Unknown foundation model {name!r}; choose from {tuple(FOUNDATION_MODELS)}"
        )
    return name, FOUNDATION_MODELS[name]


def foundation_model_spec(
    cfg: dict[str, Any] | None = None,
) -> tuple[str, FoundationModelSpec]:
    resolved = _resolved_config_arg(cfg)
    return _foundation_model_spec_resolved(resolved)


def load_frozen_encoder(
    cfg: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device, FoundationModelSpec]:
    """Download/load a pathology encoder and freeze all of its parameters."""
    cfg = _resolved_config_arg(cfg)
    name, spec = _foundation_model_spec_resolved(cfg)
    dev = resolve_device(
        device or _foundation_config_resolved(cfg).get("device", "auto")
    )
    print(
        f"  Loading frozen encoder: {name} ({spec.repo_id}) | "
        f"device: {device_label(dev)}"
    )
    if spec.backend == "timm":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "This foundation encoder requires timm. Install "
                "requirements-pharma.txt first."
            ) from exc
        model = timm.create_model(
            f"hf_hub:{spec.repo_id}",
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0,
        )
    elif spec.backend == "transformers_cls":
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "This foundation encoder requires transformers. Install "
                "requirements-pharma.txt first."
            ) from exc
        model = _TransformersCLSEncoder(AutoModel.from_pretrained(spec.repo_id))
    else:  # pragma: no cover - registry is tested
        raise ValueError(f"Unsupported foundation backend: {spec.backend!r}")
    model.requires_grad_(False).eval().to(dev)
    return model, dev, spec


def _normalize_patches(batch: torch.Tensor, spec: FoundationModelSpec) -> torch.Tensor:
    mean = torch.tensor(spec.mean, device=batch.device, dtype=batch.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(spec.std, device=batch.device, dtype=batch.dtype).view(
        1, 3, 1, 1
    )
    return (batch - mean) / std


@torch.inference_mode()
def extract_frozen_embeddings(
    patches: np.ndarray,
    model: torch.nn.Module,
    spec: FoundationModelSpec,
    device: str | torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed NCHW float patches without computing gradients."""
    if batch_size < 1:
        raise StageValidationError(
            stage="foundation_embedding",
            subject="batch size",
            observed=batch_size,
            minimum=1,
            guidance="Set batch_size to a positive integer before embedding.",
        )
    require_non_empty(
        patches,
        stage="foundation_embedding",
        subject="NCHW patch batch",
        guidance="Provide at least one patch before foundation embedding.",
    )
    dev = torch.device(device)
    embeddings = []
    for start in range(0, len(patches), batch_size):
        batch = torch.as_tensor(patches[start : start + batch_size]).to(dev)
        output = model(_normalize_patches(batch, spec))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.flatten(1)
        embeddings.append(output.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embedding_cache_path(slide_id: str, cfg: dict[str, Any]) -> Path:
    model_name, _ = _foundation_model_spec_resolved(cfg)
    patch_version = cfg.get("patches", {}).get("version", "v1")
    cache_dir = pharma_processed_dir() / "foundation_embeddings" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{safe_filename(slide_id)}_{patch_version}.npz"


def load_or_extract_slide_embeddings(
    slide_id: str,
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    encoder_bundle: tuple[torch.nn.Module, torch.device, FoundationModelSpec]
    | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return spot-aligned embeddings, reusing a validated per-slide cache."""
    cfg = _resolved_config_arg(cfg)
    cache_path = _embedding_cache_path(slide_id, cfg)
    use_cache = bool(_foundation_config_resolved(cfg).get("cache", True))

    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_spots = cached["spot_ids"]
            cached_embeddings = cached["embeddings"]
        require_non_empty(
            cached_embeddings,
            stage="foundation_embedding_cache",
            subject=f"cached embeddings for slide {slide_id}",
            guidance="Regenerate a non-empty embedding cache for this slide.",
        )
        cache_metadata = pd.DataFrame(
            {
                "slide_id": [slide_id] * len(cached_spots),
                "spot_id": cached_spots.tolist(),
            }
        )
        aligned_labels = align_labels_with_metadata(
            labels,
            cache_metadata,
            stage="foundation_embedding_cache",
            expected_slide_id=slide_id,
            value_row_count=len(cached_embeddings),
        )
        cache_rows = aligned_labels["_patch_source_row"].to_numpy(dtype=np.int64)
        return cached_embeddings[cache_rows], aligned_labels

    patches, aligned_labels = load_slide_patches(slide_id, labels, cfg=cfg)
    # Force a fixed-width Unicode dtype so caches remain readable with
    # allow_pickle=False across pandas/NumPy versions.
    spot_ids = np.asarray(aligned_labels["spot_id"].tolist(), dtype=np.str_)

    if encoder_bundle is None:
        encoder_bundle = load_frozen_encoder(cfg)
    model, dev, spec = encoder_bundle
    batch_size = int(_foundation_config_resolved(cfg).get("batch_size", 64))
    embeddings = extract_frozen_embeddings(
        patches, model, spec, dev, batch_size=batch_size
    )
    if len(embeddings) != len(aligned_labels):
        raise IdentityValidationError(
            stage="foundation_embedding_extraction",
            issues=(
                IdentityIssue(
                    code="cardinality_mismatch",
                    side="values",
                    count=abs(len(embeddings) - len(aligned_labels)),
                ),
            ),
        )
    if embeddings.shape[1] != spec.embedding_dim:
        raise ValueError(
            f"{_foundation_model_spec_resolved(cfg)[0]} returned "
            f"{embeddings.shape[1]} features; "
            f"expected {spec.embedding_dim}."
        )
    if use_cache:
        np.savez_compressed(
            cache_path,
            embeddings=embeddings.astype(np.float32),
            spot_ids=spot_ids,
        )
    return embeddings, aligned_labels


def train_eval_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: pd.DataFrame,
    val_embeddings: np.ndarray,
    val_labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit linear heads on frozen embeddings and evaluate held-out spots."""
    require_non_empty(
        train_embeddings,
        stage="foundation_probe_training",
        subject="training embeddings",
        guidance="Provide at least one training embedding before fitting probes.",
    )
    require_non_empty(
        train_labels,
        stage="foundation_probe_training",
        subject="training labels",
        guidance="Provide at least one training label before fitting probes.",
    )
    require_non_empty(
        val_embeddings,
        stage="foundation_probe_prediction",
        subject="held-out embeddings",
        guidance="Provide at least one held-out embedding before probe prediction.",
    )
    require_non_empty(
        val_labels,
        stage="foundation_probe_prediction",
        subject="held-out labels",
        guidance="Provide at least one held-out label before probe prediction.",
    )
    if cfg is None:
        cfg = load_config()
    cls_col = classification_column(cfg)
    reg_cols = regression_columns(train_labels, cfg)

    y_cls_train = train_labels[cls_col].to_numpy()
    y_cls_val = val_labels[cls_col].to_numpy()
    if np.unique(y_cls_train).size < 2:
        raise ValueError("Linear classification probe needs at least two classes.")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    classifier.fit(train_embeddings, y_cls_train)
    y_cls_pred = classifier.predict(val_embeddings)
    cls_metrics = classification_metrics(y_cls_val, y_cls_pred)

    y_reg_train = train_labels[reg_cols].to_numpy(dtype=np.float64)
    y_reg_val = val_labels[reg_cols].to_numpy(dtype=np.float64)
    reg_preds = np.full_like(y_reg_val, np.nan)
    for target_idx in range(len(reg_cols)):
        target = y_reg_train[:, target_idx]
        finite = np.isfinite(target)
        if finite.sum() < 3 or np.nanstd(target) < 1e-8:
            continue
        regressor = make_pipeline(
            StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        )
        regressor.fit(train_embeddings[finite], target[finite])
        reg_preds[:, target_idx] = regressor.predict(val_embeddings)

    reg_df = regression_metrics(y_reg_val, reg_preds, reg_cols)
    return {
        **cls_metrics,
        "mean_pearson_r": float(reg_df["pearson_r"].mean()) if len(reg_df) else 0.0,
        "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) else 0.0,
        "regression_per_target": reg_df,
        "y_cls": y_cls_val,
        "y_cls_pred": y_cls_pred,
        "y_reg": y_reg_val,
        "y_reg_pred": reg_preds,
        "reg_cols": reg_cols,
    }


def run_foundation_loso(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract each slide once, then run slide-level linear-probe CV."""
    unique_non_empty = [slide_id for slide_id in dict.fromkeys(slide_ids) if slide_id]
    require_non_empty(
        unique_non_empty,
        stage="foundation_loso_admission",
        subject="unique non-empty slide IDs",
        minimum=2,
        guidance="Admit at least two distinct slides before foundation LOSO.",
    )
    require_non_empty(
        labels,
        stage="foundation_loso_admission",
        subject="cohort label rows",
        guidance="Generate non-empty cohort labels before foundation LOSO.",
    )
    if cfg is None:
        cfg = load_config()
    seed = int(cfg.get("seed", 0))
    bundle = load_frozen_encoder(cfg)
    slide_data: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for slide_id in slide_ids:
        slide_data[slide_id] = load_or_extract_slide_embeddings(
            slide_id, labels, cfg=cfg, encoder_bundle=bundle
        )
        embeddings, aligned_labels = slide_data[slide_id]
        require_non_empty(
            embeddings,
            stage="foundation_loso_embedding",
            subject=f"embeddings for slide {slide_id}",
            guidance="Extract at least one aligned embedding for every admitted slide.",
        )
        require_non_empty(
            aligned_labels,
            stage="foundation_loso_embedding",
            subject=f"aligned labels for slide {slide_id}",
            guidance="Retain at least one aligned label for every admitted slide.",
        )

    rows = []
    for fold, (train_slides, val_slide) in enumerate(loso_folds(slide_ids)):
        train_x = np.concatenate([slide_data[s][0] for s in train_slides])
        train_y = pd.concat([slide_data[s][1] for s in train_slides], ignore_index=True)
        val_x, val_y = slide_data[val_slide]
        require_non_empty(
            train_x,
            stage="foundation_probe_training",
            subject=f"concatenated training embeddings for fold {fold}",
            guidance="Retain at least one training embedding across outer-training slides.",
        )
        require_non_empty(
            train_y,
            stage="foundation_probe_training",
            subject=f"concatenated training labels for fold {fold}",
            guidance="Retain at least one training label across outer-training slides.",
        )
        require_non_empty(
            val_x,
            stage="foundation_probe_prediction",
            subject=f"held-out embeddings for slide {val_slide}",
            guidance="Retain at least one held-out embedding before probe prediction.",
        )
        require_non_empty(
            val_y,
            stage="foundation_probe_prediction",
            subject=f"held-out labels for slide {val_slide}",
            guidance="Retain at least one held-out label before probe prediction.",
        )

        # Match the CNN/RF smoke-test bound without invalidating slide holdout.
        quick_max = 500 if os.environ.get("PHARMA_QUICK") else None
        train_x, train_y = _maybe_subsample(train_x, train_y, quick_max)
        val_x, val_y = _maybe_subsample(val_x, val_y, quick_max)
        metrics = train_eval_linear_probe(
            train_x, train_y, val_x, val_y, cfg=cfg, seed=seed
        )
        rows.append(
            {
                "model": "foundation_linear",
                "foundation_model": foundation_model_spec(cfg)[0],
                "fold": fold,
                "val_slide": val_slide,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_pearson_r": metrics["mean_pearson_r"],
                "mean_r2": metrics["mean_r2"],
            }
        )
    return rows
