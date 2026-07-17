"""LOSO training loop for Spatial Pharma DL (v2 remediated labels + modules)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import load_config, pharma_outputs_dir
from .device import device_label, resolve_device
from .labels import (
    align_labels_with_patches,
    classification_column,
    regression_columns,
    tme_class_names,
)
from .models import build_model
from .patches import SpotPatchDataset, load_patch_arrays
from .transforms import NormalizedDataset
from .validation import require_non_empty


def _maybe_subsample(
    patches: np.ndarray, labels: pd.DataFrame, max_spots: int | None
) -> tuple[np.ndarray, pd.DataFrame]:
    if max_spots is None or len(labels) <= max_spots:
        return patches, labels
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=max_spots, replace=False)
    return patches[idx], labels.iloc[idx].reset_index(drop=True)


def loso_folds(slide_ids: list[str]) -> list[tuple[list[str], str]]:
    unique_non_empty = [slide_id for slide_id in dict.fromkeys(slide_ids) if slide_id]
    require_non_empty(
        unique_non_empty,
        stage="loso_fold_admission",
        subject="unique non-empty slide IDs",
        minimum=2,
        guidance="Admit at least two distinct slides before creating LOSO folds.",
    )
    return [([s for s in slide_ids if s != v], v) for v in slide_ids]


def load_slide_patches(
    slide_id: str, labels: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    patches, meta = load_patch_arrays(slide_id, cfg=cfg)
    slide_labels = labels[labels["slide_id"] == slide_id]
    aligned = align_labels_with_patches(slide_labels, meta)
    require_non_empty(
        aligned,
        stage="patch_label_alignment",
        subject=f"aligned rows for slide {slide_id}",
        guidance=(
            "Ensure the admitted slide has overlapping label and patch spot IDs "
            "before fold execution."
        ),
    )
    order = aligned["spot_id"].tolist()
    idx_map = {s: i for i, s in enumerate(meta["spot_id"])}
    return patches[[idx_map[s] for s in order]], aligned.reset_index(drop=True)

def train_one_fold(
    train_slides: list[str],
    val_slide: str,
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    fold: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    require_non_empty(
        train_slides,
        stage="cnn_fold_training",
        subject=f"training slide IDs for held-out slide {val_slide}",
        guidance="Provide at least one outer-training slide for this fold.",
    )
    require_non_empty(
        labels,
        stage="cnn_fold_training",
        subject="cohort label rows",
        guidance="Provide non-empty admitted labels before loading fold patches.",
    )
    if cfg is None:
        cfg = load_config()
    train_cfg = cfg["training"]

    train_patches, train_labels = [], []
    for sid in train_slides:
        patches, lab = load_slide_patches(sid, labels, cfg=cfg)
        require_non_empty(
            patches,
            stage="cnn_fold_training",
            subject=f"training patches for slide {sid}",
            guidance="Rebuild a non-empty aligned patch cache for this training slide.",
        )
        require_non_empty(
            lab,
            stage="cnn_fold_training",
            subject=f"training labels for slide {sid}",
            guidance="Retain at least one aligned label row for this training slide.",
        )
        train_patches.append(patches)
        train_labels.append(lab)
    X_val, lab_val = load_slide_patches(val_slide, labels, cfg=cfg)
    require_non_empty(
        X_val,
        stage="cnn_fold_training",
        subject=f"held-out patches for slide {val_slide}",
        guidance="Rebuild a non-empty aligned patch cache for the held-out slide.",
    )
    require_non_empty(
        lab_val,
        stage="cnn_fold_training",
        subject=f"held-out labels for slide {val_slide}",
        guidance="Retain at least one aligned label row for the held-out slide.",
    )

    X_train = np.concatenate(train_patches, axis=0)
    lab_train = pd.concat(train_labels, ignore_index=True)
    require_non_empty(
        X_train,
        stage="cnn_fold_training",
        subject="concatenated training patches",
        guidance="Retain at least one aligned patch across the training slides.",
    )
    require_non_empty(
        lab_train,
        stage="cnn_fold_training",
        subject="concatenated training labels",
        guidance="Retain at least one aligned label across the training slides.",
    )

    quick_max = 500 if os.environ.get("PHARMA_QUICK") else None
    X_train, lab_train = _maybe_subsample(X_train, lab_train, quick_max)
    X_val, lab_val = _maybe_subsample(X_val, lab_val, quick_max)

    cls_col = classification_column(cfg)
    reg_cols = regression_columns(lab_train, cfg)
    dev = resolve_device(device or train_cfg.get("device", "auto"))
    device = str(dev)
    model_name = train_cfg.get("model", "resnet18")
    pretrained = bool(train_cfg.get("pretrained", True))
    print(f"  Device: {device_label(dev)} | backbone: {model_name} | cls={cls_col}")

    # Global TME classes — stable across slides (no per-slide cluster collision)
    n_classes = len(tme_class_names(cfg))
    n_genes = len(reg_cols)

    train_ds = NormalizedDataset(
        SpotPatchDataset(X_train, lab_train, cls_col=cls_col, reg_cols=reg_cols),
        augment=bool(train_cfg.get("augment", False)),
    )
    val_ds = NormalizedDataset(
        SpotPatchDataset(X_val, lab_val, cls_col=cls_col, reg_cols=reg_cols),
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = build_model(
        n_classes, n_genes, model_name=model_name, pretrained=pretrained
    ).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr = 0
    history = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yc, yr in train_loader:
            xb, yc, yr = xb.to(dev), yc.to(dev), yr.to(dev)
            opt.zero_grad()
            pred_cls, pred_reg = model(xb)
            loss = (
                train_cfg["cls_weight"] * cls_loss_fn(pred_cls, yc)
                + train_cfg["reg_weight"] * reg_loss_fn(pred_reg, yr)
            )
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yc, yr in val_loader:
                xb, yc, yr = xb.to(dev), yc.to(dev), yr.to(dev)
                pred_cls, pred_reg = model(xb)
                loss = (
                    train_cfg["cls_weight"] * cls_loss_fn(pred_cls, yc)
                    + train_cfg["reg_weight"] * reg_loss_fn(pred_reg, yr)
                )
                val_loss += loss.item() * len(xb)
        val_loss /= max(len(val_ds), 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= train_cfg["patience"]:
                break

    model.load_state_dict(best_state)
    exp = cfg.get("experiment", "v2")
    out_dir = pharma_outputs_dir() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}_{exp}_fold{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "experiment": exp,
            "pretrained": pretrained,
            "n_classes": n_classes,
            "n_reg_targets": n_genes,
            "cls_col": cls_col,
            "reg_cols": reg_cols,
            "val_slide": val_slide,
            "train_slides": train_slides,
        },
        model_path,
    )

    return {
        "fold": fold,
        "val_slide": val_slide,
        "train_slides": train_slides,
        "model_path": model_path,
        "history": history,
        "cls_col": cls_col,
        "reg_cols": reg_cols,
        "lab_val": lab_val,
        "X_val": X_val,
        "model": model,
        "model_name": model_name,
        "device": device,
    }


def train_loso(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    unique_non_empty = [slide_id for slide_id in dict.fromkeys(slide_ids) if slide_id]
    require_non_empty(
        unique_non_empty,
        stage="loso_training_admission",
        subject="unique non-empty slide IDs",
        minimum=2,
        guidance="Admit at least two distinct slides before LOSO training.",
    )
    require_non_empty(
        labels,
        stage="loso_training_admission",
        subject="cohort label rows",
        guidance="Generate non-empty cohort labels before LOSO training.",
    )
    if cfg is None:
        cfg = load_config()
    results = []
    for fold, (train_slides, val_slide) in enumerate(loso_folds(slide_ids)):
        print(f"Fold {fold}: train={train_slides}, val={val_slide}")
        results.append(
            train_one_fold(train_slides, val_slide, labels, cfg=cfg, fold=fold)
        )
    return results
