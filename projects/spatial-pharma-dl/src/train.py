"""LOSO training loop for Spatial Pharma DL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import load_config, pharma_outputs_dir
from .labels import gene_columns
from .models import build_model
from .patches import SpotPatchDataset, load_patch_arrays


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class NormalizedDataset(torch.utils.data.Dataset):
    """Wrap SpotPatchDataset with ImageNet normalization."""

    def __init__(self, base: SpotPatchDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y_cls, y_reg = self.base[idx]
        return _imagenet_normalize(x), y_cls, y_reg


def loso_folds(slide_ids: list[str]) -> list[tuple[list[str], str]]:
    """Return list of (train_slides, val_slide) for leave-one-slide-out."""
    return [([s for s in slide_ids if s != v], v) for v in slide_ids]


def _load_slide_data(
    slide_id: str, labels: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    patches, meta = load_patch_arrays(slide_id)
    sub = labels[labels["slide_id"] == slide_id].merge(
        meta, on=["slide_id", "spot_id"]
    )
    order = sub["spot_id"].tolist()
    idx_map = {s: i for i, s in enumerate(meta["spot_id"])}
    indices = [idx_map[s] for s in order]
    return patches[indices], sub.reset_index(drop=True)


def train_one_fold(
    train_slides: list[str],
    val_slide: str,
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    fold: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Train one LOSO fold; return metrics and model path."""
    cfg = cfg or load_config()
    train_cfg = cfg["training"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_patches, train_labels = [], []
    for sid in train_slides:
        p, lab = _load_slide_data(sid, labels)
        train_patches.append(p)
        train_labels.append(lab)
    val_patches, val_labels = _load_slide_data(val_slide, labels)

    X_train = np.concatenate(train_patches, axis=0)
    X_val = val_patches
    lab_train = pd.concat(train_labels, ignore_index=True)
    lab_val = val_labels

    gene_cols = gene_columns(lab_train)
    n_classes = int(lab_train["cluster_id"].nunique())
    n_genes = len(gene_cols)

    # Remap cluster ids to contiguous 0..K-1 for this fold
    uniq = sorted(lab_train["cluster_id"].unique())
    cls_map = {c: i for i, c in enumerate(uniq)}
    lab_train = lab_train.copy()
    lab_val = lab_val.copy()
    lab_train["cluster_id"] = lab_train["cluster_id"].map(cls_map)
    lab_val["cluster_id"] = lab_val["cluster_id"].map(
        lambda c: cls_map.get(c, -1)
    )
    val_mask = lab_val["cluster_id"] >= 0
    lab_val = lab_val[val_mask].reset_index(drop=True)
    X_val = X_val[val_mask.to_numpy()]

    train_ds = NormalizedDataset(
        SpotPatchDataset(X_train, lab_train, gene_cols=gene_cols)
    )
    val_ds = NormalizedDataset(
        SpotPatchDataset(X_val, lab_val, gene_cols=gene_cols)
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

    model = build_model(n_classes, n_genes).to(device)
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

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yc, yr in train_loader:
            xb, yc, yr = xb.to(device), yc.to(device), yr.to(device)
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
                xb, yc, yr = xb.to(device), yc.to(device), yr.to(device)
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
    out_dir = pharma_outputs_dir() / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"resnet18_fold{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_classes": n_classes,
            "n_genes": n_genes,
            "gene_cols": gene_cols,
            "cls_map": cls_map,
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
        "gene_cols": gene_cols,
        "lab_val": lab_val,
        "X_val": X_val,
        "model": model,
        "device": device,
    }


def train_loso(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run full LOSO training across slides."""
    cfg = cfg or load_config()
    results = []
    for fold, (train_slides, val_slide) in enumerate(loso_folds(slide_ids)):
        print(f"Fold {fold}: train={train_slides}, val={val_slide}")
        res = train_one_fold(train_slides, val_slide, labels, cfg=cfg, fold=fold)
        results.append(res)
    return results
