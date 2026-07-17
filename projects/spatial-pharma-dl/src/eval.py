"""Evaluation metrics, RF baseline, and Grad-CAM interpretability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
)
from scipy.stats import pearsonr

from .data import load_config, pharma_outputs_dir
from .device import resolve_device
from .labels import classification_column, regression_columns
from .models import get_gradcam_layer
from .patches import patch_features
from .transforms import imagenet_normalize


@torch.no_grad()
def predict_cnn(
    model: torch.nn.Module,
    patches: np.ndarray,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    dev = resolve_device(device or "cpu")
    model.eval()
    all_cls, all_reg = [], []
    for i in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[i : i + batch_size]).to(dev)
        batch = imagenet_normalize(batch)
        pred_cls, pred_reg = model(batch)
        all_cls.append(pred_cls.cpu().numpy())
        all_reg.append(pred_reg.cpu().numpy())
    return np.concatenate(all_cls), np.concatenate(all_reg)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]
) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(target_names):
        yt, yp = y_true[:, j].astype(np.float64), y_pred[:, j].astype(np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 3 or np.std(yt[mask]) < 1e-8:
            continue
        yt, yp = yt[mask], yp[mask]
        r, _ = pearsonr(yt, yp)
        if not np.isfinite(r):
            continue
        r2 = r2_score(yt, yp) if np.all(np.isfinite(yp)) else float("nan")
        label = name.replace("gene_", "").replace("module_", "")
        rows.append(
            {
                "target": label,
                "pearson_r": float(r),
                "r2": float(r2) if np.isfinite(r2) else 0.0,
                "mae": float(np.abs(yt - yp).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fold(result: dict[str, Any]) -> dict[str, Any]:
    model = result["model"].to("cpu")
    X_val = result["X_val"]
    lab_val = result["lab_val"]
    cls_col = result["cls_col"]
    reg_cols = result["reg_cols"]

    logits, reg_pred = predict_cnn(model, X_val, device="cpu")
    y_cls = lab_val[cls_col].to_numpy()
    y_cls_pred = logits.argmax(axis=1)
    cls_metrics = classification_metrics(y_cls, y_cls_pred)

    y_reg = lab_val[reg_cols].to_numpy()
    reg_df = regression_metrics(y_reg, reg_pred, reg_cols)

    return {
        "fold": result["fold"],
        "val_slide": result["val_slide"],
        **cls_metrics,
        "mean_pearson_r": float(reg_df["pearson_r"].mean()) if len(reg_df) else 0.0,
        "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) else 0.0,
        "regression_per_target": reg_df,
        "y_cls": y_cls,
        "y_cls_pred": y_cls_pred,
        "confusion_matrix": confusion_matrix(y_cls, y_cls_pred),
    }


def radiomics_from_patches(patches: np.ndarray) -> pd.DataFrame:
    rows = []
    for p in patches:
        hwc = (p.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        rows.append(patch_features(hwc))
    df = pd.DataFrame(rows)
    return df.fillna(df.mean(numeric_only=True))


def train_eval_rf_baseline(
    train_patches: np.ndarray,
    train_labels: pd.DataFrame,
    val_patches: np.ndarray,
    val_labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    cls_col = classification_column(cfg)
    reg_cols = regression_columns(train_labels, cfg)

    X_train = radiomics_from_patches(train_patches).to_numpy()
    X_val = radiomics_from_patches(val_patches).to_numpy()

    y_cls_train = train_labels[cls_col].to_numpy()
    y_cls_val = val_labels[cls_col].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_cls_train)
    y_cls_pred = clf.predict(X_val)
    cls_metrics = classification_metrics(y_cls_val, y_cls_pred)

    y_reg_train = train_labels[reg_cols].to_numpy(dtype=np.float64)
    y_reg_val = val_labels[reg_cols].to_numpy(dtype=np.float64)
    reg_preds = np.full_like(y_reg_val, np.nan)
    for j in range(len(reg_cols)):
        yt = y_reg_train[:, j]
        if not np.isfinite(yt).all() or np.nanstd(yt) < 1e-8:
            continue
        reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        reg.fit(X_train, np.nan_to_num(yt, nan=np.nanmean(yt)))
        reg_preds[:, j] = reg.predict(X_val)

    reg_df = regression_metrics(y_reg_val, reg_preds, reg_cols)
    return {
        **cls_metrics,
        "mean_pearson_r": float(reg_df["pearson_r"].mean()) if len(reg_df) else 0.0,
        "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) else 0.0,
        "regression_per_target": reg_df,
    }


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        pred_cls, _ = self.model(x)
        if target_class is None:
            target_class = pred_cls.argmax(dim=1).item()
        score = pred_cls[0, target_class]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def grad_cam_for_patch(
    model: torch.nn.Module,
    patch_chw: np.ndarray,
    target_class: int | None = None,
    device: str | None = None,
) -> np.ndarray:
    dev = resolve_device(device or "cpu")
    model.eval()
    x = torch.from_numpy(patch_chw).unsqueeze(0).to(dev)
    x = imagenet_normalize(x)
    cam = GradCAM(model, get_gradcam_layer(model))
    return cam(x, target_class)


def save_benchmark_report(
    rows: list[dict],
    path: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> Path:
    cfg = cfg or load_config()
    if path is None:
        exp = cfg.get("experiment", "v2")
        path = pharma_outputs_dir() / f"benchmark_report_{exp}.csv"
    df = pd.DataFrame(rows)
    df["experiment"] = cfg.get("experiment", "v2")
    df.to_csv(path, index=False)
    return path
