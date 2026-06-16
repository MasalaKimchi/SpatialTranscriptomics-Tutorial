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
from .labels import gene_columns
from .models import build_model
from .patches import patch_features, load_patch_arrays


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_cnn(
    model: torch.nn.Module,
    patches: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CNN inference; return (cls_logits, reg_preds)."""
    model.eval()
    all_cls, all_reg = [], []
    for i in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[i : i + batch_size]).to(device)
        batch = _imagenet_normalize(batch)
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
    y_true: np.ndarray, y_pred: np.ndarray, gene_names: list[str]
) -> pd.DataFrame:
    rows = []
    for j, gene in enumerate(gene_names):
        yt, yp = y_true[:, j], y_pred[:, j]
        if np.std(yt) < 1e-8:
            continue
        r, _ = pearsonr(yt, yp)
        rows.append(
            {
                "gene": gene.replace("gene_", ""),
                "pearson_r": float(r),
                "r2": float(r2_score(yt, yp)),
                "mae": float(np.abs(yt - yp).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fold(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one LOSO fold from train_one_fold output."""
    model = result["model"]
    device = result["device"]
    X_val = result["X_val"]
    lab_val = result["lab_val"]
    gene_cols = result["gene_cols"]

    logits, reg_pred = predict_cnn(model, X_val, device=device)
    y_cls = lab_val["cluster_id"].to_numpy()
    y_cls_pred = logits.argmax(axis=1)
    cls_metrics = classification_metrics(y_cls, y_cls_pred)

    y_reg = lab_val[gene_cols].to_numpy()
    reg_df = regression_metrics(y_reg, reg_pred, gene_cols)

    return {
        "fold": result["fold"],
        "val_slide": result["val_slide"],
        **cls_metrics,
        "mean_pearson_r": float(reg_df["pearson_r"].mean()) if len(reg_df) else 0.0,
        "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) else 0.0,
        "regression_per_gene": reg_df,
        "y_cls": y_cls,
        "y_cls_pred": y_cls_pred,
        "confusion_matrix": confusion_matrix(y_cls, y_cls_pred),
    }


def radiomics_from_patches(patches: np.ndarray) -> pd.DataFrame:
    """Extract 15 handcrafted features from CHW float patches."""
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
    seed: int = 0,
) -> dict[str, Any]:
    """RF baseline (notebook 10 parity) on radiomics features."""
    gene_cols = gene_columns(train_labels)
    X_train = radiomics_from_patches(train_patches).to_numpy()
    X_val = radiomics_from_patches(val_patches).to_numpy()

    y_cls_train = train_labels["cluster_id"].to_numpy()
    y_cls_val = val_labels["cluster_id"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_cls_train)
    y_cls_pred = clf.predict(X_val)
    cls_metrics = classification_metrics(y_cls_val, y_cls_pred)

    y_reg_train = train_labels[gene_cols].to_numpy()
    y_reg_val = val_labels[gene_cols].to_numpy()
    reg_preds = np.zeros_like(y_reg_val)
    for j in range(len(gene_cols)):
        reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        reg.fit(X_train, y_reg_train[:, j])
        reg_preds[:, j] = reg.predict(X_val)

    reg_df = regression_metrics(y_reg_val, reg_preds, gene_cols)
    return {
        **cls_metrics,
        "mean_pearson_r": float(reg_df["pearson_r"].mean()) if len(reg_df) else 0.0,
        "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) else 0.0,
        "regression_per_gene": reg_df,
    }


class GradCAM:
    """Grad-CAM for ResNet18 conv layer (layer4)."""

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

    def __call__(
        self, x: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
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
    device: str = "cpu",
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single patch."""
    model.eval()
    x = torch.from_numpy(patch_chw).unsqueeze(0).to(device)
    x = _imagenet_normalize(x)
    target_layer = model.features[-1][-1].conv2  # last conv in layer4
    cam = GradCAM(model, target_layer)
    return cam(x, target_class)


def save_benchmark_report(rows: list[dict], path: Path | None = None) -> Path:
    """Save benchmark comparison CSV."""
    path = path or pharma_outputs_dir() / "benchmark_report.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
