"""Evaluation metrics, RF baseline, and Grad-CAM interpretability."""

from __future__ import annotations

import json
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
from .validation import StageValidationError, require_non_empty, resolve_config
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactFingerprint,
    ArtifactValidationError,
    admit_artifact,
    build_fingerprint,
    publish_artifact,
    read_artifact_manifest,
)


BENCHMARK_REPORT_COLUMNS = (
    "model",
    "fold",
    "val_slide",
    "balanced_accuracy",
    "macro_f1",
    "mean_pearson_r",
    "mean_r2",
    "experiment",
)


@torch.no_grad()
def predict_cnn(
    model: torch.nn.Module,
    patches: np.ndarray,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise StageValidationError(
            stage="cnn_prediction",
            subject="batch size",
            observed=batch_size,
            minimum=1,
            guidance="Set batch_size to a positive integer before prediction.",
        )
    require_non_empty(
        patches,
        stage="cnn_prediction",
        subject="NCHW patch batch",
        guidance="Provide at least one patch before CNN prediction.",
    )
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
    require_non_empty(
        train_patches,
        stage="rf_training",
        subject="training patch rows",
        guidance="Provide at least one training patch before fitting RF estimators.",
    )
    require_non_empty(
        train_labels,
        stage="rf_training",
        subject="training label rows",
        guidance="Provide at least one training label before fitting RF estimators.",
    )
    require_non_empty(
        val_patches,
        stage="rf_prediction",
        subject="held-out patch rows",
        guidance="Provide at least one held-out patch before RF prediction.",
    )
    require_non_empty(
        val_labels,
        stage="rf_prediction",
        subject="held-out label rows",
        guidance="Provide at least one held-out label before RF prediction.",
    )
    if cfg is None:
        cfg = load_config()
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
        self._handles = (
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        )

    def close(self) -> None:
        """Remove hooks registered on the target layer."""
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

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
    with GradCAM(model, get_gradcam_layer(model)) as cam:
        return cam(x, target_class)


def save_benchmark_report(
    rows: list[dict],
    path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    upstream_lineage: dict[str, object] | None = None,
) -> Path:
    if cfg is None:
        cfg = load_config()
    else:
        cfg = resolve_config(cfg).to_dict()
    if path is None:
        exp = cfg.get("experiment", "v2")
        path = pharma_outputs_dir() / f"benchmark_report_{exp}.csv"
    df = pd.DataFrame(rows)
    df["experiment"] = cfg.get("experiment", "v2")
    df = df.loc[:, list(BENCHMARK_REPORT_COLUMNS)]
    schema = _benchmark_schema(df, cfg)
    fingerprint = _report_fingerprint(
        df, cfg, upstream_lineage=upstream_lineage or {}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_artifact(
        path,
        artifact_kind="report",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["report"],
        fingerprint=fingerprint,
        payload_format="csv",
        payload_schema=schema,
        write_payload=lambda temporary: df.to_csv(temporary, index=False),
        reader=lambda temporary: _read_benchmark_payload(temporary, cfg),
    )
    return path


def _benchmark_schema(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, object]:
    if frame.columns.tolist() != list(BENCHMARK_REPORT_COLUMNS) or frame.empty:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if frame.columns.duplicated().any() or frame[["model", "val_slide", "experiment"]].isnull().any().any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if any(type(value) is not str or not value for value in frame["model"].tolist()):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if any(type(value) is not str or not value for value in frame["val_slide"].tolist()):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    experiment = cfg.get("experiment", "v2")
    if any(type(value) is not str or value != experiment for value in frame["experiment"].tolist()):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if not pd.api.types.is_integer_dtype(frame["fold"]) or (frame["fold"] < 0).any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    numeric = list(BENCHMARK_REPORT_COLUMNS[3:7])
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in numeric):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    if frame.duplicated(["experiment", "model", "fold", "val_slide"]).any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="report")
    return {
        "columns": list(BENCHMARK_REPORT_COLUMNS),
        "rows": int(len(frame)),
        "row_identity": frame[["experiment", "model", "fold", "val_slide"]]
        .to_dict("split")["data"],
    }


def _read_benchmark_payload(path: Path, cfg: dict[str, Any]):
    frame = pd.read_csv(
        path,
        dtype={"model": str, "val_slide": str, "experiment": str},
    )
    return frame, _benchmark_schema(frame, cfg)


def _report_fingerprint(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    upstream_lineage: dict[str, object],
) -> ArtifactFingerprint:
    schema = _benchmark_schema(frame, cfg)
    return build_fingerprint(
        "report",
        {
            "configuration": cfg,
            "source": {"report_schema": list(BENCHMARK_REPORT_COLUMNS)},
            "upstream": upstream_lineage,
            "identity": {"rows": schema["row_identity"]},
        },
    )


def _manifest_expected_fingerprint(
    path: Path,
    *,
    kind: str,
    cfg: dict[str, Any],
    upstream_lineage: dict[str, object] | None = None,
):
    parsed = read_artifact_manifest(path)
    inputs = parsed.fingerprint.to_dict()["inputs"]
    inputs["configuration"] = cfg
    if upstream_lineage is not None:
        inputs["upstream"] = upstream_lineage
    return build_fingerprint(
        kind,
        {
            "configuration": inputs["configuration"],
            "source": inputs["source"],
            "upstream": inputs["upstream"],
            "identity": inputs["identity"],
        },
    )


def load_benchmark_report(
    path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    upstream_lineage: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Load an exact benchmark table only after contract and lineage admission."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    if path is None:
        path = pharma_outputs_dir() / f"benchmark_report_{resolved.get('experiment', 'v2')}.csv"
    expected = _manifest_expected_fingerprint(
        path, kind="report", cfg=resolved, upstream_lineage=upstream_lineage
    )
    admission = admit_artifact(
        path,
        expected_kind="report",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["report"],
        expected_fingerprint=expected,
        reader=lambda candidate: _read_benchmark_payload(candidate, resolved),
    )
    frame, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="report", basename=path.name
        )
    return frame


def _result_table_schema(frame: pd.DataFrame, table_name: str) -> dict[str, object]:
    columns = frame.columns.tolist()
    if type(table_name) is not str or not table_name or frame.empty:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    if any(type(column) is not str or not column for column in columns) or len(columns) != len(set(columns)):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    if frame.isnull().any().any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric_columns and not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    return {
        "table_name": table_name,
        "columns": columns,
        "rows": int(len(frame)),
        "dtypes": [str(frame[column].dtype) for column in columns],
    }


def save_result_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    table_name: str,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
) -> Path:
    """Atomically publish one named retained CSV result table."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    schema = _result_table_schema(frame, table_name)
    fingerprint = build_fingerprint(
        "summary",
        {
            "configuration": resolved,
            "source": {"table_name": table_name, "columns": schema["columns"]},
            "upstream": upstream_lineage or {},
            "identity": {"rows": schema["rows"]},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_artifact(
        path,
        artifact_kind="summary",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["summary"],
        fingerprint=fingerprint,
        payload_format="csv",
        payload_schema=schema,
        write_payload=lambda temporary: frame.to_csv(temporary, index=False),
        reader=lambda temporary: _read_result_table(temporary, table_name, schema["columns"]),
    )
    return path


def _read_result_table(path: Path, table_name: str, columns: list[str]):
    frame = pd.read_csv(path)
    if frame.columns.tolist() != columns:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    return frame, _result_table_schema(frame, table_name)


def load_result_table(
    path: Path,
    *,
    table_name: str,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
) -> pd.DataFrame:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    expected = _manifest_expected_fingerprint(
        path, kind="summary", cfg=resolved, upstream_lineage=upstream_lineage
    )
    parsed = read_artifact_manifest(path)
    declared = json.loads(parsed.payload_schema_json)
    if declared.get("table_name") != table_name or type(declared.get("columns")) is not list:
        raise ArtifactValidationError("payload_schema_mismatch", artifact_kind="summary", basename=path.name)
    admission = admit_artifact(
        path,
        expected_kind="summary",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["summary"],
        expected_fingerprint=expected,
        reader=lambda candidate: _read_result_table(candidate, table_name, declared["columns"]),
    )
    frame, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError("payload_schema_mismatch", artifact_kind="summary", basename=path.name)
    return frame


def _json_payload(value: object, result_name: str) -> tuple[bytes, dict[str, object]]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        restored = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary") from None
    if restored != value:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="summary")
    return raw, {"result_name": result_name, "keys": sorted(value)}


def save_json_result(
    value: dict[str, object],
    path: Path,
    *,
    result_name: str,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
    artifact_kind: str = "summary",
) -> Path:
    """Atomically publish a named canonical JSON result or manifest wrapper."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    raw, schema = _json_payload(value, result_name)
    kind = artifact_kind
    fingerprint = build_fingerprint(
        kind,
        {
            "configuration": resolved,
            "source": {"result_name": result_name, "inner_payload": value},
            "upstream": upstream_lineage or {},
            "identity": {"keys": schema["keys"]},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_artifact(
        path,
        artifact_kind=kind,
        contract_version=ARTIFACT_CONTRACT_VERSIONS[kind],
        fingerprint=fingerprint,
        payload_format="canonical-json",
        payload_schema=schema,
        write_payload=lambda temporary: temporary.write_bytes(raw),
        reader=lambda temporary: _read_json_result(temporary, result_name, schema["keys"], kind),
    )
    return path


def _read_json_result(path: Path, result_name: str, keys: list[str], kind: str):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind=kind) from None
    raw, schema = _json_payload(value, result_name)
    del raw
    if schema["keys"] != keys:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind=kind)
    return value, schema


def load_json_result(
    path: Path,
    *,
    result_name: str,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
    artifact_kind: str = "summary",
) -> dict[str, object]:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    expected = _manifest_expected_fingerprint(
        path, kind=artifact_kind, cfg=resolved, upstream_lineage=upstream_lineage
    )
    parsed = read_artifact_manifest(path)
    declared = json.loads(parsed.payload_schema_json)
    if declared.get("result_name") != result_name or type(declared.get("keys")) is not list:
        raise ArtifactValidationError("payload_schema_mismatch", artifact_kind=artifact_kind, basename=path.name)
    admission = admit_artifact(
        path,
        expected_kind=artifact_kind,
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS[artifact_kind],
        expected_fingerprint=expected,
        reader=lambda candidate: _read_json_result(candidate, result_name, declared["keys"], artifact_kind),
    )
    value, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError("payload_schema_mismatch", artifact_kind=artifact_kind, basename=path.name)
    return value
