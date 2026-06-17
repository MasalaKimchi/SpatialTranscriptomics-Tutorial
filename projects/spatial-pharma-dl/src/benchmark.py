"""LOSO benchmark orchestration: CNN vs Random Forest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_config, pharma_outputs_dir
from .eval import evaluate_fold, save_benchmark_report, train_eval_rf_baseline
from .train import _maybe_subsample, load_slide_patches, loso_folds, train_loso


def _benchmark_row(
    model: str, fold: int, val_slide: str, metrics: dict[str, float]
) -> dict[str, Any]:
    return {
        "model": model,
        "fold": fold,
        "val_slide": val_slide,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_pearson_r": metrics["mean_pearson_r"],
        "mean_r2": metrics["mean_r2"],
    }


def run_rf_loso_fold(
    train_slides: list[str],
    val_slide: str,
    labels: pd.DataFrame,
    fold: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Train and evaluate RF baseline for one LOSO fold."""
    train_patches, train_labels = [], []
    for sid in train_slides:
        patches, lab = load_slide_patches(sid, labels)
        train_patches.append(patches)
        train_labels.append(lab)
    val_patches, val_labels = load_slide_patches(val_slide, labels)
    quick_max = 500 if os.environ.get("PHARMA_QUICK") else None
    X_train = np.concatenate(train_patches, axis=0)
    lab_train = pd.concat(train_labels, ignore_index=True)
    X_train, lab_train = _maybe_subsample(X_train, lab_train, quick_max)
    val_patches, val_labels = _maybe_subsample(val_patches, val_labels, quick_max)
    rf = train_eval_rf_baseline(
        X_train,
        lab_train,
        val_patches,
        val_labels,
        seed=seed,
    )
    return _benchmark_row("rf", fold, val_slide, rf)


def run_loso_benchmark(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run CNN LOSO training plus RF baseline; return (benchmark_rows, cnn_results)."""
    cfg = cfg or load_config()
    seed = int(cfg.get("seed", 0))

    cnn_results = train_loso(slide_ids, labels, cfg=cfg)
    rows: list[dict[str, Any]] = []

    for result in cnn_results:
        ev = evaluate_fold(result)
        rows.append(_benchmark_row("cnn", ev["fold"], ev["val_slide"], ev))

    for fold, (train_slides, val_slide) in enumerate(loso_folds(slide_ids)):
        rows.append(run_rf_loso_fold(train_slides, val_slide, labels, fold, seed=seed))

    return rows, cnn_results


def run_and_save_benchmark(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    path: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run full benchmark and write ``benchmark_report.csv``."""
    rows, cnn_results = run_loso_benchmark(slide_ids, labels, cfg=cfg)
    report_path = save_benchmark_report(rows, path=path)
    return report_path, cnn_results
