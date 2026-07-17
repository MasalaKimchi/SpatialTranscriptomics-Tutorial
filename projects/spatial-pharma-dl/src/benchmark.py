"""LOSO benchmark orchestration: CNN, radiomics RF, and frozen embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_config
from .eval import evaluate_fold, save_benchmark_report, train_eval_rf_baseline
from .foundation import run_foundation_loso
from .train import _maybe_subsample, load_slide_patches, loso_folds, train_loso
from .validation import require_non_empty
from utils.artifacts import manifest_path, parse_manifest_bytes


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
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train and evaluate RF baseline for one LOSO fold."""
    require_non_empty(
        train_slides,
        stage="rf_fold_training",
        subject=f"training slide IDs for held-out slide {val_slide}",
        guidance="Provide at least one outer-training slide for this RF fold.",
    )
    require_non_empty(
        labels,
        stage="rf_fold_training",
        subject="cohort label rows",
        guidance="Provide non-empty admitted labels before loading RF fold patches.",
    )
    if cfg is None:
        cfg = load_config()
    train_patches, train_labels = [], []
    for sid in train_slides:
        patches, lab = load_slide_patches(sid, labels, cfg=cfg)
        require_non_empty(
            patches,
            stage="rf_fold_training",
            subject=f"training patches for slide {sid}",
            guidance="Rebuild a non-empty aligned patch cache for this training slide.",
        )
        require_non_empty(
            lab,
            stage="rf_fold_training",
            subject=f"training labels for slide {sid}",
            guidance="Retain at least one aligned label row for this training slide.",
        )
        train_patches.append(patches)
        train_labels.append(lab)
    val_patches, val_labels = load_slide_patches(val_slide, labels, cfg=cfg)
    require_non_empty(
        val_patches,
        stage="rf_fold_training",
        subject=f"held-out patches for slide {val_slide}",
        guidance="Rebuild a non-empty aligned patch cache for the held-out slide.",
    )
    require_non_empty(
        val_labels,
        stage="rf_fold_training",
        subject=f"held-out labels for slide {val_slide}",
        guidance="Retain at least one aligned label row for the held-out slide.",
    )
    quick_max = 500 if os.environ.get("PHARMA_QUICK") else None
    X_train = np.concatenate(train_patches, axis=0)
    lab_train = pd.concat(train_labels, ignore_index=True)
    require_non_empty(
        X_train,
        stage="rf_fold_training",
        subject="concatenated training patches",
        guidance="Retain at least one aligned patch across RF training slides.",
    )
    require_non_empty(
        lab_train,
        stage="rf_fold_training",
        subject="concatenated training labels",
        guidance="Retain at least one aligned label across RF training slides.",
    )
    X_train, lab_train = _maybe_subsample(X_train, lab_train, quick_max)
    val_patches, val_labels = _maybe_subsample(val_patches, val_labels, quick_max)
    rf = train_eval_rf_baseline(
        X_train,
        lab_train,
        val_patches,
        val_labels,
        cfg=cfg,
        seed=seed,
    )
    return _benchmark_row("rf", fold, val_slide, rf)


def run_loso_benchmark(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run configured LOSO benchmark arms; return rows and CNN results."""
    unique_non_empty = [slide_id for slide_id in dict.fromkeys(slide_ids) if slide_id]
    require_non_empty(
        unique_non_empty,
        stage="loso_benchmark_admission",
        subject="unique non-empty slide IDs",
        minimum=2,
        guidance="Admit at least two distinct slides before running LOSO benchmarks.",
    )
    require_non_empty(
        labels,
        stage="loso_benchmark_admission",
        subject="cohort label rows",
        guidance="Generate non-empty cohort labels before running LOSO benchmarks.",
    )
    if cfg is None:
        cfg = load_config()
    seed = int(cfg.get("seed", 0))

    cnn_results = train_loso(slide_ids, labels, cfg=cfg)
    rows: list[dict[str, Any]] = []

    for result in cnn_results:
        ev = evaluate_fold(result)
        rows.append(_benchmark_row("cnn", ev["fold"], ev["val_slide"], ev))

    for fold, (train_slides, val_slide) in enumerate(loso_folds(slide_ids)):
        rows.append(
            run_rf_loso_fold(train_slides, val_slide, labels, fold, cfg=cfg, seed=seed)
        )

    if cfg.get("foundation", {}).get("enabled", False):
        rows.extend(run_foundation_loso(slide_ids, labels, cfg=cfg))

    return rows, cnn_results


def run_and_save_benchmark(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    path: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run the configured benchmark and write its versioned report."""
    rows, cnn_results = run_loso_benchmark(slide_ids, labels, cfg=cfg)
    resolved = load_config() if cfg is None else cfg
    from .labels import _table_fingerprint
    from .patches import _patch_fingerprint

    checkpoints = []
    for result in cnn_results:
        checkpoint_path = Path(result["model_path"])
        manifest = parse_manifest_bytes(
            manifest_path(checkpoint_path).read_bytes(),
            expected_basename=checkpoint_path.name,
        )
        checkpoints.append(manifest.fingerprint.digest)
    report_path = save_benchmark_report(
        rows,
        path=path,
        cfg=resolved,
        upstream_lineage={
            "checkpoints": checkpoints,
            "labels": [
                _table_fingerprint("label_table", [sid], resolved).digest
                for sid in slide_ids
            ],
            "patches": [_patch_fingerprint(sid, resolved).digest for sid in slide_ids],
        },
    )
    return report_path, cnn_results
