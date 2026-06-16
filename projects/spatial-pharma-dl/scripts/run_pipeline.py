#!/usr/bin/env python3
"""Run the full Spatial Pharma DL pipeline (phases 1-5 smoke + benchmark)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHARMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHARMA))

import matplotlib

matplotlib.use("Agg")

from utils import st_helpers as st

st.set_seeds()

from src.data import (  # noqa: E402
    cohort_summary,
    load_config,
    pharma_outputs_dir,
    preprocess_cohort,
)
from src.eval import evaluate_fold, save_benchmark_report, train_eval_rf_baseline  # noqa: E402
from src.labels import build_labels_cohort  # noqa: E402
from src.patches import build_patch_cohort, fit_reference_stain, save_patch_index  # noqa: E402
from src.train import loso_folds, train_loso, _load_slide_data  # noqa: E402


def main() -> None:
    cfg = load_config()
    oncology = cfg["cohorts"]["oncology"]
    external = cfg["cohorts"]["external"]
    benchmark = cfg["cohorts"]["benchmark"]
    all_slides = oncology + external + benchmark

    print("=" * 60)
    print("Phase 1: Data curation")
    print("=" * 60)
    preprocess_cohort(all_slides, cfg=cfg)
    summary = cohort_summary(all_slides)
    out = pharma_outputs_dir() / "cohort_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string())
    print("Wrote", out)

    print("=" * 60)
    print("Phase 2: Label engineering")
    print("=" * 60)
    labels = build_labels_cohort(all_slides, cfg=cfg)
    print(f"Labels: {len(labels)} spots")

    print("=" * 60)
    print("Phase 3: Patch dataset")
    print("=" * 60)
    ref_stain = fit_reference_stain(oncology, cfg)
    build_patch_cohort(all_slides, ref_stain=ref_stain, cfg=cfg)
    idx_path = save_patch_index(labels)
    print("Wrote", idx_path)

    print("=" * 60)
    print("Phase 4-5: LOSO training + RF benchmark (breast cohort)")
    print("=" * 60)
    breast_labels = labels[labels["slide_id"].isin(oncology)]
    # Use reduced epochs for CI-style run if env var set
    import os

    if os.environ.get("PHARMA_QUICK"):
        cfg["training"]["epochs"] = 2
        cfg["training"]["patience"] = 1
        print("PHARMA_QUICK=1: epochs=2")

    results = train_loso(oncology, breast_labels, cfg=cfg)
    benchmark_rows = []
    for r in results:
        ev = evaluate_fold(r)
        benchmark_rows.append(
            {
                "model": "cnn",
                "fold": ev["fold"],
                "val_slide": ev["val_slide"],
                "balanced_accuracy": ev["balanced_accuracy"],
                "macro_f1": ev["macro_f1"],
                "mean_pearson_r": ev["mean_pearson_r"],
                "mean_r2": ev["mean_r2"],
            }
        )
        print(
            f"  CNN fold {ev['fold']} {ev['val_slide'][:30]}: "
            f"bal_acc={ev['balanced_accuracy']:.3f} mean_r={ev['mean_pearson_r']:.3f}"
        )

    import numpy as np
    import pandas as pd

    for fold, (train_slides, val_slide) in enumerate(loso_folds(oncology)):
        train_p, train_l = [], []
        for sid in train_slides:
            p, lab = _load_slide_data(sid, breast_labels)
            train_p.append(p)
            train_l.append(lab)
        val_p, val_l = _load_slide_data(val_slide, breast_labels)
        rf = train_eval_rf_baseline(
            np.concatenate(train_p),
            pd.concat(train_l),
            val_p,
            val_l,
            seed=cfg.get("seed", 0),
        )
        benchmark_rows.append(
            {
                "model": "rf",
                "fold": fold,
                "val_slide": val_slide,
                "balanced_accuracy": rf["balanced_accuracy"],
                "macro_f1": rf["macro_f1"],
                "mean_pearson_r": rf["mean_pearson_r"],
                "mean_r2": rf["mean_r2"],
            }
        )
        print(
            f"  RF  fold {fold} {val_slide[:30]}: "
            f"bal_acc={rf['balanced_accuracy']:.3f} mean_r={rf['mean_pearson_r']:.3f}"
        )

    report_path = save_benchmark_report(benchmark_rows)
    print("Wrote", report_path)
    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
