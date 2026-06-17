#!/usr/bin/env python3
"""Run the full Spatial Pharma DL pipeline (phases 1-5 smoke + benchmark)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[3]
PHARMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHARMA))

import src.bootstrap  # noqa: F401  # sets KMP_DUPLICATE_LIB_OK + sys.path

import matplotlib

matplotlib.use("Agg")

from utils import st_helpers as st

st.set_seeds()

from src.benchmark import run_and_save_benchmark  # noqa: E402
from src.data import (  # noqa: E402
    cohort_slide_ids,
    cohort_summary,
    load_config,
    pharma_outputs_dir,
    preprocess_cohort,
)
from src.eval import evaluate_fold  # noqa: E402
from src.labels import build_labels_cohort  # noqa: E402
from src.patches import build_patch_cohort, fit_reference_stain, save_patch_index  # noqa: E402


def main() -> None:
    cfg = load_config()
    oncology = cfg["cohorts"]["oncology"]
    all_slides = cohort_slide_ids(cfg)
    train_only = os.environ.get("PHARMA_TRAIN_ONLY")

    if not train_only:
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
    else:
        print("PHARMA_TRAIN_ONLY=1: skipping phases 1-3")
        labels = build_labels_cohort(all_slides, cfg=cfg)

    print("=" * 60)
    print("Phase 4-5: LOSO training + RF benchmark (breast cohort)")
    print("=" * 60)
    breast_labels = labels[labels["slide_id"].isin(oncology)]

    if os.environ.get("PHARMA_QUICK"):
        cfg["training"]["epochs"] = 2
        cfg["training"]["patience"] = 1
        print("PHARMA_QUICK=1: epochs=2")

    report_path, cnn_results = run_and_save_benchmark(oncology, breast_labels, cfg=cfg)
    for ev in (evaluate_fold(r) for r in cnn_results):
        print(
            f"  CNN fold {ev['fold']} {ev['val_slide'][:30]}: "
            f"bal_acc={ev['balanced_accuracy']:.3f} mean_r={ev['mean_pearson_r']:.3f}"
        )

    import pandas as pd

    for row in pd.read_csv(report_path).query("model == 'rf'").itertuples():
        print(
            f"  RF  fold {row.fold} {row.val_slide[:30]}: "
            f"bal_acc={row.balanced_accuracy:.3f} mean_r={row.mean_pearson_r:.3f}"
        )

    print("Wrote", report_path)
    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
