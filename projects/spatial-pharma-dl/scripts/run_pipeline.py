#!/usr/bin/env python3
"""Run the Spatial Pharma DL pipeline (v2 remediated experiment)."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[3]
PHARMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHARMA))

import src.bootstrap  # noqa: E402,F401

from src.data import (  # noqa: E402
    SourceAcquisitionError,
    available_processed_slide_ids,
    cohort_slide_ids,
    load_slide,
    load_config,
    pharma_outputs_dir,
    preprocess_cohort,
)
from src.validation import (  # noqa: E402
    CohortAdmissionError,
    PreprocessingManifest,
    admit_run,
    resolve_config,
)


def _load_stages() -> SimpleNamespace:
    """Load scientific and model stages only after final admission succeeds."""
    import matplotlib

    matplotlib.use("Agg")

    from src.benchmark import run_and_save_benchmark
    from src.data import cohort_summary
    from src.eval import evaluate_fold
    from src.eval import (
        load_benchmark_report,
        save_json_result,
        save_result_table,
    )
    from src.labels import build_labels_cohort
    from src.patches import (
        build_patch_cohort,
        fit_reference_stain,
        patch_cache_path,
        patch_reuse_status,
        save_patch_index,
    )
    from utils import st_helpers as st

    return SimpleNamespace(
        st=st,
        run_and_save_benchmark=run_and_save_benchmark,
        cohort_summary=cohort_summary,
        evaluate_fold=evaluate_fold,
        load_benchmark_report=load_benchmark_report,
        save_json_result=save_json_result,
        save_result_table=save_result_table,
        build_labels_cohort=build_labels_cohort,
        build_patch_cohort=build_patch_cohort,
        fit_reference_stain=fit_reference_stain,
        patch_cache_path=patch_cache_path,
        patch_reuse_status=patch_reuse_status,
        save_patch_index=save_patch_index,
    )


def _need_patch_rebuild(cfg: dict, all_slides: list[str], patch_reuse_status) -> bool:
    if os.environ.get("PHARMA_FORCE_PATCHES"):
        return True
    return any(not patch_reuse_status(sid, cfg).reusable for sid in all_slides)


def _curate_sources(cfg: dict, slide_ids: list[str]):
    """Attempt remote sources and finalize strict or partial admission once."""
    allow_partial = cfg["cohort_policy"]["allow_partial"]
    successful: list[str] = []
    failures: dict[str, str] = {}
    first_source_error: SourceAcquisitionError | None = None
    unattempted: list[str] = []
    for index, slide_id in enumerate(slide_ids):
        try:
            preprocess_cohort([slide_id], cfg=cfg)
            successful.append(slide_id)
        except SourceAcquisitionError as exc:
            if first_source_error is None:
                first_source_error = exc
            failures[slide_id] = "Source loading failed for the configured slide."
            if not allow_partial:
                unattempted = slide_ids[index + 1 :]
                break
    try:
        admission_kwargs = {
            "available_slide_ids": successful,
            "failures": failures,
        }
        if unattempted:
            admission_kwargs["unattempted_slide_ids"] = unattempted
        return admit_run(cfg, **admission_kwargs)
    except CohortAdmissionError as exc:
        if first_source_error is not None:
            raise exc from first_source_error
        raise


def _assemble_preprocessing_manifest(final_admitted) -> PreprocessingManifest:
    """Validate complete per-slide preprocessing facts in admitted order."""
    slide_ids = list(final_admitted.slide_ids)
    records = []
    for slide_id in slide_ids:
        adata = load_slide(slide_id)
        records.append(adata.uns.get("spatial_pharma_preprocessing"))
    return PreprocessingManifest(slide_ids=slide_ids, records=records)


def main() -> None:
    cfg = load_config()
    if os.environ.get("PHARMA_FOUNDATION"):
        cfg["foundation"]["enabled"] = True
    if os.environ.get("PHARMA_QUICK"):
        cfg["training"]["epochs"] = 2
        cfg["training"]["patience"] = 1

    resolved = resolve_config(cfg)
    cfg = resolved.to_dict()
    configured_slides = cohort_slide_ids(cfg)
    train_only = bool(os.environ.get("PHARMA_TRAIN_ONLY"))

    exp = cfg.get("experiment", "v2")
    print("=" * 60)
    print(f"Experiment: {exp}")
    print("=" * 60)

    if train_only:
        print("PHARMA_TRAIN_ONLY=1: skipping phase 1")
        final_admitted = admit_run(
            cfg,
            available_slide_ids=available_processed_slide_ids(configured_slides),
        )
    else:
        print("Phase 1: Data curation")
        admit_run(cfg, available_slide_ids=None)  # provisional: never published
        final_admitted = _curate_sources(cfg, configured_slides)

    all_slides = list(final_admitted.slide_ids)
    oncology = [
        record.slide_id
        for record in final_admitted.manifest.included
        if record.cohort == "oncology"
    ]
    stages = _load_stages()
    preprocessing_manifest = _assemble_preprocessing_manifest(final_admitted)
    stages.st.set_seeds(cfg["seed"])

    out_dir = pharma_outputs_dir()
    manifest_path = out_dir / "cohort_manifest.json"
    stages.save_json_result(
        final_admitted.manifest.to_dict(),
        manifest_path,
        result_name="cohort_manifest",
        cfg=cfg,
        upstream_lineage={
            "admitted_slides": list(final_admitted.slide_ids),
        },
        artifact_kind="cohort_manifest",
    )
    preprocessing_manifest_path = out_dir / "preprocessing_manifest.json"
    stages.save_json_result(
        preprocessing_manifest.to_dict(),
        preprocessing_manifest_path,
        result_name="preprocessing_manifest",
        cfg=cfg,
        upstream_lineage={
            "cohort_manifest_sha256": hashlib.sha256(
                final_admitted.manifest.canonical_json.encode("utf-8")
            ).hexdigest(),
        },
        artifact_kind="preprocessing_manifest",
    )

    if not train_only:
        summary = stages.cohort_summary(all_slides)
        out = out_dir / "cohort_summary.csv"
        stages.save_result_table(
            summary,
            out,
            table_name="cohort_summary",
            cfg=cfg,
            upstream_lineage={
                "preprocessing_manifest_sha256": hashlib.sha256(
                    preprocessing_manifest.canonical_json.encode("utf-8")
                ).hexdigest(),
            },
        )
        print(summary.to_string())
        print("Wrote", out)
    print("Phase 2: Label engineering (harmonized TME + modules)")
    labels = stages.build_labels_cohort(all_slides, cfg=cfg)
    print(f"Labels: {len(labels)} spots")

    if not train_only or _need_patch_rebuild(
        cfg, all_slides, stages.patch_reuse_status
    ):
        print(
            "Phase 3: Patch dataset (context_scale=%s, version=%s)"
            % (
                cfg["patches"].get("context_scale", 1.0),
                cfg["patches"].get("version", "v1"),
            )
        )
        ref_stain = stages.fit_reference_stain(oncology, cfg)
        stages.build_patch_cohort(all_slides, ref_stain=ref_stain, cfg=cfg)
    else:
        print("Phase 3: using cached v2 patches")

    idx_path = stages.save_patch_index(labels, cfg=cfg)
    print("Wrote", idx_path)

    benchmark_arms = "CNN + RF"
    if cfg.get("foundation", {}).get("enabled", False):
        benchmark_arms += " + frozen-FM probe"
    print(f"Phase 4-5: LOSO benchmark ({benchmark_arms}; breast cohort)")
    breast_labels = labels[labels["slide_id"].isin(oncology)]

    if os.environ.get("PHARMA_QUICK"):
        print("PHARMA_QUICK=1: epochs=2")

    report_path, cnn_results = stages.run_and_save_benchmark(
        oncology, breast_labels, cfg=cfg
    )
    for result in cnn_results:
        ev = stages.evaluate_fold(result)
        print(
            f"  CNN fold {ev['fold']} {ev['val_slide'][:30]}: "
            f"bal_acc={ev['balanced_accuracy']:.3f} mean_r={ev['mean_pearson_r']:.3f}"
        )

    report = stages.load_benchmark_report(report_path, cfg=cfg)
    for row in report.query("model != 'cnn'").itertuples():
        print(
            f"  {row.model:<17} fold {row.fold} {row.val_slide[:30]}: "
            f"bal_acc={row.balanced_accuracy:.3f} mean_r={row.mean_pearson_r:.3f}"
        )

    summary = {
        "experiment": exp,
        "classification_col": cfg["labels"]["classification_col"],
        "regression_targets": cfg["labels"]["regression_targets"],
        "context_scale": cfg["patches"].get("context_scale"),
        "patch_version": cfg["patches"].get("version"),
        "cnn_mean_balanced_accuracy": float(
            report.query("model == 'cnn'")["balanced_accuracy"].mean()
        ),
        "cnn_mean_pearson_r": float(
            report.query("model == 'cnn'")["mean_pearson_r"].mean()
        ),
        "rf_mean_balanced_accuracy": float(
            report.query("model == 'rf'")["balanced_accuracy"].mean()
        ),
        "rf_mean_pearson_r": float(
            report.query("model == 'rf'")["mean_pearson_r"].mean()
        ),
    }
    foundation_rows = report.query("model == 'foundation_linear'")
    if len(foundation_rows):
        summary["foundation_mean_balanced_accuracy"] = float(
            foundation_rows["balanced_accuracy"].mean()
        )
        summary["foundation_mean_pearson_r"] = float(
            foundation_rows["mean_pearson_r"].mean()
        )
    summary_path = out_dir / f"experiment_{exp}_summary.json"
    stages.save_json_result(
        summary,
        summary_path,
        result_name="experiment_summary",
        cfg=cfg,
        upstream_lineage={
            "report_content_sha256": hashlib.sha256(
                report.to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "cohort_manifest_sha256": hashlib.sha256(
                final_admitted.manifest.canonical_json.encode("utf-8")
            ).hexdigest(),
        },
    )
    print("Wrote", report_path)
    print("Wrote", summary_path)
    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
