---
title: Architecture
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
mapped_at: 2026-07-17
focus: architecture
---

# Architecture

## System Shape

This repository is a notebook-first spatial transcriptomics tutorial with a nested, code-oriented deep-learning extension.
The root workflow is a linear sequence of notebooks, `00_overview_spatial_transcriptomics.ipynb` through `12_summary_research_extensions.ipynb`.
The extension under `projects/spatial-pharma-dl/` reuses root utilities while adding cohort preprocessing, label engineering, image-patch extraction, model training, and slide-level evaluation.
There is no service process, database, web API, or deployed runtime; execution is local and artifact-driven.

## Architectural Layers

1. **Presentation and orchestration:** root notebooks teach the workflow interactively; `projects/spatial-pharma-dl/notebooks/` exposes the pharma workflow as seven notebooks.
2. **Command-line orchestration:** `projects/spatial-pharma-dl/scripts/run_pipeline.py` runs the pharma stages end to end and is the principal non-notebook entry point.
3. **Shared tutorial utilities:** `utils/st_helpers.py` owns repository-relative paths, random seeds, Visium loading, AnnData caching, image access, gene guards, and Leiden compatibility.
4. **Pharma domain modules:** `projects/spatial-pharma-dl/src/` separates data, labels, patches, models, training, evaluation, benchmarking, and foundation-model logic.
5. **Configuration:** `projects/spatial-pharma-dl/configs/default.yaml` is the central experiment contract for cohorts, preprocessing, targets, patches, training, foundation encoders, and metrics.
6. **Artifact storage:** ignored directories under `data/` hold downloads and intermediate caches; `outputs/` holds reports, models, figures, and experiment summaries.

## Primary Data Flow

The root tutorial fetches a Squidpy Visium object through `utils/st_helpers.py`, then moves an `AnnData` object through QC, normalization, clustering, spatial analysis, image features, and integration notebooks.
Each root notebook can persist or reload `.h5ad` intermediates in `data/processed/`, reducing coupling to an in-memory notebook session.
The pharma runner loads `projects/spatial-pharma-dl/configs/default.yaml`, expands configured cohorts, and downloads/preprocesses each slide via `projects/spatial-pharma-dl/src/data.py`.
Preprocessed slide-level `AnnData` files flow into `projects/spatial-pharma-dl/src/labels.py`, which derives harmonized TME classes, marker-gene values, and gene-module regression targets.
`projects/spatial-pharma-dl/src/patches.py` maps Visium coordinates into H&E pixels, extracts/resizes/stain-normalizes patches, derives radiomic features, and stores compressed patch arrays.
`projects/spatial-pharma-dl/src/train.py` aligns patches and labels by spot ID, creates leave-one-slide-out folds, builds multitask image models from `projects/spatial-pharma-dl/src/models.py`, and writes checkpoints.
`projects/spatial-pharma-dl/src/benchmark.py` coordinates CNN, radiomics Random Forest, and optional frozen-foundation-model arms.
`projects/spatial-pharma-dl/src/eval.py` converts fold predictions into classification/regression metrics and benchmark reports.
`projects/spatial-pharma-dl/src/foundation.py` extracts and caches frozen embeddings, while `projects/spatial-pharma-dl/src/foundation_eval.py` performs nested slide-level probe selection for the dedicated comparison workflow.
Final tabular and JSON results are written under `outputs/pharma/`; foundation embeddings are cached under `data/processed/pharma/foundation_embeddings/`.

## Entry Points

- Human-guided tutorial: open `00_overview_spatial_transcriptomics.ipynb` and proceed numerically through `12_summary_research_extensions.ipynb`.
- Pharma end-to-end CLI: run `projects/spatial-pharma-dl/scripts/run_pipeline.py` from the repository root.
- Pharma interactive workflow: run notebooks in `projects/spatial-pharma-dl/notebooks/` in numeric order.
- Gallery regeneration: run `scripts/generate_gallery_figures.py` after prerequisite root caches exist.
- Root notebook maintenance: run `scripts/patch_notebooks.py` to apply idempotent content patches to selected tutorial notebooks.
- Pharma notebook generation: run `projects/spatial-pharma-dl/scripts/build_notebooks.py` or `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`.
- Import entry point: `projects/spatial-pharma-dl/src/__init__.py` exposes a lazy package-level API so optional heavy dependencies are not imported until needed.

## Boundaries and Dependency Direction

Root notebooks depend on `utils/st_helpers.py`; the utility module does not depend on notebook state.
Pharma modules depend on root `utils/` for paths, seeds, data fetching, and common AnnData operations, making the repository root a required runtime boundary.
Within pharma code, `data.py` is foundational; `labels.py` and `patches.py` build on it; `models.py`, `device.py`, and `transforms.py` are lower-level ML utilities; `train.py` combines those pieces; `eval.py` assesses outputs; `benchmark.py` orchestrates the full comparison.
`foundation.py` shares data, device, label, evaluation, and training utilities, while `foundation_eval.py` is intentionally more self-contained and operates on already-extracted embeddings and labels.
`projects/spatial-pharma-dl/src/bootstrap.py` and `utils/st_helpers.py::setup_pharma_paths` bridge the root/subproject boundary by mutating `sys.path`.
Generated data is not source-controlled: `.gitignore` excludes `data/*`, most of `outputs/*`, `.h5ad`, `.h5`, and `.npz`, while retaining `data/.gitkeep` and gallery PNGs in `outputs/figures/`.

## Architectural Conventions

Repository-relative paths are derived with `pathlib.Path`; runtime code avoids hard-coded absolute paths.
The root workflow uses numbered filenames to encode execution order and cached artifacts to make later notebooks independently runnable.
The pharma workflow treats a whole slide as the validation unit, and `projects/spatial-pharma-dl/src/train.py::loso_folds` encodes that leakage-control boundary.
Heavy libraries such as Scanpy and Squidpy are imported inside functions where practical, preserving lightweight imports for focused utilities and tests.
Configuration is passed as nested dictionaries loaded from YAML rather than typed configuration objects.
The pharma package name is the generic `src`, so scripts and notebooks must establish import paths before importing it.

## Cross-Cutting Concerns

Reproducibility is centralized through `utils/st_helpers.py::set_seeds` and the `seed` value in `projects/spatial-pharma-dl/configs/default.yaml`.
Compute-device selection is centralized in `projects/spatial-pharma-dl/src/device.py`, with CUDA, Apple MPS, and CPU support.
Image normalization/augmentation is centralized in `projects/spatial-pharma-dl/src/transforms.py`; stain normalization and handcrafted image features live in `projects/spatial-pharma-dl/src/patches.py`.
Optional foundation encoders remain frozen, and only lightweight probes are trained; cached embeddings prevent repeated expensive inference.
Tests in `projects/spatial-pharma-dl/tests/` target core refactors, foundation loading/extraction, and nested LOSO behavior rather than the full download-to-report pipeline.
