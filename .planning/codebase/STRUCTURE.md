---
title: Structure
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
mapped_at: 2026-07-17
focus: structure
---

# Repository Structure

## Top-Level Layout

- `README.md` is the primary tutorial guide, notebook index, setup reference, glossary, and troubleshooting document.
- `00_overview_spatial_transcriptomics.ipynb` through `12_summary_research_extensions.ipynb` form the ordered core curriculum.
- `utils/` contains the small importable package shared by root notebooks and the pharma extension.
- `scripts/` contains root-level notebook maintenance and figure-gallery generation utilities.
- `projects/spatial-pharma-dl/` is a nested applied ML project with its own README, configuration, notebooks, source modules, scripts, tests, and report.
- `data/` is the runtime input/cache area for downloaded Visium slides and processed artifacts; its contents are ignored except `data/.gitkeep`.
- `outputs/` is the runtime results area; only canonical PNGs under `outputs/figures/` are intended for source control.
- `requirements.txt`, `environment.yml`, and `pyproject.toml` define the environment and minimal package metadata.

## Root Tutorial Files

The root notebooks are named with two-digit prefixes, making execution and conceptual progression visible in directory listings.
Notebooks `00`-`03` cover orientation, environment, acquisition, and AnnData/spatial metadata.
Notebooks `04`-`08` cover QC, histology preprocessing, visualization, clustering, and spatially variable genes.
Notebooks `09`-`12` cover image features, multimodal integration, optional cell-type annotation, and research extensions.
`utils/__init__.py` marks the shared utility package; nearly all implementation is in `utils/st_helpers.py`.
`scripts/generate_gallery_figures.py` reads processed tutorial artifacts and regenerates selected figures in `outputs/figures/`.
`scripts/patch_notebooks.py` represents notebook cells as dictionaries and applies targeted, idempotent insertions to selected root notebooks.

## Pharma Project Layout

- `projects/spatial-pharma-dl/README.md` describes the scientific question, workflow, current findings, usage modes, and limitations.
- `projects/spatial-pharma-dl/PROJECT_REPORT.md` is the consolidated methods/results record for the extension.
- `projects/spatial-pharma-dl/configs/default.yaml` defines every current cohort and experiment setting.
- `projects/spatial-pharma-dl/notebooks/01_data_curation.ipynb` through `projects/spatial-pharma-dl/notebooks/06_interpretability.ipynb` mirror the staged pipeline.
- `projects/spatial-pharma-dl/notebooks/07_foundation_model_comparison.ipynb` is the dedicated executed foundation-model experiment.
- `projects/spatial-pharma-dl/scripts/run_pipeline.py` is the canonical automated runner.
- `projects/spatial-pharma-dl/scripts/build_notebooks.py` generates notebooks 01-06 from embedded Markdown and Python cell sources.
- `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py` generates notebook 07.
- `projects/spatial-pharma-dl/requirements-pharma.txt` layers PyTorch, model-hub, YAML, and Parquet dependencies on the root environment.
- `projects/spatial-pharma-dl/tests/test_core_refactors.py` protects import behavior, cache handling, notebook patching, and Grad-CAM cleanup.
- `projects/spatial-pharma-dl/tests/test_foundation.py` covers frozen encoders, embedding caches, and nested leave-one-slide-out evaluation.

## Pharma Source Modules

- `projects/spatial-pharma-dl/src/__init__.py` defines the lazy public API and package version.
- `projects/spatial-pharma-dl/src/bootstrap.py` discovers repository/subproject roots and installs them on `sys.path`.
- `projects/spatial-pharma-dl/src/data.py` owns configuration loading, cohort identifiers, AnnData preprocessing, slide caches, and cohort summaries.
- `projects/spatial-pharma-dl/src/labels.py` owns TME class harmonization, marker/module targets, cohort label tables, and patch-label alignment.
- `projects/spatial-pharma-dl/src/patches.py` owns coordinate conversion, patch extraction, Macenko normalization, resizing, radiomic features, caches, and patch datasets.
- `projects/spatial-pharma-dl/src/models.py` owns multitask CNN construction, backbone adaptation, checkpoint loading, and Grad-CAM layer lookup.
- `projects/spatial-pharma-dl/src/transforms.py` owns ImageNet normalization and stochastic training augmentation.
- `projects/spatial-pharma-dl/src/device.py` owns accelerator discovery and human-readable device labels.
- `projects/spatial-pharma-dl/src/train.py` owns patch loading, leave-one-slide-out folds, fold training, early stopping, and checkpoint production.
- `projects/spatial-pharma-dl/src/eval.py` owns prediction, classification/regression metrics, radiomics baselines, Grad-CAM, and report saving.
- `projects/spatial-pharma-dl/src/benchmark.py` combines CNN, RF, and optional frozen-embedding probe results into one benchmark.
- `projects/spatial-pharma-dl/src/foundation.py` owns foundation-model specifications, loading, frozen embedding extraction/caching, and linear probes.
- `projects/spatial-pharma-dl/src/foundation_eval.py` owns task filtering, per-slide embedding transforms, inner model selection, and unbiased outer LOSO evaluation.

## Runtime Directories and Artifacts

`data/raw/` is the conceptual location for original downloads, while Squidpy/Visium material is also present beneath `data/visium/` at runtime.
`data/processed/` stores root `.h5ad` intermediates; `data/processed/pharma/` stores clustered slides, labels, patch tensors, and foundation embedding caches.
`outputs/figures/` stores README-visible tutorial plots.
`outputs/pharma/` stores benchmark CSVs, experiment JSON summaries, model checkpoints, and foundation notebook outputs.
Cache and output directories are created on demand by `utils/st_helpers.py`, `projects/spatial-pharma-dl/src/data.py`, and downstream modules.

## Naming and Placement Conventions

Notebook filenames use numeric prefixes; source modules use lowercase domain nouns; generated outputs use descriptive snake-case filenames.
Shared root behavior belongs in `utils/st_helpers.py`; pharma-specific behavior belongs in `projects/spatial-pharma-dl/src/`.
Thin operational entry points belong in `scripts/`; scientific narrative and exploratory execution belong in notebooks.
Tests live beside the pharma project in `projects/spatial-pharma-dl/tests/`, not in a repository-wide `tests/` directory.
Configuration is colocated with the pharma project rather than at repository root because only the extension consumes it.

## Navigation Guidance

For tutorial behavior, begin with `README.md`, then inspect the relevant numbered notebook and any helper it calls in `utils/st_helpers.py`.
For full pharma execution, begin at `projects/spatial-pharma-dl/scripts/run_pipeline.py` and follow imports into `projects/spatial-pharma-dl/src/`.
For experiment semantics, read `projects/spatial-pharma-dl/configs/default.yaml` before tracing implementation.
For training behavior, follow `benchmark.py` to `train.py`, `models.py`, and `eval.py`; for image preparation, follow `patches.py` and `transforms.py`.
For foundation comparisons, follow `build_foundation_notebook.py` to `foundation.py` and `foundation_eval.py`.
For generated-versus-hand-edited notebook questions, inspect the applicable builder or patcher script before changing an `.ipynb` file directly.
