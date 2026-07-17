---
title: Codebase Concerns
focus: concerns
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
mapped_at: 2026-07-17
---

# Codebase Concerns

## Priority legend

- **P0**: can invalidate reported scientific results or execute untrusted code.
- **P1**: high-probability correctness, reproducibility, or operational risk.
- **P2**: maintainability, performance, and developer-experience debt worth scheduling.

## Highest-priority risks

### 1. P0 — The outer LOSO test slide is used for early stopping

`projects/spatial-pharma-dl/src/train.py` constructs the validation loader from the nominally held-out `val_slide` and evaluates it after every epoch (lines 73, 94-109, and 143-164). The resulting validation loss selects `best_state`, after which `projects/spatial-pharma-dl/src/eval.py` reports metrics on the same slide. This makes the CNN LOSO estimate optimistic: the test domain influences model selection. Add an inner slide-level split (or a training-only stopping rule), freeze the selected epoch/hyperparameters, and evaluate the outer slide exactly once.

### 2. P0 — The configured stain normalization does not map slides to a common target

In `projects/spatial-pharma-dl/src/patches.py`, `_extract_spot_patches()` sets `stain` to the current slide's estimated matrix when `per_slide_stain_norm` is enabled, then calls `macenko_normalize(raw, stain)` (lines 154-165). `macenko_normalize()` defaults `target_stain` to the same `ref_stain` used for decomposition (lines 76-90), so the operation largely reconstructs each patch in its own basis rather than normalizing it to the cohort reference returned by `fit_reference_stain()`. Preserve separate source and target stain matrices and test that cross-slide color statistics actually converge.

### 3. P0 — Patch caches require unsafe pickle deserialization

`projects/spatial-pharma-dl/src/patches.py` saves `meta.to_dict("list")` into an object-valued NPZ (line 213) and reloads it with `np.load(..., allow_pickle=True)` (lines 223-225). A modified or untrusted cache can execute arbitrary code during loading. Store numeric arrays and strings without object dtype, or place metadata in Parquet/JSON and always use `allow_pickle=False`; document that existing caches must be regenerated.

### 4. P0 — Model checkpoint loading explicitly enables Python pickle

`projects/spatial-pharma-dl/src/models.py` calls `torch.load(..., weights_only=False)` (line 186). PyTorch checkpoints are pickle-backed, so loading an untrusted `.pt` file can execute code. Use a weights-only state dict plus validated JSON metadata, default to `weights_only=True`, and clearly restrict any legacy loader to trusted local artifacts.

### 5. P1 — Label alignment silently discards unmatched spots and does not enforce uniqueness

`projects/spatial-pharma-dl/src/labels.py` performs an unconstrained inner merge in `align_labels_with_patches()` (line 203). `projects/spatial-pharma-dl/src/train.py` then builds an `idx_map` that silently keeps the last duplicate spot ID (lines 44-49). Missing, duplicated, or cross-slide-mislabeled rows can shrink or duplicate data without an error. Validate one-to-one keys, reject null/duplicate IDs, and report exact unmatched counts before training.

### 6. P1 — Cache reuse ignores configuration and source-data changes

`projects/spatial-pharma-dl/src/data.py` reuses any existing processed slide solely by filename (lines 132-149), while `projects/spatial-pharma-dl/src/patches.py` keys patch caches only by slide ID and a manually managed `patches.version` (lines 137-140). Changes to QC thresholds, image scaling, output size, stain behavior, code, or downloaded source data can silently reuse stale artifacts. Include a content/config/code fingerprint in cache manifests and verify it before reuse.

### 7. P1 — Cache and result writes are not atomic or integrity-checked

H5AD, NPZ, Parquet, CSV, JSON, and model files are written directly to their final paths throughout `projects/spatial-pharma-dl/src/data.py`, `patches.py`, `labels.py`, `train.py`, and `eval.py`. An interrupted process can leave a file that exists and is subsequently treated as valid. Write to a same-filesystem temporary path, fsync where warranted, atomically rename, and validate shape/schema before accepting a cache.

### 8. P1 — Configuration is an unvalidated nested dictionary

`projects/spatial-pharma-dl/src/data.py` returns raw `yaml.safe_load()` output (lines 19-22), and callers immediately index nested keys such as `cfg["training"]["batch_size"]` and `cfg["patches"]["output_size"]`. Typos, null YAML, incompatible types, invalid ranges, or missing sections fail late and inconsistently. Introduce a typed schema with cross-field checks (positive dimensions, valid model/task names, PCA limits, class/target consistency) and validate at startup.

### 9. P1 — Reproducibility seeding does not cover the training stack

`utils/st_helpers.py` seeds Python and NumPy only (lines 109-118). CNN initialization, PyTorch augmentation workers, CUDA kernels, and DataLoader generators remain uncontrolled; `projects/spatial-pharma-dl/src/train.py` does not seed Torch or pass a generator/worker initializer. Add centralized Torch CPU/CUDA seeds, deterministic-mode policy, worker seeding, environment/version capture, and reproducibility tests with realistic tolerances.

### 10. P1 — Empty inputs cause low-level failures instead of actionable validation

`projects/spatial-pharma-dl/src/eval.py::predict_cnn()` concatenates empty result lists (lines 39-46); `projects/spatial-pharma-dl/src/train.py` concatenates slide arrays without confirming that slides, aligned spots, classes, and regression targets remain non-empty (lines 68-88); `projects/spatial-pharma-dl/src/patches.py` stacks an empty patch list (line 176). Validate cohort cardinality and per-fold/task support before expensive work begins.

### 11. P1 — Class coverage is not checked per LOSO fold

`projects/spatial-pharma-dl/src/train.py` always creates heads for every configured class (lines 86-88), but does not verify that each training fold contains at least two classes or that every held-out class was observed in training. Metrics can therefore reflect structurally impossible predictions without an explicit warning. Emit fold-level class counts, fail on degenerate training folds, and report unseen-test-class coverage separately.

### 12. P1 — The multi-task regression loss is sensitive to arbitrary target scale

`projects/spatial-pharma-dl/src/train.py` applies one unweighted MSE over all module/gene targets (lines 119-137). Targets are not standardized within the training fold, so high-variance targets dominate the shared encoder and `reg_weight`; fitting scalers globally would also leak. Fit per-target transformations on outer-training data only, persist them, invert predictions for reporting, and handle missing targets with masks.

### 13. P1 — RF preprocessing is fitted independently on validation data

`projects/spatial-pharma-dl/src/eval.py::radiomics_from_patches()` fills missing values using the mean of whichever split is passed (lines 111-117), and `train_eval_rf_baseline()` invokes it separately for train and validation (lines 132-133). This uses held-out feature statistics and can produce inconsistent columns when invalid patches yield `{}` in `patch_features()`. Build a fixed feature schema and fit an sklearn imputer/pipeline on training features only.

### 14. P1 — Border patches have inconsistent physical context and no tissue-quality gate

`projects/spatial-pharma-dl/src/patches.py::extract_patch()` clips crops at image boundaries (lines 41-47), then every crop is stretched to the same output size (lines 95-100). Edge spots therefore see a different field of view and geometry, and blank/background-heavy spots are still trained despite `tissue_fraction` being computed only for the RF arm. Pad to a fixed native extent, record padding, and reject or flag patches using tissue/blur/artifact QC.

### 15. P1 — Macenko estimation lacks numerical and image-shape validation

`projects/spatial-pharma-dl/src/patches.py::stain_matrix_macenko()` assumes uint8 RGB, finite covariance, stable eigendecomposition, and non-empty tail groups (lines 54-73). Degenerate or float-valued images can yield NaNs or invalid matrices that propagate into every patch. Validate RGB dtype/range, tissue pixel count, finite matrix rank/norms, and fall back with an explicit quality flag rather than silently continuing.

### 16. P1 — Cohort processing silently skips missing slides

`projects/spatial-pharma-dl/src/data.py::cohort_summary()`, `labels.py::build_labels_cohort()`, and `patches.py::build_patch_cohort()` catch `FileNotFoundError` and continue. The pipeline can complete on a partial cohort while still printing “Pipeline complete” in `projects/spatial-pharma-dl/scripts/run_pipeline.py`. Require an explicit `allow_partial` mode, persist a cohort manifest, and fail by default when configured slides are absent.

### 17. P1 — Scientific labels are heuristic and lack confidence/provenance controls

`projects/spatial-pharma-dl/src/labels.py::annotate_domain()` classifies a cluster from only its five highest-scoring marker names using substring matching (lines 67-78), then maps any unrecognized domain to `other` (lines 31-37). Substrings such as `VIM` or `COL1` can collide, and no confidence, multiple-testing threshold, or human-reviewed provenance is stored. Use explicit gene sets and enrichment statistics, add confidence/abstention, version label rules, and evaluate sensitivity to label noise.

### 18. P1 — Preprocessing has unguarded dimension and QC edge cases

`projects/spatial-pharma-dl/src/data.py` requests fixed `n_top_genes_hvg`, `n_pcs`, and neighbor dimensions after filtering (lines 75-102) without checking remaining spots/genes or ensuring `n_pcs_neighbors <= computed PCs`. Small or heavily filtered slides can fail deep in Scanpy. Validate post-QC sizes, adapt dimensions deterministically, and record actual parameters in `adata.uns` and output manifests.

### 19. P1 — No automated end-to-end notebook or pipeline verification exists

Tests under `projects/spatial-pharma-dl/tests/` cover lazy imports, notebook-patch idempotence, Grad-CAM hook cleanup, embedding extraction, and synthetic nested LOSO, but not preprocessing, label generation, cache round-trips, patch geometry, CNN folds, RF leakage, or execution of the 20 notebooks. There is also no tracked `.github/workflows/` configuration. Add small fixture data, execute representative notebooks headlessly, and establish unit/integration/scientific-regression CI tiers.

### 20. P1 — Dependency resolution is not reproducible and declarations disagree

`requirements.txt`, `environment.yml`, `pyproject.toml`, and `projects/spatial-pharma-dl/requirements-pharma.txt` use broad lower bounds without a lock or tested upper bounds. README recommends Python 3.10 or 3.11, while `pyproject.toml` requires `>=3.11`; PyArrow is `>=14.0` in `pyproject.toml` but `>=12.0` in the pharma requirements. Consolidate dependency metadata, define supported Python versions, generate lock files, and test the minimum plus locked environment.

## Maintainability and operational debt

### 21. P2 — The pharma package relies on global import-path mutation

`utils/st_helpers.py::setup_pharma_paths()`, `projects/spatial-pharma-dl/src/bootstrap.py`, `projects/spatial-pharma-dl/scripts/run_pipeline.py`, and both test modules mutate `sys.path`. The package is generically named `src`, making collisions likely and imports dependent on execution location/order. Package the subproject under a distinctive import name and install it in editable mode for development.

### 22. P2 — A global OpenMP workaround masks runtime incompatibility

`projects/spatial-pharma-dl/src/bootstrap.py` and `projects/spatial-pharma-dl/scripts/run_pipeline.py` set `KMP_DUPLICATE_LIB_OK=TRUE`. This suppresses duplicate OpenMP runtime failures rather than resolving the incompatible binary stack and can hide crashes or incorrect parallel behavior. Remove the global mutation and document/test a compatible environment.

### 23. P2 — Generated notebook sources are duplicated and prone to drift

`scripts/patch_notebooks.py` is 401 lines of embedded notebook cell strings; `projects/spatial-pharma-dl/scripts/build_notebooks.py` and `build_foundation_notebook.py` similarly duplicate executable code already present under `src/`. Numeric insertion positions in `patch_notebooks.py` are fragile when notebook layouts change, and exact-source idempotence does not detect semantically equivalent edits. Move behavior into importable functions and generate notebooks from stable cell IDs/templates.

### 24. P2 — Committed notebook outputs make review and history expensive

Several root notebooks are very large because executed outputs are committed: `05_histology_image_loading_and_preprocessing.ipynb` is about 3.9 MB, `06_spatial_visualization.ipynb` about 1.9 MB, and `08_spatially_variable_genes.ipynb` about 1.7 MB. Binary/base64 output diffs obscure code review and increase clone history. Adopt output-stripping hooks or Jupytext sources while retaining curated figures under `outputs/figures/`.

### 25. P2 — Internal helpers have become cross-module API

`projects/spatial-pharma-dl/src/benchmark.py` and `foundation.py` import `_maybe_subsample` from `train.py`. This couples evaluation and foundation-model paths to a private training implementation and creates awkward dependency direction. Move fold construction, deterministic subsampling, and shared validation into a neutral data-splitting module with public APIs.

### 26. P2 — Full patch arrays are repeatedly materialized in memory

`projects/spatial-pharma-dl/src/patches.py` builds all slide patches into a Python list and stacks them; `train.py` loads and concatenates every training slide; `benchmark.py` repeats this for each LOSO arm/fold. At 224×224 float32, memory and I/O scale poorly and the same arrays are copied repeatedly. Use chunked uint8/Zarr storage, memory mapping, lazy datasets, and per-slide feature/embedding caching.

### 27. P2 — RF feature extraction is unnecessarily recomputed per fold

`projects/spatial-pharma-dl/src/benchmark.py::run_rf_loso_fold()` reloads patches, while `eval.py::train_eval_rf_baseline()` recomputes GLCM/entropy/radiomics for every train and validation occurrence. Cache versioned, fixed-schema slide-level radiomics once and reuse them across LOSO folds.

### 28. P2 — Public API discovery is obscured by lazy dynamic exports

`projects/spatial-pharma-dl/src/__init__.py` resolves exports through module-level `__getattr__`. This avoids importing optional dependencies, but makes static discovery, typing, and refactoring less direct. Prefer a real installed package with small dependency-specific submodules and `TYPE_CHECKING` declarations or explicit lightweight facades.

### 29. P2 — Assertions are used for runtime data validation

`projects/spatial-pharma-dl/src/patches.py::SpotPatchDataset` uses `assert len(self.patches) == len(self.labels)` (line 279), and generated foundation notebook code uses assertions for artifact integrity. Python can remove assertions under optimization. Replace them with explicit exceptions carrying slide IDs, expected/actual shapes, and recovery guidance.

### 30. P2 — Output provenance is incomplete

`projects/spatial-pharma-dl/scripts/run_pipeline.py` writes a summary containing selected config values and aggregate metrics (lines 124-152), but does not persist the full resolved config, git commit, package versions, dataset checksums, seed/determinism policy, sample counts, or command/environment flags. Add a run manifest alongside every report/model and link all artifacts to a unique run ID.

## Recommended remediation order

1. Correct LOSO model selection and stain normalization; add regression tests that fail on the current behavior.
2. Remove unsafe pickle loading from NPZ and checkpoint paths.
3. Enforce cohort, alignment, class-support, target, and configuration validation.
4. Add cache fingerprints, atomic writes, and complete run provenance.
5. Fix train-only preprocessing/imputation and target scaling within each fold.
6. Establish fixture-backed pipeline/notebook CI and deterministic training controls.
7. Package the pharma project normally, consolidate dependencies, and simplify notebook generation.
8. Optimize patch/radiomics storage only after correctness and cache contracts are stable.
