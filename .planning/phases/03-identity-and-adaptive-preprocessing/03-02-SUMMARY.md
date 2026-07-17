---
phase: 03-identity-and-adaptive-preprocessing
plan: "02"
subsystem: preprocessing
tags: [python, scanpy, anndata, adaptive-dimensions, provenance, offline-testing]

requires:
  - phase: 03-identity-and-adaptive-preprocessing
    provides: Exact compound AnnData spot identity from Plan 03-01
  - phase: 02-validated-run-and-cohort-admission
    provides: Exact-type configuration and final admitted cohort contracts
provides:
  - Pure two-stage post-QC and post-HVG dimension resolution with structured scientific nonviability
  - Exact finalized HVG, PCA, neighbor, and graph-PC orchestration in preprocess_slide
  - Canonical AnnData preprocessing facts and admitted-order run-level provenance
affects: [phase-04-artifacts, phase-06-fold-admission, phase-10-environment]

tech-stack:
  added: []
  patterns:
    - Observed post-QC counts resolve the HVG call before normalization-dependent graph work
    - Actual selected HVGs resolve PCA rank and graph dimensions before PCA or neighbors
    - Exact JSON primitive admission precedes manifest comparison, serialization, and publication

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py
  modified:
    - projects/spatial-pharma-dl/src/validation.py
    - projects/spatial-pharma-dl/src/data.py
    - projects/spatial-pharma-dl/scripts/run_pipeline.py
    - projects/spatial-pharma-dl/tests/conftest.py
    - projects/spatial-pharma-dl/tests/test_cohort_admission.py

key-decisions:
  - "Post-QC resolution and post-HVG finalization are separate pure stages so actual selected genes, not requested HVGs, determine PCA rank."
  - "AnnData carries both the exact primitive tree and canonical JSON; load_slide restores the tree from safe JSON after H5AD converts scalar storage types."
  - "The run manifest is fully admitted in final cohort order before either manifest write or any label, patch, model, or report stage."

requirements-completed: [VAL-05]

duration: 20min
completed: 2026-07-17
---

# Phase 3 Plan 02: Adaptive Preprocessing and Provenance Summary

**Observed slide cardinalities now resolve legal Scanpy dimensions exactly once and remain reconstructable in AnnData and admitted-order run provenance.**

## Performance

- **Duration:** 20 min
- **Completed:** 2026-07-17
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added import-light frozen preprocessing resolutions and structured errors that reject hostile/non-exact inputs before arithmetic and distinguish post-QC from post-HVG scientific nonviability.
- Derived the HVG call from observed post-QC genes, then derived PCA rank, neighbors, and graph PCs from actual selected HVGs and retained spots using explicit deterministic caps and reason codes.
- Guarded supplied configuration and exact compound observation identity before Scanpy import, AnnData copying, seed mutation, or scientific work.
- Preserved the established QC, normalization, HVG, scale, PCA, neighbors, UMAP, and Leiden order while passing finalized values exactly once without retry or implicit fallback.
- Published fresh canonical counts, exclusions, requested/resolved values, and reasons in AnnData and restored exact primitives safely after H5AD reads.
- Added a frozen preprocessing run manifest that admits the complete candidate tree as exact finite JSON primitives, preserves final cohort order, and publishes before downstream label/model effects.

## Task Commits

1. **Task 1: Build Wave 0 dimension matrices and the pure two-stage resolver** - `8b30394`
2. **Task 2: Integrate observed counts and finalized dimensions into preprocess_slide** - `a0d0033`
3. **Task 3: Publish additive admitted-order preprocessing run provenance** - `684d93b`

## Test Evidence

- Pure resolver Wave 0: 12 focused offline tests passed; scoped Ruff passed.
- Orchestration/guard/H5AD gate: 3 focused tests passed; 77 affected identity, synthetic AnnData, and empty-boundary regressions passed; scoped Ruff passed.
- Provenance/cohort gate: 11 focused tests passed; scoped Ruff passed.
- Complete adaptive preprocessing module: 18 tests passed.
- Affected validation, AnnData, cohort, identity, and empty-boundary suite: 128 tests passed.
- Canonical repository gate: Ruff passed and all 201 offline tests passed without downloads or model weights.
- Static inspection confirmed one HVG, PCA, and neighbors call, finalized parameters, and no graph-stage retry or solver switch.

## Decisions Made

- Scientific minima are three post-QC spots, two post-QC genes, two actual HVGs, one PCA component, and two legal neighbors; failure is structured and occurs before graph work.
- Every requested dimension records either `requested_value_accepted` or a parameter-specific cap reason, making independent and joint caps distinguishable.
- The additive schema is `spatial-pharma-preprocessing-manifest-v1`; `cohort-manifest-v1`, existing configuration keys, output names, signatures, and notebook/CLI entry points remain unchanged.
- H5AD's storage-level NumPy scalar restoration is normalized only from the canonical safe JSON sibling in `load_slide`; untrusted candidate manifest values are never coerced.

## Deviations from Plan

- Updated `test_cohort_admission.py` so the existing successful runner fixture supplies the newly required per-slide preprocessing facts. This is regression-fixture maintenance, not production scope expansion.
- Added a canonical JSON sibling in AnnData to recover exact built-in primitives after H5AD scalar decoding while retaining the required structured tree under `spatial_pharma_preprocessing`.

## Issues Encountered

- The active pandas/anndata combination restores nested H5AD integer scalars as NumPy values and cannot write Arrow-backed fixture strings directly. Tests normalize fixture string storage, while production `load_slide` restores the validated exact primitive preprocessing tree from canonical JSON.
- Existing optional pandas accelerator and legacy notebook cell-ID warnings remain; all mandatory gates pass and Phase 10 owns environment reconciliation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VAL-05 has direct cap, nonviability, orchestration, H5AD, hostile-input, and runner-publication evidence.
- Plan 03-03 can add real Scanpy integration evidence without changing the resolver or manifest contracts.
- Atomic writes, checksums, completion markers, and fingerprints remain explicitly assigned to Phase 4.

## Self-Check: PASSED

- The new test file and all three task commits exist.
- Every plan verification command and the complete offline gate pass.
- Public preprocessing and runner entry points remain stable; the only new output is the documented additive preprocessing manifest.
- No artifact-durability, cache-format, fold-policy, leakage, image-science, or label-confidence work entered the diff.

---
*Phase: 03-identity-and-adaptive-preprocessing*
*Completed: 2026-07-17*
