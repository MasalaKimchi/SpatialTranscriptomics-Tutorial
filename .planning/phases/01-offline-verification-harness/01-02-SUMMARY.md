---
phase: 01-offline-verification-harness
plan: "02"
subsystem: testing
tags: [pytest, anndata, npz, parquet, pytorch, nbformat, loso]

requires:
  - phase: 01-offline-verification-harness
    provides: Strict offline tier isolation and deterministic scientific fixture factories from Plan 01-01
provides:
  - Safe primitive NPZ, Parquet, JSON, and H5AD fixture-format evidence
  - Real synthetic AnnData coverage across image, scale, coordinate, patch, and valid-alignment seams
  - Bounded CPU multi-head model and deterministic stubbed LOSO orchestration smoke coverage
  - Structural validation for all 20 committed root and pharma notebooks
affects: [phase-01-plan-03, verification, ci, artifact-safety, evaluation]

tech-stack:
  added: []
  patterns:
    - Representative offline evidence uses shared fixed-seed factories and pytest tmp_path
    - Expensive or scientifically unresolved production behavior is isolated behind test doubles

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py
    - projects/spatial-pharma-dl/tests/test_synthetic_anndata.py
    - projects/spatial-pharma-dl/tests/test_model_fold_smoke.py
    - projects/spatial-pharma-dl/tests/test_notebook_structure.py
  modified: []

key-decisions:
  - "Artifact round-trip evidence remains fixture-format-only and never invokes the unsafe production patch or checkpoint readers."
  - "LOSO orchestration is characterized with a deterministic train_one_fold stub and a forbidden output-path guard; real fold training remains later scientific work."
  - "Notebook checks validate structure and established kernel families while tolerating legacy missing cell IDs and never executing or rewriting notebooks."

patterns-established:
  - "Safe format evidence: primitive numeric/string payloads are read with object deserialization disabled and adversarial object access must raise."
  - "Bounded model evidence: use a tiny local model for optimization and one public no-pretrained backbone shape smoke."
  - "Notebook evidence: assert public numeric sequences, nbformat validity, allowed cell types, textual code source, and directory-specific kernels."

requirements-completed: [TEST-01]

duration: 7min
completed: 2026-07-17
---

# Phase 1 Plan 02: Representative Offline Evidence Summary

**Safe artifact formats, real synthetic spatial integration, bounded model/LOSO behavior, and structural checks for all 20 notebooks now run in the strict offline tier**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-17T04:23:08Z
- **Completed:** 2026-07-17T04:30:33Z
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- Added four safe primitive artifact round trips, including explicit rejection of object-valued NPZ access.
- Exercised a real deterministic AnnData across H5AD, Visium image/scalefactor access, coordinate scaling, patch extraction, and valid one-to-one label alignment.
- Added a finite one-step CPU multi-head optimization test, a public ResNet18 no-pretrained smoke, and deterministic three-slide LOSO orchestration without writes.
- Discovered and validated exactly 13 root and seven pharma notebooks without execution or modification.
- Expanded the complete strict-marker offline suite to 50 passing tests and kept the repository Ruff scope green.

## Task Commits

Each task was committed atomically:

1. **Task 1: Prove safe primitive artifact fixture round trips** - `3937a93` (test)
2. **Task 2: Exercise a real synthetic AnnData spatial path** - `e91a665` (test)
3. **Task 3: Add a bounded CPU model and LOSO orchestration smoke** - `b871278` (test)
4. **Task 4: Validate all committed notebook structures offline** - `751097d` (test)

Corrective plan-gate commit: `9ee7357` (test) removed one unused test import identified by Ruff.

**Plan metadata:** committed with this summary and sequential tracking update.

## Files Created/Modified

- `projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py` - Primitive NPZ, Parquet, JSON, H5AD, and object-rejection evidence.
- `projects/spatial-pharma-dl/tests/test_synthetic_anndata.py` - Real spatial fixture round trip, public accessor, patch-shape, and valid-alignment evidence.
- `projects/spatial-pharma-dl/tests/test_model_fold_smoke.py` - Tiny CPU optimization, no-pretrained ResNet18, and stubbed LOSO coverage.
- `projects/spatial-pharma-dl/tests/test_notebook_structure.py` - Discovery, nbformat validity, cell schema, and kernel-family checks for all notebooks.

## Decisions Made

- Kept every new module test-only so Plan 01-02 characterizes current public seams without implementing later artifact, fold-admission, evaluation, image, or label fixes.
- Guarded the public backbone smoke at the internal weight-selection boundary and passed `pretrained=False` explicitly, ensuring a regression requests no weights before a network call is possible.
- Used a test-local object-string conversion before H5AD writes to bridge the active pandas 3/AnnData 0.10 serializer mismatch without changing production fixtures or dependency policy.
- Treated legacy missing notebook cell IDs as an expected warning because adding IDs would rewrite committed teaching artifacts outside this phase.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bridged Arrow-backed inferred strings for H5AD fixture writes**
- **Found during:** Task 1 (artifact H5AD round trip)
- **Issue:** The active pandas 3 runtime inferred Arrow-backed string arrays that AnnData 0.10 cannot serialize to HDF5.
- **Fix:** Converted only the fresh test fixture's axes and slide column to object-backed strings immediately before H5AD writes.
- **Files modified:** `test_artifact_roundtrips.py`, `test_synthetic_anndata.py`
- **Verification:** Both focused H5AD suites and the complete offline suite pass using real `anndata.AnnData` objects.
- **Committed in:** `3937a93`, `e91a665`

**2. [Rule 1 - Bug] Removed an unused NumPy import caught by the plan-wide Ruff gate**
- **Found during:** Plan verification after Task 4
- **Issue:** The model smoke retained an unused import after implementation refinement.
- **Fix:** Removed the import and reran the focused model suite, repository Ruff scope, and complete offline suite.
- **Files modified:** `test_model_fold_smoke.py`
- **Verification:** Ruff reports all checks passed and all three model/fold tests pass.
- **Committed in:** `9ee7357`

---

**Total deviations:** 2 auto-fixed (1 blocking environment compatibility issue, 1 test lint bug). **Impact on plan:** Both fixes were confined to test code, preserved phase boundaries, and introduced no production or dependency-policy change.

## Issues Encountered

- The active environment emits existing pandas warnings for old optional `numexpr` and `bottleneck` accelerators.
- `nbformat` warns that six legacy pharma notebooks lack cell IDs; validation still succeeds, and Plan 01-02 intentionally does not rewrite them.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-03 can expose the now-representative 50-test offline suite through the canonical runner, documentation, and CI job.
- Network, notebook execution, model downloads, real training, repository caches, and full-cohort validation remain explicitly outside the fast evidence path.
- No high-severity T-01, T-02, or T-04 control remains open for this plan.

## Verification Results

- Artifact round trips: 4 passed.
- Synthetic AnnData integration: 3 passed.
- Model and LOSO smoke: 3 passed.
- Notebook structure: 22 passed across all 20 notebooks.
- Combined Plan 01-02 representative modules: 32 passed.
- Complete strict-marker offline suite: 50 passed.
- Repository Ruff scope: all checks passed.
- Boundary scan: no unsafe reader, model-hub, real-training, notebook-execution, or later-phase API call is present in the four new modules.

## Self-Check: PASSED

---
*Phase: 01-offline-verification-harness*
*Completed: 2026-07-17*
