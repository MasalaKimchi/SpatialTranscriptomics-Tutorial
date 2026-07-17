---
phase: 02-validated-run-and-cohort-admission
plan: "03"
subsystem: validation
tags: [python, cardinality-validation, loso, foundation-models, offline-testing]

requires:
  - phase: 02-validated-run-and-cohort-admission
    provides: Canonical configuration and cohort admission contracts from Plans 01 and 02
provides:
  - Structured StageValidationError evidence with reusable non-empty guards
  - Earliest-boundary cardinality checks across data, labels, patches, CNN, RF, and foundation paths
  - Forbidden-seam offline tests proving empty work cannot reach expensive execution
affects: [phase-03-identity, phase-06-fold-admission, phase-07-evaluation, phase-08-image-reliability]

tech-stack:
  added: []
  patterns:
    - Public scientific boundaries reject empty work with stable primitive diagnostics
    - Cardinality guards precede directory, cache, device, model, encoder, estimator, and writer work
    - Later scientific policy remains deferred after cardinality admission

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_empty_boundaries.py
  modified:
    - projects/spatial-pharma-dl/src/validation.py
    - projects/spatial-pharma-dl/src/data.py
    - projects/spatial-pharma-dl/src/labels.py
    - projects/spatial-pharma-dl/src/patches.py
    - projects/spatial-pharma-dl/src/train.py
    - projects/spatial-pharma-dl/src/benchmark.py
    - projects/spatial-pharma-dl/src/eval.py
    - projects/spatial-pharma-dl/src/foundation.py
    - projects/spatial-pharma-dl/src/foundation_eval.py
    - projects/spatial-pharma-dl/tests/test_cohort_admission.py

key-decisions:
  - "Stage errors expose only stage, subject, integer count, integer minimum, optional integer shape, and corrective guidance."
  - "LOSO admission requires two unique non-empty slide IDs but deliberately does not add class-support policy."
  - "Cardinality is rechecked at every public boundary even when canonical cohort admission already passed."
  - "Zero-row alignment fails without diagnosing or repairing keys, preserving Phase 3 ownership."

patterns-established:
  - "Guard-before-expense: validate cardinality before directories, caches, device selection, model construction, encoder loading, estimator fitting, or writes."
  - "Boundary-local guidance: every stage names the rejected subject and tells maintainers which upstream artifact or admission to correct."

requirements-completed: [VAL-03]

duration: 10min
completed: 2026-07-17
---

# Phase 2 Plan 03: Empty Scientific Boundary Summary

**Structured cardinality guards now stop zero-observation data, patch, fold, prediction, and foundation work before any expensive or output-producing seam**

## Performance

- **Duration:** 10 min
- **Started:** 2026-07-17T06:02:24Z
- **Completed:** 2026-07-17T06:12:15Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- Added a reusable `require_non_empty` contract and structured `StageValidationError` attributes for stage, subject, observed count, expected minimum, optional shape, and corrective guidance.
- Guarded cohort/data, label, regression-target, patch, stain-reference, dataset, alignment, LOSO, CNN, RF, prediction, embedding, foundation-probe, and nested-LOSO boundaries.
- Proved empty inputs stop before directory, loader, stack, cache, device, model, encoder, estimator, probe, and writer seams with deterministic offline tests.
- Preserved valid row order, patch tensor shapes, regression mode ordering, three-slide LOSO fold order, foundation task names, model outputs, and report schemas.
- Kept compound-key diagnostics, class viability, model-selection leakage, scaling, imputation, artifact durability, image normalization, and label confidence out of this phase.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the shared stage error contract and guard data, label, and patch boundaries** - `56bf407` (feat)
2. **Task 2: Guard LOSO, alignment, CNN, RF, and prediction boundaries** - `aaceaa5` (feat)
3. **Task 3: Guard foundation boundaries and run the complete regression gate** - `77f4433` (feat)

**Plan metadata:** committed with this summary and sequential GSD tracking updates.

## Files Created/Modified

- `projects/spatial-pharma-dl/src/validation.py` - Defines structured stage errors and the reusable cardinality guard.
- `projects/spatial-pharma-dl/src/data.py` - Rejects empty configured, preprocessing, and summary sequences before loaders or directories.
- `projects/spatial-pharma-dl/src/labels.py` - Rejects empty slide/cohort labels and empty regression-target selections.
- `projects/spatial-pharma-dl/src/patches.py` - Rejects empty coordinates, stain inputs, patch cohorts, datasets, and row-count mismatches.
- `projects/spatial-pharma-dl/src/train.py` - Guards LOSO admission, zero-row alignment, and CNN fold members before device/model work.
- `projects/spatial-pharma-dl/src/benchmark.py` - Guards benchmark and RF fold cardinality before estimator work.
- `projects/spatial-pharma-dl/src/eval.py` - Guards CNN prediction batches and direct RF training/prediction inputs.
- `projects/spatial-pharma-dl/src/foundation.py` - Guards embedding extraction, caches, foundation LOSO, and linear-probe inputs before encoders and estimators.
- `projects/spatial-pharma-dl/src/foundation_eval.py` - Guards zero-row task filters and nested-LOSO train/test parts without changing task mappings.
- `projects/spatial-pharma-dl/tests/test_empty_boundaries.py` - Supplies 36 focused forbidden-seam and compatibility tests.
- `projects/spatial-pharma-dl/tests/test_cohort_admission.py` - Updates the explicit-empty compatibility assertion for the new Plan 03 contract.

## Decisions Made

- Used one import-light error vocabulary across NumPy arrays, pandas frames, lists, and dataset inputs; no scientific/model dependency entered `src.validation`.
- Required two unique non-empty slide IDs at every LOSO orchestration entry while deferring class counts and unseen-class behavior to Phase 6.
- Kept unknown foundation task behavior unchanged and added cardinality checks only after existing task-enum validation.
- Kept zero-row alignment as a cardinality error only; duplicate, missing, cross-slide, and completeness diagnostics remain Phase 3 work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated a stale explicit-empty regression assertion**
- **Found during:** Task 3 (complete Phase 2 focused regression gate)
- **Issue:** Plan 02-02 correctly proved that explicit empty values do not reload defaults, but its test still expected empty data/patch helpers to return empty success values. That expectation directly contradicted VAL-03 and Plan 02-03.
- **Fix:** Retained the forbidden-default monkeypatch and changed only the affected assertions to require `StageValidationError`; the unrelated empty class-name compatibility assertion remains unchanged.
- **Files modified:** `projects/spatial-pharma-dl/tests/test_cohort_admission.py`
- **Verification:** The focused 77-test Phase 2/foundation/model command and the canonical 127-test fast gate pass.
- **Committed in:** `77f4433`

---

**Total deviations:** 1 auto-fixed (1 pre-existing test-contract bug).
**Impact on plan:** The change is required evidence for VAL-03 and does not widen production scope or alter valid behavior.

## Issues Encountered

- The active environment continues to emit the documented optional pandas accelerator warnings and legacy notebook cell-ID warnings; all gates pass and neither warning is caused by this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 now has startup configuration, cohort admission, and public cardinality contracts for all three assigned requirements.
- Phase 3 can add compound-key identity and adaptive preprocessing diagnostics on top of explicit non-empty aligned inputs.
- Class viability, train-only model selection/scaling/imputation, artifact safety, image science, and label confidence remain deliberately deferred to their roadmap phases.
- No blocker remains for independent Phase 2 verification or Phase 3 planning.

## Gate Results

- Task 1 focused gate: 16 data/label/patch/regression/dataset/stain tests passed; scoped Ruff passed.
- Task 2 focused gate: 17 LOSO/alignment/CNN/RF/prediction tests passed; 16 tests deselected by the planned selector.
- Task 2 compatibility gate: all 30 then-current empty-boundary tests and all 6 core/model-fold tests passed.
- Task 3 focused gate: 77 Phase 2 validation/admission/empty/foundation/model tests passed.
- Canonical fast gate: Ruff passed first and all 127 offline tests passed in 4.93 seconds.
- Threat review: T-01 through T-05 are closed for Phase 2; every named empty boundary fails before its expensive or output-producing seam.
- Scope review: no identity repair, class-support policy, early-stopping change, scaling, imputation, cache migration, atomic write, stain-science, border-geometry, or label-confidence behavior was introduced.

---
*Phase: 02-validated-run-and-cohort-admission*
*Completed: 2026-07-17*

## Self-Check: PASSED
