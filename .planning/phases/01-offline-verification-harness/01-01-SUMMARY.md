---
phase: 01-offline-verification-harness
plan: "01"
subsystem: testing
tags: [pytest, ruff, offline, anndata, fixtures]

requires: []
provides:
  - Strict mutually exclusive primary evidence tiers with an offline default
  - Socket and model-hub denial for offline pytest sessions
  - Deterministic valid and adversarial scientific fixture factories
affects: [phase-01-plan-02, phase-01-plan-03, verification, ci]

tech-stack:
  added: []
  patterns:
    - Exactly one primary pytest tier per test
    - Fresh fixed-seed factory fixtures isolated under tmp_path

key-files:
  created:
    - projects/spatial-pharma-dl/tests/conftest.py
    - projects/spatial-pharma-dl/tests/test_verification_contract.py
    - projects/spatial-pharma-dl/tests/test_fixture_contracts.py
  modified:
    - pyproject.toml
    - projects/spatial-pharma-dl/tests/test_core_refactors.py
    - projects/spatial-pharma-dl/tests/test_foundation.py

key-decisions:
  - "Bare pytest selects offline evidence, while command-line -m selectors retain explicit opt-in behavior."
  - "Only network and full_cohort selections can enable sockets; offline and notebook_smoke remain fail-closed."
  - "Fixture factories use local named seeds, return fresh objects, and serialize only beneath pytest tmp_path."

patterns-established:
  - "Primary-tier contract: every collected test declares exactly one of offline, notebook_smoke, network, or full_cohort."
  - "Adversarial fixture contract: unsafe object artifacts are generated only as unread adversaries and never loaded with allow_pickle=True."

requirements-completed: [TEST-01]

duration: 7min
completed: 2026-07-17
---

# Phase 1 Plan 01: Tier and Fixture Contract Summary

**Strict offline pytest isolation with deterministic AnnData, cohort, key, fold, image, and artifact fixture factories**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-17T04:12:18Z
- **Completed:** 2026-07-17T04:19:28Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Registered four mutually exclusive evidence tiers and made bare pytest select only offline evidence.
- Denied both guarded socket APIs and model-hub access for offline sessions while preserving explicit external-tier opt-ins.
- Preserved and classified all eight existing regression tests and added executable tier/network contracts.
- Added fresh deterministic factories covering valid AnnData/cohorts plus every planned key, fold, image, and artifact adversary.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement strict primary tiers and the offline execution boundary** - `725cd2a` (test)
2. **Task 2: Add the deterministic valid and adversarial fixture vocabulary** - `a687bd5` (test)

**Plan metadata:** committed with this summary and tracking update.

## Files Created/Modified

- `pyproject.toml` - Declares strict pytest tiers, offline default selection, and Python 3.11 Ruff scope.
- `projects/spatial-pharma-dl/tests/conftest.py` - Enforces tiers/network isolation and exposes deterministic factory fixtures.
- `projects/spatial-pharma-dl/tests/test_verification_contract.py` - Proves classification, environment, socket, and bare-pytest behavior.
- `projects/spatial-pharma-dl/tests/test_fixture_contracts.py` - Proves fixture repeatability, freshness, adversarial coverage, and path isolation.
- `projects/spatial-pharma-dl/tests/test_core_refactors.py` - Classifies the three existing regressions as offline evidence.
- `projects/spatial-pharma-dl/tests/test_foundation.py` - Classifies the five existing regressions as offline evidence.

## Decisions Made

- Applied the socket guard at pytest session configuration so accidental external access fails before reaching an endpoint.
- Treated `notebook_smoke` as offline by default; only explicitly selected `network` and `full_cohort` tiers may use sockets.
- Kept all fixture randomness local to fixed-seed `numpy.random.Generator` instances and all artifact writes under `tmp_path`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed a repository-declared AnnData runtime compatible with the existing scientific stack**
- **Found during:** Task 2 (deterministic fixture vocabulary)
- **Issue:** The active interpreter lacked AnnData. The newest unconstrained resolver initially selected NumPy 2.x, which was ABI-incompatible with the interpreter's prebuilt HDF5 extension.
- **Fix:** Reinstalled the compatible NumPy 1.26, SciPy 1.13, HDF5, AnnData 0.10, and Zarr 2 runtime line already allowed by repository declarations; no dependency files or production APIs changed.
- **Files modified:** None in the repository; interpreter environment only.
- **Verification:** `anndata`, `h5py`, and NumPy import together, and all six fixture-contract tests pass using real `anndata.AnnData` objects.
- **Committed in:** Not applicable (environment-only blocking fix).

---

**Total deviations:** 1 auto-fixed (1 blocking environment issue). **Impact on plan:** The fix enabled the specified real AnnData fixture without expanding repository scope; production behavior and declared dependencies were unchanged.

## Issues Encountered

- The active environment reports non-blocking pandas warnings for older optional `numexpr` and `bottleneck` accelerators. Tests do not depend on those accelerators and all acceptance commands pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The shared marker and fixture vocabulary is ready for Plan 01-02 artifact, AnnData integration, model/fold, and notebook-structure evidence.
- Plan 01-03 can build the canonical runner and CI directly on the enforced tier names.
- No high-severity T-01, T-02, or T-04 control remains open for the offline tier.

## Verification Results

- Focused Task 1 suite: 12 passed.
- Focused Task 2 suite: 6 passed.
- Combined Plan 01 offline suite: 18 passed.
- Bare `python -m pytest -q`: 18 passed with only offline evidence selected.
- Repository Ruff scope: all checks passed.

## Self-Check: PASSED

---
*Phase: 01-offline-verification-harness*
*Completed: 2026-07-17*
