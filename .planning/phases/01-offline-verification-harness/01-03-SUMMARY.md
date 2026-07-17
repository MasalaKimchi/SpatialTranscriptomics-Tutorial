---
phase: 01-offline-verification-harness
plan: "03"
subsystem: testing
tags: [pytest, ruff, github-actions, offline, ci]

requires:
  - phase: 01-offline-verification-harness
    provides: Strict offline tiers, deterministic fixtures, and representative CPU evidence from Plans 01-01 and 01-02
provides:
  - Canonical cross-platform verification runner with failure propagation
  - Required Python 3.11 offline CI and dispatch-gated opt-in evidence tiers
  - Repository and pharma documentation for verification commands and evidence boundaries
affects: [all-later-phases, verification, ci, artifact-safety]

tech-stack:
  added: [github-actions]
  patterns:
    - Local and CI verification share one canonical Python runner
    - External evidence tiers require explicit workflow dispatch inputs

key-files:
  created:
    - scripts/verify.py
    - .github/workflows/verify.yml
  modified:
    - projects/spatial-pharma-dl/tests/test_verification_contract.py
    - README.md
    - projects/spatial-pharma-dl/README.md

key-decisions:
  - "The fast tier runs Ruff before strict offline pytest and propagates the first failure status."
  - "Pytest exit status 5 is reported as explicit non-evidence only for empty opt-in tiers."
  - "Required CI is independent of dispatch-gated notebook, network, and full-cohort jobs and caches dependencies only."

patterns-established:
  - "Canonical runner: every verification tier is represented by a deterministic subprocess argument list."
  - "Evidence boundary: safe fixture round trips do not certify production cache or checkpoint migrations."

requirements-completed: [TEST-01]

duration: 6min
completed: 2026-07-17
---

# Phase 1 Plan 03: Canonical Verification Workflow Summary

**One canonical runner now drives local and required CI verification, while notebook, network, and full-cohort evidence remains visibly opt-in**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-17T04:31:30Z
- **Completed:** 2026-07-17T04:37:08Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added an import-safe runner whose fast tier executes checked-in Ruff scope before all strict offline pytest evidence and stops on the first failure.
- Added contract coverage for command arrays, selector overrides, failure propagation, empty opt-in reporting, workflow triggers, job gating, cache exclusions, and documentation.
- Added required Ubuntu/Python 3.11 offline CI plus separately named workflow-dispatch jobs for notebook-smoke, network, and full-cohort evidence.
- Documented all canonical and direct debugging commands, the CPU/offline boundary, production-migration caveat, and later-phase extension convention in both READMEs.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add and test the canonical tier runner** - `49b1e9b` (feat)
2. **Task 2: Use the canonical runner in documentation and CI** - `f2446a3` (ci)

**Plan metadata:** committed with this summary and sequential GSD tracking updates.

## Files Created/Modified

- `scripts/verify.py` - Builds and executes the four canonical verification tiers without scientific imports.
- `.github/workflows/verify.yml` - Runs required fast evidence and dispatch-gated opt-in jobs on Ubuntu/Python 3.11.
- `projects/spatial-pharma-dl/tests/test_verification_contract.py` - Machine-checks runner, CI, and documentation contracts.
- `README.md` - Documents repository-wide verification commands and evidence boundaries.
- `projects/spatial-pharma-dl/README.md` - Documents the same contract beside the pharma quickstart.

## Decisions Made

- Used `python -m ruff` and `sys.executable -m pytest` argument lists so local environments and CI resolve tools through the selected Python interpreter.
- Treated pytest's no-tests-collected status as a successful runner invocation only for explicitly selected opt-in tiers, while printing that no evidence was produced.
- Used GitHub's setup-python pip dependency cache only; no repository data, outputs, weights, model cache, or scientific artifact path is cached.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The active environment continues to emit the previously documented optional pandas accelerator and legacy notebook cell-ID warnings; neither affects verification results.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 is complete with TEST-01 enforced by 56 passing offline tests and a required canonical CI command.
- Later phases can add their evidence to the existing marker, fixture, runner, and CI contracts without changing public notebook or pipeline interfaces.
- Network and full-cohort execution remains intentionally opt-in; there are no Phase 2 blockers.

## Gate Results

- `python scripts/verify.py fast`: passed in 12.50 seconds; Ruff passed first and 56 offline tests passed.
- Bare `python -m pytest -q`: 56 passed; no notebook-smoke, network, or full-cohort evidence executed.
- Full verification contract module: 10 passed, including failure propagation and empty opt-in reporting.
- Workflow/documentation contracts: 2 passed, machine-checking triggers, Python 3.11, offline flags, exact commands, dispatch gating, job independence, cache exclusions, and both README boundaries.
- `python scripts/verify.py --help`: listed exactly `fast`, `notebook-smoke`, `network`, and `full-cohort`.
- Fast output included nonzero artifact, AnnData integration, model/fold, and notebook-structure evidence through the 56-test offline suite.
- Phase acceptance runtime remained well below the 300-second CPU limit.

---
*Phase: 01-offline-verification-harness*
*Completed: 2026-07-17*

## Self-Check: PASSED
