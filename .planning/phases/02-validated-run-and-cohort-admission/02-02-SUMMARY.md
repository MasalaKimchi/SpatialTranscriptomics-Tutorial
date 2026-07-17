---
phase: 02-validated-run-and-cohort-admission
plan: "02"
subsystem: validation
tags: [python, cohort-admission, deterministic-json, offline-testing, pipeline-startup]

requires:
  - phase: 02-validated-run-and-cohort-admission
    provides: Canonical configuration resolution and fail-closed policy validation from Plan 01
provides:
  - Immutable ordered cohort manifests and admitted-run records
  - Strict aggregate admission and explicit reason-coded partial admission
  - Provisional remote admission with final-outcome-only manifest publication
  - One admitted cohort and oncology subset propagated through every pipeline stage
  - Downstream helpers that fail instead of independently dropping admitted slides
affects: [02-03-empty-boundaries, artifact-manifests, cache-fingerprints, pipeline-provenance]

tech-stack:
  added: []
  patterns:
    - Pure admission uses injected availability and configuration-order iteration
    - Remote curation separates provisional admission from one final outcome admission
    - Heavy scientific imports and output creation follow final cohort admission
    - Admission is the sole partial-cohort policy for downstream helpers

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_cohort_admission.py
  modified:
    - projects/spatial-pharma-dl/src/validation.py
    - projects/spatial-pharma-dl/src/data.py
    - projects/spatial-pharma-dl/src/labels.py
    - projects/spatial-pharma-dl/src/patches.py
    - projects/spatial-pharma-dl/scripts/run_pipeline.py

key-decisions:
  - "Configuration order, never set iteration order, defines every manifest collection and admitted slide sequence."
  - "A remote admission with unknown availability is provisional and is never published; only complete source outcomes can produce the visible cohort manifest."
  - "Strict source errors expose deterministic concise failure evidence without exception reprs, tracebacks, host paths, or timestamps."
  - "Downstream data, label, stain, and patch helpers process exactly their supplied admitted sequence or propagate the first actionable failure."

patterns-established:
  - "Final admission gate: resolve overrides, admit complete outcomes, defer heavy imports, then create outputs."
  - "Complete partial evidence: one failed source appears independently in failed and skipped collections with stable reason codes."

requirements-completed: [VAL-04]

duration: 7min
completed: 2026-07-17
---

# Phase 2 Plan 02: Validated Run and Cohort Admission Summary

**Deterministic fail-closed cohort admission now controls pipeline startup, final manifest publication, and one ordered membership shared by every downstream stage**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-17T05:54:36Z
- **Completed:** 2026-07-17T06:01:24Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added frozen, slotted `SlideAdmission`, `CohortManifest`, and `AdmittedRun` records with canonical JSON, fresh compatibility dictionaries, stable reason codes, and configuration-order membership.
- Made strict admission aggregate every known unavailable or failed member in one `CohortAdmissionError`; explicit partial admission records configured, included, skipped, and failed outcomes and rejects empty or oncology-LOSO-unusable results.
- Refactored the CLI to re-resolve quick/foundation overrides, complete non-creating train-only preflight, keep remote admission provisional, collect partial source outcomes, and publish only the final admitted manifest.
- Deferred scientific/model imports and all output-directory work until final admission, then propagated one exact admitted sequence and its manifest-owned oncology subset through summary, labels, patches, stain fitting, and benchmarking.
- Removed downstream catch-and-continue policies and truthiness fallbacks without changing public helper signatures, successful filenames, CLI flags, report names, or terminal success messages.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement immutable cohort manifests and pure strict or partial admission** - `4da5977` (feat)
2. **Task 2: Wire startup admission and propagate one canonical cohort** - `ef8196f` (feat)
3. **Task 3: Remove independent downstream missing-slide policies** - `6a857d1` (fix)

**Plan metadata:** committed with this summary and sequential GSD tracking updates.

## Files Created/Modified

- `projects/spatial-pharma-dl/src/validation.py` - Defines immutable manifest records, canonical JSON, structured admission errors, and strict/partial admission.
- `projects/spatial-pharma-dl/src/data.py` - Adds non-creating processed-slide preflight and exact supplied-membership behavior.
- `projects/spatial-pharma-dl/src/labels.py` - Removes independent missing-slide filtering and preserves explicit config values.
- `projects/spatial-pharma-dl/src/patches.py` - Makes stain and patch cohort helpers process the admitted sequence or fail visibly.
- `projects/spatial-pharma-dl/scripts/run_pipeline.py` - Implements resolve-overrides-admit startup, provisional remote curation, final manifest publication, and downstream propagation.
- `projects/spatial-pharma-dl/tests/test_cohort_admission.py` - Proves order, strict aggregation, explicit partial evidence, runner sequencing, deferred imports, no false manifest, and helper consistency offline.

## Decisions Made

- Kept `load_config()` and all existing helpers on ordinary dict/list compatibility surfaces; immutable values remain inside the admission contract.
- Used a non-creating repository-relative processed path probe rather than any accessor that calls `mkdir`.
- Converted source exceptions to one bounded deterministic reason string so canonical evidence cannot leak host-specific exception text.
- Published the additive `cohort_manifest.json` only after final admission without claiming atomicity, checksums, fingerprints, or completion-marker durability reserved for Phase 4.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first Task 2 commit attempt was blocked by the managed sandbox's read-only Git index; the same already-verified atomic commit succeeded after the required scoped Git approval.
- The active environment continues to emit the previously documented optional pandas accelerator and legacy notebook cell-ID warnings; neither affects verification results.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-03 can apply the shared empty-boundary vocabulary to the now-canonical admitted sequences without reintroducing membership policy.
- Durable atomic manifest publication, fingerprints, cache-format migration, identity repair, class viability, leakage prevention, image science, and label confidence remain deliberately deferred to their assigned phases.
- No blocker remains for Plan 02-03.

## Gate Results

- Task 1 focused gate: 6 offline admission tests passed; scoped Ruff passed.
- Task 2 focused gate: 23 offline admission and verification-contract tests passed; scoped Ruff passed.
- Task 3 focused gate: 16 offline admission tests passed; scoped Ruff passed.
- Plan focused gate: 45 offline validation/admission/verification-contract tests passed.
- Canonical fast gate: Ruff passed first and all 91 offline tests passed in 4.49 seconds.
- Threat review: T-01 through T-05 controls assigned to this plan are closed; provisional admission cannot publish, strict failure cannot reach downstream work, partial source outcomes are complete, and unusable admitted cohorts fail closed.
- Scope review: no atomic/durable publication claim, cache fingerprint, unsafe-format migration, identity alignment, class policy, leakage/scaling/imputation change, stain-science change, or public API/CLI rename was introduced.

---
*Phase: 02-validated-run-and-cohort-admission*
*Completed: 2026-07-17*

## Self-Check: PASSED
