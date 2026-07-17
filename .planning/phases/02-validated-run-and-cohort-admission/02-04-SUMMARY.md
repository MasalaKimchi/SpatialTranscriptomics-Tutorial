---
phase: 02-validated-run-and-cohort-admission
plan: "04"
subsystem: validation
tags: [python, exact-types, adversarial-validation, foundation-models, offline-testing]

requires:
  - phase: 02-validated-run-and-cohort-admission
    provides: Aggregate configuration, cohort admission, and empty-boundary contracts from Plans 01 through 03
provides:
  - Total exact-built-in configuration admission that rejects hostile subclasses without executing overrides
  - Deterministic inert diagnostics for adversarial primitive, container, mapping, and path values
  - Explicit-None configuration semantics across foundation and benchmark-report helpers
  - Forbidden-seam evidence that malformed supplied configuration cannot reach cache, model, encoder, or report effects
affects: [phase-03-identity, phase-04-artifacts, phase-06-fold-admission, phase-07-evaluation]

tech-stack:
  added: []
  patterns:
    - Exact built-in type admission precedes every operation on caller-controlled configuration values
    - Only omitted configuration loads repository defaults; every supplied mapping is resolved first
    - Internal resolved-config helpers prevent nested foundation calls from reloading defaults

key-files:
  created: []
  modified:
    - projects/spatial-pharma-dl/src/validation.py
    - projects/spatial-pharma-dl/src/foundation.py
    - projects/spatial-pharma-dl/src/eval.py
    - projects/spatial-pharma-dl/tests/test_validation.py
    - projects/spatial-pharma-dl/tests/test_empty_boundaries.py

key-decisions:
  - "Configuration admits exact built-in dict, list, tuple, str, bool, int, float, None, and the platform concrete Path type only where the schema permits them; subclasses and arbitrary Mapping implementations fail closed."
  - "Foundation and report defaults are loaded only for cfg=None; supplied mappings always pass through resolve_config before any cache, device, model, dataframe, output-path, or writer seam."

patterns-established:
  - "Non-executing rejection: hostile values are retained only as issue evidence and rendered through bounded inert type labels."
  - "Resolved foundation flow: public helpers validate once at entry and pass a plain resolved dictionary through private config/model-spec helpers."

requirements-completed: [VAL-01, VAL-03, VAL-04]

duration: 7min
completed: 2026-07-17
---

# Phase 2 Plan 04: Verification Gap Closure Summary

**Exact-type configuration admission and explicit-None foundation/report resolution close both independent Phase 2 verification gaps before observable side effects**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-17T06:54:10Z
- **Completed:** 2026-07-17T07:01:16Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Closed G-01 by rejecting root and nested subclasses of otherwise allowed primitives, containers, mappings, and concrete paths before invoking their overridden operations.
- Preserved deterministic aggregate schema traversal and canonical JSON for exact built-in values while rendering rejected hostile values with inert bounded type labels.
- Closed G-02 at `foundation_config`, `foundation_model_spec`, `load_frozen_encoder`, `load_or_extract_slide_embeddings`, and `save_benchmark_report`; only `cfg=None` may load defaults.
- Added adversarial sentinels for `bit_length`, `strip`, representation, hashing, comparison, length, iteration, lookup, and path conversion, plus explicit-invalid-config forbidden seams for cache, directory, device, encoder, dataframe, output, and writer work.
- Preserved valid default, encoder, embedding, cache-disabled, and benchmark-report behavior without adding later-phase artifact, fold, image, or labeling policy.

## Task Commits

Each task was committed atomically:

1. **Task 1: Make aggregate configuration validation total for hostile primitive subclasses** - `cdc6903` (fix)
2. **Task 2: Reject explicit invalid configs before foundation and report side effects** - `9e28390` (fix)

**Plan metadata:** committed with this summary, the independent gap report, and sequential GSD tracking updates.

## Files Created/Modified

- `projects/spatial-pharma-dl/src/validation.py` - Uses exact-type gates before optional-default merging, schema operations, and JSON canonicalization.
- `projects/spatial-pharma-dl/tests/test_validation.py` - Proves hostile root, scalar, sequence, mapping, and concrete-path subclasses cannot execute overrides and produce deterministic aggregate errors.
- `projects/spatial-pharma-dl/src/foundation.py` - Resolves explicit configuration before cache/device/model work and carries resolved dictionaries through private foundation helpers.
- `projects/spatial-pharma-dl/src/eval.py` - Validates explicit report configuration before path resolution, DataFrame construction, or CSV writing.
- `projects/spatial-pharma-dl/tests/test_empty_boundaries.py` - Adds table-driven invalid-config forbidden seams and complete-valid-config compatibility coverage.

## Gap Closure Evidence

### G-01 — Hostile primitive subclasses

- Root `dict` and arbitrary `Mapping` subclasses raise `ConfigValidationError` without iteration, lookup, membership, length, or representation calls.
- Hostile `int`, `float`, `str`, `list`, `tuple`, nested `dict`/`Mapping`, and concrete `Path` subclasses accumulate through one deterministic exception; all operation sentinels remain zero.
- Reversed exact plain mappings containing the same adversarial values produce byte-identical exception text with no attacker-controlled text.
- Focused validation result: 23 offline tests passed; scoped Ruff passed.

### G-02 — Explicit supplied configuration

- Explicit `{}` and representative non-empty malformed mappings fail at all named foundation/report helpers before default loading or any patched side-effect seam.
- Omitted configuration still calls the default facade exactly once, while complete supplied configuration reaches existing encoder, embedding, cache-disabled, and report behavior without a default reload.
- Focused Phase 2/foundation result: 104 offline tests passed.
- Canonical fast result: Ruff passed first and all 157 offline tests passed in 10.53 seconds without network or model downloads.
- Static truthiness audit: no `cfg = cfg or load_config()` or equivalent match remains in the scoped source tree.

## Decisions Made

- Exact built-in checks are the security boundary; copying or coercing a hostile subclass would itself risk executing caller behavior.
- Trusted checked-in defaults may still be copied, but optional-default merging never copies or traverses an untrusted mapping subclass.
- Private resolved-config accessors keep existing public signatures stable and prevent nested helper calls from reinterpreting explicit mappings as absent.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The active Python 3.12 environment continues to emit the previously documented optional pandas accelerator and legacy notebook cell-ID warnings; all gates pass and Phase 10 owns environment reconciliation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- G-01 and G-02 now have direct adversarial behavioral evidence and the canonical offline gate is green.
- VAL-01, VAL-03, and VAL-04 retain valid-call and admitted-cohort compatibility coverage.
- Phase 3 can proceed with identity and adaptive preprocessing; artifact durability, class policy, leakage/scaling/imputation, image science, and label confidence remain within their assigned later phases.
- No blockers remain for independent Phase 2 re-verification.

## Sequential GSD Tracking

- Plan 02-04 executed after Plans 02-01 through 02-03 and their independent verification report.
- Production commits precede this summary commit, preserving the GSD atomic close-out invariant.
- ROADMAP, STATE, and requirement traceability are synchronized with four of four Phase 2 plans complete.

## Self-Check: PASSED

- All key files listed in frontmatter exist.
- Both task commits are present in Git history.
- Every task acceptance criterion and plan-level verification command passed.
- The diff contains no later-phase implementation or public API rename.

---
*Phase: 02-validated-run-and-cohort-admission*
*Completed: 2026-07-17*
