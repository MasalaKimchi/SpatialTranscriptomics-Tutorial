---
phase: 03-identity-and-adaptive-preprocessing
plan: "01"
subsystem: identity
tags: [python, pandas, compound-keys, alignment, foundation-cache, offline-testing]

requires:
  - phase: 02-validated-run-and-cohort-admission
    provides: Structured exact-type validation and guard-before-side-effect patterns
provides:
  - Exact built-in compound spot identity admission with deterministic bounded evidence
  - Complete metadata-order one-to-one label alignment with source-row provenance
  - Shared ordinary patch and foundation embedding consumer alignment contract
affects: [phase-04-artifacts, phase-05-safe-caches, phase-06-fold-admission, phase-09-labels]

tech-stack:
  added: []
  patterns:
    - Exact key cell types are admitted before hashing, comparison, sorting, merge, or indexing
    - Metadata owns successful row order and both label and value source ordinals remain inspectable
    - Cache hits and misses use the same compound identity and cardinality evidence as ordinary patches

key-files:
  created:
    - projects/spatial-pharma-dl/src/identity.py
    - projects/spatial-pharma-dl/tests/test_identity_alignment.py
  modified:
    - projects/spatial-pharma-dl/src/labels.py
    - projects/spatial-pharma-dl/src/patches.py
    - projects/spatial-pharma-dl/src/train.py
    - projects/spatial-pharma-dl/src/foundation.py
    - projects/spatial-pharma-dl/tests/conftest.py
    - projects/spatial-pharma-dl/tests/test_fixture_contracts.py
    - projects/spatial-pharma-dl/tests/test_empty_boundaries.py

key-decisions:
  - "The exact (slide_id, spot_id) pair is the sole alignment key; coercion, trimming, inferred indexes, lossy joins, and spot-only maps are rejected."
  - "Patch or embedding metadata defines successful output order, while _label_source_row and _patch_source_row preserve the proven mapping."
  - "Foundation cache subsets are invalid artifacts rather than a cache miss that may silently fall back to expensive extraction."

requirements-completed: [VAL-02]

duration: 18min
completed: 2026-07-17
---

# Phase 3 Plan 01: Compound Identity and Shared Alignment Summary

**Exact compound-key admission now prevents silent spot loss, multiplication, coercion, cross-slide association, and array/metadata drift across ordinary and foundation consumers.**

## Performance

- **Duration:** 18 min
- **Completed:** 2026-07-17
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Added frozen structured identity issues and a `PharmaValidationError`-compatible aggregate exception with deterministic category order, total counts, inert invalid-cell evidence, and bounded safe key samples.
- Rejected missing/reserved columns, nulls, blanks, non-built-in strings, hostile subclasses, duplicate keys, wrong-slide rows, incomplete sets, cross-slide matches, and value/metadata cardinality mismatch before pandas hashing or scientific work.
- Replaced unconstrained inner alignment with a metadata-left, pre-proven `one_to_one` merge that preserves complete target/provenance rows plus both source ordinals.
- Guarded AnnData observation identity before label marker/module/ranking/expression work and before patch coordinate/image/transform/stack work.
- Removed the ordinary spot-only index dictionary and foundation string coercion/subset cache acceptance; both paths now index only by validated patch source rows.
- Extended the shared key adversary fixture and added forbidden-seam, shuffled-success, cache-hit/miss, cardinality, and compatibility evidence.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build Wave 0 adversaries and exact compound-key admission** - `20ebd81` (feat)
2. **Task 2: Implement metadata-order alignment and guard producers** - `21e00e2` (feat)
3. **Task 3: Route patch and foundation consumers through one contract** - `37c3707` (feat)

## Verification Evidence

- Task 1 focused adversarial gate: 12 passed; scoped Ruff passed.
- Task 2 focused producer/alignment gate: 4 passed; affected AnnData/empty-boundary gate: 13 passed; scoped Ruff passed.
- Task 3 consumer and affected regression gate: 79 passed; scoped Ruff passed.
- Plan-level affected suite: 91 passed; all modified-file Ruff checks passed.
- Complete mandatory offline suite: 183 passed with no downloads or model-weight access.
- Static alignment audit found no `astype(str)`, spot-only index dictionary, set-subset cache alignment, or unconstrained production identity merge in the touched paths.

## Files Created/Modified

- `projects/spatial-pharma-dl/src/identity.py` - Owns exact key admission, structured diagnostics, AnnData identity validation, and complete metadata-order alignment.
- `projects/spatial-pharma-dl/src/labels.py` - Preserves the public two-argument facade and guards source AnnData before label science.
- `projects/spatial-pharma-dl/src/patches.py` - Guards observation identity before patch construction.
- `projects/spatial-pharma-dl/src/train.py` - Aligns the full cohort table for the expected slide and indexes arrays with `_patch_source_row` only.
- `projects/spatial-pharma-dl/src/foundation.py` - Applies the same exact contract to cache hits and verifies extraction cardinality before writes.
- `projects/spatial-pharma-dl/tests/conftest.py` - Extends fresh key adversaries through every VAL-02 defect class.
- `projects/spatial-pharma-dl/tests/test_identity_alignment.py` - Supplies hostile, deterministic, shuffled, producer, ordinary-consumer, and foundation-consumer evidence.
- `projects/spatial-pharma-dl/tests/test_fixture_contracts.py` - Verifies the expanded adversary catalog.
- `projects/spatial-pharma-dl/tests/test_empty_boundaries.py` - Updates Phase 2 expectations for the stronger Phase 3 structured identity boundary.

## Decisions Made

- Invalid key objects are identified only by row ordinal, canonical column name, and inert exact type label; their representation, hashing, comparison, coercion, string, strip, or iteration hooks are never used.
- Valid labels from other cohort slides remain admissible context for an expected-slide call and are excluded only after full-table validation; they are not classified as extras.
- A cache hit with incomplete or mismatched identity is a visible validation failure, preventing silent subset reuse and avoiding an expensive fallback that could conceal artifact corruption.

## Deviations from Plan

- Updated `test_fixture_contracts.py` and `test_empty_boundaries.py` in addition to the listed task files so the expanded fixture schema and intentionally stronger error contract remain regression-tested. No production scope was added.

## Issues Encountered

- Pandas' Arrow string columns reject non-string fixture assignment before validation. The adversary fixture explicitly converts only the attacked column to object dtype so hostile values reach the intended identity boundary without changing production behavior.
- The environment continues to emit the previously documented optional pandas accelerator and legacy notebook cell-ID warnings; all mandatory gates pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VAL-02 has direct adversarial and consumer-path evidence with all T-01 through T-03 controls closed for this plan.
- Plan 03-02 can rely on trustworthy unique spot identity while implementing adaptive preprocessing dimensions and provenance.
- Cache format migration, atomic durability, fold policy, leakage controls, image science, and label confidence remain in their assigned later phases.

## Self-Check: PASSED

- Both created files exist and all three task commits are present in Git history.
- Every task acceptance criterion and plan-level verification command passed.
- Public function signatures, valid shapes, cache names/payload fields, CLI/notebook surfaces, and public imports remain compatible.
- No later-phase artifact, fold, leakage, image, or label-policy implementation entered the diff.

---
*Phase: 03-identity-and-adaptive-preprocessing*
*Completed: 2026-07-17*
