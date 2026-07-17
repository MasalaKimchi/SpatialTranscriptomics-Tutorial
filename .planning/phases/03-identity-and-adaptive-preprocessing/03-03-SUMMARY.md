---
phase: 03-identity-and-adaptive-preprocessing
plan: "03"
subsystem: integration
tags: [python, scanpy, anndata, compound-keys, provenance, offline-testing]

requires:
  - phase: 03-identity-and-adaptive-preprocessing
    provides: Exact compound identity and adaptive preprocessing contracts from Plans 03-01 and 03-02
provides:
  - Cross-arm ordinary and foundation cache-hit/cache-miss identity equivalence
  - Mandatory real-Scanpy capped preprocessing and H5AD integration evidence
  - Exact AnnData-to-run preprocessing provenance equality with unchanged cohort schema
affects: [phase-04-artifacts, phase-06-fold-admission, phase-10-environment]

tech-stack:
  added: []
  patterns:
    - Public consumers share adversarial fixtures and prove guard ordering before merge, indexing, encoder, or cache writes
    - The fast offline gate executes real Scanpy through the declared spatial-tx environment and fails actionably if unavailable
    - Persisted AnnData facts are reloaded through production code before run-manifest comparison

key-files:
  created: []
  modified:
    - projects/spatial-pharma-dl/tests/test_identity_alignment.py
    - projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py
    - projects/spatial-pharma-dl/tests/test_cohort_admission.py

key-decisions:
  - "Real Scanpy remains mandatory fast evidence even when the outer pytest interpreter lacks Scanpy; the test locates the declared spatial-tx interpreter and fails with repair guidance rather than skipping."
  - "Sandbox-only Numba disk caching is disabled in the child scientific process without replacing Scanpy, graph construction, UMAP, or Leiden."
  - "The existing cohort-manifest-v1 schema is asserted independently from the additive preprocessing manifest."

requirements-completed: [VAL-02, VAL-05]

duration: 18min
completed: 2026-07-17
---

# Phase 3 Plan 03: Identity and Adaptive Preprocessing Integration Summary

**Every image consumer now demonstrates identical complete compound-key behavior, while real capped Scanpy output survives H5AD and reconstructs byte-stable run provenance.**

## Performance

- **Duration:** 18 min
- **Completed:** 2026-07-17
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Exercised ordinary patch loading, foundation cache miss with a local frozen encoder, and foundation cache hit using one independently shuffled complete label table and one metadata/array order.
- Proved identical compound keys, target/provenance pairing, label-source ordinals, patch-source ordinals, and repeated-barcode isolation across those outcomes.
- Added public-consumer adversaries for null, blank, hostile exact-type subclass, duplicate, label-only, metadata-only, cross-slide, wrong-slide, and row-count defects before merge, indexing, encoder, or cache publication seams.
- Executed real Scanpy QC, HVG, PCA, neighbors, UMAP, and Leiden on a heavily capped viable synthetic slide in the declared environment.
- Verified real PCA width and neighbor inputs against recorded resolved values while preserving counts, raw data, PCA metadata, and clusters.
- Round-tripped the real result through H5AD, restored it through `load_slide`, and proved exact record equality plus byte-identical repeated run manifests in admitted order.
- Strengthened nonviable and malformed-provenance forbidden-effect evidence and explicitly preserved `cohort-manifest-v1` alongside the additive preprocessing manifest.

## Task Commits

1. **Task 1: Close ordinary and foundation alignment equivalence across cache outcomes** - `53f840d`
2. **Task 2: Close viable preprocessing provenance and run the canonical Phase 3 gate** - `ebea06b`

## Test Evidence

- Cross-arm identity/foundation/empty-boundary gate: 105 passed.
- Adaptive preprocessing, synthetic AnnData, and cohort-admission gate: 52 passed, including mandatory real Scanpy execution.
- Six directly affected Phase 3 modules: 157 passed in 20.30 seconds.
- Canonical fast gate: Ruff passed first; all 230 strict offline tests passed in 21.16 seconds.
- No network, dataset download, or model-weight access occurred.

## Decisions Made

- The scientific integration test does not skip when Scanpy is absent from the outer test interpreter. It probes the declared `spatial-tx` Conda interpreter and reports exact activation/install guidance if no working interpreter exists.
- The managed sandbox prevents Numba from creating caches against installed package sources, so the child test process disables JIT and vectorizer disk caching only; all real Scanpy graph and clustering calls still execute.
- Older Scanpy writes an optional null `log1p.base` encoding unreadable by the repository's newer AnnData reader. The integration serialization omits only that optional null before H5AD write, leaving preprocessing facts and scientific artifacts unchanged.

## Deviations from Plan

- `test_foundation.py`, `test_empty_boundaries.py`, and `test_synthetic_anndata.py` required no edits because their existing assertions remained compatible and passed unchanged.
- The real Scanpy integration runs through the declared Conda interpreter as a child process because the canonical outer fast interpreter does not install Scanpy. This executes the required stack rather than substituting recorder behavior.

## Issues Encountered

- The base Python interpreter has no Scanpy, while the declared `spatial-tx` environment does. Explicit environment discovery keeps the fast gate scientifically meaningful without adding or downloading dependencies.
- Numba cache locator failures and an optional cross-version AnnData null encoding were isolated to test-runtime interoperability and resolved without production or scientific behavior changes.
- Existing pandas optional-accelerator, Copy-on-Write, and legacy notebook cell-ID warnings remain non-failing; Phase 10 owns environment reconciliation.

## User Setup Required

None for the current repository environment. A fresh machine must create the documented `spatial-tx` environment; the integration test fails with that guidance if it is absent.

## Next Phase Readiness

- VAL-02 and VAL-05 now have unit, adversarial, public-consumer, real scientific-stack, persistence, and complete fast-gate evidence.
- Phase 3 is ready for independent verification and code review.
- Artifact fingerprints, atomic publication, cache/checkpoint formats, fold policy, leakage controls, image science, labels, and environment declaration repair remain in their assigned later phases.

## Self-Check: PASSED

- Both task commits exist and the working tree was clean before summary/tracking.
- Every plan acceptance criterion and both mandatory automated commands passed.
- Real Scanpy execution was neither skipped nor replaced by a recorder.
- No production source, notebook, CLI, configuration key, output name, cache schema, model policy, or public export changed.

---
*Phase: 03-identity-and-adaptive-preprocessing*
*Completed: 2026-07-17*
