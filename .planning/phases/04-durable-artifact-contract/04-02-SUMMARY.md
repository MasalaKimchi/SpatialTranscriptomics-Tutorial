---
phase: 04-durable-artifact-contract
plan: "02"
subsystem: scientific-artifact-adapters
tags: [h5ad, parquet, csv, npz, lineage, atomic-publication, offline-testing]

requires:
  - phase: 04-durable-artifact-contract
    provides: Canonical fingerprints, generic byte admission, and manifest-last publication
  - phase: 03-identity-and-adaptive-preprocessing
    provides: Exact compound spot identity and canonical preprocessing provenance
provides:
  - Production-reader contracts for root and pharma H5AD, label/domain tables, patches/indexes, and foundation embeddings
  - Transitive processed-to-label/patch-to-embedding fingerprints with presentation-only stability
  - Typed acquisition regeneration, direct-reader failure, pure status paths, and contract-aware runner reuse
affects: [phase-04-report-adapters, phase-05-safe-formats, phase-06-folds, phase-09-labels]

tech-stack:
  added: []
  patterns:
    - Generic checksum/fingerprint admission always precedes scientific decoding
    - Production readers return exact semantic schemas compared to manifest declarations
    - Scientific writers publish temporary validated payloads and manifest-last sidecars

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_artifact_adapters.py
    - projects/spatial-pharma-dl/tests/test_artifact_orchestration.py
  modified:
    - utils/st_helpers.py
    - scripts/generate_gallery_figures.py
    - projects/spatial-pharma-dl/src/data.py
    - projects/spatial-pharma-dl/src/labels.py
    - projects/spatial-pharma-dl/src/patches.py
    - projects/spatial-pharma-dl/src/foundation.py
    - projects/spatial-pharma-dl/scripts/run_pipeline.py

key-decisions:
  - "Cache-only processed-slide reuse fingerprints stable provider/sample identity; an observed source-content digest is included only when the caller can supply it again."
  - "The object-valued patch NPZ remains explicitly trusted-local-writer-only: generic admission completes before allow_pickle=True, while Phase 5 owns its safe-format replacement."
  - "Legacy and stale caches regenerate only in acquisition paths; checksum-valid semantic corruption remains a visible failure."
  - "Device, scheduling, training, report, and presentation settings do not invalidate scientific artifacts they cannot change."

requirements-completed: [ART-03, ART-04]

duration: 13min
completed: 2026-07-17
---

# Phase 4 Plan 02: Scientific Artifact Adapter Summary

**Every root/pharma H5AD, label/domain table, patch/index, and foundation embedding now requires current lineage, stable bytes, and exact production-reader semantics before reuse.**

## Performance

- **Duration:** 13 min
- **Completed:** 2026-07-17
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments

- Preserved root tutorial H5AD filenames while encoding the raw → QC → clustered → features chain with additive manifests; gallery required and optional inputs now use contract-aware loads/status rather than filename existence.
- Added pure processed-slide paths, explicit semantic/optional observed-content source policy, atomic H5AD publication, exact spatial/image/PCA/identity/preprocessing validation, and typed acquisition regeneration.
- Added atomic Parquet/CSV label and domain adapters whose schema, rows, compound identity, label science, and processed-slide lineage are validated before return.
- Added patch and patch-index contracts with processed/stain/label transitive lineage, exact NCHW/metadata/table semantics, and an explicit local-writer-only legacy object-decode boundary.
- Added strict primitive embedding NPZ admission before patch, encoder, device, model, or download seams; exact model source, patch, dimension, dtype, finiteness, cardinality, and compound spot identity now govern reuse.
- Replaced the runner's patch existence check with production contract status and passed resolved configuration into index publication.

## Task Commits

1. **Task 1: Root/pharma H5AD and label/domain adapters** — `2b1571d`
2. **Task 2: Patch/index adapters and explicit legacy safety boundary** — `84ddeeb`
3. **Task 3: Strict foundation embedding adapter** — `9449c20`
4. **Compatibility and complete-gate fixes** — `17becb5`, `dfd3c3a`, `97c6f7d`

## Test Evidence

- Complete Plan 04-02 contract/adapter/orchestration and affected Phase 3 suite: **223 passed**; scoped Ruff passed.
- Repository fast gate: Ruff passed across all configured source/test directories and **375 offline tests passed**.
- Real Scanpy capped preprocessing roundtrip remains mandatory and passes through the new atomic processed-H5AD adapter.
- Static audit found no remaining filename-only scientific reuse in the touched root/gallery/processed/patch/embedding paths, no default-config truthiness reload, and no unsafe patch decode before generic admission.

## Compatibility

- Existing final payload names, notebook order, documented CLI, configuration keys, public positional calls, patch values/order, Phase 3 identity columns, and preprocessing inner schema remain stable.
- Keyword-only context/config additions are additive.
- Compatibility aliases remain available for prior notebook/test patch seams while all new status resolution is pure.

## Deviations from Plan

- Updated the existing import-light test to execute in an isolated subprocess, because the combined plan gate imports Torch/AnnData during collection before the generic-contract test runs.
- Updated affected Phase 2/3 fixtures to publish valid sidecars and use fully resolved configurations; legacy direct fixture writes are intentionally no longer accepted as reusable artifacts.

## Issues Encountered

- The active pandas/anndata stack restores some string columns as Arrow-backed arrays that this anndata writer cannot emit. Real H5AD fixtures normalize only those test storage dtypes while preserving values and scientific schema.
- Existing optional pandas accelerator, chained-assignment fixture, and legacy notebook cell-ID warnings remain; all required gates pass.

## Next Phase Readiness

- Plan 04-03 can reuse the same reader/schema/fingerprint/publication pattern for checkpoints, reports, summaries, and JSON wrappers.
- Phase 5 still owns authenticated-safe replacement of object patch NPZ and pickle-capable checkpoints; this plan deliberately does not claim those formats are safe for untrusted input.

## Self-Check: PASSED

- All task commits and both new evidence modules exist.
- Complete focused and repository fast gates pass.
- Worktree was clean before summary/tracking updates, and no push was performed.

---
*Phase: 04-durable-artifact-contract*
*Completed: 2026-07-17*
