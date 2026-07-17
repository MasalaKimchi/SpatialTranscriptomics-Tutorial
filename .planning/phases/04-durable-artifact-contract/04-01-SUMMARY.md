---
phase: 04-durable-artifact-contract
plan: "01"
subsystem: artifact-contract
tags: [python, canonical-json, sha256, atomic-publication, hostile-inputs, offline-testing]

requires:
  - phase: 02-validated-run-and-cohort-admission
    provides: Exact primitive admission and bounded structured errors
  - phase: 03-identity-and-adaptive-preprocessing
    provides: Canonical immutable provenance records and scientific identity boundaries
provides:
  - Import-light canonical fingerprint and manifest records for every Phase 4 artifact kind
  - Stable regular-file checksum admission before production-reader callbacks
  - Same-directory payload-first and manifest-last atomic publication with exhaustive fault evidence
affects: [phase-04-adapters, phase-05-safe-formats, phase-06-folds, phase-07-evaluation]

tech-stack:
  added: []
  patterns:
    - Exact bounded duplicate-key JSON admission precedes canonicalization and payload decoding
    - Explicit per-kind allowlisted projections separate scientific lineage from presentation settings
    - A completed canonical sidecar is the manifest-last commit marker for a validated payload generation

key-files:
  created:
    - utils/artifacts.py
    - projects/spatial-pharma-dl/tests/test_artifact_contract.py
  modified:
    - projects/spatial-pharma-dl/tests/conftest.py

key-decisions:
  - "The generic artifact layer is standard-library-only and stores admitted state as canonical JSON with fresh exact-built-in views."
  - "Every supported artifact kind owns a visible contract version and explicit configuration projection; source, upstream, identity, and contract changes always invalidate."
  - "Temporary sidecars name the trusted logical final basename and pass the exact generic admission plus production-reader callback before either final replacement."
  - "Checksums establish local integrity and lineage but do not authenticate pickle-bearing patch or checkpoint payloads; hostile deserialization remains Phase 5."

requirements-completed: [ART-03, ART-04]

duration: 16min
completed: 2026-07-17
---

# Phase 4 Plan 01: Durable Artifact Primitive Summary

**One import-light contract now owns deterministic artifact identity, bounded sidecar admission, stable byte integrity, typed reuse, and fault-safe manifest-last publication.**

## Performance

- **Duration:** 16 min
- **Completed:** 2026-07-17
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added frozen slotted fingerprint, manifest, admission, and reuse-status records backed by canonical JSON with mutation-isolated fresh views.
- Added explicit contract versions and per-kind configuration projections for root/processed H5AD, label/domain tables, patch/index, embeddings, checkpoints, reports/summaries, and cohort/preprocessing wrappers.
- Bounded manifest bytes, UTF-8/JSON decoding, duplicate keys, depth, node count, collection width, strings, integers, floats, exact key sets, basenames, versions, completion, and lowercase SHA-256 fields before payload access.
- Required regular non-symlink manifest/payload files, descriptor-stable streaming SHA-256, byte-count/checksum agreement, expected fingerprint equality, and path-generation stability around the supplied production reader.
- Added unique suffix-preserving same-directory temporaries, file and directory fsyncs, temporary validation through the exact reuse path, payload-first replacement, completed-manifest-last replacement, and exact current-call cleanup.
- Exercised all nine publication fault points against both new destinations and replacement of a valid old generation, proving old, rejected mixed, or committed-new states only.

## Task Commits

Each task was committed atomically:

1. **Task 1: Canonical fingerprint/manifest admission and fixtures** - `0a0d2f1` (feat)
2. **Task 2: Stable payload admission before production readers** - `875a36b` (feat)
3. **Task 3: Fault-injected atomic pair publication** - `483dd2a` (feat)

## Test Evidence

- Task 1 focused manifest/fingerprint gate: 72 passed.
- Task 2 complete generic contract gate: 82 passed; scoped Ruff passed.
- Task 3 focused atomic/fault gate: 20 passed.
- Full Plan 04-01 contract module: 103 passed.
- Scoped Ruff over the implementation, fixture, and contract test module passed.
- Clean-interpreter import audit confirmed NumPy, pandas, Torch, AnnData, Scanpy, torchvision, timm, and transformers remain unloaded.
- `git diff --check` passed; all evidence stayed CPU/offline under `tmp_path`.

## Files Created/Modified

- `utils/artifacts.py` - Owns exact canonical projections, manifests, strict sidecar admission, descriptor integrity, typed reuse, and atomic publication.
- `projects/spatial-pharma-dl/tests/test_artifact_contract.py` - Proves hostile manifest handling, every projection's invariants, callback ordering, races, checksums, and all publication fault states.
- `projects/spatial-pharma-dl/tests/conftest.py` - Adds fresh deterministic old/new generations, malformed sidecar bytes, and inert publication fault vocabulary.

## Decisions Made

- The generic reader accepts a production callback only after the manifest, expected lineage, stable descriptor, byte count, and checksum pass; callback-controlled parser text is never rendered.
- The publisher uses a private trusted logical-basename override only for temporary validation, so the exact sidecar bytes validated before replacement remain valid at the final name.
- Publication failure never deletes or rolls back final files. Before payload replacement an old pair remains valid, between replacements the pair rejects, and after manifest replacement the new generation remains valid even if final directory fsync reports failure.
- Phase 5 remains the sole owner of safe patch/checkpoint serialization; this plan does not deserialize attacker-authored object archives or claim checksum authenticity.

## Deviations from Plan

None - implementation remained within the generic primitive, shared fixtures, and pure contract tests named by the checked plan.

## Issues Encountered

- Git metadata writes required the workspace's approved escalated commit path; no source or test behavior was affected.
- The environment continues to emit the existing optional pandas accelerator warnings; all mandatory gates pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- T-01 through T-05 are closed for the generic primitive, so Plan 04-02 can build H5AD/table/patch/index/embedding adapters without duplicating parser, checksum, or publication logic.
- Production adapters must supply their exact semantic reader callbacks and explicit resolved configuration/source/upstream/identity projections.
- Unsafe patch/checkpoint payload migration remains explicitly deferred to Phase 5.

## Self-Check: PASSED

- Both created files exist and all three task commits are present in Git history.
- Every task acceptance criterion, focused command, full plan module, scoped Ruff command, import-light check, and diff check passed.
- No scientific adapter, cache migration, checkpoint semantic change, notebook/CLI change, or Phase 5 deserialization claim entered the diff.

---
*Phase: 04-durable-artifact-contract*
*Completed: 2026-07-17*
