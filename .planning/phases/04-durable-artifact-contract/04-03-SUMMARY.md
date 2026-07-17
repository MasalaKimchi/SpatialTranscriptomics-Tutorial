---
phase: 04-durable-artifact-contract
plan: "03"
subsystem: model-result-and-orchestration-artifacts
tags: [checkpoint, report, runner, notebooks, lineage, recovery, static-audit]

requires:
  - phase: 04-durable-artifact-contract
    provides: Generic artifact contract and scientific H5AD/table/NPZ adapters
provides:
  - Contract-admitted local checkpoint production and exact tensor/model/fold validation
  - Atomic benchmark, manifest, summary, training-history, and foundation-result adapters
  - Generated notebook consumers using named production readers and writers
  - Exact 19-artifact inventory, mixed-generation recovery, and raw-I/O bypass audit
affects: [phase-05-safe-formats, phase-06-folds, phase-07-evaluation]

tech-stack:
  added: []
  patterns:
    - Generic byte admission precedes the explicitly local-only checkpoint decoder
    - Retained CSV and canonical JSON outputs share manifest-last atomic publication
    - Static raw-I/O allowlists name individual adapter seams rather than exempt directories

key-files:
  created:
    - projects/spatial-pharma-dl/tests/test_model_fold_contracts.py
  modified:
    - projects/spatial-pharma-dl/src/models.py
    - projects/spatial-pharma-dl/src/train.py
    - projects/spatial-pharma-dl/src/eval.py
    - projects/spatial-pharma-dl/src/benchmark.py
    - projects/spatial-pharma-dl/scripts/run_pipeline.py
    - projects/spatial-pharma-dl/scripts/build_notebooks.py
    - projects/spatial-pharma-dl/scripts/build_foundation_notebook.py
    - projects/spatial-pharma-dl/tests/test_artifact_orchestration.py

key-decisions:
  - "Checkpoint weights_only=False remains a visibly local-writer-only compatibility boundary; Phase 5 owns hostile-safe deserialization."
  - "Benchmark and retained result identities include exact model/fold/slide or named-table lineage while paths, timestamps, and formatting remain irrelevant."
  - "The four regenerated notebooks use named adapters without changing numbering, kernels, commands, or final payload names."
  - "Sidecar peeking uses the same bounded regular-file manifest reader as generic admission, never an unbounded direct read."

requirements-completed: [ART-03, ART-04]

duration: 27min
completed: 2026-07-17
---

# Phase 4 Plan 03: Model, Result, and Orchestration Artifact Summary

**The complete supported pipeline now binds checkpoints, reports, manifests, summaries, and notebook-retained tables to the same deterministic, atomic artifact contract as its scientific caches.**

## Performance

- **Duration:** 27 min
- **Completed:** 2026-07-17
- **Tasks:** 3
- **Artifact inventory:** 19 retained kinds

## Accomplishments

- Added atomic local checkpoint publication with model, target, training, fold, patch, and label lineage plus exact state-key/shape/dtype declarations.
- Required bounded sidecar admission and checksum matching before the legacy local checkpoint decoder, followed by exact metadata and expected-model state-schema comparison before applying weights.
- Added atomic benchmark report production and validated loading with exact columns, row identity, numeric finiteness, duplicate policy, experiment identity, and upstream lineage.
- Added named atomic CSV and canonical JSON adapters for cohort/preprocessing manifests, cohort summary, experiment summary, training history, and both retained foundation result tables.
- Preserved the canonical inner bytes of `cohort-manifest-v1` and `spatial-pharma-preprocessing-manifest-v1` inside additive artifact wrappers.
- Routed the runner's report consumer through `load_benchmark_report` and retained typed patch-reuse status from Plan 04-02.
- Regenerated exactly notebooks 01, 04, 05, and 07 from their builders; notebook output tables and report/label consumers now use named adapters.
- Added an exact 19-kind pipeline inventory, mixed-generation and truncation recovery through production readers, and a narrow static audit covering source, scripts, emitted notebook code, and gallery generation.

## Task Commits

1. **Task 1: Contract checkpoints and benchmark/result adapters** — `421d7ee`
2. **Task 2: Route runner and generated notebooks through adapters** — `9ee3a24`
3. **Task 3: Close lineage, recovery, and static bypass inventory** — `9049eb3`

## Test Evidence

- Checkpoint/report focused gate: **4 passed**; scoped Ruff passed.
- Runner/notebook/cohort focused gate: **54 passed**; scoped Ruff passed.
- Complete Phase 4 plus affected Phase 1–3 regressions: **285 passed**.
- Generic contract, all artifact adapters, orchestration, and checkpoint gate: **119 passed** after bounded sidecar-reader hardening.
- Repository fast gate: Ruff passed and **383 offline tests passed** in about 20 seconds.
- `git diff --check` passed and the worktree was clean before summary/tracking publication.

## Compatibility

- Preserved CLI flags, config keys, public training/evaluation entry points, notebook numbering and kernels, final payload filenames, result columns, and terminal workflow.
- The generated foundation notebook was rebuilt from source, removing prior stored execution output while retaining its teaching content and public commands.
- Phase 2 cohort and Phase 3 preprocessing inner schemas remain unchanged; only sidecars are additive.

## Security Boundary

- No attacker-authored checkpoint is used in evidence.
- `torch.load(..., weights_only=False)` remains explicitly labeled `trusted-local-writer-only`; checksum and lineage do not imply authenticity.
- Phase 5 remains responsible for safe non-pickle patch/checkpoint formats and hostile deserialization safety.

## Deviations from Plan

- Updated one earlier valid-configuration regression fixture to use the established complete benchmark row schema; incomplete `{"fold": 0}` rows are now correctly rejected by the production report contract.
- Added a bounded public sidecar reader in `utils.artifacts` so fingerprint expectation reconstruction cannot perform an unbounded direct sidecar read.

## Issues Encountered

- One existing runner fixture needed explicit adapter stubs after retained outputs moved behind deferred stage loading.
- The existing optional pandas accelerator, chained-assignment fixture, and legacy notebook cell-ID warnings remain; all mandatory gates pass.

## Next Phase Readiness

- Phase 4 implementation is complete and ready for deep review and independent verification.
- Phase 5 can now replace legacy object patch archives and local pickle-compatible checkpoints without changing the artifact identity/publication layer.

## Self-Check: PASSED

- All three task commits are present.
- All focused, affected-regression, static-audit, Ruff, diff, and repository fast gates pass.
- No push was performed.

---
*Phase: 04-durable-artifact-contract*
*Completed: 2026-07-17*
