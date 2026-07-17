---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 04-03-PLAN.md
last_updated: "2026-07-17T09:56:05.000Z"
last_activity: 2026-07-17
progress:
  total_phases: 10
  completed_phases: 3
  total_plans: 13
  completed_plans: 13
  percent: 30
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-17)

**Core value:** Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.
**Current focus:** Phase 4 — Durable Artifact Contract

## Current Position

Phase: 4
Plan: 3 of 3 in current phase
Status: Implementation complete; ready for phase review and verification
Last activity: 2026-07-17

Progress: [███░░░░░░░] 30%

## Performance Metrics

**Velocity:**

- Total plans completed: 13
- Average duration: 13 min
- Total execution time: 2.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 20 min | 7 min |
| 02 | 4 | 30 min | 8 min |
| 03 | 3 | 56 min | 19 min |

**Recent Trend:**

- Last 5 plans: 10 min, 7 min, 18 min, 20 min, 18 min
- Trend: Phase 3 adds broad adversarial and scientific integration coverage

*Updated after each plan completion*

- Phase 01 Plan 03: 6 min, 2 tasks, 5 files

| Phase 02 P01 | 6 min | 2 tasks | 4 files |
| Phase 02 P02 | 7min | 3 tasks | 6 files |
| Phase 02 P03 | 10min | 3 tasks | 11 files |
| Phase 02 P04 | 7min | 2 tasks | 5 files |
| Phase 03 P01 | 18min | 3 tasks | 9 files |
| Phase 03 P02 | 20min | 3 tasks | 6 files |
| Phase 03 P03 | 18min | 2 tasks | 3 files |
| Phase 04 P01 | 16min | 3 tasks | 3 files |
| Phase 04 P02 | 13min | 3 tasks | 13 files |
| Phase 04 P03 | 27min | 3 tasks | 16 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

- [Roadmap]: Exactly 20 v1 requirements are assigned once across 10 fine-grained MVP phases.
- [Roadmap]: Verification fixtures precede contract and scientific refactors; unsafe artifact migrations use the shared manifest/atomic foundation.
- [Roadmap]: Existing notebook, CLI, configuration, output, and public-import surfaces remain stable.
- [Phase 01]: Bare pytest selects offline evidence; only explicit network and full_cohort selections enable sockets. — Fail-closed defaults preserve offline evidence while retaining deliberate external-tier opt-ins.
- [Phase 01]: Scientific fixtures are fresh fixed-seed factories isolated under pytest tmp_path. — Later phases need reproducible adversarial inputs without shared mutation or repository artifact writes.
- [Phase 01]: Artifact and model evidence remains fixture-only or stubbed so later safety and scientific fixes retain their phase boundaries. — TEST-01 needs meaningful offline coverage without blessing unsafe readers or biased real training.
- [Phase 01]: Notebook structural checks tolerate legacy missing cell IDs and never execute or rewrite notebooks. — The public teaching sequence and artifacts remain stable while fast validation detects malformed notebooks.
- [Phase 01]: Fast verification runs Ruff before strict offline pytest and propagates the first failure. — One deterministic command keeps local and CI evidence identical and fail-fast.
- [Phase 01]: Empty pytest selections are reported as non-evidence only for explicit opt-in tiers. — An unpopulated optional tier should remain visible without weakening required fast evidence.
- [Phase 01]: Required CI is independent of dispatch-gated external evidence jobs and caches dependencies only. — Pull requests stay CPU/offline while scientific artifacts and model weights cannot enter the cache.
- [Phase 02]: Configuration resolution uses explicit standard-library validators. — Invalid startup must fail before scientific and model libraries load.
- [Phase 02]: Only production-optional fields receive resolver defaults. — Required scientific sections must remain observable as missing schema defects.
- [Phase 02]: load_config returns a fresh plain dictionary decoded from canonical JSON. — Existing notebook and runner mutation remains compatible while admitted state stays immutable.
- [Phase 02]: Remote admission remains provisional until complete source outcomes are known. — Only final admission may publish the cohort manifest or release downstream stages.
- [Phase 02]: Admission is the sole partial-cohort policy for downstream helpers. — Data, label, stain, patch, and benchmark stages consume one ordered admitted sequence or fail visibly.
- [Phase 02]: Stage errors expose only bounded primitive cardinality evidence before expensive work. — Stable diagnostics close VAL-03 without capturing scientific objects or host-local state.
- [Phase 02]: LOSO entry points require two unique non-empty slides but no class-support policy. — Phase 2 owns cardinality while Phase 6 owns class viability and unseen-class coverage.
- [Phase 02]: Every public stage rechecks cardinality after cohort admission. — Direct callers and partially admitted downstream subsets must fail before expensive work.
- [Phase 02]: Zero-row alignment reports cardinality only. — Compound-key diagnosis and repair remain explicitly owned by Phase 3.
- [Phase 02]: Configuration admits exact safe built-in types before caller value operations. — Hostile subclasses and arbitrary mappings must fail deterministically without executing overrides.
- [Phase 02]: Only cfg=None loads foundation and report defaults. — Every supplied mapping must resolve before cache, device, model, dataframe, output, or writer work.
- [Phase 03]: Exact `(slide_id, spot_id)` pairs are the sole alignment identity. — Reject coercion, lossy joins, and spot-only maps before scientific work.
- [Phase 03]: Metadata owns successful aligned row order. — Source ordinals make array and label provenance inspectable.
- [Phase 03]: Post-QC and post-HVG resolution remain separate pure stages. — Actual selected HVGs must determine PCA rank and graph dimensions.
- [Phase 03]: AnnData stores a canonical preprocessing JSON sibling. — Safe JSON restores exact built-in primitives after H5AD scalar decoding.
- [Phase 03]: Preprocessing run provenance validates completely before publication. — Malformed admitted-slide metadata must reach no manifest or downstream scientific effect.
- [Phase 03]: Real Scanpy remains mandatory fast evidence through the declared spatial-tx interpreter. — Missing scientific dependencies must fail actionably rather than skip or fall back to recorder-only evidence.
- [Phase 04]: Artifact state is canonical JSON behind immutable records. — Fresh exact-built-in views prevent caller mutation while keeping the generic layer import-light.
- [Phase 04]: Artifact kinds use explicit projection allowlists and contract versions. — Scientific lineage changes invalidate deterministically without presentation-only churn.
- [Phase 04]: Temporary validation uses the final logical basename and exact reuse reader. — Validated sidecar bytes remain unchanged when the payload and manifest move to final names.
- [Phase 04]: Checksums do not authenticate pickle-bearing formats. — Phase 5 remains responsible for removing unsafe patch and checkpoint deserialization.
- [Phase 04]: Cache-only source identity never claims to detect an unobserved remote mutation. — Stable provider/sample identity is always fingerprinted and observed content is included only when it can be supplied again.
- [Phase 04]: Scientific readers validate manifest-declared schemas after generic byte admission. — A valid checksum cannot bless wrong axes, rows, dtypes, identity, dimensions, or provenance.
- [Phase 04]: Legacy object patch archives are admitted only as local-writer output. — Generic admission precedes decoding, and Phase 5 remains the required safe-format migration boundary.
- [Phase 04]: Semantic corruption is not an automatic cache miss. — Acquisition may rebuild missing, legacy, or stale artifacts while checksum-valid reader failures remain visible.
- [Phase 04]: Checkpoint compatibility remains local-writer-only until Phase 5. — Contract admission and checksums establish integrity and lineage, not hostile pickle authenticity.
- [Phase 04]: Every retained runner and generated-notebook scientific table uses a named adapter. — Static inventory rejects new raw-I/O or filename-only reuse seams without broad directory exemptions.
- [Phase 04]: Sidecar expectation reads share the bounded regular-file manifest path. — Reconstructing an expected fingerprint cannot bypass sidecar size, symlink, or generation checks.

### Pending Todos

None yet.

### Blockers/Concerns

- Full-cohort and network validation remain explicit later evidence tiers; default planning and CI must stay CPU/offline.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | Package rename/path-bootstrap removal, notebook source redesign, chunked storage, and expanded experiment bundles | Out of v1 scope | Roadmap creation |

## Session Continuity

Last session: 2026-07-17T09:56:05.000Z
Stopped at: Completed 04-03-PLAN.md
Resume file: None
