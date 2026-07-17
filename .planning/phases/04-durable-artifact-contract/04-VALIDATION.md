---
phase: 4
slug: durable-artifact-contract
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-17
---

# Phase 4 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest strict offline tier + Ruff |
| **Config file** | `pyproject.toml` and `scripts/verify.py` |
| **Quick run command** | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_contract.py projects/spatial-pharma-dl/tests/test_artifact_adapters.py projects/spatial-pharma-dl/tests/test_artifact_orchestration.py` |
| **Full suite command** | `python scripts/verify.py fast` |
| **Estimated runtime** | Under 300 seconds on CPU |

## Sampling Rate

- After every task: run the focused contract/adapter/orchestration module named by the task.
- After every plan: run all Phase 4 modules plus directly affected Phase 1–3 regressions.
- Before phase verification: `python scripts/verify.py fast` must pass.
- Max focused feedback latency: 90 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | ART-03 | T-01, T-02 | Exact bounded manifest admission and per-kind projections produce deterministic relevant invalidation and irrelevant stability | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_contract.py -k "manifest or fingerprint or hostile or projection"` | ✅ | ✅ passed |
| 4-01-02 | 01 | 1 | ART-04 | T-03, T-04 | Generic admission verifies regular-file state, fingerprint, byte count, checksum, and completion before reader callbacks | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_contract.py -k "checksum or incomplete or stale or truncated or symlink"` | ✅ | ✅ passed |
| 4-01-03 | 01 | 1 | ART-04 | T-04, T-05 | Same-directory temporary publication validates through the reader and commits payload first/manifest last at every fault point | fault-injection | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_contract.py -k "atomic or fault or replace or fsync or cleanup"` | ✅ | ✅ passed |
| 4-02-01 | 02 | 2 | ART-03, ART-04 | T-06 | Processed H5AD and label/domain tables validate lineage, schema, identity, shapes, and completion before reuse | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_adapters.py -k "h5ad or processed or label or domain"` | ✅ | ✅ passed |
| 4-02-02 | 02 | 2 | ART-03, ART-04 | T-06, T-07 | Patch/index and embedding artifacts reject legacy/stale/wrong-key/shape/dtype/identity payloads before consumer work | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_adapters.py -k "patch or index or embedding"` | ✅ | ✅ passed |
| 4-02-03 | 02 | 2 | ART-03 | T-02, T-07 | Acquisition regenerates stale/legacy scientific caches while train-only/direct reads fail with guidance and transitive lineage misses | orchestration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_adapters.py projects/spatial-pharma-dl/tests/test_artifact_orchestration.py -k "regenerate or train_only or lineage or stale"` | ✅ | ✅ passed |
| 4-03-01 | 03 | 3 | ART-03, ART-04 | T-06, T-08 | Checkpoints and benchmark/report tables validate contract/state/table schemas without claiming Phase 5 deserialization safety | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_adapters.py -k "checkpoint or benchmark or report"` | ✅ | ✅ passed |
| 4-03-02 | 03 | 3 | ART-04 | T-04, T-08 | Cohort/preprocessing/experiment manifests, summaries, training history, and foundation result tables publish atomically with unchanged inner schemas and no reader/writer bypass | orchestration/static | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_orchestration.py -k "manifest or summary or training_history or foundation_result or bypass or pipeline"` | ✅ | ✅ passed |
| 4-03-03 | 03 | 3 | ART-03, ART-04 | T-01–T-08 | Cross-artifact lineage, crash recovery, compatibility, and all prior guarantees remain green | regression | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_contract.py projects/spatial-pharma-dl/tests/test_artifact_adapters.py projects/spatial-pharma-dl/tests/test_artifact_orchestration.py projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py projects/spatial-pharma-dl/tests/test_foundation.py projects/spatial-pharma-dl/tests/test_cohort_admission.py && python scripts/verify.py fast` | ✅ | ✅ passed — Ruff + 400 offline |

## Wave 0 Requirements

- [x] `projects/spatial-pharma-dl/tests/test_artifact_contract.py` — pure manifest, fingerprint, checksum, hostile admission, and atomic fault-injection evidence.
- [x] `projects/spatial-pharma-dl/tests/test_artifact_adapters.py` — real tiny H5AD/NPZ/table production-reader schema and lineage evidence; checkpoint coverage remains Plan 04-03.
- [x] `projects/spatial-pharma-dl/tests/test_artifact_orchestration.py` — regeneration, runner publication, bypass inventory, and compatibility evidence.
- [x] Reuse Phase 1 artifact fixtures and Phase 2/3 exact primitive/identity fixtures; do not create parallel conventions.

## Manual-Only Verifications

All Phase 4 acceptance behavior is automated under `tmp_path` with synthetic local artifacts, deterministic fault injection, and production readers. No network, dataset, or model-weight download is required.

## Threat Model

| Ref | Threat | Required control |
|-----|--------|------------------|
| T-01 | Malformed/oversized/hostile sidecar executes hooks or consumes unbounded resources | Exact primitive bounded duplicate-key-rejecting admission before rendering or payload access |
| T-02 | Over- or under-inclusive fingerprints reuse stale science or invalidate on presentation changes | Explicit allowlisted per-kind projections, contract versions, upstream lineage, and positive/negative invariance tests |
| T-03 | Truncated/replaced/symlinked payload passes existence checks | Regular-file admission, streamed byte count/checksum, stat consistency, and typed rejection before payload reader |
| T-04 | Two-file publication exposes a mixed generation as valid | Same-directory temps, reader validation, fsync, payload-first/manifest-last atomic replace, checksum commit marker |
| T-05 | Writer validates a different path/parser than production reuse | Temporary publication must pass the exact production reader adapter before replacement |
| T-06 | Generic checksum admission hides wrong scientific keys/shapes/types/identity | Kind-specific production readers validate exact semantic schema after generic admission |
| T-07 | Legacy/stale artifacts silently fall back to filename-only reuse | Typed non-reusable state; acquisition regenerates, direct/train-only readers fail closed |
| T-08 | Direct library reads/writes bypass the contract or Phase 5 safety is falsely claimed | Static bypass inventory, centralized adapters, and explicit legacy-deserialization boundary |

## Validation Sign-Off

- [x] Every requirement, success criterion, and context decision has automated evidence.
- [x] No three consecutive implementation tasks lack focused automated verification.
- [x] Missing test modules are explicit Wave 0 deliverables.
- [x] Every publication fault point is exercised for new and replacement destinations, including the valid-new-generation state possible after manifest replacement but before a successful final directory fsync.
- [x] Both relevant invalidation and irrelevant stability are required per artifact kind.
- [x] Full fast-tier latency target remains under 300 seconds.
- [x] `nyquist_compliant: true` is set.

**Approval:** approved 2026-07-17
