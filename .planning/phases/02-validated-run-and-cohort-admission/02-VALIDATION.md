---
phase: 2
slug: validated-run-and-cohort-admission
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-17
---

# Phase 2 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest strict offline tier + Ruff |
| **Config file** | `pyproject.toml` and Phase 1 `scripts/verify.py` |
| **Quick run command** | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_validation.py projects/spatial-pharma-dl/tests/test_cohort_admission.py projects/spatial-pharma-dl/tests/test_empty_boundaries.py` |
| **Full suite command** | `python scripts/verify.py fast` |
| **Estimated runtime** | Under 300 seconds on CPU |

## Sampling Rate

- After every task: run the focused test module named in the task.
- After every plan: run all Phase 2 modules plus affected existing tests.
- Before phase verification: `python scripts/verify.py fast` must pass.
- Max focused feedback latency: 60 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | VAL-01 | T-01 | Aggregated config issues block all side effects | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_validation.py` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | VAL-01 | T-02 | Canonical config is deterministic and JSON-safe | unit/static | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_validation.py && python -m ruff check projects/spatial-pharma-dl/src/validation.py projects/spatial-pharma-dl/src/data.py projects/spatial-pharma-dl/tests/test_validation.py` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 2 | VAL-04 | T-03 | Strict admission aggregates missing slides before processing | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_cohort_admission.py` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 2 | VAL-04 | T-03, T-04 | Provisional remote admission publishes only the final manifest; strict source failure carries a failure manifest and partial mode re-admits all outcomes | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_cohort_admission.py projects/spatial-pharma-dl/tests/test_verification_contract.py` | ❌ W0 | ⬜ pending |
| 2-02-03 | 02 | 2 | VAL-04 | T-03 | Downstream stages consume the single admitted cohort and cannot reintroduce silent missing-slide policies | integration/static | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_cohort_admission.py && python -m ruff check projects/spatial-pharma-dl/src/validation.py projects/spatial-pharma-dl/src/data.py projects/spatial-pharma-dl/src/labels.py projects/spatial-pharma-dl/src/patches.py projects/spatial-pharma-dl/scripts/run_pipeline.py projects/spatial-pharma-dl/tests/test_cohort_admission.py` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 3 | VAL-03 | T-05 | Empty data, label, patch, regression, dataset, and stain-reference boundaries fail before expensive work | unit/integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_empty_boundaries.py -k "data or label or patch or regression or dataset or stain"` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 3 | VAL-03 | T-05 | Empty LOSO, alignment, CNN, RF, prediction, and fold inputs fail with stage diagnostics | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_empty_boundaries.py projects/spatial-pharma-dl/tests/test_model_fold_smoke.py -k "loso or align or train or rf or predict or fold"` | ❌ W0 | ⬜ pending |
| 2-03-03 | 03 | 3 | VAL-01, VAL-03, VAL-04 | T-05 | Foundation boundaries are guarded and the complete Phase 2 plus fast regression gates stay green | regression | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_validation.py projects/spatial-pharma-dl/tests/test_cohort_admission.py projects/spatial-pharma-dl/tests/test_empty_boundaries.py projects/spatial-pharma-dl/tests/test_foundation.py projects/spatial-pharma-dl/tests/test_model_fold_smoke.py && python scripts/verify.py fast` | ✅ | ⬜ pending |

## Wave 0 Requirements

- [ ] `projects/spatial-pharma-dl/tests/test_validation.py` — aggregated/canonical config contract.
- [ ] `projects/spatial-pharma-dl/tests/test_cohort_admission.py` — strict and partial manifest contract.
- [ ] `projects/spatial-pharma-dl/tests/test_empty_boundaries.py` — shared stage-boundary error contract.
- [ ] Shared Phase 1 deterministic fixtures are reused; no parallel test convention is introduced.

## Manual-Only Verifications

All Phase 2 acceptance behavior is automated with temporary synthetic configurations and cohorts.

## Threat Model

| Ref | Threat | Required control |
|-----|--------|------------------|
| T-01 | Invalid config reaches side-effectful pipeline work | Pure aggregate validation before imports/processing |
| T-02 | Canonical output hides non-JSON Python objects or unstable ordering | Frozen typed values plus sorted canonical JSON round-trip tests |
| T-03 | Missing slides are silently dropped | Full preflight and fail-closed default |
| T-04 | Partial mode obscures exclusions/failures | Explicit opt-in and complete reason-coded manifest |
| T-05 | Empty work reaches expensive/scientific code | Shared earliest-boundary guard with stage-specific diagnostics |

## Validation Sign-Off

- [x] Every requirement and decision has automated evidence.
- [x] No three consecutive implementation tasks lack a focused test.
- [x] Missing test files are explicit Wave 0 deliverables.
- [x] Full fast-tier latency target remains under 300 seconds.
- [x] `nyquist_compliant: true` is set.

**Approval:** approved 2026-07-17
