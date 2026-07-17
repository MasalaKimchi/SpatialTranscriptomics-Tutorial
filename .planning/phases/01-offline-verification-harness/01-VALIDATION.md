---
phase: 1
slug: offline-verification-harness
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-17
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest with strict markers; Ruff for static checks |
| **Config file** | `pyproject.toml` and the Phase 1 runner/config artifacts created in Wave 0 |
| **Quick run command** | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests` |
| **Full suite command** | `python scripts/verify.py fast` |
| **Estimated runtime** | A few CPU-only minutes; focused task checks should finish in seconds |

---

## Sampling Rate

- **After every task commit:** Run the task's focused offline test command.
- **After every plan wave:** Run `python scripts/verify.py fast` once the runner exists; before then run the complete strict-marker offline pytest selection.
- **Before phase verification:** `python scripts/verify.py fast` must be green in a clean process.
- **Max feedback latency:** 300 seconds for the full fast tier; focused checks should remain under 60 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | TEST-01 | T-01 / T-02 | Offline tier denies sockets/downloads and rejects unclassified tests | contract | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_verification_contract.py` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | TEST-01 | T-03 | Fixed-seed valid/adversarial fixtures are repeatable and isolated | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_fixture_contracts.py` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | TEST-01 | T-04 | Primitive artifacts round-trip without object deserialization | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | TEST-01 | — | Real synthetic AnnData preserves spatial and aligned-table contracts | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_synthetic_anndata.py` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 2 | TEST-01 | T-02 | CPU model/fold smoke uses no pretrained download and covers each holdout once | smoke | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_model_fold_smoke.py` | ❌ W0 | ⬜ pending |
| 1-02-04 | 02 | 2 | TEST-01 | — | All 20 numbered notebooks parse, validate, and retain expected kernel metadata | structural | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_notebook_structure.py` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 3 | TEST-01 | T-01 / T-02 | Canonical fast command runs Ruff and only offline evidence | end-to-end | `python scripts/verify.py fast` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 3 | TEST-01 | — | Required CI is offline; notebook-smoke, network, and full-cohort remain explicit opt-ins | configuration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_verification_contract.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Strict pytest tier registration for `offline`, `notebook_smoke`, `network`, and `full_cohort`.
- [ ] `projects/spatial-pharma-dl/tests/conftest.py` shared deterministic fixture and offline guard support.
- [ ] `projects/spatial-pharma-dl/tests/test_verification_contract.py` and `test_fixture_contracts.py` establish the harness contract before downstream evidence is added.
- [ ] A canonical fast runner and CI entry point are created no later than the final plan.

---

## Manual-Only Verifications

All Phase 1 acceptance behavior is automated. Network and full-cohort tiers are intentionally opt-in automated evidence, not manual acceptance gates for this phase.

---

## Threat Model

| Ref | Threat | Required control |
|-----|--------|------------------|
| T-01 | A test silently reaches the network | Socket-level denial plus offline environment variables in the fast tier |
| T-02 | A model library downloads weights or remote code | No-pretrained/no-download construction and the same socket guard |
| T-03 | Shared mutable or unseeded fixtures make results flaky | Fresh factories, fixed seeds, repeatability assertions |
| T-04 | Artifact tests normalize unsafe object deserialization | Primitive-only NPZ with `allow_pickle=False`; object arrays must be rejected |

---

## Validation Sign-Off

- [x] All anticipated tasks have focused automated commands or explicit Wave 0 dependencies.
- [x] Sampling continuity: every implementation task is paired with automated evidence.
- [x] Wave 0 identifies all missing harness references.
- [x] No watch-mode flags.
- [x] Full fast-tier feedback target is under 300 seconds on CPU.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** approved 2026-07-17
