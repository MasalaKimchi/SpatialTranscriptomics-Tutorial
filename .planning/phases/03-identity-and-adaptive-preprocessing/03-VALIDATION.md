---
phase: 3
slug: identity-and-adaptive-preprocessing
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-17
---

# Phase 3 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest strict offline tier + Ruff |
| **Config file** | `pyproject.toml` and Phase 1 `scripts/verify.py` |
| **Quick run command** | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_identity_alignment.py projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py` |
| **Full suite command** | `python scripts/verify.py fast` |
| **Estimated runtime** | Under 300 seconds on CPU |

## Sampling Rate

- After every task: run the focused test module named in the task.
- After every plan: run both Phase 3 modules plus directly affected regression modules.
- Before phase verification: `python scripts/verify.py fast` must pass.
- Max focused feedback latency: 60 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | VAL-02 | T-01, T-02 | Exact compound keys reject malformed and hostile values before hashing, merge, or indexing | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_identity_alignment.py -k "key or hostile or duplicate or mismatch"` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | VAL-02 | T-02, T-03 | Complete shuffled tables align one-to-one in metadata order with inspectable source rows | unit/integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_identity_alignment.py -k "shuffle or order or source"` | ❌ W0 | ⬜ pending |
| 3-01-03 | 01 | 1 | VAL-02 | T-03 | CNN/RF and foundation cache hit/miss share the strict alignment contract and validate array cardinality | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_identity_alignment.py projects/spatial-pharma-dl/tests/test_foundation.py projects/spatial-pharma-dl/tests/test_empty_boundaries.py` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 2 | VAL-05 | T-04 | Pure two-stage resolver accepts or caps HVG/PCA/neighbors deterministically and rejects nonviable counts | unit | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py -k "resolve or cap or nonviable"` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 2 | VAL-05 | T-04, T-05 | Preprocessing calls Scanpy with resolved values and stops before graph work on invalid post-QC state | integration | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py -k "orchestration or scanpy or guard"` | ❌ W0 | ⬜ pending |
| 3-02-03 | 02 | 2 | VAL-05 | T-05 | AnnData and admitted-order run provenance are JSON-safe, canonical, consistent, and H5AD-stable | integration/round-trip | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py -k "provenance or canonical or round_trip"` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 3 | VAL-02, VAL-05 | T-01–T-05 | Cross-arm compatibility and all Phase 1/2/3 guarantees remain green | regression | `python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_identity_alignment.py projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py projects/spatial-pharma-dl/tests/test_synthetic_anndata.py projects/spatial-pharma-dl/tests/test_foundation.py projects/spatial-pharma-dl/tests/test_empty_boundaries.py projects/spatial-pharma-dl/tests/test_cohort_admission.py && python scripts/verify.py fast` | ✅ | ⬜ pending |

## Wave 0 Requirements

- [ ] `projects/spatial-pharma-dl/tests/test_identity_alignment.py` — exact compound-key, one-to-one alignment, consumer-boundary, hostile-value, and deterministic-diagnostic evidence.
- [ ] `projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py` — pure resolver, call-order, scientific-minimum, provenance, and H5AD round-trip evidence.
- [ ] Extend `projects/spatial-pharma-dl/tests/conftest.py::key_adversary_factory` and reuse the existing synthetic AnnData fixture; do not introduce a parallel fixture convention.

## Manual-Only Verifications

All Phase 3 acceptance behavior is automated with DataFrames, synthetic AnnData, temporary files, and fake/call-recording Scanpy seams. No network or model downloads are required.

## Threat Model

| Ref | Threat | Required control |
|-----|--------|------------------|
| T-01 | Null, blank, wrong-type, duplicate, or hostile key values reach pandas/NumPy operations | Exact built-in string gates and inert diagnostics before hashing, comparison, merge, or indexing |
| T-02 | Inner joins silently lose/multiply rows or cross-align repeated barcodes between slides | Full compound-key uniqueness/set equality and deterministic one-to-one metadata-order alignment |
| T-03 | Ordinary and foundation consumers implement divergent alignment or accept array/metadata shape mismatch | One shared alignment contract and pre-index cardinality checks at every consumer boundary |
| T-04 | Fixed dimensions become illegal after QC/HVG selection | Pure two-stage resolution from observed counts with conservative rank/neighbor limits and explicit scientific minima |
| T-05 | Implicit Scanpy adjustment or missing provenance obscures the analysis actually run | Resolved arguments passed exactly once and canonical requested/resolved/count/exclusion/reason metadata published in AnnData and run provenance |

## Validation Sign-Off

- [x] Every requirement and context decision has automated evidence.
- [x] No three consecutive implementation tasks lack a focused test.
- [x] Missing test files and fixture extensions are explicit Wave 0 deliverables.
- [x] Guard-order tests forbid merge/index/model/graph/output side effects after invalid input.
- [x] Full fast-tier latency target remains under 300 seconds.
- [x] `nyquist_compliant: true` is set.

**Approval:** approved 2026-07-17
