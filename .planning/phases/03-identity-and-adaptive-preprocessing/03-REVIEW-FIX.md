---
status: all_fixed
phase: 03-identity-and-adaptive-preprocessing
findings_in_scope: 6
fixed: 6
skipped: 0
iteration: 2
commits:
  - c31ef9f
  - 3fc0ca1
  - 842e143
---

# Phase 3 Code Review Fix Report

All five warning findings and the informational test-evidence finding from the deep Phase 3 review were fixed across two iterations and re-reviewed.

## Fixes Applied

- **WR-01:** Persisted AnnData compound identity now admits `obs["slide_id"]` as exact, nonblank, row-aligned strings and requires it in label and patch producers. Mixed, wholly wrong, missing, null, blank, subclass, and raw-source compatibility paths have direct tests.
- **WR-02:** The preprocessing resolver rejects changes on axes that the declared Scanpy stages cannot modify. Manifest reconstruction rejects the same three impossible transitions and forged exclusion totals.
- **WR-03:** Invalid-type evidence uses fixed inert labels and never reads custom metaclass module, name, or qualified-name attributes. Parameters, both table sides, AnnData names, and persisted slide cells execute no hostile hooks.
- **WR-04:** Canonical compound keys remain complete in structured evidence, while exception text JSON-escapes controls and truncates each component deterministically to a fixed code-point budget.
- **IN-01:** The guard-order regression now requires `ConfigValidationError`, checks the exact invalid configuration path, and separately proves Scanpy import, AnnData copy, and seed effects were not reached.
- **WR-05:** DataFrame and AnnData column labels are exact-type-admitted by ordinal before required-name equality, membership, hashing, or pandas lookup. Hostile schema labels produce inert structured diagnostics without executing caller hooks.

## Atomic Commits

1. `c31ef9f fix(03): harden persisted compound identity`
2. `3fc0ca1 fix(03): validate preprocessing histories`
3. `842e143 fix(03): admit identity schema labels safely`

## Verification Evidence

- First-iteration focused adversarial selection: 17 passed.
- Second-iteration WR-05 focused adversarial selection: 5 passed.
- Latest affected identity, preprocessing, synthetic AnnData, empty-boundary, and cohort-admission suite: 177 passed.
- Scoped Ruff: passed.
- Canonical fast gate: repository Ruff passed; 255 strict offline tests passed.
- Remaining findings after re-review: none.

The warnings emitted by the canonical gate are the pre-existing optional pandas accelerator, pandas Copy-on-Write test-helper, and legacy notebook cell-ID warnings already assigned outside this review scope.
