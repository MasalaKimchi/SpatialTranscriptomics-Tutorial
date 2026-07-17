---
status: all_fixed
findings_in_scope: 4
fixed: 4
skipped: 0
iteration: 2
---

# Phase 4 Code Review Fix Report

All four findings in the second deep-review iteration were reproduced, fixed test-first, and closed. The first iteration's WR-01..WR-08 closure remains intact.

| Finding | Resolution | Commit |
|---|---|---|
| WR-01 | Typed deep JSON recursion rejection | `0ec430e` |
| WR-02 | Decoder-bound immutable admitted snapshots plus generic/patch/checkpoint ABA evidence | `0ec430e`, `2a423c0` |
| WR-03 | Observed-vs-declared schema equality before replacement | `0ec430e` |
| WR-04 | Per-kind leaf projection with embedding operational-control invariance | `0ec430e` |
| WR-05 | Independent current lineage/identity/value requirements and runner/notebook wiring | `5d64971` |
| WR-06 | Exact named table/JSON schemas and typed manifest reconstruction | `5d64971`, `2a423c0` |
| WR-07 | Actual processed/stain/label/patch parent-manifest binding, including partial cohorts | `2e8144c` |
| WR-08 | Root notebook atomic adapters, notebook static scan, and real 19-artifact chain | `da44861` |
| WR-06R | Exact cohort partitions and scientific metric bounds | `5027032` |
| WR-07R | Production-admitted child lineage and explicit stain-reference identity | `6fcf68f` |
| WR-08R | Real 19-artifact production fixture and exact source/notebook static audit | `b14e003` |
| WR-09 | Nonempty reusable writer lineage required before filesystem side effects | `5027032` |

## Verification Evidence

- Focused adversarial and affected regression gates passed after every fix batch.
- Final repository gate: `python scripts/verify.py fast` — Ruff passed; 400 offline tests passed.
- No network, dataset download, model download, or push was performed.

## Remaining Boundary

No Phase 4 review finding remains. The documented local-writer-only pickle compatibility boundary is intentionally deferred to Phase 5 and is not represented as safe for untrusted input.
