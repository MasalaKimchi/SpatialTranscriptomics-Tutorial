---
status: fixed
findings_in_scope:
  - CR-01
  - WR-01
  - WR-02
  - IN-01
fixed: 4
skipped: 0
iteration: 1
---

# Phase 1 Review Fixes

## CR-01 — Offline socket denial escape paths

Fixed in `3814d21`.

- Denied `socket.socket.connect_ex` plus `getaddrinfo`, `gethostbyname`,
  `gethostbyname_ex`, and `gethostbyaddr` during non-external pytest sessions.
- Added a `sitecustomize` guard propagated through `PYTHONPATH` and an explicit
  environment flag so ordinary child Python interpreters inherit the same
  socket and name-resolution denial.
- Added in-process and child-process regressions for connection and resolver
  paths without performing real network access.
- Documented the enforcement boundary honestly: the guard covers the pytest
  process and child Python interpreters that inherit its environment, but it is
  not OS-level isolation for arbitrary native executables.

## WR-01 — Empty opt-in tiers reported success

Fixed in `3814d21`.

- Preserved pytest exit status 5 when an opt-in tier collects no tests while
  retaining the explicit no-evidence diagnostic.
- Updated runner and workflow contract tests so empty dispatch tiers cannot be
  represented as successful verification and opt-in jobs cannot ignore errors.

## WR-02 — Unseeded model initialization

Fixed in `3814d21`.

- Moved model construction and all step randomness into a fixed-seed
  `torch.random.fork_rng()` scope.
- Added exact repeated-run loss and parameter comparisons.
- Asserted the caller's global CPU RNG state is unchanged after each run.

## IN-01 — Embedded pytest environment leakage

Fixed in `3814d21`.

- Captured the exact prior presence and value of `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE`, along with child-guard environment and patched socket
  functions, for each configured pytest session.
- Restored values exactly or removed variables that were originally absent in
  `pytest_unconfigure`.
- Added an in-process configure/unconfigure regression covering one preexisting
  custom value and one originally absent variable.

## Commits

- `3814d21` — `fix(01): close verification harness review findings`

## Verification

- Focused review regressions: 15 passed.
- `python scripts/verify.py fast`: Ruff passed; 58 offline tests passed.
- Bare `python -m pytest -q`: 58 tests passed.
- `python scripts/verify.py network` with no selected tests: exited 5 and printed
  that no evidence was produced.
- `git diff --check`: passed before commit.

All four findings were fixed; none were skipped.
