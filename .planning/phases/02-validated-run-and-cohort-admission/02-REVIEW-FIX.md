---
status: fixed
findings_in_scope:
  - WR-01
  - WR-02
  - WR-03
  - WR-04
  - WR-05
  - WR-06
  - WR-07
  - IN-01
fixed: 8
skipped: 0
iteration: 2
---

# Phase 2 Review Fixes

## Final Result

All original Phase 2 review findings and all three iteration-2 warnings are
fixed. No finding was skipped and no Phase 3 identity, Phase 6 class-support,
artifact-durability, cache-format, or leakage policy was pulled forward.

## WR-01 — Total and deterministic configuration diagnostics

Fixed in `2d8e8f9`, `de38616`, and `65cbeb0`; adversarial tests are in
`05f5173` and `65cbeb0`.

- Oversized integers are rejected without float conversion or serialization
  escape, and received-value formatting never calls arbitrary `repr` methods.
- Invalid non-string keys use deterministic tokens derived only from exact safe
  primitive values or inert type metadata; user comparison methods are never
  invoked.
- Non-string key defects are reported once rather than duplicated by schema and
  JSON-tree passes.
- Reversed insertion of invalid primitive keys now yields identical ordered
  issues and byte-identical exception text.

## WR-02 — Canonical admission evidence and pre-hash validation

Fixed in `2d8e8f9` and `65cbeb0`; adversarial tests are in `05f5173` and
`65cbeb0`.

- Failure IDs/details and availability evidence are validated against configured
  IDs before manifest construction.
- Availability members are checked as exact strings in their original iterable
  before any set construction or membership hashing.
- Scalar strings, unhashable members, hostile-hash objects, unknown IDs, and
  non-string details raise `CohortAdmissionInputError`; valid duplicates are
  normalized successfully.
- Caller failure details are replaced by fixed public guidance under the stable
  `source_load_failed` code, keeping manifests JSON-safe and sanitized.

## WR-03 — Strict fail-fast source curation

Fixed in `2d8e8f9` and `65cbeb0`; adversarial tests are in `05f5173` and
`65cbeb0`.

- Strict curation stops on the first `SourceAcquisitionError`; no later slide can
  enter preprocessing, cache publication, or a writer seam.
- Later configured members appear deterministically in the failure manifest as
  skipped with `source_not_attempted`, never as included or failed.
- Explicit partial mode retains full configured-order outcome collection.
- Strict errors remain chained from the first source acquisition error while
  canonical manifest text excludes private exception details.

## WR-04 — Narrow source acquisition policy boundary

Fixed in `2d8e8f9`; tested in `05f5173`.

- Only documented connection, timeout, URL, and Requests failures from the
  source-loader call become `SourceAcquisitionError`.
- Preprocessing, implementation, and storage exceptions propagate unchanged.

## WR-05 — Label validation before output effects

Fixed in `2d8e8f9`; tested in `05f5173`.

- Every admitted slide frame is built and validated before output-directory
  creation or any Parquet/CSV write, preventing partial cohort publication.

## WR-06 — CNN target guard before device/model seams

Fixed in `2d8e8f9`; tested in `05f5173`.

- Classification and regression columns are selected before device resolution,
  printing, dataset construction, or model setup.

## WR-07 — Valid nested-LOSO cardinality

Fixed in `2d8e8f9`; tested in `05f5173`.

- Nested LOSO requires three unique non-empty slides before task preprocessing,
  without adding later class-support policy.

## IN-01 — Behavioral adversarial evidence

Fixed in `05f5173` and `65cbeb0`.

- Tests now exercise executable-repr values, oversized integers, deterministic
  invalid keys, unsafe failure details, pre-hash availability rejection, exact
  manifest collections, narrow exception taxonomy, forbidden output seams,
  strict fail-fast behavior, and integrated fold admission.

## Commits

- `2d8e8f9` — `fix(02): close adversarial validation boundaries`
- `05f5173` — `test(02): prove adversarial phase boundaries`
- `de38616` — `fix(02): harden diagnostic rendering`
- `65cbeb0` — `fix(02): close deterministic admission gaps`

## Verification

- Iteration-2 focused Phase 2 gate: 88 offline tests passed.
- Scoped Ruff gate: passed.
- `python scripts/verify.py fast`: Ruff passed; 146 offline tests passed.
- `git diff --check`: passed before commit.

All findings are fixed; none were skipped.
