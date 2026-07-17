---
status: clean
depth: deep
files_reviewed: 14
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
resolved_findings: 7
---

# Phase 3 Code Review

**Scope:** Implementation commits after verified plans (`8256f71..4915412`), all Phase 3 context/research/validation/pattern/plan/summary artifacts, the 14 changed production and test files, and the ordinary, foundation, AnnData, H5AD, and run-manifest call chains.

**Result:** Clean after three fix and re-review iterations. All five warning findings and both informational findings are resolved, the affected regressions pass, and the canonical fast gate is green.

## Resolved Warning Findings

### WR-01 — Persisted AnnData slide identity is trusted rather than validated — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/identity.py:365-410` validates only the requested `slide_id` parameter and `adata.obs_names`; it synthesizes every admitted key as `(requested_slide_id, obs_name)` at line 385 and never inspects an existing `adata.obs["slide_id"]`. Both persisted-data consumers rely on this guard: `projects/spatial-pharma-dl/src/labels.py:140-145` and `projects/spatial-pharma-dl/src/patches.py:146-186`. A processed AnnData whose observation rows say `slide_b` can therefore be loaded as `slide_a`, relabeled as `slide_a`, and turned into `slide_a` patch metadata without a cross-slide failure.

**Reproduction:** A `SimpleNamespace` with `obs_names=["p0", "p1"]` and `obs.slide_id=["wrong", "wrong"]` was accepted by `validate_anndata_spot_identity(..., "slide_a")`.

**Impact:** This violates D-01/VAL-02's exact compound-key contract at the persisted producer boundary and can silently associate molecular targets or image patches with the wrong slide, producing scientifically incorrect training/evaluation inputs.

**Required fix:** When `adata.obs` contains `slide_id`, admit that column with the same exact-string/null/blank/hostile rules, require every row to equal the requested slide, and report deterministic `wrong_slide` evidence before label or patch science. Keep raw AnnData compatibility by defining and testing the behavior when the column is absent; processed `load_slide()` consumers should require it. Add label- and patch-producer tests with mixed and wholly wrong persisted slide IDs.

**Resolution:** `c31ef9f` adds optional raw-source compatibility plus required persisted-source admission. Label and patch producers now require `obs["slide_id"]`, reject missing/null/blank/non-exact/mixed/wholly wrong identities, and emit bounded structured `wrong_slide` evidence before marker, coordinate, image, or stack work.

### WR-02 — Scientific count validation accepts impossible preprocessing histories — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/validation.py:464-473` checks only that spots and genes are monotonically nonincreasing. It does not enforce the invariants of the actual fixed pipeline: `filter_cells` cannot change gene count, `filter_genes` cannot change spot count, and mitochondrial spot filtering cannot change gene count. `PreprocessingManifest` recomputes records through this same incomplete resolver at lines 327-360, so its advertised semantic consistency check does not close the gap.

**Reproduction:** A record with `input_genes=10`, `after_filter_cells_genes=7`, `after_filter_cells_spots=10`, `after_filter_genes_spots=8`, `after_filter_genes_genes=6`, and `post_qc_genes=5` was accepted by both the resolver and `PreprocessingManifest`, even though every cross-axis decrease is impossible under `preprocess_slide()`.

**Impact:** Malformed or forged provenance can be published as scientifically validated and byte-stable despite not describing any execution of the claimed QC pipeline. Counts and exclusions no longer reconstruct the actual filtering history required by D-04/VAL-05.

**Required fix:** Enforce exact unchanged-axis equalities (`after_filter_cells_genes == input_genes`, `after_filter_genes_spots == after_filter_cells_spots`, `post_qc_genes == after_filter_genes_genes`) in the pure resolver before resolution/canonicalization. Retain monotonic checks on the axes each stage may filter, and add focused resolver and manifest rejection tests for each impossible transition.

**Resolution:** `3fc0ca1` enforces all three fixed-axis equalities in the pure resolver while retaining the existing allowed-axis monotonic checks and structured scientific-minimum failures. Resolver and manifest tests reject each impossible transition, and a forged exclusion history is independently rejected by manifest recomputation.

### WR-03 — Invalid-type diagnostics can execute attacker-controlled metaclass hooks — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/identity.py:57-59` renders an invalid value's type by reading `type(value).__module__` and `type(value).__qualname__`. Those are attribute lookups on a caller-defined class and can be intercepted by a custom metaclass. The unsafe helper is used before rejection for parameters and key cells at lines 62-69, 115-120, 303-315, and 378-382.

**Reproduction:** An invalid key object whose class uses a metaclass that raises from `__getattribute__` for `__module__`/`__qualname__` escaped as `AssertionError("metaclass hook executed")` instead of producing `IdentityValidationError`.

**Impact:** The phase's hostile-input guarantee is incomplete: arbitrary key cells can execute caller-controlled hooks before structured admission and bypass deterministic diagnostics.

**Required fix:** Produce inert type evidence without attribute access on caller-controlled class objects (for example, a fixed safe category for non-built-in values, or another demonstrably hook-free representation). Add a custom-metaclass adversary to the shared fixture and exercise parameters, label keys, metadata keys, and AnnData observation names.

**Resolution:** `c31ef9f` replaces caller-class attribute reads with exact built-in categories and a fixed `non_builtin_object` label. Custom-metaclass adversaries cover parameters, label cells, metadata cells, and AnnData names; a separate persisted-slide subclass proves exact-type rejection without executing repr or strip hooks.

### WR-04 — Valid key samples make error diagnostics unbounded and log-injectable — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/identity.py:41-46` interpolates admitted exact strings directly into exception text. `_sample()` at lines 152-153 bounds only the number of keys; it does not bound string length or escape control characters. Exact strings containing newlines or very large payloads remain valid key components and are emitted verbatim on unmatched/duplicate/cross-slide errors.

**Reproduction:** One unmatched spot ID containing a newline plus 200,000 characters produced a 200,176-character exception containing the raw injected line break.

**Impact:** A single malformed identity can create oversized diagnostics and forge multiline log content. This contradicts the research requirement for bounded safe samples and weakens deterministic operational handling.

**Required fix:** Keep canonical keys unchanged for comparison, but render samples through a deterministic bounded encoder that escapes controls and truncates each component by code point/byte budget. Add newline/control, very-long-ID, Unicode, duplicate, unmatched, and cross-slide tests proving stable bounded output.

**Resolution:** `c31ef9f` retains complete exact strings in `IdentityIssue.sample_keys` while rendering only a JSON-escaped 64-code-point prefix plus an omitted-length marker. Duplicate, unmatched, and cross-slide tests use newline, tab, NUL, long Unicode, and 200,000-code-point IDs and prove deterministic single-line output below the fixed diagnostic budget.

## Resolved Informational Finding

### IN-01 — One guard-ordering test can pass on the forbidden failure — Resolved

**Evidence:** `projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py:237-260` installs forbidden Scanpy-import and seed sentinels, but the malformed-config branch uses `pytest.raises(Exception)` at line 257. The forbidden `AssertionError("scanpy import reached")` is also an `Exception`, so the test would remain green if production regressed and imported Scanpy before configuration validation.

**Impact:** The test does not prove the exact guard ordering claimed by Plan 03-02 and its summary, even though current production ordering is correct.

**Required fix:** Assert `ConfigValidationError` (and preferably its relevant issue path), then independently assert that import/copy/seed sentinels were not called. Avoid broad exception assertions for forbidden-seam tests.

**Resolution:** `3fc0ca1` asserts `ConfigValidationError`, verifies the `preprocessing.n_pcs` issue path, and independently proves the Scanpy import, AnnData copy, and seed sentinels remain untouched for both invalid configuration and invalid identity branches.

## Resolved Warning Finding

### WR-05 — Required-column detection executes caller-controlled equality hooks — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/identity.py:123-136` converts table columns to a tuple and then evaluates exact required names with `column not in columns`. The new persisted AnnData path repeats the same pattern at lines 451-466 with `"slide_id" in tuple(obs.columns)`. Python can fall back from the trusted string comparison to an arbitrary column label's `__eq__`, so neither boundary safely admits the schema before comparison. The subsequent `obs["slide_id"]` lookup would also hash/compare labels, but it is currently reached without first proving every column label is an exact built-in string.

**Reproduction:** A pandas DataFrame with one custom `EvilColumn` label whose `__eq__` raises was supplied to each public boundary. `validate_anndata_spot_identity(..., require_slide_id=True)` and `align_labels_with_metadata(...)` both escaped as `AssertionError("column equality hook")` instead of raising structured `IdentityValidationError`; the hook call was recorded before any identity issue was returned.

**Impact:** WR-03 closed hostile key-cell and parameter type rendering, but arbitrary table/AnnData schema labels can still execute caller-controlled code before admission. This violates the Phase 3 research rule that arbitrary identity inputs must not be hashed, compared, converted, or rendered before exact-type gates and leaves ordinary, foundation, label, and patch boundaries exposed through their shared schema checks.

**Required fix:** Admit column labels by iterating once and checking `type(label) is str` before any equality, membership, hashing, sorting, or pandas lookup. Build the set/map only from admitted exact strings, reject non-exact labels with inert bounded schema evidence, then perform required/reserved-name checks and `obs["slide_id"]` access. Add custom-column-label adversaries for labels, metadata, optional raw AnnData, and required persisted AnnData; prove no `__eq__`, `__hash__`, `__repr__`, or metaclass naming hook executes.

**Resolution:** `842e143` adds ordinal exact-type schema admission before any trusted-name comparison or pandas lookup. Label, metadata, optional raw AnnData, and required persisted AnnData adversaries now reject with structured inert `invalid_type` evidence without executing equality, hashing, repr, str, or metaclass naming hooks; exact-string missing/reserved issue ordering remains unchanged.

## Resolved Informational Finding

### IN-02 — Duplicate required key columns escape the structured schema boundary — Resolved

**Evidence:** `projects/spatial-pharma-dl/src/identity.py:123-145` exact-type-admits column labels but retains duplicates. `_schema_issues()` at lines 148-162 checks only whether each required name is present, not whether it appears exactly once. `_admit_key_rows()` then executes `frame[column].iat[row]` at lines 165-175; with duplicate `slide_id` or `spot_id` columns, pandas returns a DataFrame rather than a Series and `.iat[row]` raises an internal positional-argument `TypeError`. The AnnData path similarly treats duplicate `slide_id` columns as present at lines 477-497 and later handles the returned DataFrame as if it were a Series, producing misleading cell-type evidence rather than a schema issue.

**Reproduction:** Exact built-in DataFrames containing two `slide_id` columns or two `spot_id` columns were passed to `align_labels_with_metadata()`. Both escaped as `TypeError("DataFrame._get_value() missing 1 required positional argument: 'col'")`, not `IdentityValidationError`. An AnnData-like object with duplicate persisted `slide_id` columns produced an `invalid_type` cell issue instead of identifying the ambiguous schema.

**Impact:** The boundary remains fail-closed, so this does not silently align data, but a legal pandas schema shape bypasses deterministic actionable validation and exposes an implementation error. This weakens D-01's explicit compound-identity contract and makes malformed artifacts harder to diagnose.

**Required fix:** During exact-string schema admission, count required and reserved labels by ordinal and require each key column exactly once. Emit a deterministic schema issue such as `duplicate_column` with side/count and bounded ordinal evidence before any `frame[column]` or `obs["slide_id"]` lookup. Add label, metadata, and persisted AnnData tests for duplicate `slide_id` and `spot_id` columns while preserving the existing missing/reserved issue order.

**Resolution:** `0503519` counts excess exact required and reserved columns in fixed trusted-name order after exact-type admission and before any DataFrame selection. Labels, metadata, and persisted AnnData now emit bounded ordinal `duplicate_column` evidence for duplicate `slide_id` and `spot_id`; duplicate reserved columns report excess counts before the existing `reserved_column` issue, with missing/reserved ordering otherwise unchanged.

## Verification Performed

- Reviewed the complete `8256f71..a22fc1a` production/test/fix/evidence diff, `03-REVIEW-FIX.md`, and all Phase 3 planning artifacts.
- Traced ordinary patch loading, foundation cache hit/miss, label and patch producers, preprocessing orchestration, H5AD restoration, and admitted-order manifest assembly.
- Re-ran targeted read-only probes for persisted wrong-slide AnnData, impossible stage-count provenance, hostile custom-metaclass values, and oversized/control-character key diagnostics; all original warnings are closed.

## Fix and Re-review Verification

- Focused adversarial re-review: 17 passed for persisted identity, metaclass hooks, bounded key diagnostics, impossible count histories, forged exclusions, and exact guard ordering.
- Independent focused Phase 3 modules: 93 passed in 18.43 seconds.
- Canonical `python scripts/verify.py fast`: repository Ruff passed and all 250 strict offline tests passed in 21.19 seconds.
- Independent read-only probes confirmed WR-01 through WR-04 are closed: wrong persisted slides raise `wrong_slide`; all three impossible fixed-axis histories are rejected; custom metaclass key values execute no naming hooks; and a 200,000-character/control-bearing key produces a bounded single-line diagnostic.
- The same probe exposed WR-05 in both AnnData and table alignment schema paths despite the green suites.
- Second-iteration WR-05 focused adversarial gate: 5 passed; label, metadata, optional raw AnnData, required persisted AnnData, and exact-string compatibility paths all passed.
- Second-iteration affected Phase 3 and boundary regression suite: 177 passed; scoped Ruff passed.
- Second-iteration canonical `python scripts/verify.py fast`: repository Ruff passed and all 255 strict offline tests passed in 20.49 seconds.
- Static re-review confirmed `_schema_issues` and AnnData `slide_id` detection iterate and exact-type-admit every column label before required/reserved membership or `obs["slide_id"]` lookup.
- Independent final focused Phase 3 modules: 98 passed in 18.97 seconds.
- Independent final canonical `python scripts/verify.py fast`: repository Ruff passed and all 255 strict offline tests passed in 20.20 seconds.
- Independent probes reconfirmed WR-01 through WR-05 execute no forbidden hooks and exposed IN-02 for duplicate exact required-column labels despite the green suites.
- Third-iteration duplicate and hostile schema gate: 13 passed, covering both required columns on both alignment sides, both persisted AnnData identity names, reserved-column excess counts, exact missing/reserved ordering, and hostile labels.
- Third-iteration affected Phase 3 and boundary regression suite: 185 passed; scoped Ruff passed.
- Third-iteration canonical `python scripts/verify.py fast`: repository Ruff passed and all 263 strict offline tests passed in 19.70 seconds.
- Static re-review confirmed duplicate schema issues are complete before `frame[column]` or `obs["slide_id"]` selection and no pandas `TypeError` or misleading cell issue can escape for the covered duplicate schemas.
- Independent final-final focused Phase 3 modules: 106 passed in 18.39 seconds.
- Independent final-final canonical `python scripts/verify.py fast`: repository Ruff passed and all 263 strict offline tests passed in 20.63 seconds.
- Independent final-final probes covered duplicate `slide_id` and `spot_id` on both alignment sides and persisted AnnData, duplicate reserved columns with excess counts, hostile labels on both alignment sides and optional/required AnnData, wrong persisted slides, all three impossible preprocessing histories, custom metaclass values, and 200,000-character control-bearing diagnostics. Every boundary returned the intended structured evidence with no forbidden hook, pandas `TypeError`, raw control, or unbounded message.

## Positive Observations

- Complete shuffled label/metadata tables are aligned one-to-one in metadata order with both source ordinals retained.
- Ordinary and foundation consumers share the strict aligner and reject ordinary null/blank/subclass/duplicate/missing/cross-slide/cardinality defects before merge, indexing, encoding, or cache publication.
- PCA, neighbors, and graph-PC dimensions use the intended conservative formulas after actual HVG selection, and the real Scanpy/H5AD path is exercised in the fast gate.
- The additive preprocessing manifest preserves admitted order and leaves `cohort-manifest-v1` unchanged.
- No Phase 4+ artifact durability, safe-format migration, fold-policy, leakage, image-science, or label-confidence behavior entered the Phase 3 production diff.

## Review Conclusion

Phase 3 is **clean** after the third fix iteration. WR-01 through WR-05 and IN-01 through IN-02 are resolved with adversarial evidence, no warning or informational finding remains in the reviewed call chains, and the phase is ready for independent verification.
