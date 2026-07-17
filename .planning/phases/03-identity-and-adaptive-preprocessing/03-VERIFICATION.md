---
phase: 03-identity-and-adaptive-preprocessing
status: passed
score: "20/20"
requirements:
  - VAL-02
  - VAL-05
date: 2026-07-17
verifier: independent-gsd-verifier
---

# Phase 3 Verification

## Result

Phase 3 achieves its goal: every retained spatial observation is admitted and
aligned by exact unique `(slide_id, spot_id)` identity, and every post-QC
analysis either uses deterministic legal HVG/PCA/neighbor dimensions or stops
with structured scientific nonviability. VAL-02 and VAL-05 pass against the
current production implementation, not only the phase summaries.

The score covers 20 independent checks spanning both requirements, all four
roadmap success criteria, D-01 through D-05, T-01 through T-05, ordinary and
foundation consumers, real Scanpy/H5AD persistence, run provenance, review
closure, and the canonical regression gate. All 20 pass; no gap remains.

## Twenty Acceptance Checks

| # | Check | Result | Evidence |
|---:|---|---:|---|
| 1 | Exact compound-key admission | Passed | Both key components require exact built-in, nonblank strings before tuple/hash/compare/merge/index work. |
| 2 | Hostile value and schema safety | Passed | Raising repr/str/hash/equality/order/strip/iteration/metaclass hooks remain uncalled for parameters, cells, and column labels. |
| 3 | Deterministic bounded diagnostics | Passed | Structured issues retain totals and at most five samples; text JSON-escapes controls and bounds each component. |
| 4 | Complete defect taxonomy | Passed | Missing/duplicate columns, reserved collisions, null/type/blank values, duplicate keys, wrong-slide, label-only, metadata-only, cross-slide, and cardinality defects fail explicitly. |
| 5 | Complete shuffled alignment | Passed | Metadata order is preserved while targets and provenance move together; both source ordinals remain inspectable. |
| 6 | Full-cohort label selection | Passed | Other admitted slides remain valid context; only the expected slide is selected, and repeated barcodes cannot cross slides. |
| 7 | Persisted AnnData identity | Passed | Label and patch producers require exact persisted `obs["slide_id"]`; raw preprocessing retains documented absent-column compatibility. |
| 8 | Ordinary consumer alignment | Passed | Patch arrays are indexed only by validated `_patch_source_row`, with no spot-only dictionary or silent subset. |
| 9 | Foundation consumer alignment | Passed | Cache hits and misses use the same compound-key contract and validate embedding/metadata cardinality before indexing or publication. |
| 10 | Cross-arm equivalence | Passed | An independent shuffled probe produced identical `s2, s1` ordinary/foundation order and paired target/provenance rows. |
| 11 | Stage-count admission | Passed | Counts are exact nonnegative integers, allowed axes are monotonic, and all fixed pipeline axes must remain unchanged. |
| 12 | HVG resolution | Passed | `hvg_call = min(requested_hvg, post_qc_genes)` with explicit accepted/capped reason. |
| 13 | PCA resolution | Passed | `pca = min(requested_pca, min(post_qc_spots, actual_hvgs) - 1)` after actual HVG selection. |
| 14 | Neighbor and graph-PC resolution | Passed | `neighbors = min(requested_neighbors, post_qc_spots - 1)` and `graph_pcs = min(requested_graph_pcs, resolved_pca)`. |
| 15 | Scientific nonviability | Passed | Fewer than three spots, fewer than two genes/HVGs, impossible histories, or illegal cardinalities raise structured errors before graph/save/downstream work. |
| 16 | Exact Scanpy orchestration | Passed | QC, normalization, HVG, scale, PCA, neighbors, UMAP, and Leiden retain order; each resolved dimension is passed exactly once. |
| 17 | AnnData provenance | Passed | Counts, exclusions, requested/resolved values, and reasons are exact JSON primitives under the additive preprocessing key with byte-stable canonical JSON. |
| 18 | Real Scanpy and H5AD path | Passed | Real capped Scanpy preprocessing persisted/restored the record, PCA shape, neighbor arguments, counts layer, raw data, clusters, and canonical JSON. |
| 19 | Run provenance and compatibility | Passed | Admitted-order `preprocessing_manifest.json` recomputes and validates every record; `cohort-manifest-v1`, config, CLI, notebook, import, and established output contracts are unchanged. |
| 20 | Regression gates | Passed | Affected suite: 190 passed. Scientific seam selection: 4 passed. Fast gate: Ruff plus 263 offline tests passed. |

## Requirement Mapping

### VAL-02 — Passed

- `src/identity.py` exact-type-admits schema labels and key cells before any
  caller-controlled hashing, comparison, rendering, pandas lookup, duplicate
  operation, merge, or NumPy indexing.
- `align_labels_with_metadata()` proves per-side uniqueness, expected-slide
  membership, exact set equality, and optional value cardinality before a
  metadata-left `validate="one_to_one"` merge.
- `labels.align_labels_with_patches()` retains its public two-argument
  DataFrame facade. `train.load_slide_patches()` and foundation cache hit/miss
  paths route through the same aligner and index arrays only by validated
  `_patch_source_row` ordinals.
- Label and patch producers validate persisted AnnData compound identity before
  marker, coordinate, image, transform, row-construction, or stack work.

### VAL-05 — Passed

- `resolve_post_qc_preprocessing()` validates input and stage counts, enforces
  the real pipeline's fixed/filtered-axis history, rejects fewer than three
  spots or two genes, and resolves the HVG call from actual post-QC genes.
- `finalize_preprocessing_resolution()` uses actual selected HVGs to resolve
  legal PCA, neighbor, and graph-PC values or raises structured post-HVG
  nonviability.
- `preprocess_slide()` passes those exact values to Scanpy and publishes the
  canonical record in AnnData. `PreprocessingManifest` exact-primitive-admits,
  schema-checks, recomputes, and compares every admitted-order record before
  pipeline manifests or downstream stages are written.
- Fake/call-recording and real Scanpy executions both prove the scientific call
  contract; the real H5AD round trip proves persisted and run-level provenance
  equality.

## Decisions D-01 Through D-05

| Decision | Result | Evidence |
|---|---:|---|
| D-01 canonical compound identity | Passed | Exact nonblank compound keys, persisted slide identity, complete structured failures, and hostile-input inertness are independently proven. |
| D-02 one-to-one alignment | Passed | Complete shuffled inputs return metadata-order rows with target/provenance pairing and both source ordinals; no subset or multiplication is accepted. |
| D-03 adaptive legal dimensions | Passed | Deterministic two-stage formulas, exact Scanpy arguments, explicit reasons, and structured scientific minima are proven. |
| D-04 visible preprocessing provenance | Passed | AnnData and admitted-order run records contain canonical JSON-safe counts, exclusions, requests, resolutions, and reasons and survive H5AD. |
| D-05 compatibility and boundaries | Passed | Public facades, valid output shapes, existing analysis fields, runner entry point, cohort manifest, configuration keys, notebooks, and lazy exports remain stable. |

## Roadmap Success Criteria

1. **Passed:** null, blank, wrong-type, duplicate, unmatched, wrong-slide, and
   cross-slide identities fail with total counts and bounded representative
   evidence before alignment/indexing or scientific consumer work.
2. **Passed:** independently shuffled complete labels and patch/cache metadata
   align one-to-one in metadata order without row, target, or provenance loss.
3. **Passed:** actual post-QC and selected-HVG counts deterministically resolve
   legal HVG/PCA/neighbor/graph-PC values or raise scientific nonviability.
4. **Passed:** input/stage counts, exclusions, requested/resolved parameters,
   and reason codes remain visible and canonical in AnnData and admitted-order
   run provenance.

## Threat Controls T-01 Through T-05

| Threat | Result | Evidence |
|---|---:|---|
| T-01 hostile identity operations | Passed | Exact type/schema gates and inert diagnostics precede every relevant pandas/Python operation; direct probe recorded zero hooks. |
| T-02 silent join loss/multiplication | Passed | Complete compound-key equality and uniqueness precede one-to-one metadata-left merge. |
| T-03 divergent consumers/cardinality | Passed | Ordinary and foundation paths share the aligner and reject value-row mismatch before indexing, encoding, or writes. |
| T-04 illegal post-QC dimensions | Passed | Conservative two-stage formulas use observed counts and actual HVGs; nonviable states stop before graph/save work. |
| T-05 hidden Scanpy adjustment/provenance | Passed | Resolved arguments are passed explicitly once and the exact canonical execution record is persisted and revalidated. |

## Deep Review Closure

Independent inspection and replay confirm all recorded review findings remain
closed: persisted slide identity (WR-01), impossible count histories (WR-02),
hostile metaclass diagnostics (WR-03), bounded safe key rendering (WR-04),
hostile schema-label admission (WR-05), exact guard-order tests (IN-01), and
duplicate required/reserved column handling (IN-02). No new warning or
informational gap was found.

## Automated and Independent Evidence

| Command or probe | Result |
|---|---:|
| `python -m pytest -q --strict-markers -m offline` over identity, adaptive preprocessing, synthetic AnnData, foundation, empty-boundary, and cohort-admission modules | Passed: 190 tests in 19.13s |
| Adaptive selection covering exact fake Scanpy calls, real capped Scanpy/H5AD, metadata H5AD, and nonviable guard ordering | Passed: 4 tests in 18.23s |
| `python scripts/verify.py fast` | Passed: Ruff + 263 offline tests in 20.29s |
| Independent hostile identity probe | Passed: zero repr/str/hash/equality/order/strip/iteration/metaclass hooks |
| Independent ordinary/foundation shuffled alignment probe | Passed: identical `s2, s1` order, values, targets, provenance, and source ordinals |
| Independent resolver/nonviability/canonical JSON probe | Passed: exact formulas and structured rejection |
| Phase diff compatibility audit from verified Phase 2 baseline | Passed: additive preprocessing manifest; cohort manifest write/schema and public exports/config remain unchanged |

## Warnings

- Verification ran on Python 3.12 while the declared project runtime is Python
  3.11; Phase 10 owns environment reconciliation.
- Pandas warns about old optional `numexpr` and `bottleneck` accelerators.
- Two fake-Scanpy test helpers emit pandas Copy-on-Write chained-assignment
  warnings; production behavior and assertions pass.
- Legacy pharma notebooks still emit missing-cell-ID warnings already owned by
  the existing environment/notebook backlog.
- Network, model-download, executable-notebook, and full-cohort tiers remain
  explicit non-gating evidence by design.

## Human Verification

No human-only item or unresolved gap remains. Every Phase 3 must-have is covered
by deterministic offline evidence, independently replayed adversarial probes,
and the real local Scanpy/H5AD integration path.

---

*Independent verification completed 2026-07-17.*
