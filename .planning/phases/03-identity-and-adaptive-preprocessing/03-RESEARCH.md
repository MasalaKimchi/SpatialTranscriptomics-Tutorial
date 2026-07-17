# Phase 3: Identity and Adaptive Preprocessing - Research

**Researched:** 2026-07-17  
**Requirements:** VAL-02, VAL-05  
**Scope:** Exact spot identity, complete one-to-one label/patch alignment, legal
post-QC dimensions, and visible deterministic preprocessing provenance

## Executive Summary

Phase 3 should be implemented as two related but independently testable
contracts:

1. a shared compound-identity/alignment boundary used by the ordinary patch
   loader and the foundation-embedding cache path; and
2. a pure two-stage preprocessing resolver used by `data.preprocess_slide()`
   after QC and again after HVG selection.

The current ordinary path is unsafe because
`labels.align_labels_with_patches()` performs an unconstrained inner merge and
`train.load_slide_patches()` builds a last-write-wins `spot_id` dictionary. The
foundation-cache path independently casts IDs to strings, accepts subset
matches, and indexes labels by `spot_id` alone. Fixing only the inner merge
would therefore leave a second production path that can silently lose,
duplicate, coerce, or cross-align rows.

The preprocessing defect is concentrated in `data.preprocess_slide()`. It
applies fixed configured HVG, PCA, neighbor, and neighbor-PC dimensions after
three sequential filters without first checking retained cardinalities. The
safe design is not a catch-and-retry around Scanpy. It is deterministic
resolution from observed counts before each expensive call, followed by
verification and JSON-safe provenance.

No new dependency is needed. Reuse Phase 2's `PharmaValidationError` hierarchy,
structured attributes, exact-type discipline, canonical JSON encoding, and
guard-before-side-effect tests. Keep artifact fingerprints, atomic publication,
pickle migration, fold support, leakage policy, image QC, and label confidence
out of this phase.

## Requirement Interpretation

### VAL-02 — exact and complete compound identity

The canonical key is the exact `(slide_id, spot_id)` pair. For this repository,
both components should be exact built-in, non-blank strings. Do not call
`astype(str)`, `str(value)`, `fillna`, `drop_duplicates`, index inference, or a
repairing join. Reject:

- missing required key columns;
- null values in either component;
- non-string values and string subclasses before hashing/comparison;
- empty or whitespace-only strings without trimming accepted values;
- duplicate compound keys on either side;
- label-only and patch-only compound keys;
- the cross-slide form where an unmatched `spot_id` exists on the opposite
  side under a different `slide_id`;
- metadata rows whose `slide_id` differs from the requested slide; and
- patch/embedding array row counts that differ from metadata row counts.

Diagnostics need category counts and a bounded deterministic sample (five is
sufficient) of safe validated keys. Invalid arbitrary objects must be reported
by inert type labels, never by caller-controlled `repr`, hashing, sorting, or
string conversion. Once components are known exact strings, key samples can be
sorted lexicographically and serialized safely.

Successful alignment must preserve patch/cache row order. Labels and all label
provenance columns are reordered together to that order. The result should add
reserved source-row columns (for example `_label_source_row` and
`_patch_source_row`) so the mapping is inspectable. Reject an input collision
with those reserved names rather than overwriting caller data.

### VAL-05 — legal dimensions after actual QC

Configuration validation already proves that requested dimensions are positive
integers and `n_pcs_neighbors <= n_pcs`. Phase 3 must validate them against the
actual retained slide, not duplicate the startup schema work.

Record sequential counts around existing operations:

- input spots and genes;
- spots after `filter_cells(min_counts=...)`;
- genes after `filter_genes(min_cells=...)`;
- spots after mitochondrial filtering;
- selected HVGs; and
- final PCA components and graph dimensions.

The legal deterministic caps are:

```text
hvg_for_call       = min(requested_hvg, post_qc_genes)
pca_rank_limit     = min(post_qc_spots, actual_hvgs) - 1
resolved_pcs       = min(requested_pcs, pca_rank_limit)
neighbor_limit     = post_qc_spots - 1
resolved_neighbors = min(requested_neighbors, neighbor_limit)
resolved_graph_pcs = min(requested_graph_pcs, resolved_pcs)
```

The strict `- 1` PCA rank limit keeps the request legal for Scanpy's common
ARPACK path rather than depending on solver-specific behavior. A slide is
scientifically nonviable for the existing PCA-neighbor-UMAP-Leiden workflow if
it has fewer than three post-QC spots, fewer than two post-QC genes/HVGs, no
legal PCA component, or fewer than two legal neighbors. Fail with stage,
counts, requested values, a stable reason code, and remediation before PCA,
neighbors, UMAP, or Leiden. Do not silently switch algorithms or skip stages.

Because Scanpy determines the actual HVG mask, resolution should occur twice:

1. after QC, resolve the HVG request and reject grossly nonviable spot/gene
   counts before normalization/HVG-dependent work where possible; then
2. after `highly_variable_genes`, count the actual boolean mask and resolve PCA,
   neighbors, and neighbor PCs from that observed count.

Each accepted request or cap gets an explicit reason code, for example
`requested_value_accepted`, `hvg_capped_to_post_qc_genes`,
`pca_capped_to_rank_limit`, `neighbors_capped_to_spot_limit`, or
`neighbor_pcs_capped_to_resolved_pcs`. Never infer provenance later from the
shape of `X_pca`.

## Current Code and Exact Integration Seams

### Identity and alignment

| Seam | Current behavior | Required change |
|---|---|---|
| `src/labels.py::align_labels_with_patches` | Unconstrained inner merge on two columns; unmatched rows disappear and duplicate keys can multiply rows. | Make this the compatibility facade over one strict shared aligner; preserve its DataFrame return type. |
| `src/train.py::load_slide_patches` | Filters labels first, accepts inner-join output, builds `{spot_id: index}` so duplicates overwrite, and returns labels in merge/label order. | Validate full relevant identity, array/metadata cardinality, and expected slide; align in metadata order; index patches using recorded patch source rows rather than a lossy dictionary. |
| `src/foundation.py::load_or_extract_slide_embeddings` cache branch | Casts cached and label IDs to strings, checks only label `spot_id` uniqueness, accepts cached subsets, and aligns by `spot_id` without `slide_id`. | Synthesize cache metadata with requested `slide_id` and cached spot IDs, verify embedding row count, then call the same strict aligner. No coercion or subset acceptance. |
| `src/patches.py::_extract_spot_patches` | Derives `spot_id` positionally from `adata.obs_names` but never validates canonical/unique source identity. | Validate `(slide_id, obs_name)` before constructing patches/metadata; keep positional patch extraction only after uniqueness is proven. |
| `src/labels.py::build_labels_for_slide` | Emits identities from `sample_id` and `adata.obs_names` without source validation. | Reuse the AnnData identity guard before label row construction. |
| `src/patches.py::load_patch_arrays` | Returns arrays and metadata without row-cardinality validation; unsafe pickle loading remains. | Phase 3 validates returned shapes/identity immediately at the consuming alignment boundary. Do not migrate serialization here; Phase 5 owns that. |

The neutral contract can live in a small pandas-aware module such as
`src/identity.py`. Keep `src/validation.py` import-light and standard-library
only; importing pandas there would undo the Phase 2 startup boundary. Suggested
surface:

```python
class IdentityValidationError(PharmaValidationError):
    stage: str
    issues: tuple[IdentityIssue, ...]

@dataclass(frozen=True, slots=True)
class IdentityIssue:
    code: str
    side: str
    count: int
    sample_keys: tuple[tuple[str, str], ...]

def validate_anndata_spot_identity(adata, slide_id: str, *, stage: str) -> None: ...

def align_labels_with_metadata(
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    stage: str,
    expected_slide_id: str | None = None,
    value_row_count: int | None = None,
) -> pd.DataFrame: ...
```

The aligner should validate both complete tables before calling pandas merge.
Then add source-row ordinals, use metadata as the left/order-defining side, and
perform a left merge with `validate="one_to_one"`, `sort=False`, and an
indicator as a defensive assertion. Since set equality and uniqueness were
already proven, any pandas merge validation failure is an invariant error, not
a user repair policy. Drop only the indicator; retain canonical key and source
row provenance columns.

For a per-slide call against a cohort label table, `expected_slide_id` defines
the intended label subset while still allowing cross-slide diagnosis: validate
the full label table first, validate all metadata rows belong to the expected
slide, select exact-slide labels, and classify missing patch IDs that occur in
other slide rows as cross-slide mismatches. Other correctly labeled cohort
slides are not treated as extras.

### Adaptive preprocessing

`src/data.py::preprocess_slide` is the sole production owner of the current
fixed sequence:

```text
QC metrics -> cell filter -> gene filter -> mito filter -> normalize/log
-> HVG -> scale -> PCA -> neighbors -> UMAP -> Leiden
```

Keep that order and public signature. Add a pure resolver in `src/validation.py`
or a new import-light `src/preprocessing.py` that does not import Scanpy at
module import time. A useful value object is:

```python
@dataclass(frozen=True, slots=True)
class PreprocessingResolution:
    schema_version: str
    counts: Mapping[str, int]
    exclusions: Mapping[str, int]
    requested: Mapping[str, int | float]
    resolved: Mapping[str, int | float]
    reasons: Mapping[str, str]
    canonical_json: str
```

Prefer immutable tuples/canonical JSON internally and fresh plain dictionaries
at AnnData boundaries, following `ResolvedConfig`. Keep the calculation pure so
tests can prove every cap and failure without importing Scanpy.

`preprocess_slide()` should:

1. resolve an explicit config before importing or calling Scanpy;
2. copy the input and validate source compound identity;
3. capture input counts and counts immediately after each existing filter;
4. call a post-QC resolver before normalization/PCA graph work;
5. run HVG selection using the resolved HVG call size;
6. count the actual HVG mask and finalize PCA/neighbor dimensions;
7. call PCA and neighbors only with finalized values;
8. write canonical metadata to a stable key such as
   `adata.uns["spatial_pharma_preprocessing"]`; and
9. preserve current `X_pca`, `pca`, `clusters`, raw/count layers, and return
   behavior for ordinary viable slides.

AnnData metadata should contain only built-in JSON primitives. Avoid NumPy
scalars, arrays, timestamps, filesystem paths, warnings, exception strings, or
unordered sets. The canonical form should use Phase 2's established encoding:

```python
json.dumps(value, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False)
```

Recommended schema:

```json
{
  "schema_version": "spatial-pharma-preprocessing-v1",
  "slide_id": "slide_a",
  "counts": {
    "input_spots": 12,
    "input_genes": 10,
    "post_cell_filter_spots": 11,
    "post_gene_filter_genes": 9,
    "post_mito_filter_spots": 10,
    "selected_hvgs": 9
  },
  "exclusions": {
    "low_count_spots": 1,
    "low_support_genes": 1,
    "high_mito_spots": 1
  },
  "requested": {
    "n_top_genes_hvg": 2000,
    "n_pcs": 50,
    "n_neighbors": 15,
    "n_pcs_neighbors": 30
  },
  "resolved": {
    "n_top_genes_hvg": 9,
    "n_pcs": 8,
    "n_neighbors": 9,
    "n_pcs_neighbors": 8
  },
  "reasons": {
    "n_top_genes_hvg": "hvg_capped_to_post_qc_genes",
    "n_pcs": "pca_capped_to_rank_limit",
    "n_neighbors": "neighbors_capped_to_spot_limit",
    "n_pcs_neighbors": "neighbor_pcs_capped_to_resolved_pcs"
  }
}
```

For run-level visibility, do not mutate Phase 2's validated
`cohort-manifest-v1` schema. Add a sibling, additive preprocessing provenance
document assembled in admitted-slide order from each cached AnnData's metadata
and publish it next to the cohort manifest (or embed the identical primitive
tree in the existing experiment summary). A dedicated
`preprocessing_manifest.json` written immediately after final admission is the
cleaner seam because normal and train-only runs can both reconstruct it before
labels/models. Atomic writes, checksums, and fingerprints remain Phase 4.

## Reusable Patterns and Constraints

- `src.validation.PharmaValidationError` is already a `ValueError`; new identity
  and preprocessing errors should subclass it so compatibility is preserved.
- `ValidationIssue`, `StageValidationError`, `ResolvedConfig`, and
  `CohortManifest` demonstrate structured fields plus actionable prose and
  canonical JSON.
- Phase 2 hostile-value tests establish that arbitrary caller objects must not
  be hashed, compared, sorted, converted, or rendered before exact-type gates.
- `tests/conftest.py::key_adversary_factory` already supplies null, duplicate,
  unmatched-label, unmatched-patch, and cross-slide frames. Extend it with
  shuffled-complete, missing-column, wrong-type, blank-string, duplicate-patch,
  and reserved-provenance-column variants rather than creating a second fixture
  convention.
- `synthetic_anndata_factory` already supplies deterministic integer counts,
  compound identity, spatial coordinates, and fresh mutation-isolated AnnData.
  Add small/heavily-filtered variants or parameters for dimension tests.
- `scripts/verify.py fast` and strict `offline` markers remain the canonical
  gate. Tests must not download Visium data or model weights.
- Scanpy is declared in tutorial dependencies but may not exist in every
  lightweight developer interpreter. Most dimension evidence therefore belongs
  in pure resolver tests; one optional-dependency integration test can use
  `pytest.importorskip("scanpy")` only if the required fast CI environment is
  documented to install Scanpy. A fake-Scanpy call recorder can still prove
  resolved arguments and guard ordering in the minimal environment.

## Recommended Plan Split

### Plan 03-01 — Compound identity and shared alignment

Implement structured identity errors and the shared strict aligner; validate
AnnData source IDs; route ordinary patch loading and foundation-cache reuse
through it; preserve patch/cache row order and source-row provenance. Add all
key adversaries, shuffled success, shape mismatch, and forbidden-side-effect
tests.

### Plan 03-02 — Adaptive preprocessing and provenance

Implement pure post-QC/two-stage dimension resolution, integrate it into
`preprocess_slide`, record counts/exclusions/requested/resolved/reasons in
AnnData, and expose admitted-order run provenance without changing existing
manifest schemas or public outputs. Add resolver matrices, fake-Scanpy ordering,
AnnData metadata, H5AD round-trip, and canonical-byte tests.

### Plan 03-03 — Cross-arm integration and regression closure (if checker keeps it separate)

Exercise ordinary CNN/RF loading, foundation cache hit/miss, labels, and patch
metadata through the same contract; verify normal valid rows and ordinary
preprocessing outputs remain compatible; run the full fast gate. This may be
folded into Plans 03-01 and 03-02 if their task size remains reviewable.

### Concrete files and verification commands

Expected production edits are
`projects/spatial-pharma-dl/src/identity.py` (new),
`projects/spatial-pharma-dl/src/validation.py`, `src/labels.py`, `src/train.py`,
`src/foundation.py`, `src/patches.py`, `src/data.py`, and
`projects/spatial-pharma-dl/scripts/run_pipeline.py`. Expected test edits are
`projects/spatial-pharma-dl/tests/conftest.py` plus focused new modules
`test_identity_alignment.py` and `test_adaptive_preprocessing.py`; existing
`test_synthetic_anndata.py`, `test_foundation.py`, `test_empty_boundaries.py`,
and `test_cohort_admission.py` are the regression seams most likely to need
additive assertions.

Run focused evidence first, then the canonical gate:

```bash
python -m pytest -q --strict-markers -m offline \
  projects/spatial-pharma-dl/tests/test_identity_alignment.py \
  projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py

python -m pytest -q --strict-markers -m offline \
  projects/spatial-pharma-dl/tests/test_synthetic_anndata.py \
  projects/spatial-pharma-dl/tests/test_foundation.py \
  projects/spatial-pharma-dl/tests/test_empty_boundaries.py \
  projects/spatial-pharma-dl/tests/test_cohort_admission.py

python scripts/verify.py fast
```

## Validation Architecture

### Evidence layers

| Layer | Evidence | Purpose |
|---|---|---|
| Pure identity unit | DataFrames only, no cache/model | Prove exact key types, null/blank/missing columns, uniqueness, set equality, cross-slide classification, bounded deterministic diagnostics, and metadata-order alignment. |
| Consumer boundary | Monkeypatched `load_patch_arrays` and foundation cache | Prove failures occur before patch indexing, encoder loading, model execution, and output/cache writes; prove both arms use the same aligner. |
| Pure dimension unit | Integer counts/requested mapping only | Exhaustively prove legal caps, minima, rank limit, reason codes, canonical JSON, and deterministic replay without Scanpy. |
| Preprocessing orchestration | Synthetic AnnData plus fake/call-recording Scanpy | Prove counts are captured after the correct filters and finalized values are passed to HVG/PCA/neighbors; impossible dimensions stop before later calls. |
| Scientific integration | Real tiny AnnData with installed Scanpy | Prove a viable heavily capped slide completes, actual PCA/graph shapes match recorded values, and metadata survives H5AD round-trip. |
| Pipeline provenance | Stubbed runner stages and temporary outputs | Prove admitted slide order is retained, normal/train-only paths expose the same per-slide metadata, and no Phase 2 manifest schema is changed. |
| Regression gate | Ruff plus all strict offline tests | Protect notebook/CLI/config/public imports, existing output names, valid fold order, and all Phase 1/2 guarantees. |

### Mandatory VAL-02 cases

1. Complete labels shuffled independently from patch metadata produce exactly
   patch-order rows, unchanged target/provenance pairing, unique keys, and
   inspectable source-row ordinals.
2. Null, blank, wrong-type, and missing key columns fail before merge, set,
   dictionary, array indexing, cache write, or model/encoder work.
3. Duplicate label and duplicate metadata compound keys report side, total
   count, and stable sample; they never multiply rows or overwrite an index.
4. Label-only and patch-only keys report both counts and samples in one failure;
   no inner-join subset is returned.
5. A spot present under the wrong slide reports a cross-slide reason and both
   compound identities; exact spot strings repeated legitimately on different
   slides do not fail when both compound key sets are complete.
6. Metadata/patch or cache/embedding row mismatch fails before indexing.
7. Foundation cache hit, foundation cache miss, CNN, and RF paths return the
   same aligned key order for equivalent inputs; no path calls `astype(str)`.
8. Reversed input mapping/row construction for the same defects yields identical
   structured issue ordering and exception text after deterministic sorting.
9. Hostile string subclasses/arbitrary objects with raising `__repr__`,
   `__str__`, `__hash__`, `__eq__`, and comparison methods remain unexecuted.

### Mandatory VAL-05 cases

1. Requested values below every limit pass through unchanged with
   `requested_value_accepted` reasons.
2. Oversized HVG, PCA, neighbors, and graph-PC requests independently and
   jointly cap to the formulas above with the expected reason codes.
3. Actual HVG count lower than the call request is used for the final PCA rank
   calculation and recorded; the code never trusts only the requested count.
4. Zero/one/two post-QC spots, zero/one genes, zero/one actual HVGs, and any
   state with no legal PCA or neighbor dimension raise a structured
   preprocessing error before PCA/neighbors/UMAP/Leiden.
5. Each sequential QC exclusion count is nonnegative, totals reconcile with
   observed AnnData dimensions, and empty-after-filter errors identify the
   responsible stage and threshold.
6. The exact resolved dimensions are the values passed to
   `highly_variable_genes`, `pca`, and `neighbors`; call-recorder tests forbid
   fallback retries with different implicit values.
7. AnnData metadata and admitted-order run provenance contain the same primitive
   requested/resolved/count/reason content; semantic repeats have byte-identical
   canonical JSON.
8. H5AD round-trip retains the metadata and ordinary successful slides still
   expose `layers["counts"]`, `raw`, `obsm["X_pca"]`, `uns["pca"]`, and
   `obs["clusters"]`.
9. Valid default-scale production inputs retain their configured values and
   ordinary clustering behavior; only impossible dimensions are capped or
   rejected.

### Guard ordering

Tests should monkeypatch later seams to raise `AssertionError` and confirm they
remain unreachable:

```text
bad identity -> no pandas merge -> no NumPy indexing -> no dataset/model/encoder
nonviable post-QC -> no PCA -> no neighbors -> no UMAP -> no Leiden -> no save
bad explicit config -> no Scanpy import/call -> no AnnData mutation/output
```

The first arrow requires implementation-level injection or monkeypatching of
`pd.merge`; the latter arrows can use a fake Scanpy namespace with per-call
sentinels. Keep all paths under `tmp_path` and mark every new test `offline`.

### Requirement-to-evidence matrix

| Requirement / decision | Production seam | Required proof |
|---|---|---|
| VAL-02 / D-01 | `identity.validate_*` | Exact non-null/nonblank built-in string compound keys; deterministic counts/samples; no hostile method execution. |
| VAL-02 / D-02 | `labels.align_labels_with_patches`, `train.load_slide_patches` | Complete one-to-one set equality, metadata-order output, source-row provenance, and no last-write-wins index. |
| VAL-02 cross-arm | `foundation.load_or_extract_slide_embeddings` | Cache hit/miss uses the same compound contract with no coercion or subset reuse. |
| VAL-05 / D-03 | `data.preprocess_slide`, pure dimension resolver | Post-QC/actual-HVG counts determine legal call values or a scientific failure before expensive graph work. |
| VAL-05 / D-04 | AnnData `uns` plus sibling run provenance | Requested/resolved/count/exclusion/reason content is JSON-safe, canonical, H5AD-stable, and admitted-order stable. |
| D-05 compatibility | current public facades and fast suite | Signatures, imports, CLI flags, notebook order, config keys, output names, valid DataFrame targets, and ordinary AnnData fields remain stable. |

## Risks and Planning Warnings

- **Fixing only `labels.py` is incomplete.** The foundation cache currently has
  a separate permissive alignment implementation and must be routed through the
  shared contract.
- **Validating only `spot_id` is incorrect.** Spot barcodes may repeat between
  slides; every uniqueness, equality, index, and diagnostic operation must use
  `(slide_id, spot_id)`.
- **Filtering the cohort label table too early hides cross-slide defects.** Keep
  enough full-table evidence to distinguish ordinary other-slide rows from a
  requested slide's mislabeled spot.
- **Pandas merge validation is not input validation by itself.** Nulls, hostile
  values, and mismatch diagnostics must be handled before pandas hashing and
  comparison.
- **Scanpy may choose solver-specific behavior or emit warnings for tiny data.**
  Resolve to the conservative ARPACK rank limit and define scientific minima
  explicitly; never rely on Scanpy's implicit neighbor adjustment.
- **Actual HVGs are an observed quantity.** Do not calculate PCA only from the
  requested/resolved HVG call size without checking the mask produced.
- **AnnData `uns` can hold non-JSON objects, but this contract must not.** Cast
  all NumPy integers/booleans to built-ins before canonicalization and test with
  `allow_nan=False`.
- **Do not extend `cohort-manifest-v1` in place.** Its exact canonical behavior
  was independently verified in Phase 2. Use additive sibling provenance.
- **Do not repair unsafe patch serialization here.** Loading the current cache
  remains a Phase 5 concern; Phase 3 may validate its returned rows but must not
  create an undocumented legacy migration.

## Out-of-Scope Checks for Review

Reject Phase 3 diffs that introduce cache fingerprints/atomic writers,
`allow_pickle=False` patch migration, checkpoint loading, fold class support,
train-only scalers/imputers, inner CNN selection, stain reference changes,
border padding/image quality, heuristic-label confidence, dependency locks, or
notebook redesign. Those changes belong to Phases 4-10 and would make Phase 3's
scientific evidence harder to isolate.

## Planning Recommendation

Plan identity first because downstream model and foundation tests can then use
one trusted ordering contract. Plan adaptive preprocessing second because its
resolver and provenance are independent of patch cache format. Require a final
cross-arm regression task before verification. The phase is complete only when
all production alignment paths share the strict compound-key contract and all
PCA/neighbor calls are demonstrably derived from recorded post-QC/actual-HVG
counts.

---

*Phase: 03-identity-and-adaptive-preprocessing*  
*Research complete: implementation seams, risks, and validation architecture
mapped against current source and Phase 1/2 contracts.*
