# Phase 3: Identity and Adaptive Preprocessing — Codebase Patterns

**Mapped:** 2026-07-17  
**Requirements:** VAL-02, VAL-05  
**Purpose:** Give the planner and executors concrete repository-native analogs for strict compound identity, one-to-one source-order alignment, adaptive preprocessing, and canonical provenance.

## Pattern Summary

Phase 3 should add two narrow contracts rather than distribute new checks among consumers:

1. a pandas-aware identity/alignment module that validates exact compound keys before hashing, comparison, merge, indexing, or array construction; and
2. an import-light adaptive-preprocessing resolver whose immutable result is consumed by `data.preprocess_slide()` and published to AnnData and pipeline provenance.

The closest established style is Phase 2: frozen slotted issue/value objects, one exception carrying structured evidence, exact built-in type gates, deterministic bounded rendering, canonical JSON, and guard-before-side-effect tests. The closest test style is Phase 1: fresh deterministic factory fixtures, offline markers, temporary-path serialization, and real H5AD round trips.

## Closest Analogs

### 1. Structured validation errors

**Source:** `projects/spatial-pharma-dl/src/validation.py`

`ValidationIssue` plus `ConfigValidationError` is the closest aggregate-error model:

```python
@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    received: object
    expected: str
    guidance: str


class ConfigValidationError(PharmaValidationError):
    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        lines = [f"Experiment configuration has {len(self.issues)} issue(s):"]
        lines.extend(... for issue in self.issues)
        super().__init__("\n".join(lines))
```

`StageValidationError` is the closest stage/count/actionability model:

```python
class StageValidationError(PharmaValidationError):
    def __init__(self, *, stage, subject, observed, minimum, guidance,
                 shape=None, message=None):
        self.stage = stage
        self.subject = subject
        self.observed = int(observed)
        self.minimum = int(minimum)
        self.shape = None if shape is None else tuple(int(size) for size in shape)
        ...
```

**Phase 3 application:** define frozen `IdentityIssue` records and an `IdentityValidationError(PharmaValidationError)` containing `issues: tuple[IdentityIssue, ...]`. A preprocessing viability error should likewise expose stable primitive fields such as `stage`, `reason_code`, `counts`, `requested`, and `guidance`. Avoid exception-only prose as the evidence API. Keep `PharmaValidationError` as the common `ValueError` compatibility base.

### 2. Hostile exact-type gates and inert diagnostics

**Source:** `projects/spatial-pharma-dl/src/validation.py`

Phase 2 established that exact built-in admission precedes operations on caller-controlled values:

```python
if type(raw) is not dict:
    issues.add("config", raw, "a non-empty YAML mapping", ...)
    raise ConfigValidationError(issues.values)

valid = type(value) is str and bool(value.strip())
```

Safe rendering never invokes arbitrary `repr`:

```python
def _safe_received(value: object) -> str:
    if value is None or type(value) in (bool, float):
        return repr(value)
    ...
    return f"<{type(value).__name__}>"
```

Invalid-key ordering uses inert exact-type tokens rather than caller comparison:

```python
def _invalid_key_sort_token(value: object) -> tuple[object, ...]:
    value_type = type(value)
    if value is None:
        return (0,)
    if value_type is bool:
        return (1, int(value))
    ...
    return (99, value_type.__module__, value_type.__qualname__)
```

**Tests:** `test_validation.py` proves hostile `repr`, `__lt__`, `__hash__`, `bit_length`, iteration, lookup, and path conversion are not executed.

**Phase 3 application:** inspect each `slide_id` and `spot_id` cell by position and require `type(value) is str` before `strip`, hashing, tuple construction, set membership, sorting, `duplicated`, merge, or index creation. For rejected values, record only side, column, row ordinal, and inert type label. Only after exact-string admission may blank checks, lexicographic ordering, compound-key hashing, and key samples occur. Do not use `astype(str)`, `str(value)`, `repr(value)`, `isinstance(value, str)`, or pandas duplicate/set operations on unvalidated key columns.

### 3. Deterministic issue ordering and canonical JSON

**Source:** `projects/spatial-pharma-dl/src/validation.py`

`CohortManifest` is the canonical value-object analog:

```python
@dataclass(frozen=True, slots=True)
class CohortManifest:
    ...

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {...},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)
```

`ResolvedConfig` uses the same canonical-string/fresh-tree split so callers cannot mutate the validated value. `test_validation.py::test_canonical_json_sorts_mappings_but_preserves_cohort_lists` proves mappings are order-independent while meaningful sequence order remains significant. `test_cohort_admission.py::test_manifest_json_is_deterministic_safe_and_mutation_isolated` proves byte equality, JSON safety, and fresh mutable views.

**Phase 3 application:**

- collect issue categories in a fixed schema order: missing columns, invalid values, blanks, reserved-column collisions, duplicates, wrong-slide rows, label-only, metadata-only, cross-slide, cardinality mismatch;
- within a category, use original row ordinal for invalid-value evidence and lexicographic `(slide_id, spot_id)` order only after exact-string admission;
- cap samples at five while retaining the total count;
- construct preprocessing provenance from fixed field names and admitted slide order;
- canonicalize with the exact Phase 2 `json.dumps` arguments above; and
- return fresh built-in dictionaries at AnnData/pipeline boundaries.

### 4. DataFrame source-order alignment

**Current unsafe seam:** `labels.align_labels_with_patches()` performs:

```python
return labels.merge(meta, on=["slide_id", "spot_id"], how="inner")
```

`train.load_slide_patches()` then reverses the ownership of order and loses duplicates through a dictionary:

```python
order = aligned["spot_id"].tolist()
idx_map = {s: i for i, s in enumerate(meta["spot_id"])}
return patches[[idx_map[s] for s in order]], aligned.reset_index(drop=True)
```

The closest valid ordering evidence is `test_synthetic_anndata.py::test_patch_extraction_and_valid_alignment_preserve_spot_order`, where patch metadata is the left/order-defining table and `validate="one_to_one"` is explicit:

```python
aligned = patch_metadata.merge(
    labels, on=["slide_id", "spot_id"], how="inner", validate="one_to_one"
)
assert aligned["spot_id"].tolist() == expected_spots
```

Patch production itself is positional and stable: `_extract_spot_patches()` iterates coordinates, appends tensors and metadata together, and writes `spot_id = adata.obs_names[i]`. This is the order the strict aligner must preserve after proving identity validity.

**Phase 3 recipe:**

```python
label_rows = labels.copy()
metadata_rows = metadata.copy()
label_rows["_label_source_row"] = range(len(label_rows))
metadata_rows["_patch_source_row"] = range(len(metadata_rows))

aligned = metadata_rows.merge(
    label_rows,
    on=["slide_id", "spot_id"],
    how="left",
    validate="one_to_one",
    sort=False,
    indicator=True,
)
```

This merge is allowed only after full validation proves exact key types, per-side uniqueness, expected-slide membership, set equality, reserved-column absence, and optional value-array cardinality. Treat a pandas merge failure after those checks as an internal invariant failure. Retain `_label_source_row` and `_patch_source_row`; use `_patch_source_row` directly for array indexing. Preserve all label provenance columns as one reordered row, never rebuild target columns independently.

For per-slide consumers, validate the complete label table first, validate every metadata row belongs to `expected_slide_id`, then select exact-slide labels. Other valid cohort slides are not extras, but a metadata spot found only under another label slide must produce a cross-slide issue.

### 5. AnnData metadata and H5AD round-trip tests

**Fixtures:** `tests/conftest.py::synthetic_anndata_factory` builds fresh deterministic integer counts, exact `obs_names`, an `obs["slide_id"]` column, spatial coordinates, and Visium-shaped image metadata.

**Round-trip analogs:**

- `test_artifact_roundtrips.py::test_h5ad_axes_and_spatial_metadata_round_trip`
- `test_synthetic_anndata.py::test_real_anndata_round_trip_preserves_spatial_axes`

Both normalize pandas string storage before writing:

```python
adata.obs_names = pd.Index(adata.obs_names.to_numpy(dtype=object), dtype=object)
adata.var_names = pd.Index(adata.var_names.to_numpy(dtype=object), dtype=object)
adata.obs["slide_id"] = adata.obs["slide_id"].astype(object)
adata.write_h5ad(path)
restored = ad.read_h5ad(path)
```

**Phase 3 application:** add the preprocessing tree under one stable additive key, preferably `adata.uns["spatial_pharma_preprocessing"]`. Test both in-memory equality and H5AD-restored equality. Also assert:

- `json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` succeeds;
- all leaves are exact built-in JSON primitives, not NumPy scalars/arrays, `Path`, timestamps, sets, warnings, or exception strings;
- repeated resolution yields byte-identical canonical JSON; and
- requested values, resolved values, counts, exclusions, and reason codes survive the round trip.

Use `tmp_path` only. Preserve the existing pandas/AnnData compatibility normalization rather than inventing a separate serialization workaround.

### 6. Pipeline provenance publication

**Source:** `projects/spatial-pharma-dl/scripts/run_pipeline.py`

Phase 2 publishes the final manifest only after final admission and preserves configured/admitted order:

```python
all_slides = list(final_admitted.slide_ids)
...
manifest_path = out_dir / "cohort_manifest.json"
manifest_path.write_text(
    final_admitted.manifest.canonical_json,
    encoding="utf-8",
)
```

`test_cohort_admission.py` tracks the two admissions, proves provisional admission is never published, and checks that manifest included order equals the downstream label order.

**Phase 3 application:** do not change `cohort-manifest-v1`. Build an additive `preprocessing_manifest.json` from processed AnnData provenance in `final_admitted.slide_ids` order, after final admission and before labels/models consume the run. Use a frozen value object/canonical JSON just like `CohortManifest`. Tests should monkeypatch slide loading/output paths, assert admitted order, exact document bytes across repeated runs, no publication before all admitted metadata validates, and no label/patch/model stage after provenance assembly failure. Atomicity, checksums, completion markers, and fingerprints remain Phase 4.

### 7. Focused test fixture patterns

**Source:** `tests/conftest.py`

Factories return fresh objects rather than shared mutable fixtures:

```python
@pytest.fixture
def key_adversary_factory(cohort_factory):
    def build():
        valid = cohort_factory()
        ...
        return {"null": ..., "duplicate": ..., ...}
    return build
```

`test_fixture_contracts.py` proves deterministic contents and mutation isolation by mutating one result and constructing another. `test_empty_boundaries.py` uses a `_forbidden` sentinel plus monkeypatching to prove invalid inputs stop before loaders, caches, devices, models, estimators, and writers.

**Phase 3 extensions:** extend, do not replace, `key_adversary_factory` with:

- shuffled complete labels;
- shuffled complete metadata;
- missing `slide_id` and missing `spot_id` columns;
- exact-type violations and hostile `str` subclasses on each side;
- blank and whitespace-only exact strings;
- duplicate metadata keys;
- metadata rows on the wrong slide;
- array/metadata row-count mismatch; and
- collisions with `_label_source_row` and `_patch_source_row`.

Parameterize `synthetic_anndata_factory` or derive fresh copies for tiny/heavily filtered spot and gene counts. Keep deterministic named seeds. Add a fake/call-recording Scanpy module for orchestration tests so pure resolver and guard-order evidence stays in the mandatory offline tier. Patch PCA/neighbors/UMAP/Leiden with forbidden sentinels when post-QC state is nonviable.

## Expected Files by Role

| File | Role | Inputs | Outputs / invariant |
|---|---|---|---|
| `src/identity.py` (new) | Pandas-aware compound-key admission and alignment owner | label DataFrame, metadata DataFrame, expected slide, optional value row count; AnnData + slide ID | structured identity errors or metadata-order one-to-one aligned frame with both source-row columns |
| `src/validation.py` | Import-light shared base errors and pure preprocessing value/error objects/resolvers | exact built-in counts and requested config values | immutable canonical resolution or structured scientific failure; no pandas/Scanpy import |
| `src/labels.py` | Compatibility facade and label source guard | cached AnnData and legacy aligner arguments | current public DataFrame shapes, now validated through `src.identity` |
| `src/patches.py` | Patch source identity guard | AnnData obs identity, positional coordinates/tensors | patch rows and metadata remain positional only after unique canonical identity is proven |
| `src/train.py` | Ordinary CNN/RF consumer boundary | patch array, metadata, cohort labels | shared strict alignment; array indexed by `_patch_source_row`; aligned metadata order |
| `src/foundation.py` | Foundation cache hit/miss consumer boundary | embeddings, cached spot IDs, expected slide, labels | same shared alignment/cardinality contract; no string coercion or subset acceptance |
| `src/data.py` | Adaptive preprocessing orchestrator and AnnData provenance owner | validated config, source AnnData, observed counts/HVG mask | resolved Scanpy arguments, stable existing outputs, additive provenance in `.uns` |
| `scripts/run_pipeline.py` | Admitted-order run provenance publication | `final_admitted.slide_ids`, each processed AnnData provenance | sibling canonical preprocessing manifest; cohort manifest schema unchanged |
| `tests/conftest.py` | Shared fresh Phase 3 adversaries | named deterministic seeds/options | mutation-isolated identity and tiny-QC fixtures |
| `tests/test_identity_alignment.py` (new) | VAL-02 unit/integration evidence | adversarial tables, arrays, consumers | exact-type safety, deterministic issues, completeness, source order, common consumer contract |
| `tests/test_adaptive_preprocessing.py` (new) | VAL-05 unit/integration evidence | pure count matrices, fake Scanpy, AnnData/tmp paths | caps/minima, call arguments/order, canonical metadata, H5AD and run-manifest stability |
| existing regression tests | Compatibility seams | ordinary valid configs/data | Phase 1/2 behavior and valid public outputs remain unchanged |

## Data Flow to Preserve

### Identity path

```text
AnnData obs identity
  -> validate exact (slide_id, obs_name)
  -> patch tensors + patch metadata in the same positional order
  -> validate array/meta cardinality and both complete key tables
  -> metadata-left one-to-one alignment
  -> use _patch_source_row to index tensor/embedding array
  -> return labels + all provenance in metadata order
```

Both ordinary patch loading and foundation cache reuse must enter at the same strict alignment boundary. Cache misses may continue through ordinary patch loading; cache hits synthesize exact metadata from the requested slide plus cached spot IDs and pass that metadata to the same aligner.

### Adaptive preprocessing path

```text
resolve explicit config
  -> validate/copy source AnnData identity
  -> record input counts
  -> existing cell/gene/mito filters with counts after each stage
  -> resolve legal HVG request and reject gross nonviability
  -> existing normalize/log/HVG call
  -> count actual HVG mask
  -> finalize PCA/neighbors/neighbor-PC dimensions
  -> existing scale/PCA/neighbors/UMAP/Leiden calls using resolved values
  -> attach canonical primitive provenance to AnnData
  -> save H5AD
  -> assemble admitted-order preprocessing manifest in pipeline
```

The resolver should make acceptance versus capping explicit with stable reason codes. Do not catch Scanpy dimension errors and retry, change solver/algorithm, infer resolved values from `X_pca`, or silently skip graph stages.

## Test and Review Checklist

- [ ] Exact-type validation runs before pandas null/duplicate/merge/index operations on identity columns.
- [ ] Hostile values cannot trigger representation, hashing, comparison, coercion, iteration, or string methods.
- [ ] Every failure category has a total count and deterministic bounded sample.
- [ ] Shuffled complete inputs succeed in metadata/array order and retain both source-row ordinals.
- [ ] Missing, extra, duplicate, wrong-slide, cross-slide, or cardinality-mismatched inputs fail before array/model/cache effects.
- [ ] Ordinary and foundation cache paths call the shared contract.
- [ ] Pure preprocessing resolution covers accepted values, each cap reason, and every scientific minimum.
- [ ] Invalid post-QC state stops before PCA, neighbors, UMAP, and Leiden.
- [ ] AnnData metadata is exact-built-in JSON, canonical, mutation-isolated, and H5AD-stable.
- [ ] Run provenance is assembled in final admitted order without changing `cohort-manifest-v1`.
- [ ] Focused Phase 3 tests stay offline and the full `python scripts/verify.py fast` gate remains green.

## Explicit Non-Patterns / Scope Boundaries

- Do not repair keys by coercion, filling, trimming, dropping duplicates, or inner joining.
- Do not put pandas or Scanpy imports into import-light `src.validation` module scope.
- Do not migrate `load_patch_arrays(... allow_pickle=True)` in this phase; Phase 5 owns serialization safety.
- Do not add fingerprints, checksums, atomic publication, or completion markers; Phase 4 owns durability.
- Do not add fold class support, leakage/model-selection policy, image-science changes, or label-confidence semantics.
- Do not rename public functions, config keys, CLI entry points, notebooks, outputs, or existing AnnData analysis keys.

---

*Pattern map for Phase 03 planning; no production code changed.*
