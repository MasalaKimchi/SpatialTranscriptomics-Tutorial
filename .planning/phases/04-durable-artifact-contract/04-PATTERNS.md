# Phase 4: Durable Artifact Contract — Codebase Patterns

**Mapped:** 2026-07-17  
**Requirements:** ART-03, ART-04  
**Purpose:** Give the planner and executors repository-native implementation and test patterns for deterministic artifact identity, strict reuse admission, production-reader validation, and interruption-safe publication.

## Pattern Summary

Phase 4 should introduce one generic, import-light contract and small format/domain adapters. The contract decides whether bytes are eligible to reach a decoder; the adapter decides whether decoded content is scientifically and structurally valid. Neither layer may be optional at a supported reuse boundary.

The generic module should be reachable from both architectural surfaces. `utils/` is the only package installed by `pyproject.toml`, root notebooks already import it, and the nested pharma package already imports `utils`. Therefore the closest low-friction placement is `utils/artifacts.py` (or a similarly named module under `utils/`), not a pharma-only module that would force root `st_helpers.py` to mutate `sys.path`. Pharma-specific projection builders and payload adapters remain in `projects/spatial-pharma-dl/src/`.

The intended data flow is:

```text
exact admitted config/source/upstream values
        -> kind-specific fingerprint projection
        -> canonical strict JSON -> SHA-256 expected fingerprint
        -> generic manifest + regular-file + checksum admission
        -> kind-specific production reader and semantic validation
        -> decoded payload returned to scientific/model code

writer callback -> same-directory temporary payload -> fsync
        -> same production reader validates temporary payload
        -> canonical completed temporary sidecar -> fsync
        -> os.replace(payload) -> directory fsync
        -> os.replace(sidecar) last -> directory fsync
```

There is no close existing atomic-publication implementation in the repository. The immutable/canonical admission patterns from Phases 2–3 and the `tmp_path` artifact fixtures from Phase 1 are the analogs to reuse; `tempfile`, `os.fsync`, and `os.replace` are deliberately new infrastructure.

## Closest Analogs

### 1. Exact hostile-safe primitive admission

**Source:** `projects/spatial-pharma-dl/src/validation.py`

`_admit_manifest_json()` is the closest recursive primitive gate. It checks the exact type before invoking primitive methods and sorts keys only after every key is known to be an exact string:

```python
def _admit_manifest_json(value: object, path: str) -> object:
    value_type = type(value)
    if value is None or value_type in (str, bool):
        return value
    if value_type is int:
        if value.bit_length() > 4096:
            raise _manifest_error(...)
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _manifest_error(...)
        return value
    if value_type is list:
        return [_admit_manifest_json(item, ...) for ...]
    if value_type is dict:
        keys = list(value.keys())
        if any(type(key) is not str for key in keys):
            raise _manifest_error(...)
        return {
            key: _admit_manifest_json(value[key], ...)
            for key in sorted(keys)
        }
    raise _manifest_error(...)
```

`resolve_config()` and the identity module establish the same rule at other boundaries: exact built-in admission must occur before `strip`, hashing, equality, ordering, rendering, pandas lookup, or conversion. Tests in `test_validation.py` already provide hostile subclasses for `dict`, `Mapping`, `int`, `float`, `str`, `list`, `tuple`, and `Path`; tests in `test_identity_alignment.py` add hostile metaclass, equality, hash, repr, and schema-label cases.

**Phase 4 application:** reuse the rule, not this helper verbatim. Artifact manifests need stricter budgets and parser behavior:

- cap sidecar byte length before UTF-8 decode or JSON parse;
- parse with `object_pairs_hook` so duplicate keys are rejected rather than silently overwritten;
- cap nesting depth, total nodes, mapping width, list length, string length, and integer bit length during admission;
- require exact root/key/value types and exact key sets in fixed schema order;
- reject non-finite numbers and unsupported schema/kind/contract/algorithm values;
- validate digests as exact built-in strings with lowercase 64-character hexadecimal content;
- accept only a basename equal to the expected payload basename; and
- never include raw JSON values, raw exception text, archive members, or absolute paths in diagnostics.

Do not use `deepcopy`, generic `Mapping`, dataclass coercion, `str(value)`, `repr(value)`, `Path(value)`, or `json.dumps(value)` before admission. The Phase 3 review specifically found that even class metadata (`type(value).__module__`) and membership against hostile schema labels can invoke caller hooks. Use fixed inert categories such as `non_builtin_object` if type evidence is needed.

### 2. Immutable canonical value objects with fresh views

**Sources:** `projects/spatial-pharma-dl/src/validation.py`

`ResolvedConfig`, `PreprocessingResolution`, and `PreprocessingManifest` are the closest storage pattern:

```python
@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    canonical_json: str

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise TypeError(...)
        return value
```

```python
@dataclass(frozen=True, slots=True, init=False)
class PreprocessingManifest:
    canonical_json: str

    def __init__(self, *, slide_ids: object, records: object) -> None:
        admitted_ids = _admit_manifest_json(slide_ids, "slide_ids")
        admitted_records = _admit_manifest_json(records, "records")
        ...
        object.__setattr__(self, "canonical_json", _canonical_json(value))
```

**Phase 4 application:** store an admitted artifact manifest and fingerprint inputs as canonical JSON (or frozen tuples plus canonical JSON), never as a caller-owned mutable dictionary. `to_dict()` returns a fresh exact-built-in tree. Suggested generic records:

```python
@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    algorithm: str
    digest: str
    canonical_inputs_json: str

@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    canonical_json: str
    artifact_kind: str
    contract_version: str
    fingerprint: ArtifactFingerprint
    payload_sha256: str
    payload_byte_count: int
    payload_schema_json: str

@dataclass(frozen=True, slots=True)
class ArtifactAdmission:
    payload_path: Path
    manifest: ArtifactManifest
```

The `Path` in an internal admission record is locally resolved trusted state, not serialized manifest content. Public diagnostics and canonical JSON use only the basename.

### 3. Canonical JSON and deterministic SHA inputs

**Source:** `projects/spatial-pharma-dl/src/validation.py`

The established canonical serializer is:

```python
def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
```

`test_validation.py::test_canonical_json_sorts_mappings_but_preserves_cohort_lists` proves mapping insertion order does not change bytes while meaningful list order does. `test_adaptive_preprocessing.py` proves repeated resolution is byte-identical and mutation-isolated.

**Phase 4 application:** compute a fingerprint only from the canonical UTF-8 bytes of an explicit per-kind input object:

```python
canonical = canonical_json(admitted_inputs).encode("utf-8")
digest = hashlib.sha256(canonical).hexdigest()
```

Use explicit allowlisted projection functions. Do not hash the whole config and then remove fields. Do not hash Git state, timestamps, output paths, checkout paths, plotting options, console text, dictionary insertion order, or presentation formatting.

Recommended projection ownership:

| Artifact kind | Relevant fingerprint inputs | Explicitly irrelevant examples |
|---|---|---|
| root tutorial H5AD | fixed stage contract, dataset semantic identity, upstream stage fingerprint | plotting choices, output formatting |
| processed pharma H5AD | processed-slide contract, sample ID, `preprocessing`, seed, source semantic/content identity | labels, patches, models, reports |
| label/domain table | label contract, `labels`, marker/module panels, processed-slide fingerprint | patch/training/foundation settings |
| patch NPZ/index | patch contract, `patches`, processed-slide fingerprint, stain-reference fingerprint | epochs, model, reports |
| foundation embedding NPZ | embedding contract, model spec/source identity, patch fingerprint, spot identity | device, cache toggle, report style |
| checkpoint | checkpoint contract, model/training target schema, fold lineage, seed policy, patch/label fingerprints | report formatting |
| benchmark/report/summary | report contract, experiment/evaluation projection, exact upstream model/label/fold lineage | output directory, timestamp, plot style |
| cohort/preprocessing JSON | wrapper contract and exact canonical inner payload/upstream lineage | indentation and host path |

Each kind needs a visible `*_CONTRACT_VERSION` constant. Later Phase 5/7/8/9 semantic changes bump the affected version rather than relying on an incidental source hash.

### 4. Structured errors and bounded inert diagnostics

**Sources:** `projects/spatial-pharma-dl/src/validation.py`, `projects/spatial-pharma-dl/src/identity.py`

The repository standard is one typed error with frozen primitive evidence:

```python
@dataclass(frozen=True, slots=True)
class IdentityIssue:
    code: str
    side: str
    count: int
    sample_keys: tuple[tuple[str, str], ...] = ()
    sample_rows: tuple[tuple[int, str, str], ...] = ()
```

Identity rendering bounds and JSON-escapes admitted strings before adding them to messages. Stage/config errors retain machine-assertable fields and actionable guidance.

**Phase 4 application:** define `ArtifactValidationError` as a `ValueError`-compatible typed failure with stable fields such as `artifact_kind`, bounded trusted basename, `reason_code`, and fixed guidance. Reason codes should distinguish at least:

```text
missing_manifest, legacy_artifact, malformed_manifest, unsupported_manifest,
incomplete_manifest, stale_fingerprint, missing_payload, invalid_payload_file,
byte_count_mismatch, checksum_mismatch, unstable_payload, payload_schema_mismatch,
reader_validation_failed, publication_failed
```

Messages should be assembled from trusted enums and a basename that has already passed the exact-string/basename contract. Never interpolate JSON values, parser exceptions, absolute paths, or a decoder exception's repr. Preserve a cause only for debugging when safe; callers and tests should assert structured fields rather than parser prose.

### 5. Side-effect-free path resolution

**Closest valid source:** `projects/spatial-pharma-dl/src/data.py::available_processed_slide_ids`

This helper deliberately resolves the read path without calling the creating directory helper:

```python
base = st.project_root() / "data" / "processed" / "pharma"
return {
    sample_id
    for sample_id in sample_ids
    if (base / f"{safe_filename(sample_id)}_clustered.h5ad").is_file()
}
```

**Current unsafe seams:** `pharma_processed_dir()` and `pharma_outputs_dir()` call `mkdir`; `patch_cache_path()` calls `pharma_processed_dir()`; `_embedding_cache_path()` calls both `pharma_processed_dir()` and `mkdir`. Thus merely asking whether an artifact is reusable changes the filesystem.

**Phase 4 application:** split pure resolution from writer preparation:

```python
def pharma_processed_path() -> Path:
    return st.project_root() / "data" / "processed" / "pharma"

def patch_cache_path(...) -> Path:
    return pharma_processed_path() / filename

def _prepare_parent_for_write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
```

Sidecar resolution is likewise pure: `manifest_path(payload) -> payload.with_name(payload.name + ".manifest.json")`. Read/admission/status calls must not create parents, delete remnants, rewrite legacy files, or load default configuration when an explicit config/context was supplied.

### 6. Generic admission before production-reader validation

Phase 3's identity call chain is the closest guard-order analog. `load_slide_patches()` obtains decoded patch metadata, passes it through `align_labels_with_metadata()`, then indexes arrays only with the validated `_patch_source_row`. Invalid identity tests monkeypatch merge/model/cache seams with `_forbidden` sentinels and prove ordering.

Phase 4 needs the same two-stage shape:

```python
admission = admit_artifact(
    payload_path,
    expected_kind=...,
    expected_contract_version=...,
    expected_fingerprint=...,
)
payload, observed_schema = production_reader(admission.payload_path)
validate_schema_against_manifest(observed_schema, admission.manifest)
return payload
```

Generic admission must complete these checks before the callback runs:

1. pure path/sidecar resolution;
2. sidecar and payload are regular non-symlink files;
3. bounded strict manifest parsing/admission;
4. completion, kind, contract, basename, and expected fingerprint;
5. fingerprint digest recomputation with `hmac.compare_digest`;
6. payload `fstat` identity/size, streaming SHA-256, and post-hash stability check; and
7. byte count/checksum comparison with `hmac.compare_digest`.

Then the production reader validates exact keys, types, shapes, row counts, identities, and semantic invariants. A valid checksum cannot turn a wrong-rank embedding, wrong-slide H5AD, or malformed report into a valid artifact.

### 7. Atomic same-directory publication

There is no existing repository implementation to copy. Current writers directly call `write_h5ad`, `np.savez_compressed`, `to_parquet`, `to_csv`, `write_text`, or `torch.save` on the final path. Phase 4 should centralize the new behavior rather than partially wrapping individual calls.

Suggested orchestration seam:

```python
def publish_artifact(
    final_path: Path,
    *,
    artifact_kind: str,
    contract_version: str,
    fingerprint: ArtifactFingerprint,
    write_payload: Callable[[BinaryIO | Path], None],
    read_and_validate_payload: Callable[[Path], PayloadSchema],
) -> ArtifactManifest: ...
```

Required mechanics:

- create unique payload and sidecar temporary files with `tempfile.mkstemp` or `NamedTemporaryFile(delete=False, dir=final_path.parent, prefix=...)`;
- keep exact paths in local trusted variables; cleanup only those paths;
- adapt suffix-sensitive libraries carefully: pass an open binary handle to `np.savez_compressed`, and give H5AD/pandas/Torch a temporary path with an acceptable suffix;
- return from the library writer only after its handle is closed;
- open the temporary payload, flush if owned, and `os.fsync` it;
- checksum in bounded chunks and construct payload schema from the same production reader;
- write compact canonical completed sidecar bytes, flush, and `os.fsync` it;
- validate the temporary payload and explicit temporary sidecar through the same generic admission plus production reader path used for reuse;
- `os.replace(temp_payload, final_payload)`, then fsync the destination directory;
- `os.replace(temp_manifest, final_manifest)` last, then fsync the directory; and
- on any error, best-effort unlink only still-existing current-call temps and raise a typed publication failure.

The manifest is the commit marker. Do not publish `complete=false` at the final name and mutate it later. Observable crash states are intentionally fail-closed:

| Failure point | Observable final pair | Reader result |
|---|---|---|
| before payload replace | old valid pair or no pair | old generation valid / missing |
| after payload replace, before sidecar replace | new payload + old/missing sidecar | checksum mismatch / missing manifest |
| after sidecar replace | new payload + new completed sidecar | new generation valid |

Directory fsync is mandatory evidence, not a warning-only optimization. Opening a directory descriptor is platform-sensitive; isolate it in one small helper that tests can monkeypatch. Fault tests should assert call ordering, not depend on a real power failure.

### 8. H5AD adapters

**Root seams:** `utils/st_helpers.py::save_adata` and `load_adata`.  
**Pharma seams:** `src.data::save_slide`, `load_slide`, `preprocess_cohort`, and `available_processed_slide_ids`.

Current readers are `exists()` followed by `ad.read_h5ad`; current writers write directly. `load_slide()` additionally restores the canonical preprocessing JSON sibling:

```python
adata = ad.read_h5ad(path)
canonical = adata.uns.get("spatial_pharma_preprocessing_canonical_json")
if type(canonical) is str:
    adata.uns["spatial_pharma_preprocessing"] = json.loads(canonical)
```

The Phase 3 production validation to reuse is `validate_anndata_spot_identity(..., require_slide_id=True)` plus `PreprocessingManifest`/deterministic preprocessing reconstruction. The H5AD adapter should also validate declared and observed `n_obs`, `n_vars`, exact observation identities, required `obs`/`obsm`/`uns` fields, preprocessing slide/count consistency, and relevant image/spatial metadata before returning.

Root helpers preserve their two-argument public calls. Known tutorial filenames may map to fixed stage contracts; an unknown filename needs explicit source/upstream context or is non-reusable. Pharma internal calls should thread the already resolved config and source/upstream context rather than silently load defaults.

`preprocess_cohort()` catches typed non-reusable artifact states in acquisition mode and rebuilds. `load_slide()` and train-only availability fail with regeneration guidance. No mode falls back to filename existence.

### 9. NPZ adapters: patches and foundation embeddings

**Patch seam:** `src.patches::save_patch_arrays/load_patch_arrays`. The current payload includes an object array and uses `allow_pickle=True`:

```python
np.savez_compressed(path, patches=patches, meta=meta.to_dict("list"))
with np.load(path, allow_pickle=True) as data:
    patches = data["patches"]
    meta = pd.DataFrame(data["meta"].item())
```

Phase 4 may admit only a checksum/fingerprint-matched locally published patch payload before that legacy decode, then validate exact keys, NCHW shape, dtype/finiteness policy, metadata columns/cardinality, and compound identity. It must explicitly label this adapter local-writer-only and must not claim hostile-deserialization safety. Tests may round-trip an in-process writer payload but must never feed an attacker-authored object NPZ to this reader. Phase 5 owns format migration.

**Embedding seam:** `src.foundation::load_or_extract_slide_embeddings`. Its safe primitive cache is the best NPZ adapter analog:

```python
with np.load(cache_path, allow_pickle=False) as cached:
    cached_spots = cached["spot_ids"]
    cached_embeddings = cached["embeddings"]
```

The production adapter should require the exact key set, 2-D `float32` finite embeddings, expected model dimension, fixed-width Unicode spot IDs, row-cardinality equality, and Phase 3 compound alignment. It should reject before `load_slide_patches`, encoder/device/model loading, or cache rewrite on a cache hit. Writer-side `np.savez_compressed` should receive an open temporary binary handle so NumPy cannot append `.npz` to a different path.

`save_patch_index()` needs a paired production `load_patch_index()` even though no reader exists today; an artifact cannot be advertised as reusable if only test code knows how to validate it.

### 10. Table and JSON adapters

**Current table writers/readers:** label Parquet/domain CSV in `labels.py`, patch-index Parquet in `patches.py`, benchmark CSV in `eval.py`, cohort/report/summary CSV reads and writes in `run_pipeline.py`, direct report/label reads in notebook builders, and result CSVs in the foundation builder.

The table adapter should be thin around pandas, not a universal weak schema checker. Each production reader declares exact required columns, duplicate-column policy, dtype families, row count, finite/null policy, identity columns, and kind-specific invariants. Reuse Phase 3's `_admit_column_labels` principle: validate exact column-label types and duplicates before name membership or selection.

Add named readers such as:

```python
load_label_table(...)
load_domain_annotations(...)
load_patch_index(...)
load_benchmark_report(...)
load_cohort_summary(...)
```

Generated notebooks should call these readers through their builder-generated imports; regenerate checked-in notebooks from builders rather than editing JSON manually. Atomicity applies even to a result that is not later reused if it remains a supported published output.

For JSON payloads (`cohort_manifest.json`, `preprocessing_manifest.json`, experiment summaries), preserve existing inner schema and canonical bytes. The artifact sidecar wraps the payload; it does not rename `cohort-manifest-v1` or `spatial-pharma-preprocessing-manifest-v1`. The JSON production reader reconstructs `CohortManifest` or `PreprocessingManifest` (or otherwise revalidates the exact existing schema), not merely `json.loads`.

### 11. Checkpoint adapter boundary

**Current seams:** `src.train::train_one_fold` writes a Python dictionary with `torch.save`; `src.models::load_model_from_checkpoint` calls:

```python
ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
```

Phase 4 wraps local checkpoint creation, checksum/fingerprint admission, and post-decode schema validation. It validates model name, class/target counts, target columns, fold/slide lineage, exact state keys, tensor shapes, and dtypes against the sidecar before returning the model. However `weights_only=False` remains unsafe for malicious bytes, so:

- Phase 4 tests create and consume only a tiny in-process local checkpoint;
- generic admission occurs before legacy decoding;
- no test claims checksum is authenticity; and
- Phase 5 replaces the deserialization semantics.

The adapter should avoid constructing a pretrained model, resolving a device, or loading any model weights until generic admission succeeds.

### 12. Fault injection and forbidden-seam tests

**Closest style:** `test_empty_boundaries.py`, `test_identity_alignment.py`, and `test_adaptive_preprocessing.py` monkeypatch downstream functions to `_forbidden` callbacks. This is stronger than merely asserting the final exception.

For the atomic primitive, expose or inject narrow operations so tests can fail each point deterministically:

```text
write temporary payload
fsync temporary payload
production-reader temporary validation
write/fsync temporary sidecar
replace final payload
first directory fsync
replace final sidecar
final directory fsync
cleanup
```

Run every fault point against both a new destination and replacement of an old valid generation. After failure, call the real production reader and assert either the old generation is still valid or the observed state is rejected. Never assert that both files change atomically; prove that mixed generations are invalid.

Additional race/integrity tests should cover symlinks/non-regular files where portable, replacement between stat/hash/read, payload truncation/appending, byte-count mismatch, checksum mismatch, orphan temps, old-sidecar/new-payload, completed-sidecar/missing-payload, and legacy payloads. Temporary names are never candidates for reuse.

### 13. Static bypass regression

The current raw API inventory is finite and visible:

```text
write_h5ad/read_h5ad
np.savez_compressed/np.load
to_parquet/read_parquet
to_csv/read_csv
Path.write_text
torch.save/torch.load
```

Add a test with a checked allowlist of adapter/builder locations. It should scan `utils/`, `projects/spatial-pharma-dl/src/`, and `projects/spatial-pharma-dl/scripts/`. Test fixtures and notebook-maintenance scripts may have separately documented allowances. The allowlist should identify exact module and purpose, not blanket-exempt a directory.

The test protects architecture, but behavior tests remain authoritative: a raw API confined to an adapter is still wrong if a caller bypasses contract admission or writer validation.

### 14. Existing fixture patterns to extend

**Source:** `projects/spatial-pharma-dl/tests/conftest.py`

Reuse these factories rather than introduce parallel fixture vocabularies:

- `synthetic_anndata_factory`: fresh deterministic real AnnData with exact slide identity, spatial coordinates, image, and scale factors;
- `cohort_factory`: stable slide/spot rows and LOSO ordering;
- `key_adversary_factory`: null, blank, duplicate, wrong-type, hostile, unmatched, cross-slide, reserved-column, and row-cardinality cases;
- `artifact_adversary_factory`: missing-key, wrong-shape/dtype, object NPZ, corrupt JSON, row mismatch, and corrupt bytes, all under `tmp_path`;
- `image_adversary_factory` and `fold_adversary_factory`: retain for downstream regression boundaries rather than Phase 4 format invention.

`test_fixture_contracts.py` already proves fresh construction, deterministic contents, mutation isolation, and that artifact paths stay under `tmp_path`. `test_artifact_roundtrips.py` provides primitive NPZ/Parquet/JSON/H5AD round trips. Extend those patterns with factory helpers for a valid manifest pair, a valid old/new generation pair, malformed sidecars, and exact fault-point operation logs.

Hostile sidecar tests should operate on parsed candidate objects when testing primitive admission, and on bounded raw bytes when testing the JSON parser. Do not serialize a hostile Python object with pickle or place one in a checkpoint/patch payload.

## Expected File Classification

### Generic contract and shared tests

| Expected file | Role | Data-flow position |
|---|---|---|
| `utils/artifacts.py` | Import-light canonical manifest, exact admission, fingerprint, checksum, sidecar path, reuse status, atomic publication | Between kind-specific context and all payload adapters; no NumPy/pandas/Torch/AnnData imports |
| `projects/spatial-pharma-dl/tests/test_artifact_contract.py` | Pure hostile admission, fingerprints, streaming integrity, fault injection, cleanup | Directly exercises generic contract with byte payload callbacks |
| `projects/spatial-pharma-dl/tests/conftest.py` | Extend existing artifact factories with valid manifest generations/fault logs | Shared deterministic test inputs only |

### Scientific cache adapters

| Expected file | Role | Data-flow position |
|---|---|---|
| `utils/st_helpers.py` | Root H5AD adapter integration while preserving `save_adata/load_adata` calls | Root notebook cache producer/consumer |
| `projects/spatial-pharma-dl/src/data.py` | Processed-slide projection, pure paths, H5AD writer/reader, regeneration/admission | Raw AnnData -> processed H5AD -> label/patch stages |
| `projects/spatial-pharma-dl/src/labels.py` | Label/domain projections and production table readers/writers | Processed slide -> label tables -> training/evaluation |
| `projects/spatial-pharma-dl/src/patches.py` | Patch/index projections, legacy-local patch adapter, index reader, pure path | Processed slide + stain -> patches/index -> models |
| `projects/spatial-pharma-dl/src/foundation.py` | Embedding projection and strict primitive NPZ adapter | Patches + model identity -> embeddings -> probes |
| `projects/spatial-pharma-dl/tests/test_artifact_adapters.py` | Real tiny H5AD, NPZ, tables, and checkpoint semantic validation | Exercises the actual production readers after generic admission |

### Models, reports, and orchestration

| Expected file | Role | Data-flow position |
|---|---|---|
| `projects/spatial-pharma-dl/src/train.py` | Checkpoint publication context and upstream/fold lineage | Fold training -> checkpoint |
| `projects/spatial-pharma-dl/src/models.py` | Checkpoint admission/production reader and state-schema validation | Checkpoint -> restored model; legacy decode boundary explicit |
| `projects/spatial-pharma-dl/src/eval.py` | Benchmark report projection plus validated table writer/reader | Fold outputs -> benchmark report -> summary/notebook |
| `projects/spatial-pharma-dl/scripts/run_pipeline.py` | Replace existence/direct pandas/text seams; publish manifests/summaries through adapters | End-to-end orchestration and final artifact lineage |
| `projects/spatial-pharma-dl/scripts/build_notebooks.py` | Generated notebook report reads use production reader | Builder source for supported notebook consumers |
| `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py` | Generated notebook label reads and result publications use adapters | Builder source for foundation notebook consumers |
| `projects/spatial-pharma-dl/notebooks/*.ipynb` | Regenerated checked-in artifacts only where builder output changes | Public teaching surface; no hand-edited JSON |
| `projects/spatial-pharma-dl/tests/test_artifact_orchestration.py` | Regeneration, train-only failure, lineage propagation, runner publication, static bypass, compatibility | Cross-module contract closure |

### Existing tests expected to receive focused regressions

| Existing file | Preserve/prove |
|---|---|
| `test_artifact_roundtrips.py` | Primitive local formats and paths; do not convert it into the contract suite |
| `test_cohort_admission.py` | Final admitted order, strict/partial behavior, no downstream stages after failure |
| `test_identity_alignment.py` | Compound identity and source-order alignment after cache admission |
| `test_adaptive_preprocessing.py` | Canonical preprocessing payload/H5AD restoration and unchanged inner schema |
| `test_foundation.py` | Offline embedding/probe behavior and no model download |
| `test_model_fold_smoke.py` / `test_model_fold_contracts.py` | Tiny local checkpoint plumbing and fold orchestration only |
| `test_notebook_structure.py` | Notebook sequence/kernel/public structure unchanged after regeneration |

## Concrete Planning Boundaries

### Plan 04-01 — Generic contract and atomic primitive

Create the generic module and pure tests first. Prove hostile-safe admission, immutable manifests, explicit per-kind fingerprint projections (or a registry interface used by later adapters), regular-file/checksum admission, same-directory temporary files, fsync/replace order, cleanup, and every fault state. No scientific adapter should invent its own sidecar parser or atomic write sequence.

### Plan 04-02 — H5AD, label, patch/index, and embedding adapters

Integrate scientific caches only after 04-01 is green. Thread resolved config/source/upstream context through existing internal calls; keep public signatures compatible through optional keyword-only context where necessary. Acquisition rebuilds typed stale/legacy states; direct/train-only readers fail closed. Reuse Phase 3 identity and provenance validators after generic admission.

### Plan 04-03 — Checkpoints, reports, JSON manifests, runner, and bypass closure

Route the remaining writers/readers through the shared primitive and named production adapters. Update notebook builders and regenerate notebooks. Add end-to-end lineage, compatibility, mixed-generation recovery, and static raw-API inventory tests. Keep checkpoint/patch malicious-deserialization claims explicitly deferred to Phase 5.

## Implementation Guardrails

- A final payload without its completed sidecar is legacy/incomplete and never reusable.
- A sidecar is not sufficient: expected fingerprint, payload byte count/checksum, and kind-specific semantic schema must all pass.
- Never catch a broad exception and fall back to filename-only reuse. Catch a typed non-reusable state only where regeneration is authorized.
- Fingerprint projections accept only resolved exact primitives and explicit upstream fingerprints; no generic caller mapping is hashed directly.
- Read-path helpers create no directories and perform no cleanup.
- Validate temporary output with the same production reader contract used for final reuse.
- Publish payload first and completed sidecar last; do not mutate a final sidecar in place.
- Cleanup only exact temporary paths created by the current call.
- Preserve final filenames, CLI flags, config keys, notebook order, public exports, and successful return schemas.
- Preserve `cohort-manifest-v1` and `spatial-pharma-preprocessing-manifest-v1` as inner payload schemas.
- Keep all Phase 4 tests CPU/offline under `tmp_path`; do not download datasets or weights.
- Do not deserialize attacker-authored object NPZ or checkpoint bytes in this phase.
- A static bypass allowlist supplements, but never replaces, production-reader and fault-injection behavior tests.

## Expected Verification Flow

1. Pure tests admit only bounded exact manifest primitives and prove deterministic fingerprint invalidation/stability.
2. Generic reader tests reject missing, legacy, malformed, incomplete, stale, symlinked, truncated, replaced, and checksum-invalid pairs before callbacks.
3. Atomic tests inject every failure for new and replacement destinations and prove no mixed generation is reusable.
4. Adapter tests round-trip real tiny H5AD, primitive embedding NPZ, trusted local patch NPZ, Parquet/CSV/JSON tables, and a tiny local checkpoint through production readers.
5. Schema mutations with recomputed valid checksums still fail at the kind-specific reader.
6. Transitive lineage changes invalidate processed -> label/patch -> embedding/checkpoint -> report artifacts; unrelated presentation changes preserve fingerprints.
7. Acquisition regenerates stale/legacy caches; train-only/direct readers fail before labels, devices, models, or reports.
8. Static inventory permits raw I/O only in named adapters/builders.
9. Existing filenames, inner manifest schemas, public APIs, CLI behavior, and notebook structure remain compatible.
10. `python scripts/verify.py fast` passes the full strict offline suite.

---

*Pattern mapping complete for Phase 04 planning.*
