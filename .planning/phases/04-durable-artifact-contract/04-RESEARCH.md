---
phase: 04-durable-artifact-contract
status: researched
requirements:
  - ART-03
  - ART-04
date: 2026-07-17
researcher: gsd-phase-researcher
---

# Phase 4 Research: Durable Artifact Contract

## Research conclusion

Phase 4 should introduce one import-light artifact-contract module and make every
production reuse decision pass through it. A final payload's existence is not
evidence. Reuse is admitted only when a small, bounded, canonical JSON sidecar
is present, complete, schema-valid, fingerprint-matched, and consistent with
the payload checksum and kind-specific schema. Writers must publish a validated
temporary payload first and the completed sidecar last; the sidecar is the
commit marker.

The current code has no durable artifact contract. Processed H5AD and patch
caches are reused by filename alone, foundation NPZ checks only keys that happen
to be accessed, tables and reports are written directly, checkpoints are saved
directly, and the runner publishes JSON manifests with `Path.write_text()`.
`Path.exists()` therefore currently conflates complete, stale, truncated, and
legacy files. ART-03 and ART-04 require replacing that decision at every
production boundary, not merely adding a helper that some paths can bypass.

Phase 4 must not claim that pickle-backed patch NPZ or PyTorch checkpoint
payloads are safe against a malicious artifact author. A valid checksum proves
integrity relative to the sidecar, not trust. This phase can ensure that only a
locally published, fingerprint-admitted payload reaches the existing legacy
reader. Phase 5 must remove object deserialization and use `weights_only=True`.

## Existing patterns to preserve and reuse

- `src.validation.resolve_config()` already converts admitted configuration to
  exact JSON primitives, canonicalizes mapping order, preserves meaningful list
  order, rejects hostile subclasses before their methods run, and returns fresh
  views. Artifact fingerprint projections should accept only this resolved
  representation.
- `CohortManifest`, `PreprocessingResolution`, and `PreprocessingManifest` use
  frozen/canonical internal state plus fresh JSON-safe views. Artifact manifests
  should follow the same split rather than expose mutable dictionaries.
- `src.identity` provides exact compound identity and bounded inert diagnostics.
  Kind-specific patch, label, and embedding readers should reuse it after the
  generic contract admits bytes.
- Phase 1 fixtures already provide primitive NPZ/Parquet/H5AD round trips and
  corrupt/missing-key/wrong-shape/row-mismatch artifacts under `tmp_path`.
- Phase 3 proves real Scanpy/H5AD evidence is part of the fast offline gate and
  that the preprocessing JSON sibling survives H5AD scalar decoding.
- Preserve `cohort-manifest-v1` and
  `spatial-pharma-preprocessing-manifest-v1` as payload schemas. The new artifact
  sidecar wraps their publication; it does not replace or mutate their content.

## Complete production artifact inventory

### Root tutorial H5AD caches

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `utils/st_helpers.py::save_adata` | Direct `adata.write_h5ad(final)` | Publish through the H5AD adapter; preserve final filename. |
| `utils/st_helpers.py::load_adata` | Existence check then `ad.read_h5ad` | Require the sidecar, expected contract/fingerprint, checksum, and H5AD schema before return. |
| Root notebooks 02, 04, 07, 09 | Produce `adata_raw.h5ad`, `adata_qc.h5ad`, `adata_clustered.h5ad`, `adata_features.h5ad` | Keep names/order; propagate upstream artifact fingerprint and fixed notebook-stage contract version. |
| Root notebooks 03-11 and `scripts/generate_gallery_figures.py` | Call `load_adata` | Automatically inherit validation once the helper changes. |

The root tutorial has no YAML experiment projection. Its relevant projection is
the fixed stage name/contract version and declared upstream fingerprint. The raw
stage additionally needs the public dataset semantic identity. Do not hash
plotting options. To preserve the existing two-argument helper, optional
keyword-only contract inputs may be added; fixed known filenames can map to
stable stage contracts for unchanged notebooks. Unknown filenames need an
explicit source/upstream identity or must be treated as non-reusable rather than
receive a content-free fingerprint.

### Pharma processed H5AD

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `src.data::save_slide` | Direct H5AD write | H5AD adapter with `processed-slide-v1`, preprocessing projection, source identity, payload schema, and checksum. |
| `src.data::load_slide` | Filename existence then `ad.read_h5ad`; restores canonical preprocessing JSON | Contract admission first; then read and validate AnnData axes, identity, preprocessing schema/counts, required fields, and manifest-declared shapes. |
| `src.data::preprocess_cohort` | Reuses `*_clustered.h5ad` on `out.exists()` | Compute expected fingerprint and call the production reader; missing/stale/legacy/incomplete means regenerate in acquisition mode. |
| `src.data::available_processed_slide_ids` | Uses `is_file()` only | Count only artifacts admitted for the current resolved configuration/source identity. Train-only mode must fail with regeneration guidance rather than admit legacy bytes. |
| `src.data::cohort_summary`, `src.labels::build_labels_for_slide`, `src.patches::fit_reference_stain`, `build_patch_cohort`, runner manifest assembly | Consume `load_slide` | Pass the resolved config/source expectation through so custom configs do not silently validate against defaults. |

Relevant fingerprint projection: `preprocessing`, `seed` (because PCA/UMAP/
Leiden use it), processed-slide code-contract version, `sample_id`, source
identity, and no labels/training/foundation/report settings. The writer can
derive an actual source identity from admitted raw AnnData (shape; exact
obs/var identities; matrix, spatial-coordinate, image, and scale-factor
digests), but cache-only offline reuse cannot re-observe a mutable remote
provider without downloading it. Therefore distinguish `source_semantic_id`
(provider/sample identity used for ordinary offline expectation) from the
recorded `source_content_digest` produced when source bytes are available.
Synthetic tests must prove that supplying a different content digest misses.

### Patch NPZ and patch index

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `src.patches::save_patch_arrays` | Direct object-valued `np.savez_compressed` | Atomic NPZ publication with patch fingerprint, keys/shape/dtype/row metadata, and checksum. Keep legacy payload format until Phase 5. |
| `src.patches::load_patch_arrays` | Existence then `np.load(...allow_pickle=True)` | Admit sidecar/checksum/fingerprint before legacy load; then validate exact required keys, NCHW shape, numeric dtype/finite policy, metadata columns/cardinality/identity. |
| `src.patches::patch_cache_path` | Filename includes only manual `patches.version` | Preserve filename; sidecar owns scientific invalidation. Path construction should not create directories during a read/admission check. |
| `src.patches::save_patch_index` | Direct Parquet write | Atomic table adapter; declared columns/dtypes/row count and upstream label+patch fingerprints. |
| `src.train::load_slide_patches` and `src.foundation::load_or_extract_slide_embeddings` | Indirect patch reader | Pass expected config/upstream identity and retain Phase 3 one-to-one alignment. |
| runner `_need_patch_rebuild` | `Path.exists()` | Ask the patch reader/contract for reuse status; stale, legacy, malformed, or incomplete caches rebuild. |

Patch projection: `patches` configuration, patch code-contract version,
processed-slide fingerprint, and shared stain-reference identity/fingerprint.
It must not include training epochs, model choice, report formatting, or
foundation options. Current `save_patch_arrays` does not receive the reference
stain or processed-slide lineage, so the internal writer API must accept an
explicit artifact context from `build_patch_cohort`. This anticipates Phase 8:
source/target stain and quality algorithm changes invalidate caches by bumping
the patch contract version or changing the projection, without implementing
the Phase 8 algorithm now.

### Label and domain tables

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `src.labels::build_labels_cohort` | Direct per-slide Parquet and CSV writes | Publish every label Parquet and `domain_annotations.csv` atomically with table schema and processed-slide upstream fingerprints. |
| Generated foundation notebook / `build_foundation_notebook.py` | Direct `pd.read_parquet(labels_*.parquet)` | Use a production `load_label_table` adapter; otherwise a stale/corrupt table bypasses the contract. Regenerate the checked-in notebook from its builder. |
| `patch_index.parquet` | Written by `save_patch_index`; no production reader currently | Contract it now and add a validated reader for supported reuse/inspection. |

Label projection: `labels`, `marker_genes`, `gene_modules`, label code-contract
version, and the processed-slide fingerprint. It should not include patch,
training, foundation, or evaluation settings. `domain_annotations.csv` is a
derived view of the same label inputs. Phase 9 can bump the label contract and
extend the payload schema without changing this framework.

### Foundation embedding NPZ

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `src.foundation::_embedding_cache_path` | Creates cache directory during path calculation | Separate pure path resolution from writer directory creation. |
| `load_or_extract_slide_embeddings` cache branch | Existence then safe NPZ load; validates non-empty and identity alignment but not exact key set/dtype/dimension/finiteness/checksum/fingerprint | Admit contract first, then validate exact keys, 2-D float embeddings, expected dimension, finite values, Unicode IDs, row cardinality, and exact identity. |
| `load_or_extract_slide_embeddings` write branch | Direct `np.savez_compressed` | Atomic NPZ adapter validated through the same reader. |

Embedding projection: foundation model name plus explicit model-spec identity
(`repo_id`, backend, preprocessing mean/std, dimension, and preferably an
explicit upstream revision/weight digest), embedding code-contract version,
patch fingerprint, and spot identity. Device, cache enablement, report options,
and presentation settings are irrelevant. Batch size should be omitted only if
tests establish that it is a scheduling choice under the declared numerical
contract; otherwise include it. A model hub name without a pinned revision
cannot prove immutable weight identity, so the manifest must expose that
limitation rather than imply a content checksum it does not possess. Phase 10
may pin external dependencies, but Phase 4 should at least record the resolved
model source identity.

### Model checkpoints

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| `src.train::train_one_fold` | Direct `torch.save(final)` containing weights and Python metadata | Publish via checkpoint adapter with training/fold/input fingerprint and tensor schema metadata. Temporary payload validation is allowed only after local writer creation. |
| `src.models::load_model_from_checkpoint` | Direct unsafe `torch.load(...weights_only=False)` | Require contract/checksum/fingerprint before legacy loading, then validate model identity, expected state keys/tensor shapes/dtypes, and metadata. Phase 5 replaces the unsafe format/loader. |

Checkpoint projection: training model/hyperparameters and label target schema,
seed/determinism policy when Phase 6 supplies it, fold/train/test lineage,
upstream patch and label fingerprints, and checkpoint code-contract version.
Phase 7 selection/scaling/imputation lineage must become upstream/projection
inputs later. Do not include report formatting. Phase 4 tests must never pass an
attacker-authored pickle through the legacy loader; malicious-deserialization
evidence belongs to Phase 5.

### Cohort/preprocessing manifests, reports, and summaries

| Boundary | Current behavior | Required Phase 4 behavior |
|---|---|---|
| runner `cohort_manifest.json` and `preprocessing_manifest.json` | Direct `write_text` | Atomic JSON adapter; preserve inner schemas and canonical bytes; sidecars carry completion/checksum/fingerprint. |
| `src.eval::save_benchmark_report` | Direct CSV write | Atomic table adapter with report projection, upstream checkpoint/label/fold lineage, exact columns/types/counts. |
| runner report read | Direct `pd.read_csv(report_path)` | Use a validated benchmark-report reader. |
| runner `cohort_summary.csv` and `experiment_<exp>_summary.json` | Direct pandas/text writes | Atomic table/JSON adapters with upstream lineage and schema validation. |
| generated notebook 05 / builder | Direct report CSV read | Use the validated report reader. |
| foundation notebook result CSVs | Direct writes under figure output | If treated as supported reusable scientific results, route through the table adapter; purely disposable plotting intermediates may be explicitly non-reusable but still need atomic publication if retained. |

Run-manifest/report fingerprints include resolved experiment/evaluation
projections and exact upstream fingerprints. They must not include absolute
checkout/output paths, timestamps, incidental dictionary order, console text,
or plot styling. Phase 2/3 manifest payloads remain byte-compatible.

## Proposed shared contract architecture

Add `projects/spatial-pharma-dl/src/artifacts.py` (or equivalently named
import-light module). It should depend only on the standard library and small
validated value objects, not NumPy, pandas, Torch, Scanpy, or AnnData. Domain
modules retain their production payload readers/writers and pass adapters into
the shared layer.

### Canonical manifest

Recommended sidecar: `<payload-name>.manifest.json`, retaining every existing
payload filename. A single schema such as
`spatial-pharma-artifact-manifest-v1` should contain:

```json
{
  "schema_version": "spatial-pharma-artifact-manifest-v1",
  "artifact_kind": "processed_slide",
  "contract_version": "processed-slide-v1",
  "complete": true,
  "fingerprint": {
    "algorithm": "sha256",
    "digest": "...",
    "inputs": {
      "configuration": {},
      "source": {},
      "upstream": [],
      "contract_version": "processed-slide-v1"
    }
  },
  "payload": {
    "filename": "slide_clustered.h5ad",
    "format": "h5ad",
    "byte_count": 123,
    "sha256": "...",
    "schema": {}
  }
}
```

The fingerprint digest is SHA-256 over canonical strict JSON of `inputs`, not
over the output payload. This permits the reader to compute the expected digest
before reuse. The payload checksum independently detects corruption or
interrupted replacement. `payload.schema` is kind-specific metadata such as
required keys, axes, rows, shapes, dtypes, target columns, or model state
summary. The manifest need not checksum itself (that is recursive); strict
parsing and atomic last publication protect it.

Use per-kind projection functions with explicit allowlists, not a generic
"drop irrelevant keys" algorithm. Each should return a frozen canonical value
and have a visible `*_CONTRACT_VERSION` constant. A code behavior change that
can alter scientific meaning increments that constant. Do not use the Git
commit as the contract version: unrelated edits would invalidate caches and a
dirty tree is not a stable semantic contract.

### Hostile-safe admission

Manifest admission occurs before heavy imports or payload deserialization:

1. Resolve expected payload and sidecar paths without creating directories.
2. Reject missing paths, symlinks/non-regular files where practical, wrong
   sidecar suffix, oversized manifest bytes, invalid UTF-8, and empty files.
3. Parse JSON with duplicate-key rejection. Enforce exact root/key/value types,
   maximum nesting/node/string/list sizes, finite bounded numbers, exact key
   set, supported schema/kind/contract/digest algorithms, lowercase fixed-width
   hex digests, basename-only payload filename, and `complete is True`.
4. Canonicalize and recompute `fingerprint.digest`; compare it and the expected
   digest using `hmac.compare_digest`.
5. Stream payload SHA-256 in bounded chunks and verify byte count/checksum from
   an opened regular file. Avoid loading payload bytes merely to hash them.
6. Only then invoke the kind-specific production reader and validate declared
   keys/types/shapes/rows/identities/semantic schema before returning data.

Diagnostics should use a frozen `ArtifactValidationError` with inert fields
(`artifact_kind`, bounded basename, stable reason code, guidance). Never include
raw JSON values, exception reprs, absolute host paths, archive member names, or
attacker-controlled strings. Distinguish at least: missing manifest, legacy
artifact, malformed/unsupported manifest, incomplete, stale fingerprint,
checksum mismatch, truncated payload, payload schema mismatch, and reader
validation failure. All rejection messages should direct the user to regenerate
the named artifact kind.

Canonical JSON must set `sort_keys=True`, compact separators, ASCII policy
consistently, and `allow_nan=False`. Values entering projection builders must
already be exact admitted primitives; reuse Phase 2's validators rather than
calling arbitrary mapping/string/number methods.

### Atomic publication and ordering

One shared publication primitive should orchestrate every kind-specific writer:

1. Create unique temporary payload and manifest paths with `tempfile` in the
   final destination directory (same filesystem); never derive a predictable
   name and never treat temporary remnants as candidates.
2. Ask the adapter to write the temporary payload. Flush/close library writers,
   then open/fdatasync or fsync the completed temporary file.
3. Compute byte count/checksum and kind-specific schema metadata.
4. Construct a completed canonical temporary sidecar. Before publication, run
   the generic contract plus the same kind-specific production reader against
   the temporary payload/explicit temporary sidecar. This is the only acceptable
   writer-side validation path.
5. Flush/fsync the temporary sidecar.
6. `os.replace(temp_payload, final_payload)`, then fsync the destination
   directory.
7. `os.replace(temp_manifest, final_manifest)` last, then fsync the directory.
8. Best-effort cleanup only the exact temporary paths created by this call.

Do not publish an incomplete final sidecar and later flip it in place. The
absence of the completed final sidecar is the incomplete state. With an older
valid pair, a crash before payload replacement preserves it; a crash after
payload replacement but before manifest replacement leaves old-manifest/new-
payload checksum mismatch and therefore fails closed; after sidecar replacement
the new pair is committed. This ordering sacrifices reuse after the middle
crash but never admits a mixed generation.

Adapters whose libraries append extensions (notably `np.savez_compressed`) must
write to an already opened binary file handle or a correctly suffixed temporary
path so the validated file is the file later replaced. H5AD/Torch/pandas writers
must return only after their handles are closed. Fsync failures are publication
failures, not warnings.

### Production-reader adapters

The shared module should expose orchestration/value objects, not universal
payload decoding. Suggested adapters:

- `data.py`: `_read_processed_slide_payload` validates H5AD axes, compound
  identity, declared dimensions, required obs/obsm/uns fields, and canonical
  preprocessing provenance.
- `patches.py`: `_read_patch_payload` validates exact keys, shape/dtype/
  cardinality/identity after generic admission; its Phase 4 legacy decode is
  explicitly marked local-writer-only pending Phase 5.
- `labels.py`: `load_label_table` and domain table reader validate exact required
  columns, dtypes, row count, compound identity, and label schema.
- `foundation.py`: `_read_embedding_payload` validates exact NPZ keys, 2-D
  float32/finiteness/dimension, Unicode spot IDs, cardinality, and alignment.
- `models.py`: checkpoint reader validates manifest-declared model/state schema
  after the existing legacy load; Phase 5 replaces decoding and metadata.
- `eval.py`: `load_benchmark_report` validates exact report columns, experiment,
  fold/slide/model rows, numeric finiteness policy, and manifest lineage.
- runner: JSON/table helpers validate cohort/preprocessing/summary schemas and
  use the report reader rather than pandas directly.

Generic checksum admission must not be confused with semantic validation. For
example, a valid-checksum NPZ with a wrong embedding dimension remains invalid;
an H5AD whose declared axis counts match but whose preprocessing record names a
different slide remains invalid.

## Legacy and regeneration policy

- A final payload with no sidecar is a legacy artifact and is never reused.
- A malformed/incomplete/stale/checksum-invalid sidecar is rejected; there is no
  fallback to filename-only reuse.
- Acquisition/build paths may catch a typed non-reusable result and regenerate.
  Direct/train-only readers raise actionable `ArtifactValidationError`.
- Existing files are not migrated in place because their relevant source/
  configuration lineage cannot be reconstructed reliably. Regenerate them.
- Do not delete rejected legacy files during read admission. A successful
  atomic publication can replace them. Temporary cleanup is narrowly scoped to
  names created by the current writer.
- Bump a per-kind contract version for later Phase 5/7/8/9 semantic changes;
  old sidecars then become deterministic misses with precise guidance.

## Risks and mandatory mitigations

1. **Bypass risk:** direct `Path.exists`, `pd.read_*`, `ad.read_h5ad`,
   `np.load`, or `torch.load` can bypass the contract. Add a static regression
   inventory test or tight `rg` allowlist and update runner/notebook builders.
2. **Custom-config mismatch:** many current loaders accept only an ID and would
   default-load configuration. Thread the already resolved config/artifact
   context through internal calls; preserve existing public calls with defaults.
3. **Path helpers cause side effects:** `pharma_processed_dir()` and embedding
   path construction create directories. Separate pure read paths from
   writer-side `mkdir` so invalid/missing reads remain side-effect free.
4. **Source identity limits:** provider IDs are not content hashes. Record both
   semantic and observed-content identities and test explicit content changes;
   do not claim remote mutation detection when the source is unavailable.
5. **Checksum is not authenticity:** a malicious author can replace payload and
   sidecar together. Keep the Phase 5 boundary explicit and never run a hostile
   pickle in Phase 4 tests.
6. **TOCTOU:** checking a path and reopening it can race. Hash and validate via
   regular-file descriptors where feasible; at minimum reject symlinks and
   compare stat identity/size around hashing and reading.
7. **Atomic multi-file misconception:** payload+sidecar cannot be atomically
   replaced together. Payload-first/manifest-last plus checksum makes every
   observable mixed state invalid.
8. **Writer validation recursion:** temporary validation must accept explicit
   payload/sidecar paths and must not resolve the final path or publish again.
9. **Over-invalidation:** hashing full config or Git commit defeats ART-03's
   presentation-only stability. Per-kind allowlisted projections and negative
   invariance tests are mandatory.
10. **Under-invalidation:** omitting code-contract, upstream lineage, target
    schema, source identity, or stain/model identity silently reuses stale
    science. Every adapter needs an explicit projection table and tests for each
    relevant input.
11. **Manifest DoS/diagnostic injection:** bound bytes/depth/counts and render
    only inert enums/basenames/reason codes.
12. **Performance:** streaming checksums add I/O, especially H5AD/patches. That
    is required for reuse correctness; use a moderate fixed chunk size and do
    not materialize entire files.

## Proposed plan split

### Plan 04-01 — Shared contract, fingerprints, and atomic primitive

- Add import-light manifest/value/error types, strict hostile-safe JSON
  admission, streaming SHA-256, exact per-kind fingerprint projection registry,
  explicit contract versions, sidecar naming, and reuse-status API.
- Add same-directory temporary publication with fsync, production-reader callback,
  payload-first/manifest-last replacement, and exact cleanup.
- Test relevant/non-relevant projection changes, mapping-order invariance,
  unsupported/duplicate/oversized/hostile manifests, checksum/byte mismatches,
  symlinks where portable, and every fault-injection point.

### Plan 04-02 — Scientific caches and tables

- Integrate processed H5AD, patch NPZ/index, per-slide label/domain tables, and
  foundation embedding NPZ.
- Replace `exists()` cache reuse and ensure custom resolved config/source/
  upstream lineage reaches readers/writers.
- Add exact kind-specific schema/shape/dtype/identity validation using Phase 3
  helpers, plus legacy/stale regeneration behavior. Keep unsafe patch payload
  decoding behind admitted local-writer lineage pending Phase 5.

### Plan 04-03 — Checkpoints, reports, manifests, and orchestration closure

- Integrate checkpoint writes/reads, benchmark/cohort/domain/summary tables,
  cohort and preprocessing JSON manifests, experiment JSON, and supported
  generated-notebook readers.
- Replace direct runner pandas reads and `_need_patch_rebuild` existence checks;
  regenerate checked-in notebooks from updated builders where necessary.
- Add a fixture-backed pipeline publication/reuse test, static bypass inventory,
  interruption recovery across existing-artifact replacement, and compatibility
  assertions for filenames/inner manifest schemas/public exports/CLI/notebook
  order.

The plans are sequential because Plan 04-02/03 must use the exact primitive
proven by 04-01. Within 04-02, H5AD, patch, label, and embedding adapters can be
implemented as separate tasks but should land only with cross-artifact lineage
tests.

## Validation Architecture

### Evidence layers

1. **Pure contract unit evidence (no scientific imports):** canonical manifests,
   exact-type input admission, per-kind projections, digest recomputation,
   bounded diagnostics, checksum streaming, and reuse decisions.
2. **Atomic fault-injection evidence:** instrument writer, temp fsync, temp
   reader validation, payload replace, first directory fsync, manifest replace,
   and final directory fsync. At each injected failure assert that the final
   pair is either the previous valid generation or rejected; it is never a
   falsely valid mixed generation. Assert temp names are ignored and exact
   current-call remnants are cleaned where safe.
3. **Kind-specific production-reader evidence:** real small H5AD, primitive
   embedding NPZ, current patch NPZ (trusted writer fixture only), Parquet/CSV/
   JSON tables, and a tiny locally produced checkpoint. Mutate required keys,
   schemas, shape, dtype, rows, identities, completion, fingerprint, checksum,
   and truncation independently.
4. **Lineage/invalidation evidence:** for every kind, change each relevant
   config/source/upstream/code-contract field and assert a deterministic miss;
   change presentation/output paths/unrelated config/insertion order and assert
   identical fingerprint bytes. Upstream processed change must transitively miss
   patch/label, patch change must miss embedding/checkpoint, and model/label/
   fold lineage must miss reports.
5. **Orchestration evidence:** train-only availability admits only valid current
   processed artifacts; acquisition mode regenerates legacy/stale caches; patch
   rebuild uses contract state; runner publishes both Phase 2/3 manifest payloads
   atomically and reads its report through the production adapter. Failures
   precede labels/models/results.
6. **Compatibility evidence:** existing final filenames, notebook order, CLI
   flags, config keys, public imports/signatures, successful return schemas,
   `cohort-manifest-v1`, and
   `spatial-pharma-preprocessing-manifest-v1` remain unchanged. Sidecars are the
   only additive public files.

### Required adversarial matrix

| Dimension | Cases |
|---|---|
| Manifest | absent, zero-byte, invalid UTF-8/JSON, duplicate keys, extra/missing keys, wrong exact types, oversized/deep/wide values, NaN/Infinity representation, unsupported schema/kind/contract/algorithm, `complete=false`, hostile-looking strings |
| Fingerprint | malformed hex, digest/input disagreement, expected mismatch from config/source/upstream/code, mapping reorder, irrelevant config change |
| Payload integrity | missing, symlink/non-regular, zero-byte, truncated, appended bytes, byte-count mismatch, checksum mismatch, replacement after hash/stat |
| Payload schema | missing/extra NPZ keys, object/wrong dtype, wrong rank/dimension, non-finite values, duplicate/null/cross-slide IDs, row mismatch, H5AD axis/provenance mismatch, table columns/types mismatch, checkpoint state mismatch |
| Publication | exception before/after temporary write, payload fsync, temp validation, manifest fsync, payload replace, directory fsync, manifest replace, final fsync; new destination and replacement of old valid generation |
| Recovery | orphan temp only, final payload without sidecar, old sidecar/new payload, completed sidecar/missing payload, legacy final artifact, stale but well-formed pair |

Every adversarial artifact stays under `tmp_path`. Phase 4 must not download data
or model weights and must not deserialize attacker-authored object NPZ/pickle.
Tiny checkpoints are created and consumed in-process only to prove publication
plumbing; Phase 5 owns malicious checkpoint safety.

### Focused commands

Run from repository root with the declared spatial-tx interpreter when needed:

```bash
python -m pytest -q \
  projects/spatial-pharma-dl/tests/test_artifact_contract.py \
  projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py

python -m pytest -q \
  projects/spatial-pharma-dl/tests/test_artifact_adapters.py \
  projects/spatial-pharma-dl/tests/test_identity_alignment.py \
  projects/spatial-pharma-dl/tests/test_adaptive_preprocessing.py

python -m pytest -q \
  projects/spatial-pharma-dl/tests/test_artifact_orchestration.py \
  projects/spatial-pharma-dl/tests/test_cohort_admission.py \
  projects/spatial-pharma-dl/tests/test_foundation.py \
  projects/spatial-pharma-dl/tests/test_model_fold_contracts.py \
  projects/spatial-pharma-dl/tests/test_notebook_structure.py

python scripts/verify.py fast
```

Add a static audit to the tests (or run during review) that permits raw payload
APIs only inside named adapters/builders:

```bash
rg -n "write_h5ad|read_h5ad|np\.savez|np\.load|to_parquet|read_parquet|to_csv|read_csv|write_text|torch\.save|torch\.load" \
  utils projects/spatial-pharma-dl/src projects/spatial-pharma-dl/scripts
```

The audit is not a substitute for behavior tests, but it prevents a later direct
reader/writer from silently bypassing the contract.

### Phase acceptance

ART-03 is satisfied only if tests prove both invalidation and non-invalidation
for every supported kind, including transitive upstream changes. ART-04 is
satisfied only if production readers—not test-only parsers—reject all malformed,
mixed-generation, stale, and semantically wrong artifacts, and every supported
writer uses the shared atomic primitive. A passing checksum alone, a sidecar not
consulted by the reader, or atomic payload replacement without manifest-last
commit does not satisfy the phase.

## Planning checklist

- [ ] One canonical manifest schema and immutable/fresh-view value object.
- [ ] Explicit allowlisted fingerprint projection and contract version per kind.
- [ ] Source semantic identity and observed content identity are distinguished.
- [ ] Generic admission is import-light, bounded, inert, and before payload load.
- [ ] Same-directory temporary payload and sidecar; fsync and manifest-last order.
- [ ] Temporary validation uses the exact production reader adapter.
- [ ] Every current `exists()`/direct reader reuse bypass is removed.
- [ ] Every payload adapter validates exact schema, shape, dtype, rows, identity,
  and semantic provenance appropriate to its kind.
- [ ] Legacy artifacts reject/regenerate without unsafe migration or deletion.
- [ ] Phase 2/3 inner manifest schemas and all final payload filenames remain.
- [ ] Phase 5 unsafe-format boundary is explicit and no hostile pickle is loaded.
- [ ] Fault injection, transitive invalidation, and presentation-only stability
  are fast CPU/offline evidence.

