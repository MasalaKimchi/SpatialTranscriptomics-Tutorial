# Phase 2: Validated Run and Cohort Admission - Research

**Researched:** 2026-07-17  
**Requirements:** VAL-01, VAL-03, VAL-04  
**Scope:** Startup configuration, cohort admission, and empty-stage contracts only

## Summary

Phase 2 should add one dependency-free validation module and route the existing
pharma entry points through it. The current pipeline has no single admission
boundary: `src.data.load_config()` returns unvalidated YAML, the runner mutates
that dictionary, and data, labels, patches, training, RF, and foundation-model
paths each discover failure independently. Three cohort helpers also silently
drop missing slides. This makes the run that was actually evaluated impossible
to reconstruct from configuration alone.

The least disruptive design is:

1. Keep `src.data.load_config()` and its `dict[str, Any]` return type public.
2. Add `src.validation` with strict schema resolution, a shared exception
   hierarchy, immutable admission records, deterministic JSON conversion, and
   reusable non-empty guards.
3. Have `load_config()` parse YAML and delegate to `resolve_config()`, then
   return a fresh plain dictionary for compatibility.
4. In `scripts/run_pipeline.py`, apply environment overrides, revalidate once,
   and call `admit_run()` before output-directory creation, download, model
   loading, or training.
5. Pass the admitted slide sequence through every stage. Remove the three
   `FileNotFoundError`-and-continue paths; partial execution is allowed only by
   the new explicit `cohort_policy.allow_partial` configuration flag and every
   exclusion is represented in the manifest.

Do not add Pydantic, Hydra, JSON Schema, or another runtime dependency. Frozen
standard-library dataclasses plus explicit validators fit the small, stable YAML
surface and preserve the repository's lightweight import behavior.

## Existing Runtime Flow and Failure Seams

### Configuration and CLI

- `projects/spatial-pharma-dl/src/data.py:19-22` opens YAML and returns
  `yaml.safe_load()` directly. Empty YAML can return `None`; unknown keys,
  missing sections, wrong scalar types, and invalid ranges fail later as
  `KeyError`, `TypeError`, or library errors.
- `projects/spatial-pharma-dl/scripts/run_pipeline.py:52-59` loads configuration,
  mutates `foundation.enabled` from `PHARMA_FOUNDATION`, and derives the cohort.
  `PHARMA_QUICK` mutates training epochs and patience later at lines 102-105.
  Both overrides must be applied before the final startup validation.
- The runner creates output/data directories indirectly through current helper
  calls. Admission must run before those side effects if it is to be a true
  startup boundary.
- `src.__init__` lazily exports `load_config`, `cohort_slide_ids`,
  `preprocess_cohort`, `train_loso`, and the benchmark entry points. These names
  and their current call forms must remain valid.

### Cohort discovery and silent partial execution

- `src.data.cohort_slide_ids()` concatenates exactly `oncology`, `external`,
  and `benchmark` (`data.py:25-29`) without validating emptiness, duplicates,
  element types, or overlap.
- `src.data.preprocess_cohort()` uses `sample_ids or cohort_slide_ids(cfg)`
  (`data.py:132-154`). An explicitly empty list is therefore replaced by the
  configured cohort rather than rejected. This truthiness pattern must become
  `if sample_ids is None` throughout Phase 2 boundaries.
- `src.data.cohort_summary()` catches `FileNotFoundError` and silently continues
  (`data.py:157-177`).
- `src.labels.build_labels_cohort()` catches the same error, prints a skip, and
  can return an untyped empty `DataFrame` (`labels.py:163-199`).
- `src.patches.build_patch_cohort()` also prints and continues
  (`patches.py:229-249`). `fit_reference_stain()` silently scans past absent
  slides before raising only when all are absent (`patches.py:189-203`).
- `run_pipeline.py:156` prints `Pipeline complete` even when those helpers
  omitted configured slides.

### Empty-input low-level failures

- `_extract_spot_patches()` calls `np.stack(patches)` with no spot check
  (`patches.py:143-176`).
- `SpotPatchDataset` uses an `assert` only for length equality and accepts zero
  rows (`patches.py:263-291`). Assertions are not a public runtime contract.
- `load_slide_patches()` can return an empty inner merge and duplicate-index
  behavior (`train.py:41-49`). Phase 2 should reject only the empty result;
  key completeness and uniqueness belong to VAL-02 in Phase 3.
- `loso_folds([])` returns no folds, and `train_loso()` consequently returns an
  apparently successful empty result (`train.py:37-38,203-215`). One slide
  produces a fold with an empty training set.
- `train_one_fold()` and `run_rf_loso_fold()` concatenate lists without checking
  them (`train.py:68-76`; `benchmark.py:41-54`).
- `predict_cnn()` reaches `np.concatenate([])` for an empty patch batch
  (`eval.py:30-46`).
- `regression_columns()` can return an empty selection; CNN, RF, and foundation
  consumers handle this inconsistently (`labels.py:89-103`, `train.py:82-84`,
  `eval.py:128-156`, `foundation.py:223-280`).
- `extract_frozen_embeddings()` already rejects an empty array, but with a
  standalone `ValueError` after the encoder may have been loaded
  (`foundation.py:142-163`). `run_foundation_loso()` loads the encoder before
  validating the slide/fold shape (`foundation.py:283-302`).
- `foundation_eval.prepare_classification_task()` can filter a slide to zero
  rows and nested LOSO later fails in scikit-learn (`foundation_eval.py:73-86`).

## Recommended Module and Public Compatibility

Create `projects/spatial-pharma-dl/src/validation.py`. Keep it import-light:
standard-library `collections.abc`, `dataclasses`, `json`, `math`, and `pathlib`
only. It must not import Scanpy, Squidpy, Torch, torchvision, or foundation
backends.

Recommended public objects:

```python
class PharmaValidationError(ValueError): ...
class ConfigValidationError(PharmaValidationError): ...
class CohortAdmissionError(PharmaValidationError): ...
class StageValidationError(PharmaValidationError): ...

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    received: object
    expected: str
    guidance: str

@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    canonical_json: str
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True, slots=True)
class SlideAdmission:
    slide_id: str
    cohort: str
    status: str
    reason_code: str | None = None
    reason: str | None = None

@dataclass(frozen=True, slots=True)
class CohortManifest:
    schema_version: str
    allow_partial: bool
    configured: tuple[SlideAdmission, ...]
    included: tuple[SlideAdmission, ...]
    skipped: tuple[SlideAdmission, ...]
    failed: tuple[SlideAdmission, ...]
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True, slots=True)
class AdmittedRun:
    config: ResolvedConfig
    manifest: CohortManifest
```

`PharmaValidationError` should subclass `ValueError` so callers already catching
invalid-argument failures remain compatible. Aggregate configuration problems
as ordered `ValidationIssue` values and render one exception whose lines include
the dotted path, `repr(received)`, expected constraint, and correction. Cohort
and stage exceptions should expose structured attributes (`stage`, `observed`,
`minimum`, `manifest`) as well as the actionable message.

Do not make `load_config()` return `ResolvedConfig`; notebooks and modules use
ordinary nested dictionary operations, and the runner intentionally applies
environment overrides. Instead:

```python
def load_config(path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(...)
    return resolve_config(raw).to_dict()
```

`resolve_config()` may be public from `src.validation`; `load_config` remains
the compatibility facade. `run_pipeline.main()` should mutate the fresh dict for
environment overrides, then call `resolve_config(cfg).to_dict()` once more and
immediately admit the run. There is no need to add new names to `src.__init__`
in Phase 2, although doing so additively is safe.

## Configuration Contract

### Defaults and unknown keys

Add one section to `configs/default.yaml`:

```yaml
cohort_policy:
  allow_partial: false
```

This is the only new behavior flag needed by Phase 2. It is explicit, defaults
fail-closed, and does not mix policy scalars into the three cohort lists.

Use a checked-in nested defaults constant only for fields that are already
optional in production code (`seed`, `experiment`, `patches.version`,
`patches.context_scale`, `patches.per_slide_stain_norm`, training model/device/
pretrained/augment, foundation fields, and `cohort_policy.allow_partial`). Do
not fill absent required scientific sections wholesale from `default.yaml`, or
tests for missing required sections would be meaningless. Merge allowed defaults
first, then validate. Reject unknown keys at every known mapping level in one
aggregate pass. User-defined names under `marker_genes` and `gene_modules` are
the exception; validate their values rather than treating names as unknown.

Strict type checks must reject Python's `bool` where an integer/float is
expected (`bool` is an `int` subclass). Do not coerce strings such as `"32"` or
`"false"`; permissive coercion hides YAML mistakes.

### Exact validation surface

| Path | Contract |
|---|---|
| `seed` | integer `>= 0` |
| `experiment` | non-empty string safe to represent in an output filename |
| `cohorts` | required mapping with exactly `oncology`, `external`, `benchmark` |
| each cohort list | list of non-empty strings; no duplicate ID within or across lists |
| combined cohort | at least one slide; `oncology` at least two slides because the public benchmark is LOSO over oncology |
| `cohort_policy.allow_partial` | boolean, default `false` |
| `preprocessing.min_counts`, `min_cells`, `n_top_genes_hvg`, `n_pcs`, `n_neighbors`, `n_pcs_neighbors` | positive integers |
| `preprocessing.max_pct_mito` | finite number in `[0, 100]` |
| `preprocessing.leiden_resolution` | finite number `> 0` |
| preprocessing cross-field | `n_pcs_neighbors <= n_pcs`; post-QC feasibility remains Phase 3 |
| `labels.classification_col` | non-empty string; current default remains `tme_class_id` |
| `labels.regression_targets` | one of `modules`, `genes`, `both` |
| `labels.tme_classes` | non-empty unique string list containing `other` |
| `marker_genes` | non-empty mapping of non-empty panel names to non-empty unique gene-string lists; `breast` required because it is the runtime fallback |
| `gene_modules` | non-empty mapping of non-empty module names to at least two unique gene strings |
| `patches.version` | non-empty string |
| `patches.context_scale` | finite number `> 0` |
| `patches.min_patch_px`, `patches.output_size` | positive integers (minimum 2 is appropriate for image operations) |
| `patches.per_slide_stain_norm` | boolean |
| `training.model` | one of the checked-in model registry names from `src.models` (`resnet18`, `resnet50`, `efficientnet_b0`, `convnext_tiny`, `vit_b_16`) copied as a validation constant to avoid importing torchvision |
| `training.device` | `auto`, `cpu`, `cuda`, or `mps` |
| `training.pretrained`, `training.augment` | booleans |
| `training.batch_size`, `epochs`, `patience` | positive integers |
| `training.num_workers` | integer `>= 0` |
| `training.lr` | finite number `> 0` |
| `training.weight_decay`, `cls_weight`, `reg_weight` | finite number `>= 0`; `cls_weight` and `reg_weight` may not both be zero |
| `foundation.enabled`, `foundation.cache` | booleans |
| `foundation.model` | one of the checked-in registry keys (`kaiko_vits16`, `phikon`) |
| `foundation.device` | `auto`, `cpu`, `cuda`, or `mps` |
| `foundation.batch_size` | positive integer |
| `evaluation.primary_metrics.classification` | non-empty unique list drawn from `balanced_accuracy`, `macro_f1`, `accuracy` |
| `evaluation.primary_metrics.regression` | non-empty unique list drawn from `pearson_r`, `r2`, `mae` |

Referenced filesystem paths are not present in the current YAML. The validator
should support `Path` only when future/additive keys declare a path; it must not
invent a local path requirement for public Squidpy dataset IDs. Source/cache
availability belongs to cohort admission, not schema validation.

Canonicalization uses `json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)`. Convert tuples to lists and `Path` values
to POSIX strings at the boundary; reject sets, arbitrary objects, non-finite
floats, and non-string mapping keys. Preserve cohort-list order because it
determines stable fold/report order; sort mapping keys only. `to_dict()` should
round-trip through `json.loads(canonical_json)`, yielding a fresh mutable dict
for existing callers while the admitted representation remains immutable.

## Cohort Admission and Manifest Semantics

### Admission API

Use a pure, injectable availability seam:

```python
def admit_run(
    cfg: Mapping[str, Any],
    *,
    available_slide_ids: Collection[str] | None = None,
    failures: Mapping[str, str] | None = None,
) -> AdmittedRun: ...
```

`available_slide_ids=None` means source availability is not yet locally
knowable; all validated configured slides are admitted as pending/included.
Local-cache stages must pass the set discovered by a complete preflight scan.
Tests can pass an in-memory set and require no filesystem or network. A helper
such as `processed_slide_ids(sample_ids)` may derive that set using the same
`safe_filename()` path convention without opening H5AD files.

The full pipeline has two legitimate admission modes:

1. **Normal curation run:** configuration/cohort shape is admitted before the
   first download. `preprocess_cohort` remains fail-fast by default. If a source
   load fails, strict mode raises `CohortAdmissionError` and does not print
   completion; partial mode records the failure/exclusion and continues.
2. **Train-only/cache-backed run:** scan every configured processed-slide and
   patch path before label generation, stain fitting, model creation, or encoder
   loading. Strict mode reports all missing IDs together. Partial mode admits
   only IDs whose required local stage inputs exist.

When a caller supplies a batch of known availability, complete the scan before
processing any member. This satisfies D-03 for locally knowable missing inputs
and prevents the current “process some, discover a later miss” behavior. Remote
dataset existence cannot be proven without making the network request; do not
mislabel a syntactically valid Squidpy ID as available. Record loader failures
when they occur and keep the distinction explicit in error text and manifest.

`preprocess_cohort`, `cohort_summary`, `build_labels_cohort`,
`fit_reference_stain`, and `build_patch_cohort` should accept the admitted slide
sequence (plain list for compatibility) and must no longer catch-and-continue by
default. Partial filtering happens once, at admission, not independently in
each stage. This prevents later stages from resolving different cohorts.

### Manifest schema

The deterministic in-memory Phase 2 manifest should serialize as:

```json
{
  "schema_version": "cohort-manifest-v1",
  "allow_partial": false,
  "configured": [
    {"slide_id": "...", "cohort": "oncology", "status": "configured",
     "reason_code": null, "reason": null}
  ],
  "included": [
    {"slide_id": "...", "cohort": "oncology", "status": "included",
     "reason_code": null, "reason": null}
  ],
  "skipped": [],
  "failed": []
}
```

Rules:

- `configured` contains every unique slide in configuration order with its
  owning cohort.
- `included` contains only slides used by every downstream stage, in the same
  relative order.
- In explicit partial mode, known unavailable slides go to `skipped` with a
  stable code such as `missing_processed_slide`, `missing_patch_cache`, or
  `source_load_failed`, plus concise correction guidance.
- `failed` represents attempted operations that failed. In strict mode the
  raised exception carries a manifest with all known missing/failed members;
  no `AdmittedRun` is returned. In partial mode a source loader failure may be
  represented in both `failed` (attempt evidence) and `skipped` (admission
  consequence); records are independent, not a lossy status overwrite.
- A partial admission still fails if `included` is empty or if the oncology
  subset no longer has enough slides for the requested LOSO benchmark. Generic
  non-empty enforcement is Phase 2; detailed class/fold viability is Phase 6.
- Do not include wall-clock timestamps, host paths, unordered sets, exception
  reprs, or tracebacks in the canonical object. They make semantically identical
  admissions differ. Durable manifest location, atomic write, fingerprints,
  checksums, and completion markers are Phase 4. Phase 2 may write a human-visible
  JSON at the existing output boundary only after admission, but correctness
  must not depend on that non-atomic file yet.

## Empty-Boundary Contract

Implement one general guard and small semantic wrappers:

```python
def require_non_empty(
    value: Sized,
    *,
    stage: str,
    subject: str,
    minimum: int = 1,
    shape: tuple[int, ...] | None = None,
    guidance: str,
) -> None: ...
```

Messages should follow one stable form:

```text
<stage>: <subject> is empty (observed count=0, expected >=1). <guidance>
```

Use the following earliest public seams:

| Stage | Boundary and check | Deferred boundary |
|---|---|---|
| configuration/cohort | `cohort_slide_ids`: combined list non-empty; reject explicit `[]` rather than falling back | slide key uniqueness is config validation; class support is Phase 6 |
| fold creation | `loso_folds`/`train_loso`/`run_loso_benchmark`: at least two unique slides and non-empty labels before any model/cache access | class-count and unseen-class policy is Phase 6 |
| labels | `build_labels_for_slide`: returned frame non-empty; `build_labels_cohort`: at least one admitted frame and row | compound-key alignment is Phase 3 |
| patches | `_extract_spot_patches`: coordinates/spot list non-empty before loop/`np.stack`; `SpotPatchDataset`: rows and arrays non-empty with explicit exception | fixed border geometry and quality gate are Phase 8 |
| alignment | `load_slide_patches`: merged result non-empty before indexing | completeness, duplicates, cross-slide mismatch are Phase 3 |
| CNN/RF training | `train_one_fold` and `run_rf_loso_fold`: non-empty train slide list, each collected array/frame, concatenated train set, and held-out set before model construction | leakage-safe selection/scaling/imputation are Phase 7 |
| predictions | `predict_cnn`: positive `batch_size` and non-empty NCHW patch array before resolving device/model execution | output schema/integrity is later artifact work |
| regression targets | `regression_columns`: selected list non-empty for the supplied table; preserve the existing unknown-mode `ValueError` as a domain validation error | train-only scaling and missing masks are Phase 7 |
| foundation | `run_foundation_loso`: cohort/folds before `load_frozen_encoder`; `extract_frozen_embeddings`: shared error; `prepare_classification_task`: non-empty retained task rows | per-fold class support is Phase 6 |

Avoid over-validating Phase 3/6/7 behavior here. In particular, do not repair or
reorder label keys, fit any preprocessing object, require two observed classes,
or change early stopping. Phase 2 should establish only cardinality and admitted
membership.

## Offline Test Plan

Add Phase 2 tests under the existing primary `offline` marker. Reuse
`cohort_factory`, `fold_adversary_factory`, `tmp_path`, and monkeypatches. Do not
load Squidpy data, model weights, real H5AD caches, or repository outputs.

Recommended files:

- `tests/test_config_validation.py`
- `tests/test_cohort_admission.py`
- `tests/test_empty_boundaries.py`

### VAL-01 evidence

1. Default YAML resolves and canonical JSON round-trips to an equal fresh dict.
2. Reordered mapping keys produce byte-identical canonical JSON; cohort list
   order remains unchanged.
3. One malformed config containing missing sections, unknown root/nested keys,
   stringified numbers, invalid enum values, negative/zero ranges, duplicate
   slide IDs, `n_pcs_neighbors > n_pcs`, both training weights zero, missing
   `other`, and non-finite numbers raises one `ConfigValidationError` containing
   every dotted path and guidance.
4. Boolean values are rejected for numeric fields; arbitrary objects, sets,
   non-string keys, NaN, and infinity cannot enter canonical output.
5. The existing `load_config()` return remains a mutable plain dict, and two
   calls return independent nested objects.
6. Importing `src.validation` does not import Scanpy, Squidpy, Torch, torchvision,
   timm, or transformers.

### VAL-04 evidence

1. Given configured `slide_a`, `slide_b`, `slide_c` and availability
   `{slide_a, slide_c}`, default admission raises before an injected processing
   callback is invoked. The exception lists `slide_b` and carries a deterministic
   manifest.
2. With `cohort_policy.allow_partial=true`, the same input admits `slide_a` and
   `slide_c`, records `slide_b` under `skipped` with a stable reason, and never
   silently drops it.
3. Partial mode still raises when no slide is available or when the admitted
   oncology set cannot form a LOSO benchmark.
4. Duplicate IDs across cohort groups fail as configuration errors, so each
   manifest row has exactly one owning cohort.
5. Repeated admission of semantically identical config/availability produces
   byte-identical canonical config and manifest JSON.
6. `cohort_summary`, `build_labels_cohort`, and `build_patch_cohort` no longer
   catch and discard a monkeypatched `FileNotFoundError` in strict mode.

### VAL-03 evidence

1. Every `fold_adversary_factory()["empty"]` path fails with
   `StageValidationError` before a monkeypatched trainer/model/encoder/output
   helper runs.
2. One-slide LOSO fails with expected minimum 2 and identifies `fold_admission`.
3. Empty patch coordinates fail before `np.stack`; empty `SpotPatchDataset`
   fails without relying on `assert`.
4. Empty aligned rows fail before patch indexing (without asserting Phase 3's
   detailed mismatch diagnostics).
5. `predict_cnn` rejects an empty NCHW array before model forward/device work.
6. Module/gene/both selections that resolve to zero regression columns raise
   with the mode and corrective guidance in CNN, RF, and foundation seams.
7. Existing valid synthetic model/fold smoke remains unchanged, proving public
   call compatibility.

Run the canonical Phase 1 evidence command after focused tests:

```bash
python -m pytest -q projects/spatial-pharma-dl/tests/test_config_validation.py \
  projects/spatial-pharma-dl/tests/test_cohort_admission.py \
  projects/spatial-pharma-dl/tests/test_empty_boundaries.py
python scripts/verify.py fast
```

The focused tests are still subject to the Phase 1 offline socket guard and
strict primary-marker rule. All filesystem probes use `tmp_path`; injected
availability sets are preferred when no real path behavior is under test.

## Validation Architecture

| Requirement / decision | Implementation command or seam | Required evidence |
|---|---|---|
| VAL-01 / D-01 | `load_config -> resolve_config -> aggregate ValidationIssue[]` | One malformed config reports all dotted paths, received values, expected constraints, and guidance; no side-effect helper called |
| VAL-01 / D-02 | `ResolvedConfig.canonical_json` and `to_dict()` | Equal semantic mappings serialize identically; output is JSON-safe; existing dict consumers still work |
| VAL-04 / D-02 | `admit_run -> AdmittedRun(config, manifest)` | Included order and owning cohorts are stable; manifest round-trips through canonical JSON |
| VAL-04 / D-03 | Full availability preflight for cache-backed stages; explicit `cohort_policy.allow_partial` for exclusions | Strict missing members aggregate and stop before processing; partial mode records every skipped/failed member and still rejects an unusable admitted cohort |
| VAL-03 / D-04 | `require_non_empty` at cohort, fold, label, patch, alignment, prediction, and target-selection public boundaries | Each adversarial empty input raises `StageValidationError` with stage, observed count/shape, minimum, and corrective action before expensive work |
| D-05 compatibility | Preserve `load_config`, `cohort_slide_ids`, pipeline command, notebook order, output filenames, lazy exports, and config keys; add only `cohort_policy.allow_partial` | Existing Phase 1 tests plus `python scripts/verify.py fast` pass; valid callers receive ordinary dicts/lists and current output names remain unchanged |
| Phase boundary | No identity repair, adaptive PCA/HVG, artifact fingerprints/atomic writes, safe cache migration, fold class policy, leakage/scaling/imputation, or image/label changes | Review source diff and tests for absence of Phase 3-9 implementation |

The implementation order should be: (1) exception/types and canonicalization,
(2) strict config schema behind `load_config`, (3) cohort admission and runner
wiring, (4) shared empty guards at existing low-level failure seams, (5) offline
tests and the full fast gate. This order gives each later edit a validated config
and a common error vocabulary.

## Risks and Guardrails

- **Over-validating environment-specific availability:** do not require CUDA,
  model weights, network connectivity, or existing processed files merely to
  parse configuration. Availability is an injected admission concern.
- **Breaking config mutation:** never expose a nested `MappingProxyType` as the
  current `load_config()` result. The immutable representation belongs inside
  `AdmittedRun`; compatibility callers receive fresh plain dicts.
- **Nondeterministic manifests:** do not store timestamps, absolute temporary
  paths, exception reprs, unordered sets, or stack traces in canonical fields.
- **Silent partial behavior migrating downstream:** filtering must occur once at
  admission. Domain stages receive the included list and do not maintain their
  own skip policy.
- **Conflating empty with malformed identity:** Phase 2 may report zero aligned
  rows but must not implement partial-key diagnostics or merge repair; VAL-02 is
  Phase 3.
- **Conflating cohort viability with class viability:** require at least two
  admitted LOSO slides now; per-class fold support and unseen-class metrics are
  Phase 6.
- **Premature durable-manifest claims:** Phase 2 defines and exposes the schema.
  Atomic publication, fingerprints, checksums, and completed-artifact validation
  are Phase 4.
- **Heavy imports in validation:** copy small allowed-value constants rather than
  importing `models` or `foundation` registries, and add a parity test if useful.

## Planning Recommendation

Use three plans:

1. **Configuration contract:** implement exceptions, canonical JSON,
   `ResolvedConfig`, strict aggregate schema validation, default YAML policy,
   and config tests.
2. **Cohort admission:** implement manifest/admission types, strict and partial
   semantics, wire the runner and current silent-skip helpers, and add cohort
   tests.
3. **Empty boundaries:** add the shared stage guard to the identified public
   seams, add focused offline regressions, and run the complete Phase 1 fast
   verification gate.

This keeps each plan reviewable and maps every Phase 2 requirement to direct
behavioral evidence without taking ownership of later scientific or artifact
contracts.

---

*Research complete for Phase 2 planning.*
