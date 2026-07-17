# Phase 2: Validated Run and Cohort Admission - Pattern Mapping

**Mapped:** 2026-07-17  
**Requirements:** VAL-01, VAL-03, VAL-04  
**Scope:** Closest repository analogs and concrete integration seams for planning

## Recommended Shape

Add one import-light domain module, `projects/spatial-pharma-dl/src/validation.py`,
and preserve `src.data.load_config()` as the compatibility facade. The new module
should own only configuration resolution, canonical JSON, cohort admission, and
non-empty boundary errors. It should depend on the standard library, not on
Scanpy, Squidpy, Torch, torchvision, timm, transformers, pandas, or NumPy.

The closest repository pattern is not a single existing validation module. It is
a combination of:

- Phase 1's pure orchestration boundary in `scripts/verify.py`, where argument
  construction is deterministic and side effects occur only after validation;
- frozen value objects in `src/foundation_eval.py::ProbeCandidate` and
  `src.foundation::FoundationModelSpec`;
- the lazy public facade in `src/__init__.py` and ordinary-dictionary contract in
  `src.data.load_config()`;
- stable configured order in `src.train.loso_folds()` and Phase 1's
  `cohort_factory`;
- sorted JSON round trips in `tests/test_artifact_roundtrips.py`;
- actionable `ValueError` and `FileNotFoundError` messages already used at model,
  cache, and device boundaries.

Do not introduce Pydantic, Hydra, JSON Schema, a new package layout, or a second
test convention. Phase 1 already established strict `offline` markers,
`tmp_path`, deterministic fixtures, and `python scripts/verify.py fast` as the
repository contract.

## Closest Analogs

### Pure validation and orchestration

`scripts/verify.py:27-49` separates pure command construction from execution.
`build_commands()` validates the requested tier and returns deterministic plain
lists; `run_tier()` performs subprocess side effects only afterward. Phase 2
should mirror this separation:

```text
YAML parse -> resolve_config(raw) -> plain mutable dict for compatibility
environment overrides -> resolve_config(cfg) -> admit_run(resolved)
admitted slide list -> data/label/patch/train stages
```

`projects/spatial-pharma-dl/tests/test_verification_contract.py:199-210`
monkeypatches the side-effect function and proves failure stops later work. Use
the same testing technique for startup validation and strict cohort admission:
monkeypatch directory, loader, model, and writer seams to raise if called.

### Frozen records and deterministic names

`src/foundation_eval.py:31-42` uses a frozen dataclass whose derived `name`
property is deterministic and contains only primitive values. `src/foundation.py`
uses a frozen model-spec dataclass for an immutable registry. These are the
closest value-object patterns for `ValidationIssue`, `ResolvedConfig`,
`SlideAdmission`, `CohortManifest`, and `AdmittedRun`.

Use `@dataclass(frozen=True, slots=True)` for Phase 2 records. Store tuples rather
than mutable lists inside admitted records. Expose fresh JSON-derived dictionaries
at compatibility boundaries instead of `MappingProxyType`; the runner currently
mutates nested config values for environment overrides.

### Stable ordering and fresh fixtures

`src.train.loso_folds()` preserves input order and derives one held-out fold per
input slide. `tests/conftest.py::cohort_factory` and
`test_fixture_contracts.py:47-65` explicitly protect stable slide/fold order and
fresh, mutation-isolated returned objects. Cohort manifests should follow the same
rule: preserve configuration order for `configured` and the same relative order
for `included`, `skipped`, and `failed`; never iterate an availability set to
construct output records.

### JSON-safe serialization

`tests/test_artifact_roundtrips.py:62` writes a manifest with
`json.dumps(..., sort_keys=True)` and verifies primitive JSON behavior. The current
runner writes a summary with `json.dumps(summary, indent=2)` at
`scripts/run_pipeline.py:124-152`. Phase 2 should make the in-memory contract
stronger and deterministic:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
```

`ResolvedConfig.to_dict()` and `CohortManifest.to_dict()` should return fresh
plain objects by `json.loads(canonical_json)`. Keep durable/atomic publication
out of Phase 2; Phase 4 owns that artifact contract.

### Domain errors

The codebase already uses `ValueError` for invalid public choices
(`src.device`, `src.models`, `src.foundation`, `src.labels`) and
`FileNotFoundError` with corrective guidance for absent caches (`src.data:120-129`,
`src.patches:217-226`). The closest compatible hierarchy is therefore:

```python
class PharmaValidationError(ValueError): ...
class ConfigValidationError(PharmaValidationError): ...
class CohortAdmissionError(PharmaValidationError): ...
class StageValidationError(PharmaValidationError): ...
```

Subclasses should retain structured attributes as well as messages. Existing
callers catching `ValueError` continue to work, while tests can assert issues,
stage, observed count/shape, minimum, and manifest without parsing prose.

## Concrete Signatures

### New `src.validation` surface

```python
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
    @property
    def canonical_json(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

@dataclass(frozen=True, slots=True)
class AdmittedRun:
    config: ResolvedConfig
    manifest: CohortManifest
    @property
    def slide_ids(self) -> tuple[str, ...]: ...

def resolve_config(raw: Mapping[str, Any] | None) -> ResolvedConfig: ...

def admit_run(
    cfg: Mapping[str, Any],
    *,
    available_slide_ids: Collection[str] | None = None,
    failures: Mapping[str, str] | None = None,
) -> AdmittedRun: ...

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

`resolve_config()` should aggregate all issues in schema traversal order and
raise once. It must distinguish `bool` from numeric fields, reject coercion,
reject non-finite numbers and non-string mapping keys, merge only documented
optional defaults, reject unknown keys at known mappings, and preserve arbitrary
panel/module names only under `marker_genes` and `gene_modules`.

### Existing signatures to preserve

Keep these public call forms intact:

```python
def load_config(path: Path | None = None) -> dict[str, Any]: ...
def cohort_slide_ids(cfg: dict[str, Any] | None = None) -> list[str]: ...
def preprocess_cohort(
    sample_ids: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Path]: ...
def cohort_summary(sample_ids: list[str] | None = None) -> pd.DataFrame: ...
def build_labels_cohort(
    sample_ids: list[str], cfg: dict[str, Any] | None = None
) -> pd.DataFrame: ...
def fit_reference_stain(
    sample_ids: list[str], cfg: dict[str, Any] | None = None
) -> np.ndarray: ...
def build_patch_cohort(
    sample_ids: list[str],
    ref_stain: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray: ...
```

Internally change every `sample_ids = sample_ids or ...` to an explicit
`if sample_ids is None`; a supplied empty list is invalid input, not a request to
reload the configured cohort. Existing valid callers receive the same return
types and output names.

An additive internal helper can perform local preflight without reading H5AD:

```python
def processed_slide_ids(sample_ids: Collection[str]) -> set[str]:
    return {
        sid
        for sid in sample_ids
        if (pharma_processed_dir() / f"{safe_filename(sid)}_clustered.h5ad").exists()
    }
```

Avoid calling `pharma_processed_dir()` for a pure startup check because it creates
the directory. Prefer a non-creating path helper or inject availability from the
runner. The current creating accessor at `src.data:32-43` is a side effect.

## Startup Data Flow

### Normal curation run

1. Parse and resolve `default.yaml` through `load_config()`.
2. Apply `PHARMA_FOUNDATION` and `PHARMA_QUICK` overrides to the returned fresh
   dictionary. Today quick overrides happen at `run_pipeline.py:102-105`, after
   labels and patches; move them before the final resolution.
3. Call `resolve_config(cfg)` again after all overrides.
4. Call `admit_run(resolved.to_dict(), available_slide_ids=None)`. This validates
   configured membership but intentionally does not claim remote Squidpy
   availability.
5. Pass `list(admitted.slide_ids)` through preprocessing, summary, labels,
   patches, and benchmark stages. Do not independently re-expand cohorts later.
6. A remote source loader failure raises `CohortAdmissionError` in strict mode.
   In explicit partial mode, update the run admission once with reason code
   `source_load_failed`, then pass the revised included sequence downstream.

### Train-only/cache-backed run

1. Resolve overrides exactly as above.
2. Scan every configured processed-slide and required patch-cache path before
   label construction, stain fitting, model creation, or encoder loading.
3. Call `admit_run()` with the complete availability result. Strict mode raises
   one aggregate error before any member is processed. Partial mode returns one
   filtered slide sequence plus a complete manifest.
4. Derive `oncology` by filtering admitted records by their owning cohort; do not
   continue to use the original `cfg["cohorts"]["oncology"]` at
   `run_pipeline.py:57` after partial admission.

### Runner import boundary

The runner currently imports `benchmark`, `eval`, `labels`, and `patches` at
module import time (`run_pipeline.py:28-43`). Those imports transitively load
Torch, scikit-learn, image libraries, and foundation code before `main()` can
validate config. To satisfy “before expensive work,” keep only bootstrap and the
light config/admission imports at module scope, then import stage modules inside
`main()` after successful admission. `st.set_seeds()` also currently runs at
module import (`run_pipeline.py:24-26`); perform it after resolved seed validation.

## Stage Boundary Placement

Use one `StageValidationError` vocabulary, but place checks at the earliest
existing public seam. Do not repair Phase 3 identity issues or Phase 6/7 class and
leakage issues.

| Stage | Existing seam | Check before | Compatibility note |
|---|---|---|---|
| configured cohort | `data.cohort_slide_ids` | concatenating/returning an empty cohort | preserve list order and return type |
| preprocessing | `data.preprocess_cohort` | loop and any directory/download call | explicit `[]` must fail |
| cohort summary | `data.cohort_summary` | loop; fail missing members rather than catch/continue | return DataFrame on valid input |
| label slide | `labels.build_labels_for_slide` | return of a zero-row frame | do not change label identity semantics |
| label cohort | `labels.build_labels_cohort` | output-dir creation and concat | remove `FileNotFoundError` skip at lines 171-176 |
| regression targets | `labels.regression_columns` | returning zero selected columns | retain invalid-mode error, use shared subtype |
| spot patches | `patches._extract_spot_patches` | image work and `np.stack` | include coordinates/spot count and stage |
| patch dataset | `patches.SpotPatchDataset.__init__` | assigning zero/mismatched rows | replace runtime `assert` at line 279 |
| stain reference | `patches.fit_reference_stain` | scanning/estimating an empty admitted list | remove silent missing-slide scan policy |
| patch cohort | `patches.build_patch_cohort` | extraction/write loop | remove catch/continue at lines 240-245 |
| aligned slide | `train.load_slide_patches` | indexing patches after inner merge | only reject zero rows; Phase 3 owns mismatch detail |
| LOSO admission | `train.loso_folds`, `train.train_loso`, `benchmark.run_loso_benchmark` | fold generation, trainer, or cache access | require >=2 unique slides and non-empty labels |
| CNN fold | `train.train_one_fold` | device resolution, concatenation, model build | reject empty train list/member/held-out set |
| RF fold | `benchmark.run_rf_loso_fold` | concatenation and estimator construction | same stage vocabulary as CNN |
| prediction | `eval.predict_cnn` | device resolution and `model.eval()` | require positive batch size and non-empty NCHW input |
| embeddings | `foundation.extract_frozen_embeddings` | iteration/model forward | replace standalone `ValueError` with shared subtype |
| foundation LOSO | `foundation.run_foundation_loso` | `load_frozen_encoder()` at line 291 | validate cohort/folds/labels first |
| task filter | `foundation_eval.prepare_classification_task` | returning zero retained rows | preserve task enum behavior |

The minimum-two-slide LOSO rule belongs at orchestration boundaries as well as
the fold helper so direct callers fail consistently. Do not require two classes,
repair compound keys, scale targets, or change model selection in this phase.

## Files to Create

- `projects/spatial-pharma-dl/src/validation.py` — exceptions, frozen records,
  canonicalization, aggregate schema validation, admission, and empty guard.
- `projects/spatial-pharma-dl/tests/test_validation.py` — malformed aggregate,
  defaults, canonical JSON, fresh dict, and lightweight-import evidence.
- `projects/spatial-pharma-dl/tests/test_cohort_admission.py` — strict/partial
  availability, manifest order/reasons, fail-before-side-effect evidence.
- `projects/spatial-pharma-dl/tests/test_empty_boundaries.py` — empty cohort,
  fold, aligned rows, patches, predictions, targets, and foundation seams.

The validation strategy names these exact Wave 0 files. Do not create the
research document's earlier `test_config_validation.py` alias; use
`test_validation.py` so plan, validation matrix, and implementation agree.

## Files to Modify

- `projects/spatial-pharma-dl/configs/default.yaml` — add only
  `cohort_policy.allow_partial: false`.
- `projects/spatial-pharma-dl/src/data.py` — validate behind `load_config`, fix
  `None` semantics, centralize admitted membership, remove silent summary skip.
- `projects/spatial-pharma-dl/scripts/run_pipeline.py` — apply all overrides,
  re-resolve, admit, then import/run stages; propagate admitted slides.
- `projects/spatial-pharma-dl/src/labels.py` — non-empty targets/frames and no
  local partial-cohort policy.
- `projects/spatial-pharma-dl/src/patches.py` — non-empty extraction/dataset and
  no local missing-slide policy.
- `projects/spatial-pharma-dl/src/train.py` — fold, alignment, train/validation
  cardinality checks before device/model/output work.
- `projects/spatial-pharma-dl/src/benchmark.py` — reject empty benchmark/RF folds
  before cache/model work.
- `projects/spatial-pharma-dl/src/eval.py` — prediction and target cardinality
  checks before device/model/estimator work.
- `projects/spatial-pharma-dl/src/foundation.py` — validate before encoder load
  and reuse the shared empty error.
- `projects/spatial-pharma-dl/src/foundation_eval.py` — reject empty filtered
  task rows without changing Phase 6 class-support policy.
- `projects/spatial-pharma-dl/tests/conftest.py` — only add a config factory if it
  is broadly reused; keep all new tests on the existing `offline` tier.

No `src/__init__.py` change is required. Additive lazy exports are safe, but the
phase can import `src.validation` directly and avoid expanding the public API.
Do not rename existing exports or alter notebook imports.

## Compatibility Hazards

1. **`cfg or load_config()` conflates empty with absent.** This appears across
   modules. A malformed `{}` must reach validation, not silently reload defaults.
   Prefer `if cfg is None` in every touched boundary.
2. **`sample_ids or cohort_slide_ids()` silently replaces explicit empty input.**
   Replace with an explicit `None` branch before adding `require_non_empty`.
3. **Path accessors create directories.** `pharma_processed_dir()` and
   `pharma_outputs_dir()` call `mkdir`; tests proving startup failure precedes
   side effects must monkeypatch or avoid these functions until admission.
4. **Heavy imports precede `main()`.** Runner module imports transitively load
   ML/image stacks. Defer stage imports until after validation/admission.
5. **Environment overrides are split in time.** `PHARMA_FOUNDATION` is applied
   at startup, but `PHARMA_QUICK` is applied after patch work. Validate all
   overrides together before admission.
6. **Partial filtering can diverge by cohort.** If downstream helpers keep their
   own catches, labels, patches, stain reference, and benchmark can use different
   cohorts. Filter exactly once and remove catch-and-continue behavior.
7. **Original oncology list is unsafe after partial admission.** Derive the
   benchmark subset from admitted records or missing oncology slides can re-enter
   stain fitting and LOSO.
8. **Canonicalization must not reorder cohort lists.** Sort mapping keys only;
   list order is observable in folds, output rows, and reports.
9. **Mutable compatibility is required.** `load_config()` callers modify nested
   dictionaries. Return a fresh plain dict each time, while keeping the admitted
   representation immutable.
10. **YAML booleans are integers in Python's type hierarchy.** Numeric validators
    must explicitly reject `bool`; do not coerce strings or numbers.
11. **Unknown-key validation needs dynamic-map exceptions.** Panel and module
    names are user-defined; validate their values without rejecting their keys.
12. **No timestamp/path/traceback in canonical manifests.** Such values make
    semantically identical admissions differ and leak machine-local state.
13. **Reason codes must be stable.** Use codes such as
    `missing_processed_slide`, `missing_patch_cache`, and `source_load_failed`;
    keep human guidance in a separate field.
14. **A failed attempt and an exclusion are distinct facts.** In partial mode a
    source failure may appear in both `failed` and `skipped`; do not collapse the
    two manifest collections into one status.
15. **Do not claim durable manifest safety yet.** Phase 2 defines deterministic
    in-memory serialization. Atomic writes, checksums, completion markers, and
    cache fingerprints are Phase 4.
16. **Do not widen the scientific scope.** Zero-row alignment can fail here, but
    duplicate/missing identity diagnostics are Phase 3; class viability is Phase
    6; scaling, imputation, and leakage are Phase 7.
17. **Tests must not hit repository data/output paths.** Reuse Phase 1's
    `tmp_path`, `cohort_factory`, `fold_adversary_factory`, and monkeypatch style.
18. **Every new test needs exactly one primary marker.** The suite enforces this
    globally; set `pytestmark = pytest.mark.offline` in each new module.

## Suggested Plan Boundaries

### Plan 02-01 — Configuration contract

Create `src.validation`, implement exceptions, canonicalization, immutable
records, aggregate schema/default validation, `load_config()` delegation, the
new default policy key, and `tests/test_validation.py`. The must-have behavioral
proof is that one malformed config reports every issue and no side-effect seam is
called, while valid reordered mappings produce byte-identical canonical JSON and
fresh mutable dictionaries.

### Plan 02-02 — Cohort admission

Implement manifest/admission semantics, pure injected availability, strict and
partial policies, and runner wiring. Remove independent missing-slide skips from
summary/labels/patches. Add `tests/test_cohort_admission.py`. The must-have proof
is a complete preflight: strict mode aggregates all missing members before work;
partial mode records every configured/included/skipped/failed member and passes
one admitted sequence to all stages.

### Plan 02-03 — Empty boundaries

Add `require_non_empty` and place it at the tabled seams in labels, patches,
training, evaluation, RF, and foundation paths. Add
`tests/test_empty_boundaries.py`, preserve the Phase 1 model/fold smoke, and run
the complete fast gate. The must-have proof is that injected expensive helpers
are never reached for empty inputs and errors expose stable structured fields.

## Verification Commands

Focused commands should follow the already-established Phase 1 convention:

```bash
python -m pytest -q --strict-markers -m offline \
  projects/spatial-pharma-dl/tests/test_validation.py
python -m pytest -q --strict-markers -m offline \
  projects/spatial-pharma-dl/tests/test_cohort_admission.py
python -m pytest -q --strict-markers -m offline \
  projects/spatial-pharma-dl/tests/test_empty_boundaries.py
python scripts/verify.py fast
```

The last command remains the acceptance gate because it runs Ruff first and all
offline evidence second, preserving Phase 1's CPU/no-network contract.

---

*Pattern mapping complete for Phase 2 planning.*
