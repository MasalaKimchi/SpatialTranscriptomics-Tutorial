# Technology Stack

**Project:** Spatial Transcriptomics Tutorial Reliability Upgrade
**Research type:** Stack for a subsequent brownfield reliability milestone
**Researched:** 2026-07-17
**Confidence:** HIGH

## Executive Recommendation

Keep the current Python 3.11 scientific stack and add reliability through small, explicit contracts around it. The safest path is not a framework migration: use standard-library dataclasses and validators for configuration, standard-library hashing and atomic replacement for artifact plumbing, NumPy/Parquet/JSON for data artifacts, PyTorch state dictionaries with restricted loading for model weights, scikit-learn pipelines for learned tabular preprocessing, and synthetic AnnData fixtures for offline tests.

One small development-only addition is justified: `pip-tools` to compile a hash-pinned Python 3.11 lock from the canonical dependency metadata. No runtime dependency is needed for typed configuration, atomic writes, fingerprints, provenance, or deterministic seeding. This keeps the milestone focused on the 20 active correctness and security requirements rather than introducing a packaging or storage redesign.

## Recommended Stack

### Core Runtime

| Technology | Version contract | Purpose | Why |
|------------|------------------|---------|-----|
| CPython | `3.11.*` for this milestone | Runtime and standard-library reliability primitives | The Conda environment already pins 3.11 and `pyproject.toml` already requires 3.11+. Selecting one tested minor removes the current 3.10/3.11 ambiguity without forcing scientific-package upgrades. |
| NumPy | Retain project floor `>=1.23`; lock an exact resolved version | Tensor-like patch artifacts and numerical tests | Existing code and artifacts already use NumPy. Numeric/string arrays can be loaded with `allow_pickle=False`, which avoids adding a new artifact library. |
| pandas + PyArrow | Unify on existing stricter floor `pyarrow>=14`; lock exact versions | Tabular metadata, label tables, patch indexes, manifests | Parquet already exists in the project and preserves typed columns without Python object deserialization. |
| PyTorch + torchvision | Retain current floors `torch>=2.0`, `torchvision>=0.15`; lock a tested pair | CNN training, deterministic policy, tensor checkpoints | Explicit `weights_only=True` is available on the supported API; saving a plain `state_dict` is PyTorch's recommended pattern. No model framework migration is warranted. |
| scikit-learn | Retain `>=1.2`; lock exact version | Train-only transformations, RF baseline, split orchestration, metrics | `Pipeline`, `SimpleImputer`, and `StandardScaler` directly solve the current leakage risks when fit only inside each outer fold. |
| AnnData / Scanpy | Retain current floors; lock exact versions | Spatial object model, preprocessing, fixture-backed integration tests | Synthetic `AnnData` can reproduce the relevant `X`, `obs`, `var`, `obsm`, and `uns` contracts without downloads. |
| PyYAML | Retain `>=6.0`; lock exact version | Parse the existing YAML surface | Preserve user-facing configuration. Continue `safe_load`, then immediately convert into validated typed dataclasses. |

### Standard-Library Reliability Layer

| Module/pattern | Purpose | Required use |
|----------------|---------|--------------|
| `dataclasses` with `frozen=True`, `slots=True` | Immutable resolved configuration sections | Define one dataclass per existing YAML section plus a root `ExperimentConfig`; construct through explicit parsing functions. |
| Explicit validators | Runtime type, range, enum, path, and cross-field checks | Do not rely on annotations alone: Python dataclasses do not validate annotated types. Collect all configuration errors and raise one actionable `ConfigError` before pipeline work. |
| `tempfile` + `os.replace` + `os.fsync` | Atomic artifact publication | Create the temporary file in the destination directory, fully write and close it, reopen/validate when practical, `fsync`, then `os.replace`. Clean stale temporary files safely. |
| `hashlib.sha256` | Artifact and input fingerprints | Hash canonical resolved configuration, schema/contract version, relevant source files, and source-data bytes or trusted upstream checksums. |
| `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)` | Canonical manifest encoding | Serialize only JSON-native scalar/list/map values; include schema version, fingerprint inputs, shapes, dtypes, row counts, checksums, and provenance. |
| `pathlib.Path` | Explicit path resolution | Resolve all configured paths relative to the repository/config root before hashing or I/O; avoid current-working-directory-dependent behavior. |
| `random`, `platform`, `importlib.metadata` | Seed and provenance capture | Record Python/platform/package versions and effective seed/determinism policy in every run manifest. |

### Development and CI

| Tool | Version contract | Purpose | Notes |
|------|------------------|---------|-------|
| pytest | Pin in the development lock | Unit, integration, scientific-regression, and notebook smoke tiers | Keep default tests CPU-only and offline; use markers for `integration`, `notebook`, `network`, and `slow`. |
| pip-tools | `7.5.x` development-only | Compile a reproducible, hash-pinned Python 3.11 environment | Generate a checked-in lock with `pip-compile --generate-hashes`; do not import it at runtime. |
| GitHub Actions | Pin actions by release tag initially, then SHA if repository policy requires | CI contract | One fast required job on Python 3.11; optional scheduled/manual notebook and network jobs. Use `setup-python` pip caching keyed from the lock. |
| Ruff | Existing repository tool/version in the lock | Static quality gate | Keep scope to Python source and scripts; avoid style-only notebook rewrites in this milestone. |
| nbformat + nbclient (or `jupyter execute` already available) | Pin in dev/full lock | Representative notebook execution | Execute only small synthetic/offline notebooks in required CI. Full public-data notebooks remain opt-in. |

## Prescribed Patterns for the 20 Improvements

### 1. Typed Configuration Validation

Use nested frozen dataclasses, not a new validation framework. Keep YAML keys and CLI behavior stable, but change `load_config()` to return `ExperimentConfig` after these stages:

1. `yaml.safe_load` and require a mapping, not `None` or a scalar.
2. Reject unknown keys at every section so typos cannot be silently ignored.
3. Parse values without permissive coercion (`"8"` is not an integer).
4. Validate local invariants: positive patch sizes, nonnegative worker counts, supported task/model names, probability ranges, nonempty slides and targets.
5. Validate cross-field invariants: neighbor PCs do not exceed computed PCs, class heads match configured labels, partial-cohort behavior is explicit, and deterministic policy values are supported.
6. Emit the fully resolved config as canonical JSON in the run manifest.

`dataclasses` supplies structure and immutability, but annotations are not runtime validation. Therefore constructors should be private to the parser or followed by explicit `validate()` functions. A domain-specific `ConfigError` should include dotted paths such as `training.batch_size: expected positive int, got 0` and report all detected errors in one pass.

### 2. Safe NumPy and Tabular Artifacts

Use a two-file patch-cache contract:

- `patches.npz`: only fixed-shape numeric arrays and fixed-width Unicode/byte arrays; always load with `allow_pickle=False` and a context manager.
- `patches.parquet`: spot/slide identifiers, quality flags, padding, tissue fraction, blur metrics, and other row metadata using an explicit ordered schema.
- `manifest.json`: schema version, fingerprint, array keys/shapes/dtypes, table columns/dtypes/row count, and SHA-256 for both payload files.

Validate that patch count equals metadata row count and that required keys, shapes, dtypes, finite-value rules, unique identifiers, and checksums match before accepting the cache. Existing pickle-backed NPZ caches should fail with regeneration guidance; do not silently invoke a legacy unsafe loader.

Use JSON, rather than NPZ object arrays, for small structured metadata. Use Parquet, rather than JSON, for row-oriented metadata whose schema and nullability matter. Preserve H5AD for AnnData rather than introducing Zarr in this milestone.

### 3. Safe PyTorch Checkpoints

Save a plain tensor `state_dict` to the checkpoint file and put non-tensor metadata in a validated JSON sidecar. Load with an explicit `torch.load(..., weights_only=True, map_location=...)`, even where newer PyTorch releases default to restricted loading. Then validate:

- metadata schema and model/task identifiers before model construction;
- exact or explicitly documented allowed state-dict keys;
- tensor shapes and dtypes against the constructed model;
- finite floating tensors where required;
- checkpoint and metadata SHA-256 values from the manifest.

Do not auto-fallback to `weights_only=False`. A separately named legacy conversion command may accept only explicitly trusted local files, emit a warning, and write the new safe contract; it must not run in the normal pipeline or CI.

### 4. Atomic Writes and Fingerprints

Create one shared artifact utility with `atomic_write(path, writer, validator)` and format-specific wrappers. The safe sequence is:

1. Create a uniquely named temporary file in `path.parent` so replacement stays on one filesystem.
2. Invoke the format writer on that exact file handle/path.
3. Flush and `os.fsync` the file descriptor where accessible.
4. Close, then validate the temporary artifact by reopening it.
5. Use `os.replace(temp, final)` to publish atomically.
6. For durability-sensitive manifests/checkpoints on POSIX, fsync the parent directory after replacement.

For multi-file artifacts, payloads are immutable fingerprint-named files; write them first, then atomically publish the small manifest last. A cache is complete only when the manifest exists and all referenced payload hashes/schema checks pass. This avoids pretending that several independent renames form a transaction.

Fingerprint content should be canonical and explicit:

```text
SHA256(
  artifact_schema_version
  + canonical_resolved_config_subset
  + ordered_source_file_sha256s
  + ordered_relevant_code_contract_sha256s
  + preprocessing/label/stain rule versions
)
```

Do not use Python's process-randomized `hash()`, file timestamps alone, or a manually bumped version as the cache key. Hash only the relevant config subset per artifact so unrelated display changes do not invalidate expensive caches.

### 5. Deterministic PyTorch Policy

Centralize reproducibility in one function returning a provenance record. It should seed Python, NumPy, `torch.manual_seed`, and CUDA when present; configure a seeded `torch.Generator`; and use the PyTorch-documented DataLoader `worker_init_fn` based on `torch.initial_seed() % 2**32` to seed NumPy and Python in every worker.

Expose a validated policy rather than a boolean:

- `strict`: `torch.use_deterministic_algorithms(True)`, deterministic backend settings, no silent fallback; intended for CI and reproducibility checks.
- `warn`: `torch.use_deterministic_algorithms(True, warn_only=True)` where platform limitations must be tolerated, with warnings captured in provenance.
- `off`: performance-oriented, but still seed all RNGs and record that deterministic algorithms were not required.

Record seed, generator seeds, policy, device, PyTorch/CUDA/cuDNN versions, and backend flags. Tests should assert repeatability on the same locked CPU stack, not bitwise identity across releases, platforms, or CPU/GPU; PyTorch explicitly does not guarantee the latter.

### 6. Leakage-Safe Evaluation

Treat slide ID as the indivisible group throughout evaluation.

For CNN LOSO:

- Outer split: hold out exactly one slide for final testing.
- Inner selection: choose validation slide(s) only from outer-training slides, deterministically by configured seed; never inspect the outer test slide during training or early stopping.
- Fit regression `StandardScaler` values on outer-training labels only, train in standardized target space, persist scaler parameters, and inverse-transform predictions before reports.
- Evaluate the outer slide once after epoch/model selection is frozen.

For RF/radiomics:

- Establish one fixed feature-name order before folds.
- Put `SimpleImputer`, any scaling, and the estimator in a scikit-learn `Pipeline`.
- Call `fit` only on outer-training rows; call `predict`/`transform` on held-out rows.
- Keep feature extraction deterministic and free of cohort statistics; cache per-slide raw features separately from learned preprocessing.

Before fitting each fold, validate nonempty partitions, unique aligned IDs, finite/masked targets, at least two training classes, per-class counts, and held-out classes absent from training. Unseen held-out classes should be reported as explicit coverage failures, not folded into an ordinary accuracy value.

### 7. Synthetic AnnData Testing

Build a small deterministic pytest fixture using `anndata.AnnData` directly, with no Scanpy dataset download:

- small count matrix `X` with known zero/low-quality rows and genes;
- unique string `obs_names` and `var_names`;
- `obs` columns for slide, array row/column, labels, and QC expectations;
- `obsm["spatial"]` with center and border coordinates;
- `uns["spatial"][library_id]["images"]` containing a tiny synthetic RGB image;
- matching `scalefactors` needed by patch extraction;
- controlled marker genes and target values for label/regression tests.

Use `tmp_path` for H5AD, NPZ, Parquet, JSON, and checkpoint round trips. Tests should cover malformed config aggregation, one-to-one alignment, border padding, Macenko fallback provenance, safe cache round trip, corruption detection, atomic-write interruption, deterministic CPU training, inner/outer split isolation, train-only imputation/scaling, missing slides, empty inputs, and adaptive post-QC dimensions.

Do not mock AnnData itself: real tiny objects exercise axis alignment and H5AD serialization cheaply. Mock only network access and heavyweight pretrained encoders.

### 8. CI and Dependency Contract

Make `pyproject.toml` the canonical list of direct dependencies and supported Python (`>=3.11,<3.12` for this milestone). Align `requirements.txt`, `environment.yml`, and `requirements-pharma.txt` to derive from or reference that contract rather than maintaining conflicting floors. In particular, resolve the PyArrow 12/14 mismatch and remove README support claims for Python 3.10 unless it becomes a tested target.

Generate a checked-in Python 3.11 Linux lock/constraints file with hashes using `pip-compile --generate-hashes`. Keep a human-maintained direct-dependency input and a machine-generated lock; never hand-edit transitive pins. Because PyTorch wheels can be platform/device specific, document the CPU CI index/source separately from user CUDA installation instructions and regenerate locks in a controlled job.

Recommended CI tiers:

| Tier | Trigger | Environment | Required checks |
|------|---------|-------------|-----------------|
| Fast | Every PR/push | Python 3.11, Ubuntu, CPU, locked deps | Ruff, unit tests, safe artifact tests, config tests, deterministic miniature fold, notebook JSON validation. |
| Integration | Every PR if runtime remains acceptable; otherwise merge/scheduled | Python 3.11, Ubuntu, CPU | Synthetic AnnData preprocessing, H5AD/cache round trips, label/patch alignment, RF/CNN miniature pipeline. |
| Notebook smoke | Merge/scheduled/manual | Python 3.11, Ubuntu, CPU | Execute representative offline notebooks with synthetic/local fixtures and bounded timeouts. |
| Full/network | Manual/scheduled only | Documented environment | Public dataset downloads, full tutorial or foundation-model checks; never required for ordinary PRs. |
| Dependency contract | PR when dependency files change + scheduled | Python 3.11 | Regenerate/check lock drift, install with hashes, import smoke test, minimum-direct-dependency compatibility job where practical. |

CI must disable downloads in required jobs and fail if code attempts network access. Cache downloaded packages, not generated scientific result artifacts. Upload test reports/logs as CI artifacts when failures occur.

## Supporting Design Choices by Requirement

| Active improvement | Primary stack/pattern |
|--------------------|-----------------------|
| Outer LOSO isolation | Slide-grouped outer loop plus training-only inner validation |
| Shared stain basis | Validated source and target stain matrices; cohort reference fit only from allowed source slides |
| Safe patch metadata | Numeric/string NPZ + Parquet + JSON manifest, `allow_pickle=False` |
| Safe checkpoints | Plain `state_dict`, JSON sidecar, explicit `weights_only=True` |
| Complete alignment | pandas merge validation, explicit uniqueness/completeness checks |
| Cache fingerprints | Canonical JSON + SHA-256 over relevant config/data/code contracts |
| Atomic writes | Same-directory temp file, validate, `os.replace`; manifest published last |
| Config validation | Frozen dataclasses + strict explicit parser and aggregate errors |
| Training determinism | Central seed policy, DataLoader generator/worker seeding, deterministic algorithms |
| Empty inputs | Shared precondition validators with domain-specific errors |
| Fold class support | Fold manifest with training/test counts and unseen-class reporting |
| Regression target scaling | Training-fold `StandardScaler`, persisted parameters, inverse transform for reports |
| RF leakage | Fixed feature schema + train-fitted sklearn `Pipeline` |
| Border context/quality | Fixed-size padding plus Parquet quality/provenance columns |
| Macenko robustness | dtype/range/rank/finiteness validation and explicit fallback flag |
| Missing slides | Validated `allow_partial` config plus cohort manifest; fail by default |
| Label confidence | Versioned JSON-native rule metadata, confidence and abstention columns |
| Adaptive preprocessing | Deterministic dimension resolver recorded in `adata.uns` and manifest |
| Test/CI tiers | pytest synthetic AnnData fixtures + GitHub Actions CPU/offline tiers |
| Dependency contract | Canonical `pyproject.toml` + hash-pinned Python 3.11 lock |

## Alternatives Considered

| Category | Recommended | Alternative | Why not now / when alternative fits |
|----------|-------------|-------------|------------------------------------|
| Config schema | Standard-library frozen dataclasses + explicit validation | Pydantic v2 | Pydantic gives richer automatic parsing, but adds a runtime dependency and permissive coercion must still be controlled. Adopt only if schema complexity grows beyond this fixed tutorial config. |
| Array cache | Numeric/string NPZ + Parquet metadata | Zarr | Zarr is useful for chunked, concurrent, out-of-core workloads, but this milestone explicitly excludes large storage migrations. |
| Weight artifact | PyTorch `state_dict` + restricted loader | safetensors | Safetensors is a strong future option for tensor-only interchange, but `weights_only=True` plus strict tensor/schema validation addresses the current code with no new runtime dependency. |
| Locking | pip-tools hash lock for Python 3.11 | Poetry, PDM, uv migration | Any could manage environments well, but migrating project/CLI workflows is unnecessary scope. Revisit in a packaging milestone. |
| Validation | Explicit exceptions and validators | Assertions | Assertions may be removed with optimization and are unsuitable for runtime data integrity. |
| Cross-validation | Explicit nested slide loops | Row-random `train_test_split` | Row-random splits leak slide/domain information and do not estimate held-out-slide generalization. |
| Atomic multi-file output | Immutable payloads + manifest-last commit marker | Rename each final file independently | Several renames are not a transaction; manifest-last gives a clear completeness boundary. |
| Synthetic tests | Real tiny AnnData objects | Mocked AnnData/Scanpy API | Mocks miss axis-alignment, H5AD, and spatial metadata contracts. |

## What Not to Use

| Avoid | Why | Use instead |
|-------|-----|-------------|
| `allow_pickle=True` for NPZ/NPY | Object arrays can execute arbitrary code and are not portable | Numeric/string arrays plus Parquet/JSON |
| Normal-path `torch.load(..., weights_only=False)` | General pickle loading can execute code | Plain state dict and explicit restricted loading |
| Python `pickle`, `joblib`, or whole sklearn-object persistence for externally supplied artifacts | These formats inherit pickle trust requirements | Persist primitive scaler/imputer parameters and schemas in JSON/NPZ; refit RF models when needed |
| Dataclass annotations without explicit checks | Dataclasses do not enforce annotated types | Strict parser and validators |
| `tempfile.mktemp` or a temp directory on another filesystem | Race-prone naming or non-atomic cross-filesystem moves | Secure same-directory temporary files and `os.replace` |
| File existence as cache validity | Interrupted/stale artifacts can exist | Manifest, fingerprint, checksum, schema, and shape validation |
| mtime-only fingerprints | Timestamps can be preserved, rounded, or changed without semantic relevance | Content SHA-256 plus explicit contract versions |
| Global preprocessing before splitting | Leaks held-out statistics | Fit every learned transform inside the outer-training fold |
| Test-slide early stopping | Uses the final evaluation domain for model choice | Training-only inner slide validation |
| Broad Python/dependency upgrade during reliability work | Multiplies behavior changes and obscures causal review | Lock the working 3.11 stack, then update separately |

## Version and Compatibility Contract

| Component | Contract | Verification |
|-----------|----------|--------------|
| Python | Exactly one supported minor for milestone: `3.11.*` | CI setup and README/pyproject/environment agreement |
| NumPy / pandas / SciPy / scikit-learn | Exact transitive versions in lock; direct floors in canonical metadata | Import smoke plus fast/integration tests |
| PyTorch / torchvision | Lock a known-compatible CPU CI pair; document separate accelerator installation | Deterministic CPU miniature training and checkpoint round trip |
| Scanpy / AnnData / Squidpy | Exact versions in full lock | Synthetic AnnData preprocessing and representative notebook smoke |
| PyArrow / pandas | Exact pair in lock; one direct PyArrow floor everywhere | Parquet schema/round-trip tests |
| Lock generation | Same Python minor and target platform as CI | CI checks that generated lock is unchanged and installs with hashes |

The lock is an environment-specific reproducibility artifact, not a universal claim that one wheel set covers Linux, macOS, CUDA, and Apple MPS. Keep accelerator guidance explicit and record the actual resolved package/device versions in run manifests.

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Preserve current scientific stack | HIGH | Directly grounded in the mapped repository and avoids unnecessary migration risk. |
| Safe NumPy artifact pattern | HIGH | NumPy explicitly documents pickle security risk and `allow_pickle=False`; current project already uses NPZ, Parquet, and JSON. |
| Safe PyTorch checkpoint pattern | HIGH | PyTorch official serialization guidance recommends state dictionaries and documents restricted `weights_only` loading. |
| Atomic-write pattern | HIGH on local filesystems | Python documents atomic successful replacement on POSIX; multi-file completeness still requires manifest-last semantics. Network filesystems need phase-specific validation. |
| Deterministic training pattern | HIGH for same locked CPU platform | PyTorch documents generator/worker seeding and deterministic algorithms, while warning against cross-release/platform guarantees. |
| Leakage-safe evaluation | HIGH | scikit-learn explicitly recommends split-first and pipelines; slide grouping follows the scientific unit of independence. |
| Synthetic AnnData fixtures | HIGH | AnnData officially supports direct construction of the exact aligned containers needed by the pipeline. |
| pip-tools lock approach | MEDIUM-HIGH | Official pip-tools supports hash generation and stable compiled outputs; platform-specific PyTorch/Conda resolution still requires documented contracts. |

## Open Questions for Phase Planning

- Which exact source files constitute each artifact's “relevant code contract” should be enumerated per cache type; hashing the entire repository would cause needless invalidation.
- Shared stain-reference fitting must be reconciled with outer-fold isolation: if the reference is estimated from image data, phase design must decide whether unlabeled held-out images are legitimately available at inference. The conservative evaluation default is training slides only.
- Confirm whether all supported PyTorch 2.0-era environments accept every desired restricted-loading validation behavior; the implementation should explicitly test the locked version rather than relying on newer defaults.
- Parent-directory fsync behavior and atomic replacement guarantees on users' network/cloud-synced filesystems may differ from local POSIX filesystems; document local-filesystem guarantees and test the supported CI filesystem.
- Decide whether minimum-direct-dependency CI is feasible for the full spatial stack. The required contract should prioritize the exact locked environment; a minimums job can be advisory if scientific dependencies do not jointly resolve at declared floors.

## Sources

- Python `dataclasses` documentation — annotations define fields but are not runtime type validation: https://docs.python.org/3/library/dataclasses.html
- Python `os.replace` documentation — same-filesystem replacement and POSIX atomicity: https://docs.python.org/3/library/os.html#os.replace
- Python `tempfile` documentation — securely created temporary files: https://docs.python.org/3/library/tempfile.html
- Python `hashlib` documentation — SHA-256 content digests: https://docs.python.org/3/library/hashlib.html
- NumPy `load` documentation — object-array pickle security warning and `allow_pickle=False`: https://numpy.org/doc/stable/reference/generated/numpy.load.html
- NumPy I/O guidance — safe NPY/NPZ use and pickle avoidance: https://numpy.org/doc/stable/user/how-to-io.html
- PyTorch serialization semantics — state-dict best practice and `weights_only` security model/limits: https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch reproducibility guidance — deterministic limits and DataLoader worker/generator seeding: https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn common pitfalls — split-before-preprocessing and Pipeline use to prevent leakage: https://scikit-learn.org/stable/common_pitfalls.html
- AnnData API — direct construction and aligned `obs`, `var`, `obsm`, and `uns` containers: https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html
- GitHub Actions Python guide — Python matrices, pytest, and setup-python dependency caching: https://docs.github.com/en/actions/tutorials/build-and-test-code/python
- pip-tools `pip-compile` documentation — reproducible compilation and `--generate-hashes`: https://pip-tools.readthedocs.io/en/stable/cli/pip-compile/

---
*Stack research for: reliability hardening of a Python spatial transcriptomics/computational pathology tutorial*
*Researched: 2026-07-17*
