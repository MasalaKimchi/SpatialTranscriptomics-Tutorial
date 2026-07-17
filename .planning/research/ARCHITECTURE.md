---
title: Architecture Research
research_type: architecture
milestone: Spatial Transcriptomics Tutorial Reliability Upgrade
status: complete
date: 2026-07-17
source_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
---

# Architecture Research

## Research Question

What is the smallest architectural change that can deliver the 20 active reliability requirements without redesigning the notebook-first tutorial or breaking its supported CLI, configuration keys, outputs, and public Python exports?

## Executive Summary

Keep the current notebook and module topology. Add a thin reliability spine shared by the pharma CLI and notebooks: validated configuration, explicit cohort/run manifests, safe artifact adapters, fold-scoped preprocessing state, and reusable validation functions. Domain modules continue to own biology and modeling; orchestration owns sequencing and manifests; artifact code owns serialization and integrity; validation code owns preconditions but not transformations.

The critical architectural rule is that every learned quantity has an explicit fit scope. Cohort-level reference stain may use only inputs allowed by the experiment declaration; outer-test slides must never influence model selection, target scaling, RF imputation, feature schema, or stopping epoch. Within each outer LOSO fold, an inner split uses training slides only. The held-out slide is loaded for final transformation and one final evaluation only.

Secure cache and checkpoint changes are format migrations, not permissive compatibility shims. New readers reject legacy pickle-backed artifacts and instruct users to regenerate them. Atomic writes, manifests, fingerprints, and post-write validation must land before cache reuse is expanded. Synthetic fixtures and contract tests should land first so every later migration can prove both leakage isolation and artifact safety offline.

## Architectural Drivers

### Required qualities

1. **Scientific isolation:** learned preprocessing and selection state is fitted only on the declared training partition.
2. **Safe deserialization:** default artifact reads cannot construct arbitrary Python objects.
3. **Fail-fast execution:** invalid configuration, incomplete cohorts, misaligned identities, empty partitions, and unsupported classes stop before expensive work.
4. **Reproducibility:** resolved configuration, seeds, deterministic policy, environment, source identity, and artifact lineage are captured.
5. **Recoverability:** interrupted writes never masquerade as completed artifacts.
6. **Offline verifiability:** the default test suite exercises contracts with tiny synthetic data and no downloads or model weights.
7. **Compatibility:** notebooks, `scripts/run_pipeline.py`, existing config keys, and output names remain the user-facing surfaces.

### Explicit non-goals

- Renaming or repackaging the generic `src` package.
- Replacing notebooks with a service or workflow engine.
- Introducing Zarr, distributed training, or a new dataset.
- Preserving transparent reads of unsafe legacy pickle artifacts.
- Reworking the tutorial narrative or all notebook-generated source.

## Recommended Minimal Component Boundaries

The following are logical boundaries. They may begin as small modules within `projects/spatial-pharma-dl/src/`; no broad directory migration is required.

| Component | Owns | Must not own | Existing integration points |
|---|---|---|---|
| **Configuration contract** | YAML parsing, defaults, types, ranges, cross-field checks, immutable resolved config | File downloads, data transformations, model construction | Replace raw config handoff from `data.py`; called once by CLI/notebooks |
| **Run/cohort manifest** | Run ID, configured and resolved slides, partial-cohort decision, config/code/environment fingerprints, seed policy, artifact lineage | Serialization implementation of domain arrays; metric computation | Created by `run_pipeline.py`, extended at each stage |
| **Validation layer** | Reusable explicit precondition and postcondition checks; actionable typed errors | Silent repair, fitting, or scientific policy decisions | Called at stage entry/exit by `data.py`, `labels.py`, `patches.py`, `train.py`, `eval.py` |
| **Artifact adapter** | Safe schemas, fingerprint manifests, atomic same-filesystem writes, integrity checks, cache acceptance/rejection | Domain feature engineering; global orchestration | Used by H5AD, patch, label, checkpoint, feature, prediction, and report writers |
| **Cohort preprocessing** | Download/input identity, QC, dimension resolution, per-slide preprocessing, resolved parameters | Fold-specific model/RF/target transformations | Remains in `data.py` |
| **Label contract** | Spot identity, explicit gene sets, label-rule version, confidence, abstention, provenance, target columns | Patch row ordering, model selection | Evolves `labels.py`; emits schema-checked table |
| **Image/patch contract** | Coordinate mapping, fixed physical crop, border padding, source/target stain transform, QC flags, patch metadata | Label inference, fold selection | Evolves `patches.py`; emits safe arrays plus tabular/JSON metadata |
| **Split and fold context** | Outer LOSO split, inner training-only selection split, deterministic subsampling, class-support report | Model internals or metrics | Neutral public API consumed by `train.py`, `benchmark.py`, `foundation.py` |
| **Fold preprocessing** | Train-only target scaler; train-only RF feature schema and imputer; persisted transform state | Outer-test statistics; cross-fold mutable state | Constructed per fold by training/benchmark orchestration |
| **Model training** | Model creation, seeded loaders/workers, training, inner selection, weights-only state dict | Testing outer slide per epoch; report-scale inversion | Evolves `train.py` and `models.py` |
| **Evaluation/reporting** | One-shot outer prediction, inverse target transform, confidence/coverage metrics, report artifacts | Fitting imputers/scalers/schema or choosing epochs | Evolves `eval.py` and `benchmark.py` |
| **Test fixtures** | Tiny deterministic cohort, images, coordinates, labels, corrupt/legacy artifacts, minimal notebooks | Production data downloads | New test fixture layer under existing test tree |

Dependency direction should be one way:

```text
notebooks / run_pipeline.py
        |
        v
validated config + run manifest
        |
        v
data -> labels -> patches -> split context -> fold preprocessing -> train -> evaluate
        \________________ safe artifact + validation services ________________/
                                      |
                                      v
                               immutable manifests
```

Configuration, validation, manifests, and artifact adapters are lower-level services. They may not import training, evaluation, notebooks, or orchestration. Evaluation may consume saved fold state but must not fit it. This keeps leakage constraints visible in the dependency graph rather than dependent on call-site discipline.

## Data Flow and State Ownership

### 1. Startup and cohort resolution

1. CLI or notebook loads YAML through the configuration contract.
2. Validation resolves defaults and performs all type, range, enum, dimension, task, target, and cross-field checks.
3. A run manifest records the full resolved configuration, code revision, dependency versions, seed/determinism policy, command, and source-data identities.
4. Configured slides are resolved before processing. Missing slides fail by default. Explicit partial mode records configured, present, missing, and excluded slides and changes the cohort fingerprint.

No stage receives an unvalidated dictionary. Compatibility can be preserved by exposing a mapping view or serializing the typed configuration back to the existing nested shape.

### 2. Slide preprocessing

Each source slide is validated, QC-filtered, and checked for non-empty spots/genes. HVG, PCA, and neighbor dimensions are deterministically clamped to legal post-QC values. Both requested and resolved parameters are stored in `adata.uns` and the slide artifact manifest. A processed-slide fingerprint covers source identity, relevant resolved config, schema/code contract versions, and upstream dependencies.

### 3. Label derivation

Label generation emits a table keyed by `(slide_id, spot_id)` with a declared unique key. Scientific labels carry rule-set version, enrichment inputs/statistic, confidence, abstention reason, and provenance. Regression targets remain in report units at this stage. Alignment is a separate checked operation: both sides must have non-null unique keys, and complete one-to-one coverage is required unless an explicitly named filtering policy produces and records exclusions.

### 4. Patch creation and stain flow

Patch extraction always requests the same native physical extent. Border regions are padded rather than clipped and stretched. Metadata records original bounds, padding, tissue fraction, blur/artifact measurements, acceptance flag, and rejection reason.

Stain estimation has two distinct values and names:

- `source_stain`: estimated from the source slide or validated patch input.
- `target_stain`: the declared shared cohort reference basis.

Normalization decomposes using `source_stain` and reconstructs using `target_stain`. Macenko validates RGB shape, dtype/range, tissue count, finite values, rank, and matrix norms. Failure uses a configured deterministic fallback and records its reason and provenance; it never silently substitutes an invalid matrix.

The target reference's fit population must be declared in the experiment manifest. For strict LOSO scientific evaluation, the safest default is a fold-specific reference fitted from outer-training slides and then applied to the outer-test slide. If a fixed external/reference slide is used instead, it must be independent of all evaluated slides and identified by checksum. A reference fitted from all cohort slides is transductive and must not be presented as strict held-out evaluation.

### 5. Fold construction and training

For every outer held-out slide:

1. Create an immutable `FoldContext` containing outer-train slide IDs, outer-test slide ID, seed, and fingerprint.
2. Validate non-empty samples, targets, and class counts; fail on degenerate training support and explicitly record any class appearing only in outer test.
3. Split outer-training slides into inner-train and inner-validation partitions at slide level. With too few slides, use a predeclared training-only rule such as fixed epochs selected outside the fold; never fall back to the outer-test slide.
4. Fit target scaling only on outer-training data (or inner-training data while selecting, then refit according to declared policy). Persist means/scales, masks, and target order under the fold fingerprint.
5. Fit RF feature schema and imputer only on outer-training data. Test rows are reindexed and transformed without computing test-derived fill values.
6. Seed Python, NumPy, Torch CPU/CUDA, DataLoader generator, and workers from a centralized policy; record deterministic backend settings and exceptions.
7. Select epoch/hyperparameters using only the inner validation partition. Save weights-only model state plus safe JSON metadata.
8. Load the frozen selected state and evaluate the outer-test slide once. Invert regression scaling before reports and retain standardized-space diagnostics only as secondary metrics.

### 6. Reporting

Predictions retain `(run_id, fold_id, slide_id, spot_id)`, class/target schema, checkpoint fingerprint, transform-state fingerprint, and input artifact fingerprints. Reports aggregate only completed, validated fold artifacts. Empty prediction batches fail before concatenation. Coverage includes abstained labels and unseen-test classes; metrics do not silently discard them.

## Leakage Boundaries

Leakage prevention is a contract enforced at fit time. Every fittable object receives a `FitScope` derived from `FoldContext`, stores it in its manifest, and rejects data whose slide IDs fall outside that scope.

| Quantity or decision | Permitted fit data | Forbidden influence | Enforcement evidence |
|---|---|---|---|
| Outer fold model selection | Inner partitions drawn from outer-training slides | Outer-test loss, labels, predictions, or statistics | Fold context plus selection trace containing only allowed slide IDs |
| Training epoch/hyperparameters | Inner validation slides or predeclared external policy | Repeated outer-test evaluation | Outer-test prediction artifact absent until selected checkpoint is sealed |
| Regression target scaler | Outer-training target rows only | Outer-test mean, variance, missingness decisions | Persisted target order, fit slide IDs, parameters, fingerprint |
| RF schema and imputer | Outer-training radiomics only | Outer-test columns, validity frequency, means | Pipeline state and fit-scope manifest |
| Stain target | External fixed reference, or outer-training slides for strict LOSO | Cohort-wide target including held-out slide unless declared transductive | Reference-source IDs/checksum and evaluation-mode flag |
| Label-rule definition | Versioned rule declared before fold evaluation | Tuning rules against held-out performance | Rule version and provenance in label manifest |
| QC thresholds and dimension rules | Resolved configuration/predetermined deterministic rules | Ad hoc adjustment after viewing held-out outcomes | Resolved config and requested/resolved parameter record |
| Foundation probe selection | Inner training-only slide groups | Outer-test probe metrics | Existing nested LOSO pattern generalized through FoldContext |

Identity is the primary barrier: all fit/transform APIs accept tables or arrays with slide identity retained until after scope validation. Concatenating arrays and dropping slide IDs before preprocessing is prohibited. Subsampling occurs after fold assignment, deterministically, and records selected keys.

## Artifact Contracts

Every persisted artifact has a payload and a small safe manifest. The manifest uses JSON-compatible primitives, includes a schema version, and is validated before payload acceptance.

### Common manifest fields

- `artifact_type`, `schema_version`, `created_at`, `complete`
- `run_id`, optional `fold_id`, producing stage and code-contract version
- full upstream artifact fingerprints and relevant resolved-config fingerprint
- source file identities/checksums where applicable
- row/sample counts, shapes, dtypes, key schema, target/class ordering
- seed/determinism policy and producing package versions when relevant
- payload checksum and safe relative payload path
- fit scope for learned state; fallback/QC provenance for derived scientific artifacts

### Format recommendations

| Artifact | Payload | Metadata | Read acceptance |
|---|---|---|---|
| Processed slide | H5AD | JSON manifest plus required `adata.uns` fields | Fingerprint matches; required observations/variables and dimensions valid |
| Labels | Parquet or CSV with explicit scalar columns | JSON manifest | Unique `(slide_id, spot_id)`, declared schema, confidence/provenance present |
| Patches | NPZ numeric/string arrays with `allow_pickle=False`, or numeric array plus Parquet metadata | JSON manifest | No object dtype; aligned row counts; fixed shape; checksum valid |
| Stain reference | Numeric array in safe NPZ | JSON manifest | Shape/rank/norm/finite checks and source scope valid |
| Fold transforms | Numeric safe NPZ or JSON primitives | JSON manifest | Feature/target order and FitScope exactly match fold |
| Model checkpoint | Weights-only state dict, preferably safetensors if dependency policy permits | JSON manifest | `weights_only=True`; architecture/config allowlist; tensor names/shapes checked |
| Predictions/metrics | Parquet/CSV and JSON | JSON manifest | Complete keys, non-empty expected folds, finite/declared missing values |
| Run summary | JSON | Self-contained | References only complete validated child artifacts |

### Fingerprint composition

Fingerprints should be deterministic hashes of canonical serialized inputs, not timestamps. Include only relevant configuration subsections to avoid needless invalidation, but always include upstream fingerprints and an explicit code-contract/schema version. Source data identity should use content hashes where feasible; for large files, a documented size/mtime/content-sample strategy may be an interim compromise but must be labeled weaker.

### Atomic commit protocol

1. Validate in-memory payload and manifest candidate.
2. Write payload and manifest to unique temporary paths in the destination directory.
3. Flush and close; fsync important artifacts.
4. Reopen with the production safe reader and validate checksum/schema/shape.
5. Atomically replace payload, then atomically replace a final `complete` manifest as the commit marker.
6. Readers accept only artifacts with a complete manifest whose checksum and fingerprint match.

Never infer completeness from payload existence. Stale temporary files are ignored and may be cleaned separately.

### Security migration policy

- Patch cache schema v1 (object-valued NPZ) is rejected by the new reader without calling `np.load(..., allow_pickle=True)`. The error names regeneration instructions.
- Legacy pickle-capable model checkpoints are not auto-opened. Users regenerate them, or use an explicitly separate trusted-local conversion command outside normal pipeline reads.
- New formats use new schema versions and preferably distinct suffixes or manifest types, so downgrade/fallback cannot occur accidentally.
- Cache invalidation is expected and documented; compatibility means preserving commands and regenerated outputs, not preserving unsafe bytes.

## Validation Layers

Validation should be layered so failures occur at the earliest boundary with domain context.

### Layer 1: Configuration validation

Runs once at startup. Checks non-null YAML, required sections, types, enums, positive sizes, probabilities/ranges, task and class definitions, target uniqueness, PCA/neighbor relationships, stain policy, partial-cohort policy, deterministic policy, and artifact schema versions. Produces an immutable resolved configuration.

### Layer 2: Source and cohort validation

Checks configured slide uniqueness, presence, source identity, readable image/AnnData shape, coordinate columns, RGB image contract, and explicit partial-mode behavior. Produces the cohort manifest before transformation.

### Layer 3: Stage preconditions and postconditions

- Preprocessing: non-empty pre/post-QC dimensions; legal resolved HVG/PCA/neighbors.
- Labels: unique keys, known label schema, finite/declared-missing targets, confidence and provenance.
- Patches: fixed dimensions, row alignment, finite pixels, QC flags, safe dtypes, valid stain outputs.
- Alignment: one-to-one complete key match with exact missing/duplicate examples and counts.
- Folding: adequate number of slides, samples, class support, targets, and inner-split feasibility.
- Prediction: non-empty batches, output dimension/order match, finite values or explicit masks.

### Layer 4: Artifact validation

Safe reader validates schema before loading large payloads, then validates checksum, fingerprint, dtypes, shapes, keys, and upstream lineage. Cache mismatch causes regeneration, not warning-and-reuse.

### Layer 5: Scientific validation

Tests fit-scope isolation, fold class coverage, stain convergence/fallback behavior, inverse scaling, abstention reporting, and reproducibility tolerance. These are distinct from file/schema correctness.

Use explicit exceptions, never `assert`, for runtime data validation. Errors should include stage, slide/fold/run identity, expected versus observed values, representative offending keys, and recovery guidance.

## Test Fixture Architecture

Build fixtures before implementation changes. They should be small enough for fast CPU CI and rich enough to expose each reliability failure.

### Core synthetic cohort

Create three or four slides with deterministic IDs and disjoint spot keys. Each contains a tiny AnnData-like matrix (or real small AnnData when available), spatial coordinates including center and border spots, and a small RGB image. Encode:

- at least two supported classes in every valid outer-training set;
- one optional unseen class on a held-out slide;
- regression targets with intentionally different scales and missing masks;
- duplicated, null, missing, and cross-slide keys as opt-in invalid variants;
- slide-specific color casts and one degenerate stain image;
- tissue-rich, blank, blurred, and padded border patches;
- post-QC small dimensions that force safe HVG/PCA/neighbors adaptation.

### Artifact fixtures

- Valid schema-v2 patch cache with numeric/string dtypes.
- Legacy object-NPZ marker that must be rejected without deserialization.
- Truncated payload, mismatched checksum, incomplete manifest, stale fingerprint, and leftover temporary files.
- Valid weights-only checkpoint plus wrong architecture, wrong tensor shape, and legacy pickle checkpoint markers.
- Train/test feature tables where test means differ dramatically, proving imputation uses train state.
- Train/test target tables proving scale fit and inversion use train state only.

### Leakage sentinels

Instrument synthetic outer-test labels/features with extreme sentinel values. Fitted scaler, imputer, schema, stain reference, and selected epoch must remain unchanged when only sentinels change. Record accessed slide IDs in a spy `FitScope` and assert that outer-test IDs never appear in fit or selection traces. Assert the outer prediction function is called once after checkpoint selection.

### Notebook and pipeline fixtures

Maintain one minimal headless notebook per critical surface: root helper/cache path and pharma CLI/library path. Mock downloads and heavy encoders. Run a tiny end-to-end pipeline producing manifests, patch cache, labels, one fold checkpoint, predictions, and summary. Full cohort, network, GPU, and all-notebook runs remain opt-in tiers.

### CI tiers

| Tier | Contents | Default |
|---|---|---|
| Unit | config, validation, fingerprint, atomic writer, alignment, stain numerics, crop geometry, scaler/imputer, seed helpers | Every change, CPU/offline |
| Integration | synthetic slide-to-artifact flow, cache round trip/invalidation, fold train/evaluate, manifest lineage | Every change where runtime permits |
| Notebook smoke | selected notebooks headlessly with fixture/mocked data | Pull requests or scheduled based on runtime |
| Scientific regression | leakage sentinels, stain convergence, deterministic tolerance, class/abstention coverage | Pull requests and release gate |
| Full environment/cohort | locked environment, downloads, all notebooks, optional encoders/GPU | Scheduled/manual, not default |

## Build and Migration Order

The order below minimizes rework and keeps unsafe or scientifically invalid artifacts from being blessed by later infrastructure.

### Phase 1 — Test harness and core contracts

1. Add the synthetic cohort, artifact-corruption fixtures, leakage sentinels, and CI tier markers (requirement 19).
2. Add typed resolved configuration and startup validation while preserving existing config keys (requirement 8).
3. Add common explicit validation errors and fail-fast empty/cohort/fold/prediction checks (requirement 10).
4. Enforce complete one-to-one `(slide_id, spot_id)` alignment (requirement 5).
5. Resolve configured cohort membership and fail on missing slides unless explicit partial mode is recorded (requirement 16).

Rationale: subsequent changes need stable identities, legal inputs, and tests. Do not build fingerprints over unvalidated or silently partial state.

### Phase 2 — Secure artifact foundation

6. Introduce versioned manifests, canonical fingerprints, common safe readers, and cache acceptance rules (requirement 6).
7. Introduce atomic write/validate/commit helpers and migrate stage writers (requirement 7).
8. Migrate patch metadata away from object arrays; reject legacy caches and regenerate (requirement 3).
9. Migrate model saves/loads to weights-only state plus validated metadata; reject legacy normal-path loads (requirement 4).

Rationale: format migrations should use the final manifest/fingerprint and atomic-write contract. Implementing them earlier would create a second migration.

### Phase 3 — Scientific isolation

10. Add `FoldContext`, inner slide-level selection, and one-shot outer evaluation (requirement 1).
11. Add fold class-support and unseen-test-class validation/reporting (requirement 11).
12. Add training-only regression scaling, masks, persisted state, and report-unit inversion (requirement 12).
13. Add fixed RF schema and train-fitted imputation pipeline (requirement 13).
14. Centralize complete Torch/DataLoader/backend seeding and record policy (requirement 9).

Rationale: fold identity precedes every train-only transform. Class feasibility precedes training. Determinism is added after stable split/state ownership so reproducibility tests compare the intended computation.

### Phase 4 — Image and biological input reliability

15. Separate source/target stain matrices and define strict-LOSO versus external reference scope (requirement 2).
16. Validate Macenko inputs/outputs and persist fallback provenance (requirement 15).
17. Pad fixed-native-extent border patches and record tissue/blur/artifact quality flags (requirement 14).
18. Add versioned explicit label rules with confidence, abstention, and provenance (requirement 17).
19. Adapt post-QC preprocessing dimensions deterministically and record requested/resolved values (requirement 18).

Rationale: safe artifact schemas already exist to carry the new stain, patch-QC, label, and preprocessing provenance. Stain source/target semantics must be corrected before fallback logic is finalized.

### Phase 5 — Reproducible environment and end-to-end gate

20. Consolidate supported Python/dependency declarations, generate a reproducible lock/environment contract, and test minimum plus locked environments (requirement 20).
21. Run unit, integration, notebook-smoke, scientific-regression, and artifact-security gates; regenerate invalidated caches and checkpoints; publish migration notes.

The numbered implementation steps include a final integration gate, while requirements 1–20 remain mapped exactly once. Environment consolidation comes last because code and test dependency needs must first stabilize, but CI should use a temporary known-good environment throughout development.

## Requirement-to-Architecture Traceability

| Requirement | Primary boundary | Required proof |
|---:|---|---|
| 1 | Split/FoldContext + training | Outer-test sentinel cannot alter selected checkpoint/epoch; evaluated once |
| 2 | Patch/stain + FitScope | Distinct source/target matrices; held-out exclusion or external checksum |
| 3 | Artifact adapter | `allow_pickle=False`; legacy cache rejected; safe round trip |
| 4 | Model artifact adapter | Weights-only load; metadata/tensor validation; legacy rejected |
| 5 | Label/alignment validation | Duplicate/null/unmatched variants fail with counts and keys |
| 6 | Manifest/fingerprint | Relevant config/source/contract change invalidates cache |
| 7 | Atomic writer | Interrupted/truncated/incomplete artifact never accepted |
| 8 | Configuration contract | Invalid YAML/types/ranges/cross-fields fail at startup |
| 9 | Reproducibility service | Repeated seeded CPU runs agree within declared tolerance |
| 10 | Stage validation | Empty cohort/fold/patch/prediction variants fail early |
| 11 | Fold validation/reporting | Degenerate train class fails; unseen test class is reported |
| 12 | Fold preprocessing | Extreme test targets do not alter scaler; inverse units restored |
| 13 | Fold preprocessing | Extreme test features do not alter schema/imputation |
| 14 | Patch contract | Border patch retains fixed extent; padding and QC recorded |
| 15 | Patch/stain validation | Invalid image/matrix produces explicit deterministic fallback provenance |
| 16 | Cohort manifest | Missing slide fails by default; partial run is explicit and fingerprinted |
| 17 | Label contract | Rule version, confidence, abstention, and provenance survive round trip |
| 18 | Cohort preprocessing | Tiny post-QC fixture resolves legal dimensions and records them |
| 19 | Test fixture/CI | Offline tiered tests cover contracts and representative notebook/pipeline path |
| 20 | Environment contract | Metadata agrees; minimum and locked environments install and run gates |

## Migration and Compatibility Notes

### User-visible stability

- Keep notebook ordering, CLI command, configuration keys, output directories, and public exports.
- Add fields to outputs/manifests; avoid renaming existing report columns unless ambiguity threatens correctness.
- Where typed configuration replaces dictionaries internally, expose the existing mapping shape at compatibility boundaries.

### Intentional invalidation

- Existing pickle-backed patch caches must be deleted/regenerated by documented command; never auto-convert during ordinary reads.
- Existing checkpoints saved with unrestricted pickle semantics must be retrained or explicitly converted only in a separately invoked trusted-local tool.
- Fingerprint introduction invalidates filename-only caches once. A migration notice should explain why and estimate regeneration cost.
- Changes to patch geometry, stain reference, QC flags, label rules, and preprocessing dimensions each bump their relevant contract version and invalidate only dependent downstream artifacts.

### Downstream invalidation graph

```text
resolved config / source identity
  -> processed slide
      -> labels --------------------\
      -> stain + patches ------------+-> aligned fold input
                                         -> fold transforms
                                         -> model / RF / probe
                                         -> predictions
                                         -> metrics / run summary
```

A label-rule change need not invalidate raw patches. A patch geometry/stain change must invalidate radiomics, embeddings, models, and predictions. A report-format-only change need not retrain. Encode these dependencies through upstream fingerprints instead of deleting directories broadly.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Typed config breaks notebook dictionary usage | Provide mapping-compatible access and snapshot resolved YAML in manifests |
| Too few slides for inner validation | Validate feasibility and require a declared training-only fixed-epoch/external policy; never use outer test |
| Fold-specific stain references increase compute/cache volume | Cache safe stain references by training-slide-set fingerprint |
| Atomic multi-file artifacts expose payload before manifest | Treat the complete manifest as sole commit marker; readers ignore orphan payloads |
| Security migration surprises users | Fail with exact regeneration guidance and document schema cutoff prominently |
| Deterministic Torch policy is unsupported for an operation/device | Fail or explicitly record approved nondeterministic exceptions; CI uses CPU reference path |
| Confidence/abstention changes sample counts | Report denominators, coverage, and exclusions; validate fold viability after abstention |
| Broad fingerprints cause excessive rebuilding | Hash relevant config subsets and explicit contract versions, while retaining full lineage |

## Recommended Decision Record

Adopt the reliability spine and migration sequence above. In particular, treat `FoldContext`/`FitScope`, safe artifact manifests, and validated configuration as the three foundational contracts. They allow the 20 improvements to remain independently testable while preventing the most damaging failure modes: hidden outer-test influence, unsafe object loading, stale or partial artifact reuse, and silent cohort/key loss.

The milestone should not absorb the P2 packaging and notebook-generation redesign. Once the reliability contracts are stable and the 20 requirements pass their fixture-backed gates, those maintainability changes can proceed without changing scientific semantics or artifact trust rules.
