# Research Summary

**Project:** Spatial Transcriptomics Tutorial Reliability Upgrade  
**Domain:** Scientific-software reliability for a notebook-first spatial transcriptomics and computational pathology tutorial  
**Researched:** 2026-07-17  
**Confidence:** HIGH

## Executive Summary

The repository should keep its current notebook-first teaching flow and pharma CLI/library surfaces, but add a thin reliability spine beneath them. The milestone is not a redesign: it is a focused correction of exactly 20 high-priority scientific, security, reproducibility, artifact, and validation weaknesses.

Three contracts should anchor the work:

1. A validated, immutable experiment configuration and explicit run/cohort manifest.
2. A fold context and fit-scope record that prove every learned quantity was fitted without outer-test influence.
3. Versioned safe artifact manifests that bind schema, fingerprints, checksums, lineage, and atomic publication.

Synthetic fixtures and tiered tests must be established first and expanded continuously. Input identity and cohort completeness then become explicit, followed by secure artifact migrations, leakage-free model evaluation, image and label trust, and finally a locked reproducible environment plus the full end-to-end gate. Unsafe legacy caches and checkpoints must fail closed with regeneration or explicit trusted-conversion guidance; they must never be loaded through silent compatibility fallbacks.

## Recommended Direction

Preserve notebook order, documented CLI entry points, existing configuration keys, output names, and public Python exports. Add small shared modules for configuration, validation, manifests/artifacts, reproducibility, and fold scope, while leaving biological preprocessing, patch generation, label engineering, training, and evaluation in their current domain modules.

The scientific invariant is stronger than ordinary train/test separation: outer LOSO test slides may not affect model selection, stain-reference fitting when fold-specific, target scaling, RF imputation, feature schema, thresholds, or any fitted state. Every fitted object must record its fitting slide IDs, and held-out perturbation tests must prove isolation.

## Exactly 20 Active Improvements

The milestone scope is fixed to the following requirements. Each appears once and all are required for completion.

| ID | Active improvement | Priority | Primary proof |
|----|--------------------|----------|---------------|
| R01 | Prevent outer LOSO test slides from influencing CNN model selection. | P0 | Outer-test perturbation cannot alter selection; the held-out slide is evaluated once after selection. |
| R02 | Normalize stains from each source slide into a shared cohort reference basis. | P0 | Distinct source/target matrices and an allowed fit cohort are recorded; cross-slide color distance decreases. |
| R03 | Replace pickle-backed patch cache metadata with safe serialization. | P0 | New caches load with `allow_pickle=False`; legacy object caches fail closed with regeneration guidance. |
| R04 | Load model checkpoints without enabling arbitrary pickle execution. | P0 | Normal loading is weights-only with validated metadata, keys, tensor shapes, and dtypes. |
| R05 | Enforce one-to-one, complete label/patch alignment with actionable errors. | P1 | Compound keys are unique and complete; duplicates and unmatched rows fail with counts and examples. |
| R06 | Fingerprint caches against configuration, source data, and relevant code contracts. | P1 | Relevant changes miss the cache; presentation-only changes do not; lineage is manifest-visible. |
| R07 | Make cache and result writes atomic and validate completed artifacts. | P1 | Interrupted or invalid writes are never accepted; a validated manifest is the commit marker. |
| R08 | Validate the complete experiment configuration before pipeline execution. | P1 | Unknown keys, bad types/ranges, missing sections, and cross-field conflicts fail together at startup. |
| R09 | Seed PyTorch, data loaders, workers, and deterministic backend policy centrally. | P1 | Repeated locked CPU runs agree within declared tolerances and record complete seed/backend provenance. |
| R10 | Reject empty cohorts, folds, patch sets, and prediction batches early. | P1 | Empty inputs fail at their boundary with stage identity, observed count, and corrective action. |
| R11 | Validate class support and unseen-class coverage for every LOSO fold. | P1 | Degenerate training folds fail; held-out unseen classes are reported separately from ordinary metrics. |
| R12 | Fit regression-target scaling only on training data and invert it for reports. | P1 | Held-out target shifts cannot alter scaler state; reported predictions return to original units. |
| R13 | Fit RF imputation and feature schema only on training data. | P1 | Held-out feature shifts cannot alter schema or imputer; missing/extra columns obey an explicit contract. |
| R14 | Preserve fixed physical context at image borders and record patch-quality flags. | P1 | Border patches are padded, not stretched; padding, tissue, blur, and artifact flags remain aligned. |
| R15 | Validate Macenko inputs and numerical outputs with explicit fallback provenance. | P1 | Invalid image/rank/tissue cases deterministically fail or fall back with reason and quality metadata. |
| R16 | Fail on missing configured slides unless partial-cohort mode is explicitly enabled. | P1 | Default execution reports all missing slides; partial mode fingerprints and records every exclusion. |
| R17 | Add confidence, abstention, and provenance to heuristic scientific labels. | P1 | Versioned exact rules emit evidence, confidence, abstention reason, and sensitivity/noise results. |
| R18 | Adapt preprocessing dimensions safely after QC and record resolved parameters. | P1 | Legal dimensions resolve deterministically; nonviable analyses fail; requested/resolved values are recorded. |
| R19 | Add fixture-backed unit, integration, notebook, and CI validation tiers. | P1 | Offline CPU gates cover contracts and representative execution; slow/network/full-cohort tiers are explicit. |
| R20 | Consolidate supported Python/dependency declarations and produce a reproducible environment contract. | P1 | One Python policy and exact reference environment agree across metadata, docs, notebooks, and CI. |

## Recommended Implementation Order

### Phase 1 — Verification and input contracts

1. **R19:** establish deterministic synthetic cohort, artifact-corruption, leakage-sentinel, and notebook fixtures; keep extending these through every phase.
2. **R08:** parse existing YAML into validated frozen dataclasses and emit canonical resolved configuration.
3. **R10:** add shared typed boundary errors and empty-input guards.
4. **R16:** resolve cohort membership explicitly and create the initial cohort manifest.
5. **R05:** enforce canonical `(slide_id, spot_id)` identity and complete one-to-one alignment.

This phase prevents later work from building on ambiguous identities, silently partial cohorts, or unvalidated dictionaries.

### Phase 2 — Secure and durable artifacts

6. **R06:** define schema-versioned manifests, canonical fingerprints, upstream lineage, and cache acceptance rules.
7. **R07:** add same-filesystem atomic writers, temporary validation, checksums, and manifest-last publication.
8. **R03:** migrate patch caches to safe numeric/string arrays plus Parquet/JSON metadata; reject legacy object NPZ.
9. **R04:** migrate checkpoints to tensor state dictionaries plus validated JSON metadata and restricted loading.

The shared manifest and atomic protocol should precede concrete cache/checkpoint migrations so there is only one format transition.

### Phase 3 — Leakage-free and reproducible learning

10. **R11:** validate fold viability, class counts, and unseen held-out coverage before training.
11. **R09:** centralize seeds, loader generators/workers, and strict/warn/off deterministic backend policy.
12. **R01:** introduce explicit outer `FoldContext`, training-only inner selection, fit lineage, and one-shot outer evaluation.
13. **R12:** add fold-scoped regression target scaling and inverse reporting.
14. **R13:** add fixed RF feature order and train-fitted imputation pipeline.

Fold identity must exist before learned transformations. Each fitted artifact records training slide IDs and is tested with extreme held-out sentinels.

### Phase 4 — Image and preprocessing reliability

15. **R15:** validate RGB/range/tissue/rank/numerical Macenko inputs and outputs with explicit policy and provenance.
16. **R02:** separate source estimation from a shared target fitted only from an allowed reference scope.
17. **R14:** preserve native physical crop extent at borders and carry synchronized padding and quality metadata.
18. **R18:** centralize post-QC dimension resolution and record requested/resolved parameters and viability decisions.

Validated stain estimation must precede claims of normalization. Image and preprocessing changes must update only their relevant artifact contract versions.

### Phase 5 — Scientific label trust

19. **R17:** replace implicit heuristic certainty with versioned exact gene sets, evidence, confidence, abstention, provenance, and sensitivity reporting.

Label abstention and filtering must trigger renewed alignment and fold-support validation, and reports must retain denominators and exclusion reasons.

### Phase 6 — Reproducible environment and release gate

20. **R20:** reconcile Python support and dependency declarations, compile a hash-pinned Python 3.11 reference environment, and verify both minimum-supported and locked environments.

Run the complete unit, integration, notebook-smoke, scientific-regression, artifact-security, and scheduled/manual full-cohort gates. Publish cache/checkpoint migration guidance and distinguish synthetic evidence from full biological validation.

## Major Architectural Boundaries

| Boundary | Owns | Must not own |
|----------|------|--------------|
| Configuration contract | YAML parsing, defaults, types, ranges, cross-field validation, immutable resolved config | Downloads, scientific transformations, or model construction |
| Run/cohort manifest | Resolved cohort, partial-mode decision, run identity, config/code/environment fingerprints, seeds, lineage | Domain-array serialization or metric computation |
| Validation layer | Reusable stage preconditions/postconditions and actionable typed errors | Silent repair, fitting, or hidden policy decisions |
| Artifact adapter | Safe schemas, canonical fingerprints, checksums, atomic publication, integrity validation | Domain feature engineering or pipeline orchestration |
| Cohort preprocessing | Input identity, QC, legal adaptive dimensions, per-slide transforms | Fold-specific learned model/RF/target state |
| Label contract | Compound sample identity, exact rules, confidence, abstention, provenance | Patch ordering or model selection |
| Image/patch contract | Coordinates, native crop geometry, padding, stain transforms, image QC, safe patch metadata | Label inference or fold selection |
| Split and fold context | Outer LOSO, inner training-only selection, group disjointness, fit lineage, class coverage | Model internals or report formatting |
| Fold preprocessing | Training-only scaler, feature schema, imputer, persisted transform state | Outer-test statistics or cross-fold mutable state |
| Training/evaluation | Optimization, checkpoint selection, one-shot held-out prediction, report-unit metrics | Cohort resolution or unsafe artifact I/O |

Orchestration owns sequencing and manifest updates. Domain modules own transformations. Artifact code owns bytes and integrity. Validation code checks contracts but does not mutate data. This separation is the minimum architecture needed; broad repackaging, notebook replacement, distributed training, and Zarr migration remain out of scope.

## Data and Leakage Boundaries

The pipeline should resolve state in this order:

```text
validated config
  -> resolved cohort + source identities
  -> per-slide QC/preprocessing
  -> labels and image patches with compound keys
  -> complete keyed alignment
  -> outer FoldContext
       -> inner training-only selection
       -> fold-fitted stain/scaler/imputer/schema state
       -> frozen model selection
       -> one final outer transform and evaluation
  -> predictions, metrics, and run summary with lineage
```

Any transform with a `fit` operation must receive an explicit allowed `FitScope` and persist `fit_slide_ids`. Slide or patient identity is indivisible for validation and selection; random spot-level validation is not acceptable evidence of generalization.

## Stack Decisions

- **Runtime:** standardize this milestone on CPython `3.11.*`; retain current scientific-library floors but lock an exact tested resolution.
- **Configuration:** use standard-library frozen/slotted dataclasses plus explicit strict validators; preserve YAML and current keys; reject permissive coercion and unknown keys.
- **Tabular artifacts:** use Parquet with `pyarrow>=14` for typed row metadata and JSON for small canonical manifests.
- **Array artifacts:** use NumPy only for numeric and fixed-width string arrays and always read with `allow_pickle=False`.
- **Model artifacts:** save plain PyTorch `state_dict` tensors and separate validated JSON metadata; load explicitly with `weights_only=True`.
- **Learned tabular preprocessing:** use scikit-learn `Pipeline`, `SimpleImputer`, and `StandardScaler`, fitted separately inside each outer fold.
- **Integrity and publication:** use `hashlib.sha256`, canonical JSON, `tempfile`, `fsync`, and `os.replace`; publish multi-file manifests last.
- **Testing:** use pytest with deterministic AnnData/image fixtures, `nbformat`/`nbclient` for representative notebook execution, and Ruff as the static gate.
- **Environment:** add `pip-tools` as a development-only dependency to compile a hash-pinned reference lock; do not add a runtime configuration or workflow framework.

Avoid Pydantic/Hydra, pickle/joblib artifacts, `safetensors` solely as a migration detour, Zarr, workflow orchestrators, spot-level split helpers, and automatic unsafe legacy fallbacks. These add migration cost or preserve the wrong trust boundary without improving the milestone's core value.

## Artifact Contract

Every accepted artifact should be bound to a manifest containing at least:

- schema and contract version;
- artifact type and producer version;
- canonical relevant configuration fingerprint;
- source and upstream artifact fingerprints;
- code-contract fingerprint;
- payload names, sizes, checksums, shapes, dtypes, and row counts;
- compound identity/schema information;
- fit scope where applicable;
- seed, determinism, environment, and hardware provenance where relevant;
- creation state and migration/regeneration guidance.

For multi-file artifacts, write immutable fingerprint-named payloads first and atomically publish the validated manifest last. Readers treat the manifest as the only completion marker and reject schema, checksum, lineage, or fingerprint mismatches. Legacy unsafe files are never opened merely to identify or convert them in a normal execution path.

## Testing Strategy

### Fixture design

Use a deterministic three- or four-slide synthetic cohort with disjoint compound spot keys, border and center coordinates, small RGB images, slide-specific color casts, one degenerate stain case, two supported classes, an optional unseen held-out class, multi-scale regression targets, missing masks, and post-QC dimensions small enough to exercise adaptation.

Generate invalid variants for duplicate/null/cross-slide keys, missing slides, empty stages, unsupported folds, unsafe object arrays, truncated payloads, stale fingerprints, incomplete manifests, malformed checkpoints, blank/saturated/grayscale/RGBA images, and interrupted writes.

### Required tiers

| Tier | Coverage | Cadence |
|------|----------|---------|
| Unit | Config, guards, identity/alignment, fingerprints, atomic writers, stain numerics, crop geometry, scaling/imputation, seeding | Every change; CPU/offline |
| Integration | Synthetic slide-to-artifact flow, invalidation, cache round trip, fold training/evaluation, manifest lineage | Every change where runtime permits |
| Notebook smoke | Representative root tutorial and pharma paths using fixture/mocked data from documented working directories | Pull requests or scheduled by runtime |
| Scientific regression | Held-out sentinels, group disjointness, stain convergence, determinism tolerance, class/abstention coverage | Pull requests and release gate |
| Security | No unsafe deserialization path; malicious/legacy/truncated artifact rejection and source scan | Pull requests and release gate |
| Full environment/cohort | Clean locked install, real downloads, all notebooks, optional encoders/GPU | Scheduled/manual with last-success date |

Synthetic tests establish software and scientific-boundary correctness, not biological validity. Release notes and manifests should state the evidence tier and date of the last schema-faithful or real-data run.

## Top Risks and Mitigations

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Indirect outer-test leakage remains after the obvious early-stopping fix | Invalidates claimed generalization while appearing nested | Create outer split first; require fit scopes; perturb held-out labels/features and compare fitted state and selection lineage |
| Spot-level pseudo-independence | Spatial neighbors and slide signatures inflate validation | Split all model-selection decisions by slide/patient and persist group assignments |
| Safe cache/checkpoint migration retains hidden pickle fallbacks | Untrusted artifacts can execute code | Fail closed; separate tensor/data payloads from JSON/Parquet metadata; make trusted conversion explicit and offline |
| Atomic files are structurally complete but semantically stale | Corrupt or incompatible results can be reused confidently | Validate production schemas, checksums, fingerprints, lineage, and commit markers before promotion and reuse |
| Shared stain normalization is fitted incorrectly or on invalid inputs | Produces plausible images without cohort harmonization and can leak test slides | Validate Macenko first; separate source/target operations; record fit cohort, quality, fallback, and convergence |
| Alignment repair silently drops or reorders samples | Labels and pixels can pair incorrectly even when counts match | Use canonical compound keys, one keyed table, complete merge diagnostics, and no incidental inner joins |
| Seeding is mistaken for determinism | Loader/backend nondeterminism defeats reproducibility claims | Centralize RNGs, workers, generators, backend policy, environment, and repeated-process tests |
| Border padding becomes a predictive shortcut | Fixes geometry while adding slide/location leakage | Preserve padding masks/fractions, test association with slide/target, and report quality-filter balance |
| Confidence turns heuristic labels into apparent ground truth | Overstates biological evidence and hides ambiguity | Version exact rules, distinguish abstention/missing evidence, run sensitivity/noise tests, limit claims |
| Dependency lock gives false portability | One solved machine can still diverge from documented and CI surfaces | Keep one support policy, test minimum and locked environments, record platform and solved environment |
| Synthetic fixtures create false confidence | Fixtures omit real tissue, sparsity, scale, and version behavior | Maintain scheduled schema-faithful/full-cohort runs and report evidence tiers separately |

## Scope Guardrails

- Do not redesign the 20-notebook narrative or replace notebooks with a workflow engine.
- Do not add biological datasets, efficacy claims, foundation-model fine-tuning, distributed training, or a Zarr migration.
- Do not rename the package or broadly remove path bootstrapping during this milestone.
- Do not retain unsafe readers for convenience or fit any learned state from outer-test data.
- Do not make default CI depend on network access, accelerators, or large model weights.
- Do not introduce improvements beyond R01–R20 until these requirements pass their declared gates.

## Completion Standard

The milestone is complete only when all 20 requirements have observable behavior, automated evidence proportional to risk, manifest-visible provenance, and migration notes where artifacts were invalidated. Passing unit tests alone is insufficient: representative notebook execution, leakage sentinels, artifact-security tests, locked-environment verification, and an explicitly dated full-cohort validation tier are part of the release evidence.

## Sources

- `.planning/PROJECT.md` — milestone purpose, constraints, fixed 20 active requirements, and compatibility decisions.
- `.planning/research/STACK.md` — technology choices, safe artifact formats, deterministic policy, and environment contract.
- `.planning/research/FEATURES.md` — requirement catalog, priorities, dependencies, acceptance behavior, and delivery waves.
- `.planning/research/ARCHITECTURE.md` — component boundaries, fit scope, artifact contracts, migration order, and traceability.
- `.planning/research/PITFALLS.md` — critical failure modes, warning signals, phase gates, and mitigations.

---
*Research synthesis for: Spatial Transcriptomics Tutorial Reliability Upgrade*  
*Researched: 2026-07-17*
