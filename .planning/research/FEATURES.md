# Feature Research

**Domain:** Reliability upgrade for a notebook-first spatial transcriptomics and computational pathology tutorial
**Researched:** 2026-07-17
**Confidence:** HIGH — classification is grounded in the mapped repository and the fixed Active requirements in `PROJECT.md`

## Research Boundary

This milestone contains exactly 20 high-priority improvements. The catalog below preserves the wording and order of those Active requirements while classifying each once. “Table stakes” are baseline scientific/software correctness, “differentiators” make reliability unusually visible and teachable, and “anti-features” identify unsafe behavior that must be removed rather than expanded.

## Feature Landscape

### Table Stakes (Users Expect These)

| ID | Capability | Why Expected | Complexity | Observable Acceptance Behavior |
|----|------------|--------------|------------|--------------------------------|
| R01 | Prevent outer LOSO test slides from influencing CNN model selection. | A held-out test domain must remain untouched until final evaluation. | HIGH | For every outer fold, logs/manifests identify a training-only selection split or stopping rule; changing outer-test labels or samples cannot change the chosen epoch/hyperparameters; the outer slide is evaluated once after selection. |
| R05 | Enforce one-to-one, complete label/patch alignment with actionable errors. | Silent row loss or duplication invalidates sample counts and targets. | MEDIUM | Valid unique keys align in a stable order; nulls, duplicate keys, cross-slide keys, and unmatched rows stop execution with slide IDs and exact left/right unmatched counts. |
| R07 | Make cache and result writes atomic and validate completed artifacts. | File existence alone is not evidence that an interrupted write completed. | HIGH | A simulated write interruption leaves no accepted final artifact; successful writes are atomically promoted only after schema/shape checks; readers reject truncated or invalid artifacts with recovery guidance. |
| R08 | Validate the complete experiment configuration before pipeline execution. | Configuration errors should fail before costly data or model work begins. | HIGH | Missing sections, unknown keys, wrong types, invalid ranges, unsupported task/model names, and cross-field conflicts produce one startup validation report; a valid resolved config is available to all stages. |
| R09 | Seed PyTorch, data loaders, workers, and deterministic backend policy centrally. | A reproducible tutorial must control the entire stochastic training stack. | MEDIUM | Repeated CPU fixture runs under the same seed produce equivalent splits, batches, initial weights, and metrics within declared tolerances; manifests record seeds, determinism policy, hardware, and relevant versions. |
| R10 | Reject empty cohorts, folds, patch sets, and prediction batches early. | Empty scientific inputs should not surface as low-level stack/concatenate errors. | LOW | Empty configured cohorts, post-QC slides, aligned folds, patch lists, and prediction loaders fail at their boundary with the stage, slide/fold, observed count, and corrective action. |
| R11 | Validate class support and unseen-class coverage for every LOSO fold. | Classification metrics are misleading when a fold cannot learn a configured class. | MEDIUM | Each fold records train/test counts by class; degenerate training folds fail before training; unseen test classes are explicitly separated from ordinary metrics and coverage is reported. |
| R12 | Fit regression-target scaling only on training data and invert it for reports. | Target scale must neither dominate loss nor leak test statistics. | HIGH | Fold scalers are fit from outer-training targets only, persisted with masks/statistics, and never refit on test data; reported predictions and metrics are in original units; a shifted test fixture does not alter scaler parameters. |
| R13 | Fit RF imputation and feature schema only on training data. | Validation/test feature statistics must not influence preprocessing. | HIGH | A fixed ordered feature schema and imputer are fitted on training features only and reused unchanged for held-out data; missing/extra columns are handled by contract; a shifted held-out fixture does not change fitted values. |
| R19 | Add fixture-backed unit, integration, notebook, and CI validation tiers. | A tutorial cannot claim reproducibility without automated checks across its executable surfaces. | HIGH | Offline CPU CI runs unit tests, representative synthetic pipeline integration, and selected headless notebook execution; slower network/full-cohort tiers are explicit opt-ins; failures identify their tier and artifact. |
| R20 | Consolidate supported Python/dependency declarations and produce a reproducible environment contract. | Conflicting broad dependency declarations prevent repeatable installation. | HIGH | One authoritative support policy drives package/environment files; minimum and locked environments resolve in CI; Python and key dependency versions agree across documentation and metadata; a lock or equivalent exact environment artifact is generated and documented. |

### Differentiators (Reliability Made Visible and Teachable)

| ID | Capability | Value Proposition | Complexity | Observable Acceptance Behavior |
|----|------------|-------------------|------------|--------------------------------|
| R02 | Normalize stains from each source slide into a shared cohort reference basis. | Teaches cohort-level stain harmonization rather than visually plausible per-slide reconstruction. | HIGH | Source and target stain matrices are distinct and recorded; all slides use the same fitted cohort target; a synthetic cross-slide color fixture measurably converges after normalization without using the held-out slide to fit fold-specific references. |
| R06 | Fingerprint caches against configuration, source data, and relevant code contracts. | Makes cache reuse auditable instead of relying on filenames and manually bumped versions. | HIGH | Cache manifests contain canonical config, source-data, schema/code-contract fingerprints; identical inputs hit the cache; changing any relevant input produces a miss and regeneration; irrelevant presentation changes do not invalidate scientific caches. |
| R14 | Preserve fixed physical context at image borders and record patch-quality flags. | Keeps morphological context comparable while teaching the quality consequences of edge/background patches. | HIGH | Every patch represents the configured native physical extent through deterministic padding rather than clipped stretching; metadata records padding, tissue, blur, and artifact flags; policy can reject or retain flagged patches reproducibly. |
| R15 | Validate Macenko inputs and numerical outputs with explicit fallback provenance. | Converts a fragile image transform into an inspectable scientific decision. | MEDIUM | Non-RGB, invalid-range, low-tissue, rank-deficient, and non-finite inputs are detected; a deterministic fallback or explicit failure occurs by policy; output matrices are finite/valid and metadata records method, quality, reason, and fallback. |
| R17 | Add confidence, abstention, and provenance to heuristic scientific labels. | Demonstrates uncertainty-aware label engineering instead of presenting heuristics as ground truth. | HIGH | Labels use versioned explicit gene sets and enrichment/statistical evidence; every label carries confidence and rule provenance; low-confidence cases abstain by configured policy; sensitivity/noise behavior is measurable on fixtures. |
| R18 | Adapt preprocessing dimensions safely after QC and record resolved parameters. | Makes the tutorial robust to small datasets while showing exactly how analytical choices were resolved. | MEDIUM | After QC, HVG/PCA/neighbor dimensions are deterministically bounded by remaining data; impossible analyses fail clearly; requested and resolved values plus QC counts are stored in `adata.uns` and the run manifest. |

### Anti-Features (Unsafe Behaviors to Remove)

These are counted requirements, not additional scope. Each positive replacement is the acceptance target.

| ID | Active Improvement | Tempting but Problematic Behavior | Complexity | Observable Acceptance Behavior |
|----|--------------------|-----------------------------------|------------|--------------------------------|
| R03 | Replace pickle-backed patch cache metadata with safe serialization. | Keeping object-valued NPZ metadata is convenient and backward-compatible, but loading it permits arbitrary Python object deserialization. | MEDIUM | New caches contain only non-object arrays and safe JSON/Parquet-style metadata and load with `allow_pickle=False`; malformed schemas are rejected; legacy caches are not silently loaded and receive explicit regeneration guidance. |
| R04 | Load model checkpoints without enabling arbitrary pickle execution. | A single legacy `torch.load(..., weights_only=False)` path accepts rich objects but can execute attacker-controlled code. | MEDIUM | Default checkpoint loading uses weights-only state dictionaries and separately validated safe metadata; unexpected keys/shapes fail; legacy unsafe loading is absent from normal paths and any trusted-only migration is explicit and opt-in. |
| R16 | Fail on missing configured slides unless partial-cohort mode is explicitly enabled. | Silently skipping missing slides lets demos finish, but changes the cohort without scientific acknowledgement. | MEDIUM | Default runs fail with all missing slide IDs; explicit partial mode records requested, included, excluded, and reasons in a cohort manifest; summaries and “complete” status reflect the resolved cohort. |

## Feature Dependencies

```text
R08 Configuration validation
├──enables──> R09 Deterministic training policy
├──enables──> R10 Empty-input guards
├──enables──> R11 Fold class-support validation
├──enables──> R16 Explicit partial-cohort policy
└──enables──> R18 Safe adaptive preprocessing

R19 Fixture-backed test tiers
├──verifies──> R01 Leakage-free model selection
├──verifies──> R02 Shared stain target
├──verifies──> R03/R04 Safe artifact loading
├──verifies──> R05 Alignment contract
├──verifies──> R07 Atomic artifacts
├──verifies──> R12/R13 Train-only transformations
└──verifies──> R14/R15 Patch and stain quality

R05 Alignment + R10 non-empty guards + R11 class support
└──precede──> R01 CNN training and R12 target scaling

R15 validated Macenko estimation
└──precedes──> R02 shared-reference normalization

R06 cache fingerprints
├──requires──> R08 canonical resolved configuration
├──requires──> R20 reproducible code/environment contract
└──enhances──> R07 atomic artifact acceptance

R16 cohort manifest
├──feeds──> R02 cohort reference selection
├──feeds──> R06 source fingerprint
└──feeds──> R17 label provenance

R14 patch-quality flags
└──feed──> R05 alignment and R13 fixed RF schema
```

### Dependency Notes

- **R19 should begin first and grow with every capability:** synthetic fixtures provide inexpensive proof of leakage boundaries, malformed artifacts, image edge cases, and configuration failures.
- **R08 precedes R06 and most pipeline guards:** fingerprints and manifests need a canonical, fully resolved configuration rather than raw nested dictionaries.
- **R05, R10, and R11 precede model changes:** reliable fold composition and target cardinality are prerequisites for proving R01 and R12.
- **R15 precedes R02:** shared-reference normalization is only trustworthy after source/target matrix estimation has explicit validity and fallback rules.
- **R06 and R07 share an artifact contract:** a fingerprint decides whether reuse is permitted; atomic promotion and validation decide whether the artifact is complete enough to reuse.
- **R20 supports R09 and R19:** deterministic claims and CI tiers require declared, testable runtime versions.
- **R03 and R04 are migration boundaries:** safe readers should land with regeneration/conversion guidance and tests before unsafe loaders are removed from supported paths.

## Milestone Definition

### Required for Completion

All R01–R20 capabilities are milestone requirements. None are deferred to a later release. Completion requires the observable behavior in the catalog plus offline automated evidence appropriate to the risk.

### Recommended Delivery Waves

1. **Verification foundation:** R19, R20, R08, R09.
2. **Input and cohort contracts:** R10, R16, R05, R11, R18.
3. **Secure and durable artifacts:** R03, R04, R06, R07.
4. **Image reliability:** R15, R02, R14.
5. **Leakage-free learning:** R13, R12, R01.
6. **Scientific label trust:** R17, followed by full tiered regression validation.

## Prioritization Matrix

| IDs | User/Scientific Value | Implementation Cost | Priority | Rationale |
|-----|-----------------------|---------------------|----------|-----------|
| R01–R04 | HIGH | MEDIUM–HIGH | P0 | Scientific leakage and arbitrary-code execution can invalidate results or compromise users. |
| R05–R20 | HIGH | LOW–HIGH | P1 | These close correctness, reproducibility, observability, and verification gaps required by the milestone contract. |

## Anti-Scope Guardrails

- Do not redesign the 20-notebook teaching narrative while adding validation hooks.
- Do not add new biological datasets or make new biological efficacy claims.
- Do not fit shared references, scalers, imputers, or selection criteria using outer-test data.
- Do not retain unsafe loaders as silent compatibility fallbacks.
- Do not turn offline CI into a network- or accelerator-dependent workflow.
- Do not broaden this milestone into package renaming, distributed training, Zarr migration, or general performance work.

## Sources

- `.planning/PROJECT.md` — milestone goal, fixed Active requirements, constraints, and scope.
- `.planning/codebase/CONCERNS.md` — mapped P0/P1 evidence, affected modules, and recommended remediation order at commit `1c2d0739bbb2b724a4eaef1cdbb16d865bff7580`.

---
*Feature research for: Spatial Transcriptomics Tutorial Reliability Upgrade*
*Researched: 2026-07-17*
