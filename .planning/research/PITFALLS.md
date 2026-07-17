# Pitfall Research

**Domain:** Reliability upgrade for a notebook-first spatial transcriptomics and computational pathology tutorial  
**Researched:** 2026-07-17  
**Confidence:** HIGH for repository-specific risks; MEDIUM for full-cohort behavior until real-data validation is run

## Research Boundary

This document identifies failure modes that can survive apparently successful implementations of the 20 Active requirements in `PROJECT.md`. The focus is not whether a feature exists, but whether its scientific boundary, artifact contract, and verification evidence are trustworthy. The phase labels follow the delivery waves in `FEATURES.md`; roadmap numbering can be assigned later without changing the sequencing guidance.

## Critical Pitfalls

### 1. Fixing Outer LOSO Leakage While Retaining Indirect Test Influence

**What goes wrong:** The held-out slide is removed from early stopping, but still affects preprocessing, class vocabulary, stain-reference fitting, target scaling, imputation, feature selection, threshold selection, or hyperparameter choice. The resulting pipeline looks nested while the test domain still informs model construction.

**Warning signs:**

- A helper receives all slide IDs before the outer split is finalized.
- A transform exposes `fit_transform()` on train and test paths rather than separate `fit()` and `transform()` calls.
- Fold artifacts do not record exactly which slide IDs fitted each learned object.
- Perturbing held-out labels, target values, or feature distributions changes the chosen epoch, scaler, imputer, reference matrix, or hyperparameters.
- Results improve materially after adding a slide even when that slide is always nominally held out.

**Prevention:**

- Create the outer split first and pass explicit train/test cohort objects through every downstream stage.
- Give every fitted artifact a `fit_slide_ids` field and assert that it is disjoint from the outer-test IDs.
- Use an inner slide-level split for selection, or a predefined training-only stopping rule when too few training slides remain.
- Test invariance: mutate outer-test labels and statistics and prove that all fitted state and model-selection decisions remain byte-equivalent or numerically equivalent.
- Evaluate the outer slide once after selection; do not expose per-epoch outer metrics to the training loop.

**Phase:** Leakage-free learning (R13, R12, R01), with invariant fixtures established in Verification foundation (R19).

### 2. Treating Spatial Spots as Independent Samples

**What goes wrong:** Inner validation or test construction splits spots randomly, allowing neighboring spots, near-duplicate morphology, or slide-specific acquisition signatures to appear on both sides. This produces optimistic validation even though the outer loop is slide-based.

**Warning signs:**

- `random_split`, row-level stratification, or spot-level cross-validation appears in model-selection code.
- Validation batches contain barcodes from slides used to train the same stopping decision.
- Reported validation performance is much higher than outer-slide performance with low variance across seeds.
- Patch overlap or spatial autocorrelation is not considered when constructing fixture assertions.

**Prevention:**

- Split at slide or patient level for every decision intended to generalize across slides.
- If only one training slide is available, use a documented training-only epoch policy rather than claiming independent validation.
- Persist group membership and verify group disjointness in fold manifests.
- Report sample counts at both spot and slide/patient levels.

**Phase:** Input and cohort contracts (R10, R11) defines grouping invariants; Leakage-free learning (R01) enforces them.

### 3. Migrating Cache Formats Without a Strict Compatibility Boundary

**What goes wrong:** Safe cache serialization is added, but old object-valued NPZ files are silently accepted, partially converted, or mistaken for new artifacts. Alternatively, a schema migration changes row ordering, string types, or metadata precision without invalidating dependent caches.

**Warning signs:**

- A reader retries with `allow_pickle=True` after safe loading fails.
- Cache format is inferred from file extension alone.
- No schema version or migration status is present in the manifest.
- Old and new cache files share the same final path and manual version string.
- Round-trip tests check only shapes, not key order, dtypes, values, and provenance.

**Prevention:**

- Define a versioned cache schema with required fields, dtypes, dimensions, ordering rules, and checksums.
- Make legacy caches fail closed with explicit regeneration guidance; do not use an automatic unsafe fallback.
- Write new artifacts to a temporary path, validate them with the production reader, then atomically promote them.
- Fingerprint schema version, relevant configuration, source inputs, and code-contract version.
- Test malicious/object arrays, truncated files, missing fields, reordered metadata, Unicode IDs, and interrupted writes.

**Phase:** Secure and durable artifacts (R03, R06, R07); schema-fixture coverage begins in Verification foundation (R19).

### 4. Replacing One Unsafe Checkpoint Call but Preserving Pickle Elsewhere

**What goes wrong:** `weights_only=True` is used in one loader, but optimizer state, auxiliary metadata, legacy conversion scripts, notebooks, or generic serialization helpers still deserialize arbitrary objects. A “trusted local only” path becomes the default through convenience or undocumented fallback behavior.

**Warning signs:**

- Any normal execution path contains `weights_only=False`, `pickle.load`, `joblib.load`, or object-valued NumPy loading.
- Model metadata and tensors are stored in the same opaque artifact.
- Loader validation checks key names but not tensor shapes, dtypes, or architecture identity.
- Documentation implies downloaded checkpoints can be loaded without a trust decision.

**Prevention:**

- Store tensor state separately from validated JSON metadata.
- Enforce expected keys, tensor shapes/dtypes, model family, task heads, and schema version before `load_state_dict`.
- Keep any legacy converter explicit, offline, opt-in, and prominently marked as trusted-input-only; exclude it from normal pipeline APIs.
- Scan source and generated notebooks for unsafe deserialization patterns in CI.

**Phase:** Secure and durable artifacts (R04, R07), verified by the security fixture tier in R19.

### 5. Calling Per-Slide Reconstruction “Stain Normalization”

**What goes wrong:** Source stain bases are estimated correctly, yet each patch is reconstructed into its own source basis or each slide gets a different target. Images look plausible but cohort color variability is not reduced, and morphology may be distorted by unstable concentration estimates.

**Warning signs:**

- Source and target stain matrices are equal for every normalization call.
- The target matrix is fitted inside a per-slide or per-patch loop.
- Tests assert output shape/range or visual plausibility but not cross-slide convergence.
- Reference fitting includes the outer-test slide in a fold-specific experiment.
- Output provenance does not identify source matrix, target matrix, fit cohort, or fallback reason.

**Prevention:**

- Model source estimation and target selection as separate typed operations.
- Define whether the target is a fixed external reference or training-cohort reference; never silently mix policies.
- For fold-fitted targets, use outer-training slides only and record their IDs. For a fixed project reference, checksum and version it.
- Test with synthetic stain mixtures whose true bases differ, then assert reduced color-statistic distance while preserving spatial structure.
- Inspect representative real slides because synthetic optical-density assumptions are incomplete.

**Phase:** Image reliability: validated Macenko first (R15), shared-reference normalization second (R02).

### 6. Applying Macenko to Invalid or Non-Tissue Pixels

**What goes wrong:** The numerical routine receives float images on an unexpected scale, RGBA/grayscale inputs, background-dominated crops, saturated pixels, or too few optical-density samples. It may return finite but meaningless matrices, so checking only for NaNs is insufficient.

**Warning signs:**

- Inputs are cast to `uint8` without validating original range and channel semantics.
- Tissue-pixel count, covariance rank, condition number, vector orientation, and concentration tails are not recorded.
- A generic reference is substituted without a provenance flag.
- Border or blank patches disproportionately trigger normalization artifacts.

**Prevention:**

- Validate RGB shape, dtype/range, finite values, tissue-mask size, matrix rank, normalization, and sign/orientation conventions.
- Define quality thresholds and a deterministic fail/fallback policy before processing the cohort.
- Record source quality metrics, method, target ID, fallback status, and reason per slide or patch.
- Include blank, near-blank, saturated, low-contrast, grayscale, RGBA, and float-range fixtures.

**Phase:** Image reliability (R15), before R02 and patch quality R14.

### 7. Claiming Determinism From Seeding Alone

**What goes wrong:** Python, NumPy, and Torch seeds are set, but data-loader workers, generators, augmentation libraries, CUDA kernels, threading, or nondeterministic algorithms remain uncontrolled. Conversely, “strict determinism” silently changes hardware compatibility or performance without being recorded.

**Warning signs:**

- A global `set_seed()` exists but loaders lack a seeded generator and worker initializer.
- Repeated runs produce different sample orders, initial weights, or first-epoch losses.
- Determinism tests require exact equality across different hardware/software stacks.
- Backend flags are mutated in scattered modules and absent from run manifests.
- The environment still relies on `KMP_DUPLICATE_LIB_OK=TRUE` while making reproducibility claims.

**Prevention:**

- Centralize seed derivation for Python, NumPy, Torch CPU/CUDA, loaders, workers, folds, and transforms.
- Define explicit strict and best-effort policies, including expected errors for unsupported nondeterministic operations.
- Capture hardware, thread counts, backend flags, library versions, and derived seeds in run manifests.
- Test exact structural invariants and declared numeric tolerances separately; do not promise cross-platform bitwise identity.
- Run repeated-process tests, not only repeated calls in one Python process.

**Phase:** Verification foundation (R09, R20, R19), then enforced in Leakage-free learning (R01/R12/R13).

### 8. Letting Heuristic Labels Masquerade as Biological Ground Truth

**What goes wrong:** Substring rules or top-marker heuristics receive confidence scores without becoming scientifically calibrated. Abstention is tuned to maximize downstream accuracy, rule changes are not versioned, and evaluation treats pseudo-labels as independent truth.

**Warning signs:**

- Labels are named as definitive cell states rather than heuristic domains.
- Confidence is a transformed rank with no null distribution, enrichment statistic, or calibration study.
- Gene symbol matching uses substrings, aliases inconsistently, or ignores species/case normalization.
- `other` combines true negative evidence, ambiguity, missing genes, and algorithmic failure.
- CNN evaluation uses the same H&E-correlated signals that influenced heuristic label creation and calls agreement biological validation.

**Prevention:**

- Use exact, versioned gene sets and record preprocessing, scoring method, thresholds, multiple-testing policy, and rule version.
- Separate positive calls, low-confidence abstentions, insufficient-evidence cases, and out-of-scope biology.
- Keep label confidence/provenance attached through alignment, training, and reporting.
- Evaluate threshold sensitivity and label-noise robustness; avoid claiming external biological validation without independent annotations.
- Require human-reviewable summaries for rule revisions.

**Phase:** Scientific label trust (R17), supported by Input and cohort contracts (R05/R10/R16).

### 9. “Fixing” Alignment by Dropping Data More Carefully

**What goes wrong:** One-to-one merge validation is added, but the recovery path still drops missing patches/labels, coerces duplicate IDs, or sorts keys differently across targets. Counts may match while the wrong samples are paired, especially when barcodes repeat across slides.

**Warning signs:**

- Barcode alone is used as the key instead of a compound slide/spot identity.
- Duplicate resolution uses `keep="last"`, aggregation, or implicit index overwrites.
- Alignment tests use already sorted, unique IDs only.
- Classification and regression arrays are aligned in separate passes.
- Post-QC patch filtering occurs without applying the same keyed selection to labels and provenance.

**Prevention:**

- Define a canonical compound sample key and validate nullness, uniqueness, and slide consistency at each boundary.
- Produce left-only, right-only, and duplicate diagnostics before any merge.
- Align all targets and metadata through one keyed table, then derive ordered arrays once.
- Treat partial alignment as a named policy with manifest counts, never an incidental inner join.
- Test shuffled rows, duplicate barcodes across slides, nulls, mismatched slide IDs, and quality-filtered patches.

**Phase:** Input and cohort contracts (R05, R10, R16), before model or cache work consumes aligned arrays.

### 10. Data-Dependent Adaptive Dimensions That Change Analysis Semantics Silently

**What goes wrong:** PCA, HVG, and neighbor dimensions are clipped to avoid crashes, but clipping rules are inconsistent across slides or allow scientifically meaningless settings. Cache keys retain requested values while artifacts reflect resolved values.

**Warning signs:**

- Deep Scanpy errors disappear, but no resolved parameters appear in `adata.uns` or manifests.
- `min()` expressions are scattered through preprocessing code.
- A fold with very few spots completes with one component or a degenerate neighbor graph without warning.
- Cache fingerprints use raw YAML rather than the resolved analysis contract.

**Prevention:**

- Validate post-QC counts, define minimum viable analysis sizes, and centralize deterministic resolution rules.
- Fail when dimensions fall below scientifically meaningful minima; adapt only within documented bounds.
- Record requested/resolved HVGs, PCs, neighbors, input counts, exclusions, and reason codes.
- Include resolved values and QC summaries in cache fingerprints and reports.

**Phase:** Input and cohort contracts (R18/R10), integrated with Secure and durable artifacts (R06).

### 11. Border Padding That Preserves Pixel Size but Introduces a New Shortcut

**What goes wrong:** Fixed-extent padding corrects geometric stretching, but constant black/white borders encode spot location or slide geometry. The model learns padding patterns; tissue-quality filtering then changes class or slide balance.

**Warning signs:**

- Padding fraction predicts slide ID, class, or target value.
- All padded pixels use a value outside the normalized tissue distribution.
- Quality-filter exclusion rates differ substantially by slide/class and are omitted from reports.
- Augmentations transform tissue but not padding masks consistently.

**Prevention:**

- Record padding masks/fractions and evaluate their association with labels, slide, and predictions.
- Choose and document a padding policy appropriate to the image domain; keep padding metadata available for sensitivity analysis.
- Report pre/post-QC class and slide counts and fail if filtering creates unsupported folds.
- Add tests for physical extent, coordinate transforms, mask synchronization, and no unintended stretching.

**Phase:** Image reliability (R14), followed by fold-support revalidation in Leakage-free learning (R11/R01).

### 12. Atomic Writes That Are Atomic but Not Trustworthy

**What goes wrong:** Temporary-file rename prevents partial final files, but corrupted or semantically incompatible artifacts are atomically promoted. Concurrent writers can race, and readers accept a complete file whose manifest belongs to different inputs.

**Warning signs:**

- Validation occurs after rename or checks only that the file opens.
- Data and manifest are separate promotions with no shared commit marker.
- Cache readers ignore fingerprints when a final path exists.
- Two processes can build the same cache key without locking or collision detection.

**Prevention:**

- Validate schema, shape, dtype, identifiers, checksums, and fingerprint using the production reader before promotion.
- Treat multi-file artifacts as a transaction with a final manifest/commit marker written last.
- Use same-filesystem temporary paths, explicit overwrite policy, and safe concurrency handling.
- Simulate interruption at each stage and concurrent writers in integration tests.

**Phase:** Secure and durable artifacts (R06/R07).

### 13. Dependency Consolidation That Merely Moves Drift Into a Lockfile

**What goes wrong:** An exact environment is produced for one machine, but package metadata, optional pharma dependencies, notebooks, CI, and documented Python support still disagree. Binary scientific packages resolve differently across operating systems or architectures.

**Warning signs:**

- The lockfile is generated but never installed from scratch in CI.
- Minimum-supported and locked environments are conflated.
- CPU CI omits Scanpy, PyTorch, image, or notebook paths exercised by users.
- Tutorials still recommend a Python version outside package metadata.
- Platform-specific workarounds remain globally enabled.

**Prevention:**

- Declare one authoritative support policy and derive or mechanically check all environment declarations against it.
- Test both a minimum-supported environment and the locked reference environment.
- Separate offline core, pharma, notebook, and optional accelerator extras while testing the supported combinations.
- Capture the solved environment in run provenance and schedule periodic resolution tests.

**Phase:** Verification foundation (R20/R19), before reproducibility claims under R09.

### 14. CI Drift From the Tutorial’s Real Execution Surface

**What goes wrong:** Fast CI validates isolated modules while notebook kernels, path setup, optional dependencies, data contracts, or generated notebook code diverge. Green checks become evidence only for a small synthetic subsystem.

**Warning signs:**

- Notebook tests parse JSON but do not execute cells.
- CI imports modules through test-only `sys.path` mutations unlike documented user commands.
- Generated notebooks duplicate production behavior in embedded strings.
- Network/full-cohort tiers exist only in documentation and are never scheduled or recorded.
- CI uses unpinned actions or dependency resolution that changes without review.

**Prevention:**

- Define explicit unit, fixture-integration, headless-notebook, and opt-in real-data tiers with owners and expected runtimes.
- Execute representative notebooks from the same working directory and kernel contract documented for users.
- Keep core behavior in importable modules and make notebook cells thin orchestration layers.
- Pin CI actions/environments and publish tier status plus last successful real-data validation date.

**Phase:** Verification foundation (R19/R20), with notebook checks expanded after each delivery wave.

### 15. False Confidence From Synthetic Tests

**What goes wrong:** Carefully designed fixtures prove code paths but omit real spatial sparsity, tissue artifacts, barcode conventions, batch effects, class imbalance, gene dropout, large artifact sizes, and Scanpy/version behavior. Passing fixtures are mistaken for scientific validation.

**Warning signs:**

- All fixtures are balanced, dense, sorted, RGB uint8, and comfortably above dimensional thresholds.
- Tests assert exact performance improvements on data generated from the model’s own assumptions.
- No test artifact derives from the schema and edge cases of a real public slide.
- Release notes say “validated” without distinguishing unit, fixture, notebook, and full-cohort evidence.
- Full-cohort tests are permanently skipped because they are optional.

**Prevention:**

- Use adversarial synthetic fixtures for individual invariants, not as evidence of biological validity.
- Add small, redistributable, schema-faithful fixtures or metadata/image crops sampled from permitted public data.
- Maintain a scheduled or release-gated real-data smoke tier with checksums and explicit resource requirements.
- Report evidence by tier and date; state what remains untested.
- Include failure-oriented fixtures: empty/degenerate slides, imbalanced classes, missing genes, duplicate IDs, border crops, low tissue, shifted test distributions, and corrupted caches.

**Phase:** Verification foundation (R19), revisited as the completion gate for every later phase.

## Cross-Cutting Warning Matrix

| Pitfall | Fastest Detection Signal | Must Be Recorded | Primary Phase |
|---------|--------------------------|------------------|---------------|
| Indirect LOSO leakage | Held-out perturbation changes fitted state | Fit slide IDs and selection lineage | Leakage-free learning |
| Spot-level pseudo-independence | Same slide/patient on both sides of selection split | Group assignments and counts | Input/cohort contracts |
| Unsafe cache migration | Safe reader falls back to pickle | Schema/version and regeneration reason | Secure artifacts |
| Unsafe checkpoint compatibility | Normal path accepts rich Python objects | Model/schema identity and trust policy | Secure artifacts |
| False stain normalization | Cross-slide stain distance does not decrease | Source/target matrices and fit cohort | Image reliability |
| Invalid Macenko estimate | Finite but low-rank/low-tissue basis | Quality metrics and fallback provenance | Image reliability |
| Partial determinism | Repeated-process order/loss differs | Seeds, backend, hardware, versions | Verification foundation |
| Overconfident heuristic labels | `other`/confidence hides ambiguity | Rule version, evidence, abstention reason | Scientific label trust |
| Silent alignment loss | Compound-key counts change across merge | Unmatched/duplicate diagnostics | Input/cohort contracts |
| Silent dimension adaptation | Requested and resolved values differ invisibly | QC counts and resolved parameters | Input/cohort contracts |
| Padding shortcut | Padding fraction predicts outcome/slide | Padding/QC flags and exclusion balance | Image reliability |
| Semantically invalid atomic artifact | File opens but fingerprint/schema mismatches | Checksums, fingerprint, commit marker | Secure artifacts |
| Environment drift | Clean install differs from declared policy | Exact solved environment | Verification foundation |
| CI/tutorial divergence | Green unit tier, failing headless notebook | Tier result and execution contract | Verification foundation |
| Synthetic-test overclaim | No recent schema-faithful/real-data run | Evidence tier and validation date | All phase gates |

## Phase Gate Checklist

### Verification Foundation

- Test tiers are named, reproducible, and do not conflate synthetic correctness with biological validation.
- Configuration and environment contracts are canonical before fingerprints or determinism claims depend on them.
- Repeated-process determinism and held-out perturbation helpers exist before model refactors.

### Input and Cohort Contracts

- Compound sample identity, cohort completeness, grouping, class support, and post-QC viability fail early.
- Resolved preprocessing parameters and all exclusions are manifest-visible.
- Quality filtering triggers class/fold revalidation rather than silently changing the experiment.

### Secure and Durable Artifacts

- No supported reader silently falls back to unsafe object deserialization.
- Schema, fingerprint, integrity, atomic promotion, and concurrency behavior are tested together.
- Migration guidance favors regeneration and makes trust decisions explicit.

### Image Reliability

- Numerical validity precedes stain normalization.
- Source and target bases are distinct, provenance-rich, and fitted under the declared fold policy.
- Fixed physical context does not introduce unmeasured padding shortcuts.

### Leakage-Free Learning

- Every learned transform carries fit lineage disjoint from the outer test set.
- Selection uses slide/patient groups, and outer evaluation occurs only after model state is frozen.
- Predictions and regression reports are returned to original scientific units.

### Scientific Label Trust

- Heuristics are versioned evidence-generating rules, not presented as independent ground truth.
- Confidence, abstention, ambiguity, missing evidence, and rule provenance remain distinct.
- Claims are limited to what sensitivity, noise, and independent-validation evidence support.

## Sources

- `.planning/PROJECT.md` — milestone objective, 20 Active requirements, constraints, and delivery decisions.
- `.planning/codebase/CONCERNS.md` — repository-specific P0/P1 failure evidence and affected modules at commit `1c2d0739bbb2b724a4eaef1cdbb16d865bff7580`.
- `.planning/research/FEATURES.md` — requirement classification, dependency graph, and recommended delivery waves.

---
*Pitfall research for: Spatial Transcriptomics Tutorial Reliability Upgrade*  
*Researched: 2026-07-17*
