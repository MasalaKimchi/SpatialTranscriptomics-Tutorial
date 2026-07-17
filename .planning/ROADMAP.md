# Roadmap: Spatial Transcriptomics Tutorial Reliability Upgrade

## Overview

This milestone adds a thin reliability spine beneath the existing notebook-first tutorial and pharma extension. Work proceeds from offline verification and validated inputs through durable artifacts, reproducible folds, leakage-free evaluation, image and label trust, and finally a reconciled environment contract. The sequence preserves existing notebook, CLI, configuration, output, and public-import surfaces while making scientific and serialized results explicit, safe, and reviewable.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work.
- Decimal phases (2.1, 2.2): Urgent insertions, if later required.

- [x] **Phase 1: Offline Verification Harness** - Establish representative, tiered, CPU/offline evidence for every later reliability change. (completed 2026-07-17)
- [x] **Phase 2: Validated Run and Cohort Admission** - Reject invalid experiments and resolve cohort membership before expensive work. (completed 2026-07-17)
- [x] **Phase 3: Identity and Adaptive Preprocessing** - Guarantee sample alignment and scientifically viable post-QC dimensions. (completed 2026-07-17)
- [ ] **Phase 4: Durable Artifact Contract** - Accept only fingerprint-matched, complete, schema-valid artifacts.
- [ ] **Phase 5: Safe Cache and Checkpoint Formats** - Remove unsafe deserialization from supported patch and model artifact paths.
- [ ] **Phase 6: Reproducible Fold Admission** - Make each LOSO fold deterministic, viable, and explicit about class coverage.
- [ ] **Phase 7: Leakage-Free Evaluation** - Isolate model selection and learned preprocessing from every held-out slide.
- [ ] **Phase 8: Image Reliability** - Normalize stains safely and preserve physical patch context with quality provenance.
- [ ] **Phase 9: Trustworthy Heuristic Labels** - Make scientific labels evidence-backed, confidence-aware, and able to abstain.
- [ ] **Phase 10: Reproducible Environment Contract** - Reconcile and verify the supported and locked execution environments.

## Phase Details

### Phase 1: Offline Verification Harness

**Goal:** Maintainers can verify representative tutorial and pharma reliability behavior quickly on CPU without downloads or private data.
**Mode:** mvp
**Depends on:** Nothing (first phase)
**Requirements:** TEST-01
**Success Criteria** (what must be TRUE):

  1. Maintainers can run a documented fast CI tier that includes Ruff, unit tests, safe artifact round trips, synthetic AnnData integration, model/fold smoke tests, and notebook structural checks without network access.
  2. Maintainers can run deterministic fixtures covering valid and adversarial cohorts, keys, images, folds, and serialized artifacts.
  3. Maintainers can distinguish fast synthetic evidence from explicit notebook-smoke, network, and full-cohort tiers in CI configuration and test output.
  4. Later phases can add regression evidence to the same tiers without introducing separate test conventions.

**Plans:** 3/3 plans complete

### Phase 2: Validated Run and Cohort Admission

**Goal:** Maintainers can start only experiments whose configuration, stage inputs, and resolved cohort satisfy explicit contracts.
**Mode:** mvp
**Depends on:** Phase 1
**Requirements:** VAL-01, VAL-03, VAL-04
**Success Criteria** (what must be TRUE):

  1. Invalid configuration sections, keys, types, values, and cross-field combinations fail together at startup with actionable paths and expected values.
  2. Empty cohorts, folds, aligned sets, patch sets, prediction batches, and regression-target selections fail at their boundary with stage identity and corrective guidance.
  3. Missing configured slides fail before processing by default; explicit partial-cohort mode records configured, included, skipped, and failed slides.
  4. Each admitted run exposes a canonical resolved configuration and cohort manifest for downstream provenance and fingerprints.

**Plans:** 4/4 plans complete

### Phase 3: Identity and Adaptive Preprocessing

**Goal:** Maintainers can trust that every retained spot is uniquely aligned and every post-QC analysis uses legal, recorded dimensions.
**Mode:** mvp
**Depends on:** Phase 2
**Requirements:** VAL-02, VAL-05
**Success Criteria** (what must be TRUE):

  1. Null, duplicate, unmatched, or cross-slide `(slide_id, spot_id)` keys fail before array construction with counts and representative offending keys.
  2. Shuffled complete label and patch tables align one-to-one without silent row loss, duplication, or target/provenance reordering.
  3. Post-QC spot and gene counts either resolve deterministic legal HVG, PCA, and neighbor dimensions or fail as scientifically nonviable.
  4. Requested and resolved preprocessing parameters, input counts, exclusions, and reason codes remain visible in AnnData and run provenance.

**Plans:** 3/3 plans complete

### Phase 4: Durable Artifact Contract

**Goal:** Maintainers can reuse only complete artifacts whose schema, lineage, and relevant inputs match the current run.
**Mode:** mvp
**Depends on:** Phases 2 and 3
**Requirements:** ART-03, ART-04
**Success Criteria** (what must be TRUE):

  1. Relevant configuration, source-data identity, upstream lineage, or code-contract changes cause a deterministic cache miss, while presentation-only changes do not invalidate scientific artifacts.
  2. Maintainers can inspect each artifact manifest for schema version, fingerprint inputs, payload metadata, checksums, and completion state.
  3. Interrupted, truncated, stale, wrong-shape, wrong-schema, or incomplete artifacts are rejected by production readers and never promoted as valid results.
  4. Cache, model, table, and manifest publication uses same-filesystem temporary files, validation through the production reader, and atomic replacement with the completed manifest as commit marker.

**Plans:** TBD

### Phase 5: Safe Cache and Checkpoint Formats

**Goal:** Maintainers can exchange patch caches and model checkpoints without supported readers executing serialized Python objects.
**Mode:** mvp
**Depends on:** Phase 4
**Requirements:** ART-01, ART-02
**Success Criteria** (what must be TRUE):

  1. Patch arrays, compound identities, quality fields, and scalar metadata round-trip through numeric/string and JSON/tabular schemas loaded with `allow_pickle=False`.
  2. Legacy object-valued patch caches fail closed without object deserialization and provide precise regeneration guidance.
  3. Model state loads with `weights_only=True`, while separate safe metadata is validated for schema, model identity, expected keys, tensor shapes, and dtypes.
  4. Legacy or malformed checkpoint artifacts are rejected on normal execution paths without an unsafe compatibility fallback.

**Plans:** TBD

### Phase 6: Reproducible Fold Admission

**Goal:** Maintainers can create LOSO folds that are reproducible, training-viable, and explicit about held-out class coverage before fitting begins.
**Mode:** mvp
**Depends on:** Phases 1, 3, and 5
**Requirements:** REPRO-01, EVAL-02
**Success Criteria** (what must be TRUE):

  1. One seeding API deterministically derives and applies Python, NumPy, PyTorch CPU/CUDA, data-loader generator, and worker seeds for each run and fold.
  2. Strict, best-effort, and disabled deterministic-backend policies have documented behavior and record seeds, backend flags, hardware, threads, and approved exceptions.
  3. Every LOSO fold reports training and test sample/class counts before training, and degenerate training support fails with the affected fold and classes.
  4. Classes present only in the held-out slide remain evaluable through an explicit unseen-class coverage metric instead of being silently dropped or conflated with ordinary accuracy.

**Plans:** TBD

### Phase 7: Leakage-Free Evaluation

**Goal:** Maintainers can evaluate CNN and RF models knowing that no held-out slide influenced selection or fitted preprocessing state.
**Mode:** mvp
**Depends on:** Phases 5 and 6
**Requirements:** EVAL-01, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):

  1. Inner model selection uses only slide-disjoint outer-training partitions or a predeclared training-only policy, and the frozen model evaluates the outer slide once after selection.
  2. Perturbing only held-out labels, targets, or features cannot change selected epochs, hyperparameters, training-only stain/scaler/imputer state, or feature schema.
  3. Regression scaling is fitted on outer-training observations only, preserves missing-target masks and target order, and restores predictions to original report units.
  4. RF features are reindexed to a training-defined schema and transformed by an imputer fitted only on outer-training rows, with explicit handling for missing or extra columns.
  5. Fold artifacts expose fit slide IDs and selection lineage proving disjointness from the outer-test slide.

**Plans:** TBD

### Phase 8: Image Reliability

**Goal:** Maintainers can produce fixed-context patches whose stain transformation and quality decisions are valid, comparable, and provenance-rich.
**Mode:** mvp
**Depends on:** Phases 4 and 6
**Requirements:** IMG-01, IMG-02, IMG-03
**Success Criteria** (what must be TRUE):

  1. Grayscale, RGBA, invalid-range, non-finite, low-tissue, low-rank, or numerically invalid Macenko inputs deterministically fail or use the configured fallback with quality metrics and a recorded reason.
  2. Each normalized patch distinguishes its validated source stain matrix from a shared target matrix fitted only from the declared allowed scope or identified by an external checksum.
  3. Controlled cross-slide fixtures show reduced color-statistic distance after shared-reference normalization without changing patch shape or spatial layout.
  4. Border spots preserve the configured native field of view through explicit padding rather than stretching, and synchronized metadata records bounds, padding fraction/mask, tissue fraction, blur/artifact flags, acceptance, and rejection reason.
  5. A configurable quality gate reports pre/post-filter counts and cannot silently create an unsupported fold.

**Plans:** TBD

### Phase 9: Trustworthy Heuristic Labels

**Goal:** Maintainers can interpret heuristic domain labels as versioned evidence with confidence and abstention rather than unqualified biological ground truth.
**Mode:** mvp
**Depends on:** Phases 3, 4, and 6
**Requirements:** LABEL-01
**Success Criteria** (what must be TRUE):

  1. Label generation uses exact normalized gene symbols and a versioned rule set with recorded scoring, thresholds, evidence, and preprocessing provenance.
  2. Each label result distinguishes positive evidence, low confidence, insufficient evidence, out-of-scope biology, and abstention instead of forcing an assignment.
  3. Confidence, rule version, evidence, and abstention reason survive safe serialization, one-to-one alignment, quality filtering, training inputs, and reports.
  4. Maintainers can inspect threshold-sensitivity and label-noise summaries with retained denominators and renewed fold-support validation after abstention.

**Plans:** TBD

### Phase 10: Reproducible Environment Contract

**Goal:** Maintainers can install and verify the tutorial from declarations that agree on one supported Python policy and one exact reference environment.
**Mode:** mvp
**Depends on:** Phases 1-9
**Requirements:** ENV-01
**Success Criteria** (what must be TRUE):

  1. `pyproject.toml`, requirements files, Conda configuration, CI, notebooks, and documentation agree on the supported Python version and dependency partitions.
  2. Maintainers can create a hash-pinned reference environment from scratch and run the applicable fast reliability gates in it.
  3. CI mechanically detects drift between the authoritative support policy and committed environment declarations and verifies the declared minimum-supported and locked reference environments.
  4. Run provenance records the solved environment and platform, while optional network, accelerator, and full-cohort dependencies remain explicit tiers rather than default requirements.

**Plans:** TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Offline Verification Harness | 3/3 | Complete    | 2026-07-17 |
| 2. Validated Run and Cohort Admission | 4/4 | Complete    | 2026-07-17 |
| 3. Identity and Adaptive Preprocessing | 3/3 | Complete    | 2026-07-17 |
| 4. Durable Artifact Contract | 0/TBD | Not started | - |
| 5. Safe Cache and Checkpoint Formats | 0/TBD | Not started | - |
| 6. Reproducible Fold Admission | 0/TBD | Not started | - |
| 7. Leakage-Free Evaluation | 0/TBD | Not started | - |
| 8. Image Reliability | 0/TBD | Not started | - |
| 9. Trustworthy Heuristic Labels | 0/TBD | Not started | - |
| 10. Reproducible Environment Contract | 0/TBD | Not started | - |
