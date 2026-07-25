# Roadmap: Spatial Transcriptomics Tutorial Reliability Upgrade

## Overview

This milestone adds a focused reliability spine beneath the existing notebook-first tutorial and pharma extension. Work proceeds from offline verification and validated inputs through durable, lineage-aware artifacts. The sequence preserves existing notebook, CLI, configuration, output, and public-import surfaces while making the implemented validation and artifact guarantees explicit and reviewable.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work.
- Decimal phases (2.1, 2.2): Urgent insertions, if later required.

- [x] **Phase 1: Offline Verification Harness** - Establish representative, tiered, CPU/offline evidence for every later reliability change. (completed 2026-07-17)
- [x] **Phase 2: Validated Run and Cohort Admission** - Reject invalid experiments and resolve cohort membership before expensive work. (completed 2026-07-17)
- [x] **Phase 3: Identity and Adaptive Preprocessing** - Guarantee sample alignment and scientifically viable post-QC dimensions. (completed 2026-07-17)
- [x] **Phase 4: Durable Artifact Contract** - Accept only fingerprint-matched, complete, schema-valid artifacts. (completed 2026-07-25)

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

**Plans:** 4/4 plans complete

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Offline Verification Harness | 3/3 | Complete | 2026-07-17 |
| 2. Validated Run and Cohort Admission | 4/4 | Complete | 2026-07-17 |
| 3. Identity and Adaptive Preprocessing | 3/3 | Complete | 2026-07-17 |
| 4. Durable Artifact Contract | 4/4 | Complete | 2026-07-25 |
