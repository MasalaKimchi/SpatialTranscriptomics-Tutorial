# Requirements: Spatial Transcriptomics Tutorial Reliability Upgrade

**Defined:** 2026-07-17
**Core Value:** Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.

## v1 Requirements

Eight implemented requirements define the final four-phase milestone.

### Artifact Safety and Provenance

- [x] **ART-03**: Processed-slide, patch, embedding, model, and report caches include a deterministic fingerprint of relevant configuration, input identity, and artifact schema, and stale fingerprints are rejected.
- [x] **ART-04**: Cache, model, table, and manifest writes use same-filesystem temporary files plus atomic replacement, and readers validate required keys, schemas, shapes, and completion metadata.

### Input and Cohort Validation

- [x] **VAL-01**: The resolved experiment configuration is validated at startup for required sections, allowed values, types, positive ranges, and cross-field constraints with actionable errors.
- [x] **VAL-02**: Label and patch metadata must have non-null unique `(slide_id, spot_id)` keys and align one-to-one without silent row loss, duplication, or cross-slide mismatch.
- [x] **VAL-03**: Empty configured cohorts, folds, aligned spot sets, patch sets, prediction batches, and regression-target selections fail before expensive execution with domain-specific errors.
- [x] **VAL-04**: The pipeline fails when configured slides are missing unless explicit partial-cohort mode is enabled, and every run records included, skipped, and failed slides in a cohort manifest.
- [x] **VAL-05**: Preprocessing validates post-QC spot/gene counts, safely resolves HVG/PCA/neighbor dimensions, and records the actual parameters used in AnnData and run provenance.

### Verification

- [x] **TEST-01**: Fast CI runs Ruff, unit tests, artifact round trips, synthetic AnnData integration, model/fold smoke tests, and notebook structural checks, while network/full-cohort execution remains an explicit slow tier.

## v2 Requirements

### Maintainability and Scale

- **PKG-01**: Install the pharma extension under a distinctive package name and remove runtime `sys.path` mutation.
- **NOTE-01**: Replace large embedded notebook-source strings with stable templates or text-based notebook sources.
- **PERF-01**: Introduce chunked/lazy patch storage and versioned per-slide radiomics caching after correctness contracts stabilize.
- **PROV-01**: Expand run manifests into complete experiment bundles with unique run IDs and navigable artifact lineage.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New datasets or biological claims | This milestone validates existing public workflows rather than expanding scientific scope. |
| Foundation-model fine-tuning | Frozen encoders are sufficient for the current tutorial and avoid a separate training/security surface. |
| Distributed training | Not required for CPU-compatible correctness and test validation. |
| Broad notebook redesign | Educational narrative and numbering remain stable. |
| Safe non-pickle cache/checkpoint migration | Removed with Phases 5–10; compatibility readers accept trusted local artifacts only. |
| Leakage-free evaluation and fold admission | Removed with Phases 5–10; the pharma extension remains educational/research-oriented. |
| Expanded image, label, seeding, and locked-environment contracts | Removed with Phases 5–10 at project finalization. |

## Definition of Done

- All eight final v1 requirements map to exactly one roadmap phase and are verified by automated tests or explicit documented evidence.
- Default tests run without network access, model downloads, or private data.
- Patch caches and checkpoints are documented as trusted-local artifacts, not safe containers for untrusted files.
- Existing public imports, documented commands, notebook order, and intended output names remain compatible or have an explicit migration note.
- `python -m pytest -q`, Ruff, notebook structural validation, and the configured CI workflow pass.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ART-03 | Phase 4 | Complete |
| ART-04 | Phase 4 | Complete |
| VAL-01 | Phase 2 | Complete |
| VAL-02 | Phase 3 | Complete |
| VAL-03 | Phase 2 | Complete |
| VAL-04 | Phase 2 | Complete |
| VAL-05 | Phase 3 | Complete |
| TEST-01 | Phase 1 | Complete |

**Coverage:**

- v1 requirements: 8 total
- Mapped to phases: 8
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-17*
*Last updated: 2026-07-25 after project scope ended at Phase 4*
