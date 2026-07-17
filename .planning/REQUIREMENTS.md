# Requirements: Spatial Transcriptomics Tutorial Reliability Upgrade

**Defined:** 2026-07-17
**Core Value:** Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.

## v1 Requirements

Exactly 20 high-priority updates are committed to this milestone.

### Evaluation Integrity

- [ ] **EVAL-01**: Maintainers can run outer LOSO CNN evaluation without using the held-out slide for early stopping, epoch selection, hyperparameter selection, preprocessing, or any other learned decision.
- [ ] **EVAL-02**: Maintainers can inspect every LOSO fold’s training/test class counts and receive an error for degenerate training support plus an explicit metric for test classes unseen during training.
- [ ] **EVAL-03**: Multi-task regression targets are scaled from outer-training observations only, missing targets are masked, and reported predictions are restored to original target units.
- [ ] **EVAL-04**: The RF baseline uses a fixed feature schema and fits missing-value imputation from outer-training features only before applying it to the held-out slide.

### Artifact Safety and Provenance

- [ ] **ART-01**: Patch arrays and metadata round-trip through a non-pickle format, and untrusted cache loading never enables NumPy object deserialization.
- [ ] **ART-02**: Model checkpoints load weights with `weights_only=True` and validate separately stored metadata without executing pickle-backed Python objects.
- [x] **ART-03**: Processed-slide, patch, embedding, model, and report caches include a deterministic fingerprint of relevant configuration, input identity, and artifact schema, and stale fingerprints are rejected.
- [x] **ART-04**: Cache, model, table, and manifest writes use same-filesystem temporary files plus atomic replacement, and readers validate required keys, schemas, shapes, and completion metadata.

### Input and Cohort Validation

- [x] **VAL-01**: The resolved experiment configuration is validated at startup for required sections, allowed values, types, positive ranges, and cross-field constraints with actionable errors.
- [x] **VAL-02**: Label and patch metadata must have non-null unique `(slide_id, spot_id)` keys and align one-to-one without silent row loss, duplication, or cross-slide mismatch.
- [x] **VAL-03**: Empty configured cohorts, folds, aligned spot sets, patch sets, prediction batches, and regression-target selections fail before expensive execution with domain-specific errors.
- [x] **VAL-04**: The pipeline fails when configured slides are missing unless explicit partial-cohort mode is enabled, and every run records included, skipped, and failed slides in a cohort manifest.
- [x] **VAL-05**: Preprocessing validates post-QC spot/gene counts, safely resolves HVG/PCA/neighbor dimensions, and records the actual parameters used in AnnData and run provenance.

### Reproducibility and Image Contracts

- [ ] **REPRO-01**: One seeding API controls Python, NumPy, PyTorch CPU/CUDA, data-loader generators/workers, and the documented deterministic-backend policy, with reproducibility metadata captured per run.
- [ ] **IMG-01**: Macenko normalization estimates a validated source stain matrix per slide and maps patches to a separate shared target matrix, with tests showing cross-slide color convergence.
- [ ] **IMG-02**: Patch extraction preserves a fixed native field of view at image borders through explicit padding, records padding/tissue/quality fields, and supports a configurable tissue-quality gate.
- [ ] **IMG-03**: Stain estimation validates RGB shape, dtype/range, tissue-pixel count, covariance rank, finite eigenvectors, and normalized output; fallback use is explicit and recorded.

### Scientific Labels, Verification, and Environment

- [ ] **LABEL-01**: Heuristic domain labels use versioned explicit gene rules and enrichment evidence, store confidence and provenance, and abstain instead of forcing low-confidence assignments.
- [x] **TEST-01**: Fast CI runs Ruff, unit tests, artifact round trips, synthetic AnnData integration, model/fold smoke tests, and notebook structural checks, while network/full-cohort execution remains an explicit slow tier.
- [ ] **ENV-01**: Python support and dependency declarations agree across `pyproject.toml`, requirements files, Conda configuration, and documentation, with a reproducible locked/tested environment contract.

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

## Definition of Done

- All 20 v1 requirements map to exactly one roadmap phase and are verified by automated tests or explicit documented evidence.
- Default tests run without network access, model downloads, or private data.
- Learned transforms and model-selection decisions are proven to use outer-training data only.
- Cache/checkpoint readers do not enable arbitrary object execution.
- Existing public imports, documented commands, notebook order, and intended output names remain compatible or have an explicit migration note.
- `python -m pytest -q`, Ruff, notebook structural validation, and the configured CI workflow pass.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | Phase 7 | Pending |
| EVAL-02 | Phase 6 | Pending |
| EVAL-03 | Phase 7 | Pending |
| EVAL-04 | Phase 7 | Pending |
| ART-01 | Phase 5 | Pending |
| ART-02 | Phase 5 | Pending |
| ART-03 | Phase 4 | Complete |
| ART-04 | Phase 4 | Complete |
| VAL-01 | Phase 2 | Complete |
| VAL-02 | Phase 3 | Complete |
| VAL-03 | Phase 2 | Complete |
| VAL-04 | Phase 2 | Complete |
| VAL-05 | Phase 3 | Complete |
| REPRO-01 | Phase 6 | Pending |
| IMG-01 | Phase 8 | Pending |
| IMG-02 | Phase 8 | Pending |
| IMG-03 | Phase 8 | Pending |
| LABEL-01 | Phase 9 | Pending |
| TEST-01 | Phase 1 | Complete |
| ENV-01 | Phase 10 | Pending |

**Coverage:**

- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-17*
*Last updated: 2026-07-17 after roadmap traceability mapping*
