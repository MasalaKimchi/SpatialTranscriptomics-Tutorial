# Spatial Transcriptomics Tutorial Reliability Upgrade

## What This Is

This repository is a notebook-first spatial transcriptomics tutorial with an optional pharma-facing deep-learning extension for predicting tumor-microenvironment molecular state from H&E patches. This completed milestone strengthens the existing implementation through eight focused validation, artifact-provenance, and test-infrastructure requirements while preserving its educational workflow and public outputs.

## Core Value

Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.

## Requirements

### Validated

- ✓ Users can follow a numbered, cache-backed Visium tutorial from data loading through spatial analysis — existing.
- ✓ Users can run a pharma extension with cohort preprocessing, label engineering, patch extraction, LOSO benchmarking, and foundation-model probes — existing.
- ✓ The pharma package exposes lightweight imports and focused foundation-model tests pass without eagerly importing Scanpy — existing.
- ✓ Repository-relative paths, deterministic NumPy/Python seeds, configurable experiments, and committed figure previews are established conventions — existing.
- ✓ Fast CPU/offline verification now runs Ruff plus 58 fixture-backed unit, artifact, synthetic AnnData, model/fold, and notebook-structure checks; network and full-cohort tiers are explicit opt-ins — validated in Phase 1.
- ✓ Experiment startup now aggregates configuration defects, admits cohorts fail-closed by default, records explicit partial-cohort outcomes, and rejects empty work at public scientific boundaries — validated in Phase 2.
- ✓ Label, patch, CNN/RF, and foundation paths now enforce exact unique `(slide_id, spot_id)` identity with complete one-to-one metadata-order alignment and bounded hostile-safe diagnostics — validated in Phase 3.
- ✓ Preprocessing now derives legal HVG/PCA/neighbor dimensions from observed post-QC and actual-HVG counts, records canonical AnnData/run provenance, and is covered by real Scanpy/H5AD evidence within a 263-test offline gate — validated in Phase 3.
- ✓ Scientific artifacts now use deterministic fingerprints, admitted-parent lineage, strict schema checks, checksums, and atomic publication — validated in Phase 4.

### Active

None. Development stops after Phase 4 by project-owner decision.

### Out of Scope

- Full replacement of the tutorial notebooks or their teaching narrative — this milestone improves reliability without redesigning the curriculum.
- New biological datasets or claims — validation uses current public cohorts and synthetic fixtures.
- Fine-tuning commercial pathology foundation models — current encoders remain frozen and research-oriented.
- Large storage/performance migrations such as Zarr or distributed training — correctness and artifact contracts come first.
- Broad package renaming and removal of all path bootstrapping — useful follow-up work, but outside the finalized milestone.
- Safe non-pickle patch caches and `weights_only=True` checkpoints — current compatibility readers are restricted to artifacts produced locally by this repository and must not receive untrusted files.
- Leakage-free nested model selection, fold-class admission, train-only target scaling/imputation, expanded seeding, image-quality contracts, confidence-aware labels, and a locked environment contract — intentionally not included in the final four-phase milestone.

## Context

The GSD codebase map identifies a linear notebook workflow at the repository root and a more conventional Python module stack under `projects/spatial-pharma-dl/src/`. The highest-risk issues concentrate at scientific evaluation boundaries: the held-out LOSO slide currently participates in early stopping, stain normalization does not use a true shared target, RF preprocessing observes validation statistics, and multi-task targets are unscaled. Artifact boundaries also need hardening because patch caches and checkpoints use pickle-backed loaders, cache reuse is weakly keyed, and writes are not atomic. Existing automated coverage is fast but narrow: eight tests cover import behavior, Grad-CAM cleanup, and foundation-probe logic, while preprocessing, patch geometry, cache round trips, training, and notebooks lack representative verification.

## Constraints

- **Behavioral compatibility**: Keep notebook order, documented CLI entry points, config keys, output names, and public Python exports stable unless a security-safe artifact migration is explicitly documented.
- **Scientific interpretation**: The optional pharma extension remains educational/research-oriented; Phase 4 does not certify leakage-free model selection or train-only learned transforms.
- **Offline tests**: Default automated tests must not download datasets or model weights.
- **Resource budget**: Fast CI must remain CPU-compatible; network and full-cohort tests are opt-in tiers.
- **Security boundary**: Patch caches and checkpoints are trusted-local artifacts only; never load untrusted cache or checkpoint files.
- **Traceability**: Every improvement maps to one requirement, targeted tests, and a reviewable GSD phase/commit.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| End the milestone after Phase 4 | The project owner chose to finalize the implemented validation and durable-artifact scope and remove Phases 5–10 | ✓ Finalized 2026-07-25 |
| Build synthetic fixtures before expensive integration tests | Enables leakage and artifact-contract tests without downloads | ✓ Validated in Phase 1: strict offline tiers, deterministic fixtures, and 58 tests |
| Keep cache/checkpoint compatibility local-writer-only | The safe-format migration was removed from scope; documentation must not imply hostile-input safety | ✓ Documented limitation |
| Preserve existing notebook and CLI surfaces | Reliability work should not become a user-facing redesign | ✓ Preserved through Phases 1-3 while tightening validation, identity, and preprocessing admission |
| Treat `(slide_id, spot_id)` as the sole spot identity | Spot barcodes may repeat across slides and row order is not identity | ✓ Validated in Phase 3 across ordinary and foundation consumers |
| Resolve scientific dimensions from observed retained data | Requested dimensions can become illegal after QC or actual HVG selection | ✓ Validated in Phase 3 with canonical provenance and real Scanpy/H5AD evidence |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-25 after Phase 4 finalization*
