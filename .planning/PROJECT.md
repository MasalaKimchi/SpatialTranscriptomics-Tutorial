# Spatial Transcriptomics Tutorial Reliability Upgrade

## What This Is

This repository is a notebook-first spatial transcriptomics tutorial with an optional pharma-facing deep-learning extension for predicting tumor-microenvironment molecular state from H&E patches. This milestone strengthens the existing implementation through exactly 20 high-priority correctness, security, reproducibility, validation, and test-infrastructure improvements while preserving its educational workflow and public outputs.

## Core Value

Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.

## Requirements

### Validated

- ✓ Users can follow a numbered, cache-backed Visium tutorial from data loading through spatial analysis — existing.
- ✓ Users can run a pharma extension with cohort preprocessing, label engineering, patch extraction, LOSO benchmarking, and foundation-model probes — existing.
- ✓ The pharma package exposes lightweight imports and focused foundation-model tests pass without eagerly importing Scanpy — existing.
- ✓ Repository-relative paths, deterministic NumPy/Python seeds, configurable experiments, and committed figure previews are established conventions — existing.

### Active

- [ ] Prevent outer LOSO test slides from influencing CNN model selection.
- [ ] Normalize stains from each source slide into a shared cohort reference basis.
- [ ] Replace pickle-backed patch cache metadata with safe serialization.
- [ ] Load model checkpoints without enabling arbitrary pickle execution.
- [ ] Enforce one-to-one, complete label/patch alignment with actionable errors.
- [ ] Fingerprint caches against configuration, source data, and relevant code contracts.
- [ ] Make cache and result writes atomic and validate completed artifacts.
- [ ] Validate the complete experiment configuration before pipeline execution.
- [ ] Seed PyTorch, data loaders, workers, and deterministic backend policy centrally.
- [ ] Reject empty cohorts, folds, patch sets, and prediction batches early.
- [ ] Validate class support and unseen-class coverage for every LOSO fold.
- [ ] Fit regression-target scaling only on training data and invert it for reports.
- [ ] Fit RF imputation and feature schema only on training data.
- [ ] Preserve fixed physical context at image borders and record patch-quality flags.
- [ ] Validate Macenko inputs and numerical outputs with explicit fallback provenance.
- [ ] Fail on missing configured slides unless partial-cohort mode is explicitly enabled.
- [ ] Add confidence, abstention, and provenance to heuristic scientific labels.
- [ ] Adapt preprocessing dimensions safely after QC and record resolved parameters.
- [ ] Add fixture-backed unit, integration, notebook, and CI validation tiers.
- [ ] Consolidate supported Python/dependency declarations and produce a reproducible environment contract.

### Out of Scope

- Full replacement of the tutorial notebooks or their teaching narrative — this milestone improves reliability without redesigning the curriculum.
- New biological datasets or claims — validation uses current public cohorts and synthetic fixtures.
- Fine-tuning commercial pathology foundation models — current encoders remain frozen and research-oriented.
- Large storage/performance migrations such as Zarr or distributed training — correctness and artifact contracts come first.
- Broad package renaming and removal of all path bootstrapping — useful follow-up work, but lower priority than the 20 P0/P1 findings.

## Context

The GSD codebase map identifies a linear notebook workflow at the repository root and a more conventional Python module stack under `projects/spatial-pharma-dl/src/`. The highest-risk issues concentrate at scientific evaluation boundaries: the held-out LOSO slide currently participates in early stopping, stain normalization does not use a true shared target, RF preprocessing observes validation statistics, and multi-task targets are unscaled. Artifact boundaries also need hardening because patch caches and checkpoints use pickle-backed loaders, cache reuse is weakly keyed, and writes are not atomic. Existing automated coverage is fast but narrow: eight tests cover import behavior, Grad-CAM cleanup, and foundation-probe logic, while preprocessing, patch geometry, cache round trips, training, and notebooks lack representative verification.

## Constraints

- **Behavioral compatibility**: Keep notebook order, documented CLI entry points, config keys, output names, and public Python exports stable unless a security-safe artifact migration is explicitly documented.
- **Scientific validity**: All learned preprocessing, model selection, imputation, and scaling must use training data only within each outer fold.
- **Offline tests**: Default automated tests must not download datasets or model weights.
- **Resource budget**: Fast CI must remain CPU-compatible; network and full-cohort tests are opt-in tiers.
- **Security**: Untrusted cache and checkpoint reads must not execute Python objects.
- **Traceability**: Every improvement maps to one requirement, targeted tests, and a reviewable GSD phase/commit.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Scope exactly the top 20 P0/P1 GSD findings | Matches the requested count and prioritizes scientific/security risk over polish | — Pending |
| Use fine-grained GSD planning with automatic advancement | Keeps 20 changes reviewable while allowing autonomous execution | — Pending |
| Build synthetic fixtures before expensive integration tests | Enables leakage and artifact-contract tests without downloads | — Pending |
| Treat cache/checkpoint safety as documented migrations | Secure formats may invalidate legacy artifacts and require regeneration | — Pending |
| Preserve existing notebook and CLI surfaces | Reliability work should not become a user-facing redesign | — Pending |

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
*Last updated: 2026-07-17 after GSD initialization*
