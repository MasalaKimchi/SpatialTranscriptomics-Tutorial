---
last_mapped_commit: a687bd5
mapping_focus: quality
---

# Testing and Validation

## Current Test Suite

- Automated tests live only under `projects/spatial-pharma-dl/tests/`; the root tutorial helpers and notebooks do not have a dedicated test directory.
- The suite uses pytest-style functions and plain `assert`, with NumPy assertions for numerical arrays; shared tier enforcement and deterministic factories live in `projects/spatial-pharma-dl/tests/conftest.py`.
- `projects/spatial-pharma-dl/tests/test_core_refactors.py` contains three regression tests for lazy imports, idempotent notebook patching, and Grad-CAM hook cleanup.
- `projects/spatial-pharma-dl/tests/test_foundation.py` contains five tests for embedding normalization, probe behavior, encoder registry independence, slide preprocessing, and nested leave-one-slide-out classification.
- Test doubles are small in-memory Torch modules such as `MeanEncoder` and `_TinyCamModel`; synthetic NumPy/Pandas data avoids network access and real slide downloads.
- `conftest.py` centralizes the repository and pharma import paths needed for the uninstalled nested `src` package.
- The suite is deterministic: synthetic random data uses `np.random.default_rng()` with explicit seeds and CPU-sized models.
- Bare `python -m pytest -q` defaults to the `offline` primary tier and completes with 18 passing tests after Plan 01-01.
- `test_verification_contract.py` proves tier classification, offline environment flags, socket denial, and default deselection of opt-in evidence.
- `test_fixture_contracts.py` proves fresh deterministic AnnData, cohort, key, fold, image, and artifact adversary factories confined to pytest temporary paths.
- The current environment emits two dependency warnings during collection: Pandas reports outdated local `numexpr` and `bottleneck` versions; these warnings do not fail the suite.

## Static and Structural Checks

- `ruff check utils scripts projects/spatial-pharma-dl/src projects/spatial-pharma-dl/scripts projects/spatial-pharma-dl/tests` passes at the mapped commit.
- `pyproject.toml` declares strict `offline`, `notebook_smoke`, `network`, and `full_cohort` pytest markers plus the repository's Python 3.11 Ruff source/rule scope.
- No GitHub Actions workflow, pre-commit configuration, tox/nox setup, Makefile, or checked-in CI command was found.
- There is no configured coverage report or minimum threshold, so the repository cannot currently quantify line or branch coverage.
- There is no static type-checking gate; type hints improve readability but are not verified by mypy or pyright.
- Notebook JSON validity and executable notebook behavior are separate concerns: the repository contains 20 `.ipynb` files, but the pytest suite does not execute them.
- `scripts/patch_notebooks.py` has a focused idempotence regression test, but its remaining notebook-specific patch functions and generated cell positions are not individually checked.

## Well-Covered Behavior

- Lightweight package import behavior is protected against accidental eager Scanpy imports in `projects/spatial-pharma-dl/tests/test_core_refactors.py`.
- Grad-CAM resource lifecycle is covered by comparing hook counts before and after `grad_cam_for_patch()`.
- Foundation embedding extraction checks output shape and documented normalization behavior without downloading a real model.
- Linear-probe testing verifies strong synthetic classification/regression performance and ensures training does not mutate caller-owned embeddings.
- Nested LOSO testing verifies one unseen slide per fold and validates a high-signal synthetic classification scenario.
- Foundation registry testing ensures Kaiko and Phikon retain distinct dimensions and backends.

## Major Coverage Gaps

- `utils/st_helpers.py` has no unit tests for path creation, AnnData image metadata guards, cache errors, gene filtering, or the Leiden compatibility fallback.
- `projects/spatial-pharma-dl/src/data.py` lacks tests for configuration loading, preprocessing, cache naming, cohort skipping, tumor-type mapping, and malformed AnnData inputs.
- `projects/spatial-pharma-dl/src/labels.py` lacks tests for marker annotation, class harmonization, regression-column modes, label construction, duplicate identifiers, and label/patch merge loss.
- `projects/spatial-pharma-dl/src/patches.py` lacks tests for edge crops, stain estimation/normalization, feature extraction, cache round trips, metadata integrity, empty slides, and its optional Torch import branch.
- `projects/spatial-pharma-dl/src/models.py` and `projects/spatial-pharma-dl/src/train.py` lack tests for supported backbones, invalid model configuration, checkpoint loading, multitask losses, early stopping, sampling, and LOSO orchestration.
- `projects/spatial-pharma-dl/src/eval.py` lacks direct metric edge-case tests, RF baseline tests, prediction batching tests, and benchmark-report persistence tests.
- `projects/spatial-pharma-dl/src/benchmark.py` and `projects/spatial-pharma-dl/scripts/run_pipeline.py` have no orchestration tests or mocked end-to-end smoke test.
- Real foundation backends, Hugging Face authentication failures, model-cache behavior, and unsupported dependency combinations are not exercised; offline socket/model-hub denial is now enforced and tested.
- Root tutorial notebooks are not executed in order, so cache handoffs, downloaded data, plots, optional exercises, and compatibility across Scanpy/Squidpy versions remain unverified.
- No tests cover `scripts/generate_gallery_figures.py` or the complete behavior of the three notebook-builder scripts.
- Security-sensitive patch cache loading in `projects/spatial-pharma-dl/src/patches.py` still uses pickle-backed NumPy metadata and is not tested against malformed or untrusted cache files.

## Recommended Testing Priorities

1. Add fast unit tests around `utils/st_helpers.py`, config validation, patch geometry, label alignment, and cache serialization because these are shared data-boundary utilities.
2. Add synthetic AnnData integration tests that run preprocessing, label construction, patch extraction, and model input assembly without downloading public data.
3. Add model/training smoke tests for one CPU batch, checkpoint save/load, deterministic sampling, and each supported backbone with pretrained downloads disabled.
4. Add notebook structural tests for required kernels, cell identifiers/content, idempotent generation, and valid JSON for all 20 notebooks.
5. Add an opt-in slow test that executes the tutorial pipeline on a tiny cached fixture, plus separate network/model-download markers.
6. Configure coverage and CI in `pyproject.toml` and `.github/workflows/`, enforcing Ruff, pytest, notebook validation, and a documented Python version matrix.
7. Treat dependency warnings as actionable in CI after environments are pinned, so local scientific-stack drift does not silently accumulate.
