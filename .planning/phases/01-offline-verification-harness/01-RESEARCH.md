# Phase 1: Offline Verification Harness - Research

**Researched:** 2026-07-17
**Requirement:** TEST-01
**Confidence:** High

## Summary

Phase 1 should turn the repository's existing eight deterministic tests into one explicit verification contract: every collected test belongs to a named evidence tier, the required tier is CPU-only and denied network access, and later phases extend the same fixtures and markers. The repository already demonstrates the right low-cost testing style in `test_core_refactors.py` and `test_foundation.py`: small in-memory NumPy/Pandas/Torch objects, explicit random seeds, no real model loading, and public API assertions. The missing pieces are shared fixtures, representative AnnData/image/artifact/fold contracts, notebook-wide structure checks, a tier-aware runner, and CI.

The implementation should remain verification infrastructure. It should not fix the unsafe production patch/checkpoint formats, leakage, preprocessing dimensions, border geometry, or label semantics assigned to later phases. "Safe artifact round trip" in this phase should prove that the shared fixture formats (numeric NPZ with `allow_pickle=False`, Parquet, JSON, and H5AD) can be produced and read safely; Phases 4 and 5 will connect the production cache APIs to those contracts. Likewise, model/fold smoke tests should verify tensor and LOSO orchestration contracts without claiming that the current outer-fold early-stopping behavior is scientifically valid.

## Current Repository Evidence

- Tests live only in `projects/spatial-pharma-dl/tests/`; there is no root test harness, `conftest.py`, marker configuration, or CI workflow.
- The current command `python -m pytest -p no:cacheprovider projects/spatial-pharma-dl/tests -q` passes eight tests and performs no downloads.
- Ruff currently passes when invoked over `utils`, root `scripts`, pharma `src`, pharma `scripts`, and pharma `tests`, but neither its paths nor rules are checked into configuration.
- `pyproject.toml` declares package metadata and a `pharma` optional dependency group only. It has no pytest or Ruff configuration.
- There are exactly 20 committed notebooks: 13 root tutorial notebooks (`00` through `12`) and seven pharma notebooks (`01` through `07`). Root notebooks use the `python3` kernel; pharma notebooks use `spatial-tx`. Older generated pharma notebooks do not all have cell IDs, so Phase 1 must not make cell IDs a compatibility-breaking structural requirement.
- Real AnnData construction is available through the base requirements. The reusable patch functions accept AnnData-like objects directly, so a tiny real `AnnData` with `obsm["spatial"]` and `uns["spatial"]` can test axis alignment, image lookup, coordinate scaling, H5AD serialization, patch tensors, and metadata without public data.
- `src.models.build_model(..., pretrained=False)` avoids weight downloads. Existing `MeanEncoder` and `_TinyCamModel` test doubles are suitable patterns for cheaper contract tests.
- Production patch caches currently embed object metadata and load with `allow_pickle=True`; production checkpoints use `weights_only=False`. Phase 1 tests must not bless those paths as safe. They should establish safe fixtures that later phases use to drive the production migration.
- `src.train.loso_folds()` is cheap and deterministic, but `train_one_fold()` writes to repository-derived output directories and currently uses the held-out slide for early stopping. A Phase 1 smoke should isolate fold enumeration/orchestration and one CPU training step rather than treat the current full training routine as a validity oracle.

## Recommended Harness Shape

### One canonical runner

Add a small cross-platform Python entry point, preferably `scripts/verify.py`, rather than introducing Make, tox, or nox. It should expose these stable commands:

```text
python scripts/verify.py fast
python scripts/verify.py notebook-smoke
python scripts/verify.py network
python scripts/verify.py full-cohort
```

`fast` should run Ruff first and then the offline pytest tier. The other commands should invoke only their declared pytest tier and should never be called by the required PR job. The runner should use `subprocess.run(..., check=True)` with argument lists, print the exact tier being run, and propagate the first nonzero exit code. Keep individual underlying Ruff/pytest commands documented for debugging.

Configure markers in `pyproject.toml` with `--strict-markers`. Use exactly one primary tier per test:

- `offline`: required, deterministic, CPU-safe, no downloads or private data.
- `notebook_smoke`: opt-in execution of representative notebooks with local/synthetic inputs and bounded timeouts.
- `network`: opt-in checks that intentionally download public data or model weights.
- `full_cohort`: opt-in end-to-end execution over the configured cohort; may also need credentials/network, but remains one primary tier for unambiguous selection.

Optional descriptive markers (`unit`, `integration`, `artifact`, `model`, `notebook_structure`) are useful for diagnostics, but primary-tier enforcement matters more. A collection hook should fail when a test has zero or multiple primary-tier markers. Mark the two existing modules `offline` so the convention applies immediately and later phases cannot accidentally add an unclassified default test.

Do not set a broad pytest `addopts` marker expression that makes opt-in tiers difficult to select. Let `scripts/verify.py` and CI pass the exact `-m` expression. `python -m pytest` may remain a developer discovery command, while the documented acceptance command is `python scripts/verify.py fast`.

### Network denial, not merely offline intent

The offline tier needs an autouse network guard in `conftest.py`. Patch `socket.socket.connect` and `socket.create_connection` to raise an actionable error naming the active test and directing intentional external checks to the `network` tier. Also set established offline environment flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) for the test session. This catches ordinary HTTP clients because they eventually use sockets; it also makes Hugging Face failures immediate.

Provide a narrowly scoped escape only when the selected primary tier is `network` or `full_cohort`; do not expose a casual per-test bypass in the offline tier. A test of the guard itself should attempt a loopback/socket connection and assert that collection/execution fails before any external connection is made. CI should additionally set the offline environment flags, but the pytest guard is the enforceable local contract.

Never instantiate a backbone with `pretrained=True` in offline tests. Patch weight-loader entry points in the dedicated model smoke test so a future accidental pretrained default yields a clear failure even before socket access.

### Shared deterministic fixtures

Create `projects/spatial-pharma-dl/tests/conftest.py` and a small fixture helper module if needed. Fixtures should be factories, not large checked-in binary files, and every random value should come from a local `np.random.default_rng(fixed_seed)`.

The canonical synthetic AnnData fixture should include:

- 8-16 spots and 8-20 genes with an integer count matrix; known zero/low-quality observations and genes can be toggled by the factory.
- Unique string `obs_names` and gene-symbol `var_names`, including human mitochondrial and marker/module genes.
- `obs` fields used by downstream tests (`slide_id`, array row/column or label columns as appropriate).
- `obsm["spatial"]` coordinates containing both center and border locations.
- `uns["spatial"][library_id]["images"]["hires"]` containing a small deterministic RGB image, plus `scalefactors` with `tissue_hires_scalef` and `spot_diameter_fullres`.
- A small image with nonuniform tissue-like pixels so image and Macenko functions are exercised without relying on a real slide.

Build cohort/metadata factories on top of it:

- A valid three-slide cohort with stable ordering, at least two classes in each training partition, and one finite regression target.
- Key adversaries: null ID, duplicate `(slide_id, spot_id)`, unmatched label, unmatched patch, and cross-slide spot collision.
- Cohort/fold adversaries: empty cohort, one slide, single training class, unseen held-out class, empty aligned set, and missing configured slide.
- Image adversaries: grayscale/wrong-channel, invalid dtype/range, all-white/no-tissue, too few tissue pixels, rank-deficient stain data, and border coordinates.
- Artifact adversaries: missing keys, wrong dtype/shape, object-valued NPZ, truncated or invalid JSON, mismatched row counts, and corrupted bytes. These fixtures may initially be consumed only by fixture-contract tests and later production readers.

Return fresh objects from every fixture so tests cannot leak mutations. Use `tmp_path` for every H5AD, NPZ, Parquet, JSON, checkpoint, and report. Never write to `data/` or `outputs/` in the required tier.

### Representative fast evidence

Add focused modules under the existing pharma test directory rather than creating a second root suite:

1. `test_fixture_contracts.py`: determinism, fresh-object behavior, valid/adversarial variants, and cohort/key/image/fold coverage.
2. `test_artifact_roundtrips.py`: numeric/string NPZ loaded explicitly with `allow_pickle=False`, Parquet metadata schema/values, JSON primitive metadata, and H5AD axis/spatial metadata. Include an object-array NPZ rejection test. This is fixture-format evidence, not a call to the current unsafe `load_patch_arrays()`.
3. `test_synthetic_anndata.py`: real AnnData H5AD round trip, image/scalefactor retrieval, coordinate scaling, patch tensor/metadata shape, and simple label/patch alignment on a valid fixture. Avoid asserting behavior that later validation phases are supposed to change.
4. `test_model_fold_smoke.py`: one CPU forward/backward/optimizer step on a tiny model and patch batch; public `build_model(..., pretrained=False)` output-shape smoke for the default backbone; deterministic `loso_folds()` coverage; and `train_loso()` orchestration with `train_one_fold` monkeypatched so each held-out slide appears once without writes or expensive fitting.
5. `test_notebook_structure.py`: discover all committed notebooks, assert the root `00..12` and pharma `01..07` sequences, parse with `nbformat`, validate notebook format, verify nonempty cells and allowed cell types, ensure code-cell source is text, and verify the established kernel family by directory. Do not require execution, outputs to be empty, exact cell counts, or cell IDs.
6. `test_verification_contract.py`: primary-tier classification and network-guard behavior; unit-test the command construction in `scripts/verify.py` without recursively launching the full suite.

Preserve the existing tests and their direct `sys.path` setup. Moving or installing the pharma package is PKG-01 and outside this milestone. Shared path setup may be centralized in `conftest.py` only if existing imports continue to behave identically.

### CI layout

Add `.github/workflows/verify.yml` with one required `fast` job on pull requests and pushes to `main`. It should use Python 3.11 and Ubuntu CPU, install the existing base/pharma requirements plus explicit test tools, and run exactly `python scripts/verify.py fast`. Set Hugging Face/Transformers offline flags at job scope. Use dependency caching only; do not cache `data/`, `outputs/`, pretrained weights, or generated scientific artifacts.

Declare separate named jobs for `notebook-smoke`, `network`, and `full-cohort`, triggered only by `workflow_dispatch` inputs and/or schedules. In Phase 1 these may contain only marker-selection smoke/placeholder coverage when no executable tests exist, but the workflow and documented commands must distinguish them. Do not let empty marker selections silently pass as evidence: the runner should report "no tests defined for this opt-in tier" distinctly, while only `fast` is required now.

Installation can initially follow the repository's existing declarations because ENV-01 owns dependency reconciliation and locking. Do not solve the Python-version/PyArrow declaration mismatch in this phase. CI may install `requirements.txt`, `requirements-pharma.txt`, `pytest`, `ruff`, and `nbformat` explicitly; Phase 10 will replace this with the authoritative locked contract.

## Suggested Plan Decomposition

### Plan 01-01: Tier and fixture contract

- Configure pytest markers and Ruff paths/rules without changing public runtime imports.
- Add the shared path setup, primary-tier enforcement, socket denial, offline environment, deterministic RNG, AnnData/cohort/image/key/fold/artifact factories.
- Mark existing tests as offline and add fixture/network-contract tests.
- Verify that all tests use temporary paths and the offline tier cannot open a socket.

### Plan 01-02: Representative offline evidence

- Add safe fixture-format round trips, synthetic AnnData integration, model/fold smoke, and notebook structure tests.
- Keep model weights disabled, training bounded to one tiny CPU step, and notebook checks structural only.
- Add adversarial fixture assertions without encoding the later phases' desired production behavior prematurely.

### Plan 01-03: Runner, documentation, and CI tiers

- Add `scripts/verify.py` and document both canonical and underlying commands.
- Add the required GitHub Actions fast job and opt-in tier entry points.
- Run the exact required command from a clean process and confirm Ruff plus all offline categories appear in output.

This order gives Plans 01-02 and 01-03 a stable marker/fixture contract and keeps commits independently reviewable.

## Validation Architecture

Nyquist validation should treat TEST-01 as an infrastructure requirement with observable coverage across commands, collection, runtime boundaries, and representative behaviors.

### Validation layers

| Layer | Evidence | Required tier | Failure detected |
|---|---|---|---|
| Static | Ruff over `utils`, root scripts, pharma source/scripts/tests | fast/offline | syntax, imports, style regressions |
| Harness contract | marker enforcement, runner command construction, socket denial, offline env | fast/offline | unclassified tests, accidental network/model download, tier drift |
| Fixture contract | repeatable valid/adversarial AnnData, cohort, key, image, fold, artifact factories | fast/offline | nondeterminism, mutation leakage, missing adversarial coverage |
| Artifact fixture round trip | numeric NPZ (`allow_pickle=False`), Parquet, JSON, H5AD | fast/offline | object deserialization, schema/value loss, spatial-axis loss |
| Synthetic integration | AnnData → image/scales/coordinates → patch tensor and metadata → aligned table | fast/offline | incompatible spatial metadata, tensor/table shape mismatch |
| Model/fold smoke | one CPU optimization step, no-weight-download public model shape, one holdout per LOSO orchestration | fast/offline | tensor contract breakage, download regression, fold omission/duplication |
| Notebook structure | exactly `00..12` root and `01..07` pharma; `nbformat` validation and established kernels | fast/offline | missing/renamed/malformed notebook, broken kernel metadata |
| Executable notebooks | representative bounded execution with synthetic/local caches | notebook-smoke | runtime notebook/import/cache handoff failures |
| External integration | public downloads and real foundation weight loading | network | upstream/API/auth/model availability failures |
| Scientific scale | complete configured cohort and reports | full-cohort | scale-, cohort-, or resource-dependent failures |

### Requirement-to-test map

| TEST-01 clause | Automated evidence |
|---|---|
| Fast CI runs Ruff | `scripts/verify.py fast` command unit test plus CI invocation log |
| Unit tests | Existing eight tests retained and classified `offline`; new fixture/harness unit tests |
| Safe artifact round trips | `test_artifact_roundtrips.py`, explicit `allow_pickle=False`, object array rejected |
| Synthetic AnnData integration | `test_synthetic_anndata.py` using real `anndata.AnnData` and `tmp_path` H5AD |
| Model/fold smoke | one tiny CPU step, public no-pretrained shape check, deterministic LOSO orchestration |
| Notebook structural checks | discovery/sequence, `nbformat.validate`, kernel/source/cell-type checks for all 20 notebooks |
| No downloads/private data | socket guard test, offline env, no repository data paths, no pretrained weights |
| Explicit slow tiers | strict primary markers, four runner commands, separate CI job names/triggers |
| Later-phase extension contract | collection fails for missing/multiple tier markers; fixture factories are shared |

### Per-plan verification commands

After Plan 01-01:

```bash
python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_core_refactors.py projects/spatial-pharma-dl/tests/test_foundation.py projects/spatial-pharma-dl/tests/test_fixture_contracts.py projects/spatial-pharma-dl/tests/test_verification_contract.py
```

After Plan 01-02:

```bash
python -m pytest -q --strict-markers -m offline projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py projects/spatial-pharma-dl/tests/test_synthetic_anndata.py projects/spatial-pharma-dl/tests/test_model_fold_smoke.py projects/spatial-pharma-dl/tests/test_notebook_structure.py
```

After Plan 01-03 and for phase acceptance:

```bash
python scripts/verify.py fast
```

The phase acceptance run must demonstrate all of the following in one clean process:

1. Ruff succeeds over every configured Python path.
2. Pytest collects only `offline` tests for the required tier and reports nonzero counts for artifact, AnnData integration, model/fold, and notebook-structure evidence.
3. No network socket or model download occurs; the guard's self-test passes.
4. Tests write only under pytest temporary directories.
5. Repeated fixture/test runs produce equal data and fold ordering.
6. Notebook discovery reports 13 ordered root notebooks and seven ordered pharma notebooks without modifying them.
7. Opt-in commands select their own primary marker and are absent from the required CI job.

### Sampling and escalation policy

- Run focused tests after each task, the complete offline tier after each plan, and the runner/CI-equivalent command at phase completion.
- Treat any test that requires network, a pretrained weight, repository `data/`, repository `outputs/`, GPU, or more than a small CPU fixture as mis-tiered and move it out of `offline`.
- Do not xfail a missing TEST-01 behavior in the required tier. Xfails for later requirements may be added only in their owning phases, with requirement IDs and removal conditions.
- Runtime should be recorded on the first CI run. Aim for a few minutes, with individual fixture tests in seconds and the model smoke bounded to one small batch/step. If the default torchvision backbone dominates runtime, retain one public shape smoke and use the tiny model for optimization tests rather than dropping model coverage.

## Pitfalls to Avoid

- Do not call the current production `load_patch_arrays()` as the safe artifact proof; it requires pickle and belongs to ART-01 migration.
- Do not run `train_one_fold()` as an unbiased evaluation test; its current held-out-slide early stopping is explicitly EVAL-01 work.
- Do not execute all notebooks in the required tier. Many intentionally download data or depend on earlier caches; Phase 1 requires structural validation and a separately named notebook-smoke tier.
- Do not require new cell IDs in generated pharma notebooks. That would rewrite notebook artifacts without improving TEST-01.
- Do not build torchvision models with pretrained weights, import real timm/Transformers backends, or rely on a populated user cache.
- Do not let tests write to repository-relative processed/output helpers. Monkeypatch path boundaries or test lower-level pure functions with `tmp_path`.
- Do not create a second independent root test convention. The existing pharma test directory is the stable suite; the runner can orchestrate repository-wide static and notebook checks from there.
- Do not reconcile all dependencies or introduce a lock here. CI should expose existing declarations; ENV-01 owns the durable dependency contract.
- Do not use global RNG state in fixture construction or assert bitwise equivalence across platforms. Assert reproducibility for repeated construction in the same locked CPU environment.

## Files Likely to Change During Phase 1

- `pyproject.toml` — pytest markers/strictness and Ruff configuration.
- `scripts/verify.py` — canonical tier runner.
- `.github/workflows/verify.yml` — required fast job and explicit opt-in tier entry points.
- `projects/spatial-pharma-dl/tests/conftest.py` — imports, tier enforcement, network denial, and fixture factories.
- `projects/spatial-pharma-dl/tests/test_core_refactors.py` and `test_foundation.py` — offline tier marker only, preserving test behavior.
- New focused test modules described above.
- `README.md` and/or `projects/spatial-pharma-dl/README.md` — concise verification commands and tier meanings.

Production modules under `projects/spatial-pharma-dl/src/` should not need behavioral changes for this phase. If a small seam is truly required for temporary-path injection or command testing, keep it backward-compatible and do not implement a later requirement incidentally.

## Research Conclusion

TEST-01 is best satisfied by an enforceable tier contract and deterministic fixture vocabulary, not by broad end-to-end execution. The required job can be meaningfully representative while staying offline: safe primitive artifact round trips, real tiny AnnData spatial integration, a bounded CPU model/fold smoke, all-notebook structure validation, existing unit regressions, and Ruff. The main architectural guardrail is that Phase 1 supplies the evidence machinery while Phases 2-10 supply the production behavior those fixtures will increasingly verify.

---

*Phase: 01-offline-verification-harness*
*Research completed: 2026-07-17*
