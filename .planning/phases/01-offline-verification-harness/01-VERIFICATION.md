---
phase: 01-offline-verification-harness
status: passed
score: "21/21"
requirement: TEST-01
date: 2026-07-17
verifier: independent-gsd-verifier
---

# Phase 1 Verification

## Result

Phase 1 achieves its goal: maintainers can run representative tutorial and
pharma reliability checks quickly on CPU without public downloads, private
data, pretrained weights, or full-cohort inputs. TEST-01 is verified against
the repository behavior, not only against the plan summaries.

The score covers all 12 plan-level must-have truths, four threat controls,
four roadmap success criteria, and the TEST-01 requirement itself.

## Goal and Requirement Evidence

### TEST-01 — Passed

`python scripts/verify.py fast` ran the checked-in Ruff scope first and then
58 strict-marker offline tests. The command exited 0 in 16.06 seconds in the
verification environment. The suite contains nonzero evidence for:

- existing unit regressions;
- safe primitive NPZ, Parquet, JSON, and H5AD round trips;
- a real synthetic `anndata.AnnData` spatial integration path;
- a bounded CPU model step and deterministic LOSO orchestration;
- structural validation of all 20 committed notebooks;
- tier, network-denial, CI, documentation, and fixture contracts.

No Phase 1 production module was changed. The evidence deliberately does not
claim that the later production cache/checkpoint migrations are complete.

## Plan Must-Haves

### Plan 01-01 — 4/4

- **D-01:** `pyproject.toml` registers exactly the four primary tiers and bare
  pytest defaults to `offline`. `conftest.py` requires exactly one primary
  marker, sets both model-hub offline flags, and denies socket connection and
  resolver APIs in the pytest process and inheriting child Python processes.
- **D-02:** fixed-seed factories return fresh CPU-small AnnData, cohort, key,
  fold, image, and artifact objects. Serialized adversaries are confined to
  pytest `tmp_path`; a repository `data/`/`outputs/` state test passes.
- **D-03:** bare `python -m pytest -q` exited 0 with 58 offline tests. Explicit
  notebook-smoke, network, and full-cohort selections each deselected all 58
  offline tests and exited 5, so an empty opt-in cannot appear as evidence.
- **D-04:** the shared `conftest.py`, four markers, factories, runner, and CI
  command form one extension contract. Notebook numbering, pipeline commands,
  package discovery, and public runtime imports were not changed.

### Plan 01-02 — 4/4

- **D-01:** the fast suite includes 4 artifact, 3 synthetic AnnData, 3
  model/fold, and 22 notebook-structure tests in addition to the harness and
  existing regression tests.
- **D-02:** these tests consume the shared deterministic factories, use small
  arrays/models, run on CPU, and store artifacts only beneath temporary paths.
- **D-03:** notebook structure is checked offline for exactly root `00..12`
  and pharma `01..07`; notebook execution remains a distinct opt-in tier.
- **D-04:** tests exercise existing public image, scale, coordinate, patch,
  model, and fold seams. Source scans found no unsafe production reader, real
  training, model-hub, or later-phase migration call in the representative
  Phase 1 modules.

### Plan 01-03 — 4/4

- **D-01:** `scripts/verify.py` builds argument-list commands, runs Ruff before
  pytest, uses `subprocess.run(..., check=True)`, and propagates the first
  nonzero status.
- **D-02:** the canonical fast run completed in 16.06 seconds, well below the
  300-second target, with 58 tests passing.
- **D-03:** `--help` exposes exactly `fast`, `notebook-smoke`, `network`, and
  `full-cohort`. GitHub Actions requires only fast on pull requests and pushes
  to `main`; the other three jobs are separately dispatch-gated.
- **D-04:** both READMEs document all canonical and direct debugging commands,
  the offline/opt-in boundary, the safe-fixture limitation, the child-Python
  guard boundary, and the shared later-phase extension convention.

## Threat Controls

- **T-01 — Passed:** in-process tests prove `connect`, `connect_ex`,
  `create_connection`, `getaddrinfo`, and hostname lookup fail closed. The
  child-Python probe proves the inherited `sitecustomize.py` guard is active.
- **T-02 — Passed:** offline flags are set and restored correctly; the public
  ResNet18 smoke passes `pretrained=False` and fails if weight selection asks
  for pretrained weights. No foundation/model-hub backend is invoked.
- **T-03 — Passed:** fixture repeatability and mutation isolation are tested.
  Model construction and optimization run inside fixed-seed
  `torch.random.fork_rng()`, produce exactly repeated losses/parameters, and
  preserve the caller's RNG state.
- **T-04 — Passed:** NPZ reads use `allow_pickle=False`; object payload access
  raises. Source inspection found no `allow_pickle=True`, `weights_only=False`,
  `load_patch_arrays`, or `load_model_from_checkpoint` usage in the four
  representative evidence modules.

## Roadmap Success Criteria

1. **Passed:** the documented fast tier runs Ruff, unit regressions, safe
   primitive artifact round trips, real synthetic AnnData integration,
   model/fold smoke, and all-notebook structural checks under offline guards.
2. **Passed:** deterministic valid/adversarial factories cover cohorts, keys,
   images, folds, and serialized artifacts with fresh returned objects.
3. **Passed:** fast, notebook-smoke, network, and full-cohort are distinct in
   runner help, pytest markers, documentation, and GitHub Actions jobs.
4. **Passed:** strict primary-marker enforcement and shared fixtures/commands
   give later phases one established convention to extend.

## Automated Checks

| Check | Result | Evidence |
|---|---:|---|
| `python scripts/verify.py fast` | Passed | Ruff passed; 58 tests passed; 16.06 s |
| `python -m pytest -q` | Passed | 58 offline tests passed; external tiers absent |
| `python scripts/verify.py --help` | Passed | Exactly four documented tier choices |
| `python scripts/verify.py notebook-smoke` | Expected non-evidence | Exit 5; 58 deselected; explicit diagnostic |
| `python scripts/verify.py network` | Expected non-evidence | Exit 5; 58 deselected; explicit diagnostic |
| `python scripts/verify.py full-cohort` | Expected non-evidence | Exit 5; 58 deselected; explicit diagnostic |
| Safe-format boundary scan | Passed | No prohibited unsafe reader/load tokens in representative modules |
| Working-tree check after verification | Passed | Test execution created no tracked/untracked repository artifacts |

## CI and Documentation Contract

`.github/workflows/verify.yml` has pull-request and push-to-`main` triggers, a
required Ubuntu/Python 3.11 fast job, both model-hub offline variables, and the
exact `python scripts/verify.py fast` command. Opt-in jobs are independently
dispatch-gated and do not use `continue-on-error`. Only pip dependency caching
is configured; scientific data, outputs, weights, and artifacts are not cached.

The workflow contract was validated statically by the passing offline suite.
The identical canonical command was executed locally; this verification did
not independently dispatch a remote GitHub Actions run.

## Warnings

- The local verification interpreter was Python 3.12, while the required CI
  contract specifies Python 3.11. Phase 10 owns environment reconciliation;
  the Python 3.11 workflow setting is machine-checked here.
- Pandas reported outdated optional `numexpr` and `bottleneck` accelerators.
  They are not required by the verified behavior and caused no failures.
- Six legacy pharma notebooks lack cell IDs and emit `nbformat` warnings.
  They still validate successfully; Phase 1 intentionally avoids rewriting
  notebook metadata for this non-blocking compatibility warning.
- The network control is a Python test-harness guard, not an operating-system
  sandbox for arbitrary native executables. Both READMEs state this boundary.

## Gaps and Human Items

No Phase 1 gaps or required human verification items remain. Network,
executable-notebook, and full-cohort evidence are intentionally empty opt-in
tiers and are not acceptance gates for TEST-01.

---

*Independent verification completed 2026-07-17.*
