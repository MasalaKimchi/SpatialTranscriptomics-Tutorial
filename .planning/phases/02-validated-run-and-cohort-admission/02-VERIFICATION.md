---
phase: 02-validated-run-and-cohort-admission
status: passed
score: "22/22"
requirements:
  - VAL-01
  - VAL-03
  - VAL-04
date: 2026-07-17
verifier: independent-gsd-verifier
---

# Phase 2 Verification

## Result

Phase 2 achieves its goal: only experiments whose configuration, stage inputs,
and final resolved cohort satisfy explicit contracts may start expensive or
output-producing work. VAL-01, VAL-03, and VAL-04 are verified against current
source behavior, including the Plan 02-04 closure of both gaps found during the
first independent verification.

The score covers 22 independent behavioral checks across all four plan
must-haves, D-01 through D-05, T-01 through T-05, the four roadmap success
criteria, three requirements, prior review closure, compatibility, and the
canonical regression gates. All 22 pass.

## Gap Closure

### G-01 — Closed: hostile primitive subclasses fail without execution

`resolve_config()` now admits an exact built-in dictionary at the root and
exact safe built-in primitive/container types at their permitted schema
positions. Root dict/Mapping subclasses and nested subclasses of `int`,
`float`, `str`, `list`, `tuple`, `dict`, arbitrary `Mapping`, and the concrete
platform `Path` type are rejected through `ConfigValidationError` before their
overridden operations can run.

The checked-in adversarial tests attach raising sentinels to representation,
hashing, comparison, length, iteration, lookup, `strip`, `bit_length`, and
path conversion. All sentinels remain zero. Equivalent reversed exact mappings
with the same hostile values produce byte-identical exception text containing
only inert bounded type labels and schema guidance.

Independent replay of the original failures produced:

```text
EvilInt training.batch_size -> ConfigValidationError
EvilStr experiment          -> ConfigValidationError
```

Neither hostile method executed and neither attacker message entered the
diagnostic. Exact built-in oversized integers, non-finite floats, invalid
primitive keys, list order, canonical mapping order, and fresh mutable
`to_dict()` behavior remain covered and unchanged.

### G-02 — Closed: explicit invalid configs cannot reload defaults

`foundation_config`, `foundation_model_spec`, `load_frozen_encoder`,
`load_or_extract_slide_embeddings`, and `save_benchmark_report` reserve default
loading for `cfg is None`. Every supplied configuration is resolved before any
cache-path, directory, path-existence, NumPy cache, device, diagnostic, encoder,
model, slide-patch, DataFrame, output-path, or writer seam.

Independent probes monkeypatched default loading and all relevant side effects
to fail, then called the four originally affected public helpers with `{}`.
Each raised `ConfigValidationError` before a forbidden seam. Checked-in tests
repeat this for both `{}` and a non-empty malformed mapping, including an
explicit report path, and prove no output is created.

The static audit below returned no match:

```text
rg -n "cfg\s*=\s*cfg\s+or\s+load_config\(\)|cfg\s+or\s+load_config\(\)" \
  projects/spatial-pharma-dl/src
```

Omitted configuration still loads the validated default facade once. Complete
supplied configurations preserve the existing encoder, cache-disabled
embedding, report filename, and return behavior without default reload.

## Twenty-Two Acceptance Checks

| # | Check | Result | Evidence |
|---:|---|---:|---|
| 1 | Aggregate schema defects | Passed | Plain invalid sections, keys, types, values, and cross-field combinations raise one ordered `ConfigValidationError`. |
| 2 | Hostile value rejection | Passed | Primitive, container, mapping, and Path subclasses are rejected without invoking overrides. |
| 3 | Canonical resolved configuration | Passed | Mapping keys sort, configured lists retain order, JSON is strict, and `to_dict()` views are fresh. |
| 4 | Configuration facade compatibility | Passed | Default/custom YAML paths and mutable plain-dict returns remain stable; only fail-closed cohort policy was added. |
| 5 | Immutable admitted run | Passed | Frozen records combine `ResolvedConfig` with ordered `cohort-manifest-v1`. |
| 6 | Strict known-availability admission | Passed | All known missing/failed configured members aggregate before processing. |
| 7 | Provisional remote admission | Passed | Provisional results publish nothing; strict source failure stops immediately with no false success. |
| 8 | Explicit partial admission | Passed | All configured outcomes are collected, re-admitted once, reason-coded, and unusable cohorts are rejected. |
| 9 | Deterministic sanitized manifest | Passed | Canonical JSON excludes caller tracebacks, paths, timestamps, reprs, sets, NaN, and infinity. |
| 10 | One downstream membership | Passed | Summary, labels, patches, stain fitting, and benchmarking consume the final admitted sequence and manifest-owned oncology subset. |
| 11 | Structured stage diagnostics | Passed | `StageValidationError` exposes stage, subject, observed count/shape, minimum, and corrective guidance. |
| 12 | Data/label/patch empty boundaries | Passed | Empty inputs fail before loader, directory, stack, cache, or writer seams. |
| 13 | LOSO and fold boundaries | Passed | Empty/one-slide LOSO fails early; nested LOSO requires three slides before task preprocessing. |
| 14 | Prediction and target boundaries | Passed | Empty CNN/RF inputs and regression selections fail before device/model/estimator work. |
| 15 | Foundation boundaries | Passed | Empty embeddings/tasks/probes and malformed explicit configs fail before encoder/cache/model work. |
| 16 | Guard-before-expense ordering | Passed | Forbidden directory, cache, device, model, encoder, estimator, probe, DataFrame, output, and writer seams remain unreachable. |
| 17 | Label publication ordering | Passed | Every admitted slide frame validates before output-directory creation or any table write. |
| 18 | CNN target-before-device ordering | Passed | Classification/regression target selection precedes device resolution and model construction. |
| 19 | Public compatibility | Passed | Valid signatures, lazy exports, environment flags, output names, fold order, task names, and return schemas remain stable. |
| 20 | Deep review closure | Passed | WR-01 through WR-07 and IN-01 are closed against actual code and adversarial tests. |
| 21 | Focused Phase 2 gate | Passed | 119 offline tests passed in 5.66 seconds. |
| 22 | Canonical fast gate | Passed | Ruff passed first; all 157 offline tests passed in 6.92 seconds. |

## Plan Must-Haves

### Plan 02-01 — Configuration contract

- **D-01 — Passed:** every safely discoverable defect aggregates once in
  deterministic schema order; hostile allowed-type subclasses cannot escape
  the domain boundary.
- **D-02 — Passed:** successful resolution is deterministic and strictly
  JSON-safe, preserves list order, rejects unsupported values without executing
  them, and returns fresh mutable compatibility trees.
- **D-05 — Passed:** `load_config(path=None)`, custom paths, existing keys, and
  ordinary dictionary behavior remain compatible.

### Plan 02-02 — Cohort admission

- **D-02 — Passed:** every admitted run contains canonical resolved config and
  an ordered deterministic manifest.
- **D-03 — Passed:** strict known-missing admission aggregates before work;
  strict remote source failure is fail-fast; explicit partial mode collects all
  outcomes and publishes only final admission.
- **D-05 — Passed:** the runner entry point, flags, stage order, output names,
  helper signatures, and public imports remain stable while one admitted
  sequence controls downstream work.

### Plan 02-03 — Empty scientific boundaries

- **D-04 — Passed:** every named empty cohort, fold, aligned set, patch set,
  prediction batch, and regression-target selection fails at its earliest
  public boundary with structured evidence.
- **D-03 — Passed:** empty or undersized LOSO inputs cannot reach cache, device,
  encoder, model, estimator, output-directory, or writer work.
- **D-05 — Passed:** valid non-empty signatures, outputs, task modes, schemas,
  notebook order, and public imports remain compatible.

### Plan 02-04 — Verification gap closure

- **G-01 / D-01 / D-02 / T-02 — Passed:** exact-type gates precede every
  caller-controlled operation and hostile subclasses fail deterministically.
- **G-02 / T-01 / D-05 — Passed:** only omitted config loads defaults; explicit
  invalid mappings fail before all foundation/report effects.
- **VAL-03 / VAL-04 compatibility — Passed:** the closure changes neither
  admitted cohort behavior nor valid empty-boundary/public behavior.

## D-01 Through D-05

| Decision | Result | Evidence |
|---|---:|---|
| D-01 aggregate startup validation | Passed | Aggregate plain and hostile-type diagnostics are deterministic and precede side effects. |
| D-02 canonical admitted run | Passed | Resolved config and manifest are immutable/canonical internally and JSON-safe. |
| D-03 fail-closed cohort policy | Passed | Strict and partial source/cache admission preserve complete ordered evidence. |
| D-04 empty-boundary errors | Passed | All named stage inputs fail with stable structured diagnostics before expensive work. |
| D-05 compatibility | Passed | Existing valid notebook, CLI, config, import, output, and function contracts remain stable. |

## Threat Controls T-01 Through T-05

| Threat | Result | Evidence |
|---|---:|---|
| T-01 invalid config reaches side effects | Passed | Startup and direct foundation/report helpers resolve explicit mappings before observable effects. |
| T-02 arbitrary or unstable canonical state | Passed | Exact-type admission rejects hostile subclasses without executing their behavior; canonical values contain only safe primitives. |
| T-03 missing slides silently dropped | Passed | Strict admission and downstream helpers fail visibly; partial policy exists only at admission. |
| T-04 partial mode obscures exclusions | Passed | Explicit opt-in retains complete configured, included, skipped, and failed collections with stable reasons. |
| T-05 empty work reaches expensive code | Passed | All tested empty boundaries stop before scientific, model, cache, or writer seams. |

## Roadmap Success Criteria

1. **Passed:** invalid sections, keys, exact/plain and hostile-subclass types,
   values, and cross-field combinations fail together at startup with
   actionable paths, expected constraints, and guidance.
2. **Passed:** empty cohorts, folds, aligned sets, patch sets, prediction
   batches, and regression-target selections fail at their public boundary.
3. **Passed:** missing configured slides fail before locally knowable work by
   default; explicit partial mode records configured, included, skipped, and
   failed outcomes; strict remote failure publishes no false manifest.
4. **Passed:** every admitted run exposes canonical resolved configuration and
   a deterministic cohort manifest for later provenance and fingerprints.

## Requirement Status

- **VAL-01 — Passed:** startup resolution validates the complete configuration
  contract, aggregates actionable errors, safely rejects hostile subclasses,
  and admits only deterministic JSON-safe state.
- **VAL-03 — Passed:** all required empty scientific boundaries fail before
  expensive execution with domain-specific structured errors.
- **VAL-04 — Passed:** strict missing-slide behavior, explicit partial mode,
  complete reason-coded manifests, final-membership propagation, and remote
  source fail-fast/no-false-manifest behavior are verified.

## Deep Review Closure

- **WR-01:** exact-type admission, oversized-number handling, inert rendering,
  and deterministic invalid-key ordering are all closed.
- **WR-02:** availability/failure evidence validates before normalization and
  manifests contain only sanitized deterministic details.
- **WR-03:** strict curation stops at the first documented source error and
  later members are `source_not_attempted`.
- **WR-04:** only documented acquisition failures enter cohort policy;
  preprocessing, programming, and storage defects propagate.
- **WR-05:** all label frames validate before publication.
- **WR-06:** CNN targets resolve before device/model work.
- **WR-07:** nested LOSO requires three slides before task/probe work.
- **IN-01:** adversarial behavior now covers hostile allowed-type subclasses,
  explicit malformed configs, forbidden seams, strict/partial exact manifests,
  sanitization, and valid compatibility paths.

## Automated Evidence

| Command or probe | Result |
|---|---:|
| `python scripts/verify.py fast` | Passed: Ruff + 157 offline tests |
| Focused validation/empty/foundation/admission/CLI/model gate | Passed: 119 tests |
| Original `EvilInt` and `EvilStr` probes | Passed: deterministic `ConfigValidationError`, no hostile execution |
| Four explicit `{}` foundation/report probes with forbidden seams | Passed: four early `ConfigValidationError` results |
| Phase 2 truthiness-fallback `rg` audit | Passed: no matches |
| `git diff --check 65cbeb0..HEAD` | Passed |

## Warnings

- Verification ran on Python 3.12 while required CI declares Python 3.11;
  Phase 10 owns environment reconciliation.
- Pandas reports old optional `numexpr` and `bottleneck` accelerators.
- Six legacy pharma notebooks still emit missing-cell-ID warnings.
- Network, executable-notebook, model-download, and full-cohort tiers remain
  explicit non-gating evidence, as required by Phase 1.

## Human Verification

No human-only item or unresolved gap remains. All Phase 2 acceptance behavior is
covered by deterministic offline tests and independently replayed adversarial
probes.

---

*Independent re-verification completed 2026-07-17.*
