---
phase: 02-validated-run-and-cohort-admission
status: gaps_found
score: "20/22"
requirements:
  - VAL-01
  - VAL-03
  - VAL-04
date: 2026-07-17
verifier: independent-gsd-verifier
---

# Phase 2 Verification

## Result

Phase 2 is close, but it does not yet fully achieve its goal. Strict and partial
cohort admission, deterministic manifests, source-failure handling, and all
named empty scientific boundaries behave as planned. Two independent contract
gaps remain:

1. the public configuration resolver can execute methods overridden by hostile
   subclasses of JSON primitive types instead of rejecting them through
   `ConfigValidationError`;
2. foundation-stage configuration helpers still use truthiness fallback, so an
   explicitly supplied empty configuration is silently replaced with the
   repository default and can proceed toward cache or model side effects.

The score is based on 22 independent behavioral checks spanning plan
must-haves, D-01 through D-05, T-01 through T-05, roadmap success criteria,
requirements, review closure, compatibility, and regression gates. Twenty
checks passed and the two root gaps above failed. Overlapping consequences are
not double-counted in the score.

## Blocking Gaps

### G-01 — Hostile primitive subclasses escape the aggregate validation boundary

**Affected:** VAL-01, D-01, D-02, T-02, Plan 02-01 must-haves, roadmap success
criterion 1, and the claimed closure of WR-01.

`validation._string`, `_number`, and `_validate_json_tree` use `isinstance`
before calling primitive methods or comparisons. A caller can therefore pass a
subclass of `int`, `float`, `str`, `list`, `tuple`, `Mapping`, or `Path` whose
overridden operation executes during validation. This contradicts the plan's
requirement to reject arbitrary non-JSON Python objects without invoking their
behavior.

Independent probes against the current source produced:

```text
training.batch_size = EvilInt(4), EvilInt.bit_length raises
-> RuntimeError: executed hostile numeric method

experiment = EvilStr("safe"), EvilStr.strip raises
-> RuntimeError: executed hostile string method
```

Neither probe reached `ConfigValidationError`, so defects were not aggregated
and the advertised domain boundary was bypassed. The existing hostile tests
cover ordinary objects, representations, invalid-key comparisons, and hashes,
but not subclasses of allowed primitive types.

**Required remediation:** accept exact safe primitive/container types before
performing operations, or copy values through a non-executing exact-type gate;
reject subclasses as unsupported values. Add regression tests for hostile
`int`, `float`, `str`, sequence, mapping, and `Path` subclasses and require one
deterministic `ConfigValidationError` without executing their overrides.

### G-02 — Explicit empty foundation configuration reloads defaults

**Affected:** T-01 and the D-05 rule that explicit supplied values are not
reinterpreted as absent.

The following touched/public helpers still use `cfg = cfg or load_config()`:

- `src.foundation.foundation_config`;
- `src.foundation.load_frozen_encoder`;
- `src.foundation.load_or_extract_slide_embeddings`;
- `src.eval.save_benchmark_report`.

An independent call to `foundation_config({})` returned the default foundation
configuration rather than preserving or rejecting the explicit empty mapping.
More importantly, `load_frozen_encoder({})` and
`load_or_extract_slide_embeddings(..., cfg={})` can reload defaults and advance
toward device resolution, model download, or cache-directory work. The Phase 2
explicit-empty regression covers data, labels, and patches only, so it does not
detect this remaining path.

**Required remediation:** use `if cfg is None: cfg = load_config()` consistently
in the affected helpers, then validate or fail the explicit mapping before any
cache, device, encoder, output-directory, or writer seam. Add forbidden-seam
tests for explicit `{}` at the foundation and report boundaries.

## Plan Must-Haves

### Plan 02-01 — Configuration contract

- **D-01 — Gap:** ordinary YAML and plain-dictionary defects aggregate in
  deterministic order, but hostile primitive subclasses raise their own
  exceptions and escape `ConfigValidationError`.
- **D-02 — Gap:** valid plain mappings canonicalize deterministically and remain
  JSON-safe, but subclasses of admitted primitives are operated on before the
  unsupported-object rejection pass.
- **D-05 — Passed for the facade:** `load_config(path=None)` and custom paths
  return fresh plain dictionaries; existing keys remain and only
  `cohort_policy.allow_partial=false` was added to the default YAML.

### Plan 02-02 — Cohort admission

- **D-02 — Passed:** `AdmittedRun` combines canonical `ResolvedConfig` with an
  ordered, JSON-safe `cohort-manifest-v1`; configuration order controls every
  manifest collection.
- **D-03 — Passed:** known missing members aggregate in strict mode; provisional
  remote admission publishes nothing; strict source acquisition stops at the
  first documented source failure; partial mode collects all outcomes and
  re-admits once.
- **D-05 — Passed:** the runner path, environment flags, stage order, public
  imports, helper signatures, and output names remain compatible for valid
  calls, and one final admitted sequence drives downstream stages.

### Plan 02-03 — Empty scientific boundaries

- **D-04 — Passed:** configured/preprocessing/summary cohorts, label frames,
  regression targets, spot coordinates, patch cohorts, stain inputs, datasets,
  aligned sets, LOSO inputs, CNN/RF members, prediction batches, embedding
  batches, task filters, probe inputs, and nested LOSO admission raise
  structured `StageValidationError` at their tested public boundaries.
- **D-03 — Passed:** empty or one-slide LOSO inputs cannot reach cache, device,
  model, encoder, estimator, output-directory, or writer seams; nested LOSO
  requires three unique non-empty slides before task preprocessing.
- **D-05 — Passed for valid non-empty calls:** signatures, fold order, task
  names, return schemas, patch shapes, output names, notebook order, and lazy
  exports remain compatible. The explicit-empty configuration defect is
  separately recorded as G-02/T-01.

## D-01 Through D-05

| Decision | Result | Evidence |
|---|---:|---|
| D-01 aggregate startup validation | Gap | Plain configs aggregate; hostile primitive subclasses escape with `RuntimeError`. |
| D-02 canonical admitted run | Gap | Plain canonical output and manifests are deterministic; hostile allowed-type subclasses are invoked before rejection. |
| D-03 fail-closed cohort policy | Passed | Strict known availability aggregates; strict remote failure is fail-fast; partial mode records complete outcomes. |
| D-04 empty-boundary errors | Passed | All named boundary and forbidden-seam tests pass with structured primitive diagnostics. |
| D-05 compatibility | Passed with warning | Valid public behavior is stable; G-02 remains for explicit empty foundation/report configs. |

## Threat Controls T-01 Through T-05

| Threat | Result | Evidence |
|---|---:|---|
| T-01 invalid config reaches side effects | Gap | The runner validates normally, but foundation helpers still reload defaults for explicit `{}` and may approach model/cache work. |
| T-02 arbitrary or unstable canonical state | Gap | Canonical plain data is stable, but hostile primitive subclasses can execute overridden methods during admission validation. |
| T-03 missing slides silently dropped | Passed | Strict admission and strict downstream helpers fail visibly; no catch-and-continue policy remains in the reviewed cohort helpers. |
| T-04 partial mode obscures exclusions | Passed | Explicit opt-in preserves configured, included, skipped, and failed records with stable sanitized reasons. |
| T-05 empty work reaches expensive code | Passed | Forbidden directory, loader, stack, cache, device, model, encoder, estimator, probe, and writer seams remain unreachable in focused adversarial tests. |

## Roadmap Success Criteria

1. **Gap:** normal invalid sections, keys, plain types, values, and cross-field
   combinations fail together with actionable paths; hostile subclasses of
   otherwise allowed primitives do not.
2. **Passed:** every named empty cohort/fold/aligned/patch/prediction/target
   boundary has structured early-failure evidence and corrective guidance.
3. **Passed:** strict known-missing admission fails before member processing;
   explicit partial mode records configured, included, skipped, and failed
   slides, while strict remote source failure publishes no false manifest.
4. **Passed:** successful runs expose canonical resolved configuration and a
   deterministic cohort manifest suitable as later provenance/fingerprint input.

## Requirement Status

- **VAL-01 — Gap:** the ordinary YAML/startup surface is comprehensive, but the
  public resolver's type contract is not total for hostile primitive subclasses.
- **VAL-03 — Passed:** all required empty scientific boundaries fail before
  expensive execution with stage-specific diagnostics.
- **VAL-04 — Passed:** strict missing-slide behavior, explicit partial mode,
  complete manifest outcomes, final-membership propagation, source fail-fast,
  and no-false-manifest behavior are verified.

## Deep Review Closure

WR-02 through WR-07 and IN-01 were replayed and are closed:

- admission evidence is exact-type checked before hashing for availability;
- failure details are sanitized and canonical manifests contain no caller
  traceback, path, timestamp, exception representation, NaN, or infinity;
- strict source acquisition stops on the first documented source error and
  later slides are `source_not_attempted`;
- programming, preprocessing, and storage failures are not converted into
  partial-cohort policy;
- all slide label frames validate before output-directory or writer access;
- CNN regression-target selection precedes device/model construction;
- nested LOSO requires three slides before preprocessing or probe fitting;
- exact strict/partial manifest collections and forbidden seams are covered.

WR-01 is only partially closed. Oversized built-in integers, hostile ordinary
objects, invalid-key representations/comparisons, and deterministic primitive
key ordering are fixed, but hostile subclasses of allowed primitives still
execute methods. G-01 is therefore a residual WR-01 gap.

## Automated Checks

| Check | Result | Evidence |
|---|---:|---|
| `python scripts/verify.py fast` | Passed | Ruff passed first; 146 offline tests passed in 5.31 s. |
| Focused Phase 2 plus compatibility suite | Passed | 108 offline tests passed in 4.63 s. |
| Adversarial aggregate/canonicalization suite | Passed with uncovered gap | Existing hostile repr/key/oversized tests pass; independent primitive-subclass probes fail as G-01. |
| Strict and partial cohort admission | Passed | Exact ordered manifests, unusable partial rejection, fail-fast strict curation, and complete partial outcomes pass. |
| Provisional/no-false-manifest behavior | Passed | Provisional admission writes nothing; strict source failure reaches no stage/output seam and prints no completion. |
| Empty public-boundary suite | Passed | 36 focused boundary tests plus affected foundation/model tests pass. |
| Public import compatibility | Passed | Existing lazy exports remain callable and the runner import defers the named heavy model/scientific stages. |
| `git diff --check e3866f0^..HEAD` | Warning | Review metadata contains trailing whitespace and a final extra blank line; no source defect. |

## Warnings

- The local verifier ran Python 3.12 while required CI declares Python 3.11;
  Phase 10 owns environment reconciliation.
- Pandas warned that optional `numexpr` and `bottleneck` accelerators are old.
- Six legacy pharma notebooks still emit missing-cell-ID warnings.
- The direct runner has no `--help` parser; the established CLI contract is the
  path and environment-flag entry point, which remains unchanged.
- Review documentation has minor whitespace errors reported by
  `git diff --check`; implementation source and tests are clean.

## Human Verification

No human-only acceptance item is required. Both gaps are deterministic and can
be closed with offline tests. Network, full-cohort, and model-download execution
remain explicit non-gating tiers.

---

*Independent verification completed 2026-07-17.*
