# Phase 2: Validated Run and Cohort Admission - Context

**Gathered:** 2026-07-17
**Status:** Ready for planning
**Source:** Automatic smart-discuss defaults authorized by the user

<domain>
## Phase Boundary

Create the validation and admission boundary that every pharma pipeline run crosses before expensive work. This phase validates configuration/cohort presence and rejects empty stage inputs; it does not implement identity alignment, cache fingerprinting, leakage fixes, or scientific preprocessing owned by later phases.

</domain>

<decisions>
## Implementation Decisions

### D-01 — Aggregate startup validation
- Resolve defaults first, then validate required sections, unknown keys, types, enumerated values, positive ranges, referenced paths, and cross-field constraints in one pass.
- Raise one domain-specific error containing every discovered issue, with dotted configuration paths, received values, expected constraints, and corrective guidance.

### D-02 — Canonical admitted run
- Successful admission returns an immutable/canonical resolved configuration plus a cohort manifest suitable for later provenance and cache fingerprints.
- Canonical output must be deterministic for semantically identical input and JSON-serializable without arbitrary Python objects.

### D-03 — Fail-closed cohort policy
- Missing configured slides fail before any slide processing by default.
- Partial-cohort execution requires an explicit configuration flag; it may skip unavailable slides but must never silently discard them.
- The manifest records configured, included, skipped, and failed slides with reasons, and an admitted cohort must still contain usable slides.

### D-04 — Empty-boundary errors
- Empty cohorts, folds, aligned spot sets, patch sets, prediction batches, and regression-target selections raise a shared domain-specific validation error at the earliest public boundary.
- Messages identify the stage, observed count/shape, expected minimum, and likely corrective action.

### D-05 — Compatibility
- Preserve current config keys, CLI entry points, notebook order, output names, and public imports; additive validation/default keys are allowed.
- Default behavior becomes stricter where current behavior silently skips missing inputs or continues with empty work.

### the agent's Discretion
- Validation implementation style (typed dataclasses, explicit schema helpers, or another dependency-free approach), exact exception hierarchy, and manifest filename/location, provided public compatibility and deterministic serialization are preserved.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 2 goal and success criteria.
- `.planning/REQUIREMENTS.md` — VAL-01, VAL-03, and VAL-04.
- `.planning/PROJECT.md` — compatibility, scientific, offline, and traceability constraints.
- `.planning/codebase/ARCHITECTURE.md` — pipeline/config flow.
- `.planning/codebase/CONCERNS.md` — configuration, empty-input, and missing-cohort risks.
- `.planning/phases/01-offline-verification-harness/01-VERIFICATION.md` — required offline evidence convention.

</canonical_refs>

<specifics>
## Specific Ideas

- Prefer a pure validation layer that can be tested with temporary files and synthetic cohorts before pipeline side effects begin.
- Add Phase 2 tests to the existing `offline` tier and shared deterministic fixtures.

</specifics>

<deferred>
## Deferred Ideas

- Barcode identity enforcement and adaptive dimensions are Phase 3.
- Durable manifests/atomic writes and cache/checkpoint migration are Phases 4-5.
- Fold scientific correctness is Phases 6-7.

</deferred>

---

*Phase: 02-validated-run-and-cohort-admission*
*Context gathered: 2026-07-17 via automatic smart-discuss defaults*
