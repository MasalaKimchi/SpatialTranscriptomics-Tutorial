# Phase 1: Offline Verification Harness - Context

**Gathered:** 2026-07-17
**Status:** Ready for planning
**Source:** Automatic smart-discuss defaults authorized by the user

<domain>
## Phase Boundary

Establish one documented, CPU-only, offline verification harness that later phases extend. This phase supplies deterministic synthetic fixtures and tiered commands; it does not implement the later scientific or artifact-contract fixes themselves.

</domain>

<decisions>
## Implementation Decisions

### D-01 — Default evidence tier
- The default test command must require no network, model download, private data, GPU, or full public cohort.
- It must run Ruff plus focused unit, artifact round-trip, synthetic AnnData integration, model/fold smoke, and notebook-structure checks.

### D-02 — Deterministic fixtures
- Shared fixtures must use fixed seeds and cover both valid and adversarial cohorts, keys, images, folds, and serialized artifacts.
- Fixtures must be small enough for routine local and CI execution on CPU.

### D-03 — Tier boundaries
- Fast synthetic, notebook-smoke, network, and full-cohort checks must have explicit markers or commands and must be distinguishable in CI output.
- Slow or external tiers remain opt-in and may not be collected by the default offline command.

### D-04 — Extension contract
- Later phases must add evidence through the same fixture, marker, and command conventions instead of creating separate test harnesses.
- Existing notebook numbering, CLI behavior, and public imports remain unchanged.

### the agent's Discretion
- Exact pytest marker names, fixture module layout, and CI provider configuration, provided the decisions above and TEST-01 are met.

</decisions>

<canonical_refs>
## Canonical References

### Milestone contract
- `.planning/ROADMAP.md` — Phase goal and success criteria.
- `.planning/REQUIREMENTS.md` — TEST-01 and milestone definition of done.
- `.planning/PROJECT.md` — core value, constraints, and compatibility boundaries.

### Existing codebase evidence
- `.planning/codebase/TESTING.md` — current test commands, gaps, and conventions.
- `.planning/codebase/STRUCTURE.md` — repository layout and likely fixture/test locations.
- `.planning/research/SUMMARY.md` — cross-cutting reliability research and sequencing.

</canonical_refs>

<specifics>
## Specific Ideas

- Prefer named commands that make offline behavior obvious to maintainers.
- Treat accidental network access or model download during the fast tier as a test failure.

</specifics>

<deferred>
## Deferred Ideas

- Scientific leakage fixes, safe artifact format migrations, image reliability, label provenance, and environment locking belong to Phases 2-10.

</deferred>

---

*Phase: 01-offline-verification-harness*
*Context gathered: 2026-07-17 via automatic smart-discuss defaults*
