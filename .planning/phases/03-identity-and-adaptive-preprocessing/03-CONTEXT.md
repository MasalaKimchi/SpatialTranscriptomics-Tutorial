# Phase 3: Identity and Adaptive Preprocessing - Context

**Gathered:** 2026-07-17
**Status:** Ready for planning
**Source:** Automatic smart-discuss defaults authorized by the user

<domain>
## Phase Boundary

Make spot identity/alignment and post-QC preprocessing dimensions explicit, deterministic contracts. This phase validates and aligns existing label/patch/AnnData data and records resolved preprocessing parameters; it does not add cache fingerprints, artifact migrations, fold class policy, leakage fixes, image normalization, or label-confidence semantics owned by later phases.

</domain>

<decisions>
## Implementation Decisions

### D-01 — Canonical compound identity
- The canonical identity is the exact `(slide_id, spot_id)` tuple; neither component may be null, empty, coerced across types, or inferred from row order/index position.
- Duplicate, missing, unmatched, or cross-slide keys fail before array/tensor construction with total counts and a deterministic bounded sample of offending keys.

### D-02 — One-to-one alignment
- Alignment is an explicit validated join preserving the requested patch/order contract while reordering targets and provenance together.
- Complete shuffled inputs succeed deterministically; missing, extra, or duplicated rows never inner-join away silently.
- Returned aligned data must make the canonical keys and source-row provenance inspectable.

### D-03 — Adaptive legal dimensions
- Resolve requested HVG, PCA, and neighbor dimensions only after QC counts are known.
- Resolved values are deterministic and bounded by the scientific/mathematical limits of retained spots, genes, and components; invalid or scientifically nonviable inputs fail with stage, counts, requested values, and guidance.
- Do not silently reinterpret impossible requests as a different analysis; each adjustment receives an explicit reason code.

### D-04 — Visible preprocessing provenance
- Record input/post-QC spot and gene counts, exclusions, requested parameters, resolved parameters, and reason codes in JSON-safe AnnData metadata and the admitted run provenance surface.
- Repeated resolution of the same inputs/configuration produces byte-stable canonical metadata.

### D-05 — Compatibility and phase boundaries
- Preserve notebook order, config keys, CLI entry points, public imports, output names, and ordinary successful results.
- Additive metadata and stricter rejection of previously silent row loss or illegal dimensions are intended behavior changes.

### the agent's Discretion
- Exact helper/module placement and bounded offending-key sample size, provided existing module responsibilities and Phase 2 validation exception patterns are reused.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 3 goal and success criteria.
- `.planning/REQUIREMENTS.md` — VAL-02 and VAL-05.
- `.planning/PROJECT.md` — compatibility, scientific validity, offline-test, and traceability constraints.
- `.planning/phases/02-validated-run-and-cohort-admission/02-VERIFICATION.md` — validated startup/stage error vocabulary.
- `.planning/codebase/ARCHITECTURE.md` and `.planning/codebase/CONCERNS.md` — alignment and preprocessing seams.

</canonical_refs>

<specifics>
## Specific Ideas

- Reuse Phase 1 synthetic AnnData and deterministic fixtures.
- Prefer pure validation/resolution helpers that can be tested before Scanpy, NumPy array, PyTorch dataset, cache, or output side effects.

</specifics>

<deferred>
## Deferred Ideas

- Cache/result durability is Phase 4; safe cache/checkpoint formats are Phase 5.
- Fold class admission/reproducibility is Phase 6; leakage-free evaluation is Phase 7.

</deferred>

---

*Phase: 03-identity-and-adaptive-preprocessing*
*Context gathered: 2026-07-17 via automatic smart-discuss defaults*
