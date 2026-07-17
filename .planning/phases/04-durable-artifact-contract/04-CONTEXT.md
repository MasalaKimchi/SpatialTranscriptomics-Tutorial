# Phase 4: Durable Artifact Contract - Context

**Gathered:** 2026-07-17
**Status:** Ready for planning
**Source:** Automatic smart-discuss defaults authorized by the user

<domain>
## Phase Boundary

Make every supported processed-slide, patch, embedding, model, table, report, and manifest reuse decision depend on a deterministic artifact contract, and make publication interruption-safe. This phase owns fingerprints, sidecar manifests, schema/shape/completion validation, checksums, same-filesystem temporary publication, production-reader validation, and atomic replacement. It does not migrate pickle-backed payload formats or checkpoint deserialization semantics (Phase 5), change fold/evaluation science (Phases 6-7), or alter image/label algorithms (Phases 8-9).

</domain>

<decisions>
## Implementation Decisions

### D-01 — Explicit artifact manifests and commit markers
- Every reusable artifact receives a versioned, canonical JSON sidecar describing artifact kind, schema version, fingerprint inputs/digest, upstream lineage, payload metadata, checksum, and completion state.
- The completed sidecar is the commit marker. Readers reject a missing, incomplete, malformed, unsupported, or payload-inconsistent manifest before treating the final payload as reusable.
- Diagnostics expose bounded artifact path/kind/reason evidence and actionable regeneration guidance without embedding host-specific or caller-controlled representations.

### D-02 — Scientific fingerprint projections
- Fingerprints use canonical JSON and a stable cryptographic digest over only the inputs relevant to that artifact: artifact schema/code-contract version, the artifact-specific resolved configuration projection, source-data identity, and upstream artifact fingerprints.
- Presentation-only settings, output formatting, unrelated model/report options, absolute checkout paths, timestamps, and incidental mapping order do not invalidate scientific caches.
- A relevant configuration, source identity, upstream lineage, or explicit code-contract-version change must deterministically miss/reject reuse; there is no permissive stale-cache fallback.

### D-03 — Reader-owned validation
- A shared contract layer validates manifests/checksums generically, while each production reader validates its own required keys, types, shapes, row counts, identities, and semantic schema before returning the payload.
- Validation occurs before cache reuse and before downstream scientific/model work. Existing unsafe payload deserialization is not blessed or migrated here; Phase 5 must replace those formats, while Phase 4 ensures only checksum/fingerprint-admitted local artifacts reach the legacy reader.
- Valid artifact metadata remains inspectable through immutable/canonical value objects and fresh JSON-safe views.

### D-04 — Atomic same-filesystem publication
- Writers create uniquely named temporary payload and manifest files in the destination directory, flush and close them, validate the temporary payload through the same production reader contract, and only then use atomic replacement.
- Publish the payload first and the completed manifest last. Any interruption leaves no valid commit marker or a checksum mismatch, so readers fail closed; temporary remnants are never considered reusable.
- Apply one shared publication primitive across caches, H5AD, NPZ, Parquet/CSV/JSON tables, checkpoints, and reports while retaining existing public filenames and output locations.

### D-05 — Compatibility and evidence
- Preserve notebook order, CLI flags, config keys, public Python exports, final payload filenames, ordinary successful results, and Phase 2/3 manifest schemas; sidecars and stricter stale/incomplete rejection are additive intentional changes.
- Tests use `tmp_path`, deterministic synthetic artifacts, interruption/fault injection, and production readers. Default evidence remains CPU/offline and never downloads data or weights.

### the agent's Discretion
- Exact module/class names, sidecar suffix, checksum chunk size, temporary-name pattern, and bounded diagnostic sample limits, provided the shared contract is import-light and existing Phase 2/3 canonical/hostile-input patterns are reused.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 4 goal and success criteria.
- `.planning/REQUIREMENTS.md` — ART-03 and ART-04.
- `.planning/PROJECT.md` — compatibility, offline, security, and traceability constraints.
- `.planning/phases/03-identity-and-adaptive-preprocessing/03-VERIFICATION.md` — validated identity/preprocessing and canonical provenance surfaces.
- `.planning/codebase/CONCERNS.md` — stale cache reuse and non-atomic writer inventory.

</canonical_refs>

<specifics>
## Specific Ideas

- Prefer a single artifact contract/value-object layer with small typed adapters over bespoke sidecar logic in every module.
- Make fault-injection tests enumerate interruption points before/after payload replacement and before/after manifest replacement.
- Test both invalidation and non-invalidation: scientific input changes must miss; presentation-only changes must retain the same fingerprint.
- Treat the manifest as an auditable explanation of exactly why an artifact was reusable, not merely a hash file.

</specifics>

<deferred>
## Deferred Ideas

- Safe non-pickle patch and checkpoint formats are Phase 5.
- Seed/determinism and fold viability are Phase 6; leakage-free fitted preprocessing is Phase 7.
- Image normalization/quality and heuristic-label science are Phases 8-9.
- Environment locking and dependency reconciliation are Phase 10.

</deferred>

---

*Phase: 04-durable-artifact-contract*
*Context gathered: 2026-07-17 via automatic smart-discuss defaults*
