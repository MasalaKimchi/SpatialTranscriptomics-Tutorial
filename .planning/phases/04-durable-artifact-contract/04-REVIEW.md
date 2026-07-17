---
status: clean
depth: deep
files_reviewed: 41
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
---

# Phase 4 Code Review

**Scope:** All Phase 4 plans and summaries, the complete artifact contract and production adapters, runner and notebook consumers, root notebook retained outputs, and the review-fix commits `0ec430e..2a423c0`.

**Result:** Clean. The eight warning findings from the initial deep review are closed with adversarial regression evidence, and the canonical fast gate passes Ruff plus 395 offline tests.

## Closure Evidence

- **WR-01:** Excessive array/object nesting below the byte cap is translated to bounded `ArtifactValidationError(malformed_manifest)`.
- **WR-02:** Payload bytes are copied and hashed from one admitted descriptor into a private decoder snapshot. Generic, patch, and checkpoint ABA restoration tests prove unchecksummed public-path bytes never reach decoders.
- **WR-03:** Every production publisher supplies an observed-schema extractor; canonical observed and declared schemas must match before either final replacement.
- **WR-04:** Fingerprint projections are per-kind leaf allowlists. Embedding identity ignores `enabled`, `cache`, `device`, and scheduling-only `batch_size`, while model identity still invalidates.
- **WR-05:** Report and retained-result readers require independently supplied current lineage and identity/value expectations. The runner and generated evaluation notebook derive report expectations from current checkpoints, labels, patches, folds, and slides.
- **WR-06:** Named result registries reject unknown names and enforce exact columns/keys, types, identity uniqueness, cardinality/range invariants, and typed `CohortManifest`/`PreprocessingManifest` reconstruction.
- **WR-07:** Patch lineage binds the actual processed-slide and admitted shared-stain-reference manifests; per-slide normalization omits irrelevant shared lineage. Patch indexes require actual current label and patch parent sidecars in admitted order and fail before publication when a parent is absent.
- **WR-08:** Four retained root-notebook CSVs publish through named atomic adapters. Static inventory parses committed notebook code with exact-purpose allowlists. A real 19-artifact chain publishes, admits, validates bytes, and proves parent-lineage invalidation for every retained logical kind.

## Verification

- Focused generic contract, scientific adapter, checkpoint, orchestration, notebook, cohort, and identity gates passed.
- Root-notebook adapter/static/19-kind gate: 141 passed.
- Final `python scripts/verify.py fast`: Ruff passed and 395 offline tests passed in about 20 seconds.
- `git diff --check` passed before each atomic implementation commit.

## Review Conclusion

Phase 4 is clean and ready for independent requirement verification. Phase 5 remains responsible for replacing the explicitly local-only pickle-compatible patch/checkpoint payload formats.
