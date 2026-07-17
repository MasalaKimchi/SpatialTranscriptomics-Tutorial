---
phase: 02-validated-run-and-cohort-admission
plan: "01"
subsystem: validation
tags: [python, yaml, json, configuration, offline-testing]

requires:
  - phase: 01-offline-verification-harness
    provides: Strict offline pytest markers, CPU fixtures, Ruff scope, and the canonical fast verification runner
provides:
  - Dependency-free aggregate validation for every documented experiment configuration field
  - Deterministic immutable canonical JSON with fresh mutable dictionary compatibility views
  - Strict load_config routing and an explicit fail-closed partial-cohort policy default
affects: [02-02-cohort-admission, provenance, cache-fingerprints, pipeline-startup]

tech-stack:
  added: []
  patterns:
    - Pure standard-library validation precedes scientific and model imports
    - Schema issues accumulate in deterministic traversal order before one domain exception
    - Canonical state sorts mapping keys while preserving configured list order

key-files:
  created:
    - projects/spatial-pharma-dl/src/validation.py
    - projects/spatial-pharma-dl/tests/test_validation.py
  modified:
    - projects/spatial-pharma-dl/configs/default.yaml
    - projects/spatial-pharma-dl/src/data.py

key-decisions:
  - "Configuration resolution uses explicit standard-library validators so startup remains independent of scientific and model libraries."
  - "Only fields already optional in production receive defaults; required scientific sections remain required and unknown known-map keys fail closed."
  - "load_config preserves its mutable plain-dictionary facade by decoding immutable canonical JSON for every call."

patterns-established:
  - "Aggregate startup gate: resolve_config reports ordered dotted-path issues with received values, constraints, and correction guidance."
  - "Canonical compatibility: ResolvedConfig stores strict JSON and to_dict returns a fresh nested dict/list tree."

requirements-completed: [VAL-01]

duration: 6min
completed: 2026-07-17
---

# Phase 2 Plan 01: Configuration Contract Summary

**A dependency-free aggregate schema gate now rejects malformed experiments before side effects and exposes byte-stable canonical JSON through the existing mutable `load_config` facade**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-17T05:40:56Z
- **Completed:** 2026-07-17T05:47:06Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added frozen, slotted validation records and one aggregate exception containing deterministic dotted paths, received values, expected constraints, and corrective guidance.
- Validated the complete current experiment surface, including strict boolean-versus-number handling, unknown keys, cohort uniqueness, finite/range/enumeration rules, and cross-field constraints.
- Added strict JSON canonicalization that rejects arbitrary objects and non-finite values, sorts mapping keys, preserves cohort-list order, and produces fresh mutable compatibility trees.
- Routed default and custom YAML through strict resolution while preserving `load_config(path=None)`, existing config keys, public imports, and ordinary dictionary behavior.
- Added the sole Phase 2 policy default, `cohort_policy.allow_partial: false`, without changing any existing section or output name.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create the validation errors, canonical records, and aggregate schema resolver** - `e3866f0` (feat)
2. **Task 2: Route the existing configuration facade through strict resolution** - `ebcc17d` (feat)

**Plan metadata:** committed with this summary and sequential GSD tracking updates.

## Files Created/Modified

- `projects/spatial-pharma-dl/src/validation.py` - Defines the import-light exception hierarchy, immutable records, strict canonicalizer, and aggregate schema resolver.
- `projects/spatial-pharma-dl/tests/test_validation.py` - Proves aggregate diagnostics, strict types, dynamic gene maps, canonical stability, fresh dictionaries, YAML facade behavior, and fail-before-side-effect ordering.
- `projects/spatial-pharma-dl/configs/default.yaml` - Adds only the explicit fail-closed partial-cohort policy.
- `projects/spatial-pharma-dl/src/data.py` - Delegates parsed YAML and explicit helper configs through `resolve_config` while retaining existing signatures and return types.

## Decisions Made

- Copied the small model, device, foundation, and metric registries into the pure validator instead of importing heavy runtime modules.
- Applied optional defaults only to values that existing production helpers already treated as optional; missing required scientific sections still produce schema issues.
- Converted YAML parser failures into `ConfigValidationError` so syntax, empty documents, and semantic defects share one startup error boundary.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- One exploratory regression command referenced a nonexistent `test_imports.py`; the repository's actual compatibility module was identified and the corrected 20-test regression command passed.
- The active environment continues to emit previously documented optional pandas accelerator and legacy notebook cell-ID warnings; neither affects verification results.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-02 can build cohort admission and deterministic manifests directly on `ResolvedConfig` and the shared domain exception hierarchy.
- The default policy is now fail-closed, but no filtering or manifest behavior was added early; those remain exclusively in Plan 02-02.
- No blockers remain for the next plan.

## Gate Results

- Focused validation gate: 17 offline tests passed.
- Focused Ruff gate: all checks passed for `validation.py`, `data.py`, and `test_validation.py`.
- Compatibility regression gate: 20 tests passed across core refactors and configuration validation.
- Clean-interpreter import gate: `src.validation` loaded without Torch, Scanpy, Squidpy, torchvision, timm, or transformers.
- Canonical fast gate: Ruff passed first and all 75 offline tests passed in 4.10 seconds.
- Scope review: only the default config, validation/data modules, and focused test module changed; notebooks, CLI entry points, output filenames, package discovery, and public exports were untouched.

---
*Phase: 02-validated-run-and-cohort-admission*
*Completed: 2026-07-17*

## Self-Check: PASSED
