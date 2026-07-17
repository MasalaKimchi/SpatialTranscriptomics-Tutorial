---
status: clean
depth: deep
files_reviewed: 41
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
resolved_findings: 9
---

# Phase 4 Code Review

**Scope:** Second review-fix iteration through commits `5027032`, `6fcf68f`, and `b14e003`, including every WR-06R..WR-09 reproduction, the 19 retained production artifact adapters, their parent-lineage paths, and source/generated-notebook bypass inventory.

**Result:** Clean. WR-01 through WR-09 are closed with production-reader evidence and the full offline gate passes.

## Closed Second-Pass Findings

### WR-06R — Exact partitions and scientific value bounds

`eval.py` now requires each configured slide to occur exactly once across mutually exclusive outcome partitions with consistent cohort identity. Benchmark, table, and experiment-summary readers enforce metric-specific finite ranges, nonnegative losses/counts, bounded mitochondrial percentages, and the intended upper-only R2 policy. Adversarial writer/production-loader tests reject overlaps and impossible values.

### WR-07R — Child lineage derives from admitted parents

Processed slides, label tables, patch arrays, and stain references now return or consume typed `ArtifactAdmission` records; child fingerprints use only manifests obtained after complete checksum and semantic admission. The pipeline passes the actual shared stain-reference ID explicitly. Missing, corrupt, wrong-schema, and mixed-generation parents fail before patch/index publication.

### WR-08R — Real production graph and exact static closure

The generic `.bin` fixture is gone. `test_artifact_adapters.py` now round-trips all 19 retained artifacts through their real H5AD, Parquet, CSV, NPZ, PyTorch, and canonical-JSON writers/readers. Dedicated adversarial tests exercise the real dependency edges. Static inventory allowances are exact full lines, stale `path.is_file()` exemptions are removed, committed pharma notebooks are scanned, and a synthetic bypass placed inside an otherwise allowed production file is detected.

### WR-09 — Writers require reusable lineage before side effects

All three public result writers reject absent or empty lineage before directory creation or publication. Every successful writer call has a matching independently expected loader path, covered by round-trip and no-side-effect tests.

## Closed Original Findings

- **WR-01:** Deep array/object JSON nesting now returns bounded `malformed_manifest`.
- **WR-02:** Generic, patch, and checkpoint decoders consume private admitted snapshots; public-path ABA bytes do not reach decoders.
- **WR-03:** Publication compares the production reader's observed schema with the declared schema before replacement.
- **WR-04:** Embedding fingerprints ignore `enabled`, `cache`, `device`, and scheduling-only `batch_size`, while model identity remains relevant.
- **WR-05:** Loaders no longer derive current lineage/identity/value expectations from their own sidecars; runner and generated report consumers supply independent expectations.
- Root notebook retained CSV writes now call `save_root_result_table`, and gallery H5AD consumers remain contract-bound.

## Verification Performed

- Focused production adapter/static gate: **26 passed**.
- Canonical `python scripts/verify.py fast`: Ruff passed and **400 offline tests passed**.
- Test collection independently confirmed **400 tests**.
- No network, model download, dataset download, or push was performed.

## Review Conclusion

Phase 4 is **clean**. No critical, warning, or informational review finding remains.
