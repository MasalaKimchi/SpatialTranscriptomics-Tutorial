# Phase 1: Offline Verification Harness - Pattern Map

**Mapped:** 2026-07-17  
**Requirement:** TEST-01  
**Scope:** Verification infrastructure only; production scientific and artifact migrations remain in later phases.

## Placement and Dependency Direction

The repository has one established test home: `projects/spatial-pharma-dl/tests/`. Keep all pytest collection and shared fixtures there. A root `scripts/verify.py` should be a thin orchestration boundary because root `scripts/` already contains operational entry points, while `.github/workflows/verify.yml` should invoke that same public command rather than duplicate its subprocess details.

```text
.github/workflows/verify.yml
        -> python scripts/verify.py fast
             -> ruff check configured repository paths
             -> pytest -m offline projects/spatial-pharma-dl/tests
                  -> conftest.py (paths, tier enforcement, network guard, fixtures)
                  -> existing regression tests + focused contract modules
                       -> public src/utils APIs and synthetic artifacts in tmp_path
```

Production modules should remain dependencies of tests, not dependencies of test utilities. Phase 1 should not add imports from `tests/` into `src/`, write to repository `data/` or `outputs/`, or change notebook/public-runtime behavior.

## Files to Create

| File | Role | Closest existing analog |
|---|---|---|
| `scripts/verify.py` | Stable tier CLI; constructs argument lists and runs Ruff/pytest with failure propagation | `projects/spatial-pharma-dl/scripts/run_pipeline.py` is the current thin operational entry point, but verification should expose testable command construction and avoid scientific imports |
| `.github/workflows/verify.yml` | Python 3.11 CPU job for required `fast`; separately named opt-in tiers | No existing CI file; use the repository's documented requirements installation paths |
| `projects/spatial-pharma-dl/tests/conftest.py` | Shared import setup, primary-tier validation, offline environment/network denial, deterministic factories | Repeated `PHARMA`/`ROOT` path setup at the top of both existing test modules |
| `projects/spatial-pharma-dl/tests/test_fixture_contracts.py` | Fresh/deterministic valid and adversarial fixture vocabulary | Explicit-seed in-memory fixtures in `test_foundation.py` |
| `projects/spatial-pharma-dl/tests/test_artifact_roundtrips.py` | Safe primitive NPZ, Parquet, JSON, and H5AD fixture contracts | Safe embedding cache in `src/foundation.py`; H5AD helpers in `utils/st_helpers.py` |
| `projects/spatial-pharma-dl/tests/test_synthetic_anndata.py` | Tiny real AnnData spatial/image/patch integration | `st.get_image`, `st.get_scalefactors`, `coords_hires`, and `extract_all_patches_for_slide` |
| `projects/spatial-pharma-dl/tests/test_model_fold_smoke.py` | Bounded CPU step, no-pretrained model shape, deterministic LOSO orchestration | `_TinyCamModel`, `MeanEncoder`, and current `loso_folds`/`train_loso` |
| `projects/spatial-pharma-dl/tests/test_notebook_structure.py` | Discover and validate the established 13 + 7 notebook sequences | `scripts/patch_notebooks.py` JSON handling and notebook builder kernel constants |
| `projects/spatial-pharma-dl/tests/test_verification_contract.py` | Runner command construction, marker rules, offline environment/network behavior | Existing tests favor public-behavior assertions over implementation snapshots |

## Files to Modify

| File | Change boundary |
|---|---|
| `pyproject.toml` | Add strict pytest marker declarations and checked-in Ruff scope/rules; do not alter package discovery |
| `projects/spatial-pharma-dl/tests/test_core_refactors.py` | Apply one module-level `offline` primary marker; preserve all three test bodies |
| `projects/spatial-pharma-dl/tests/test_foundation.py` | Apply one module-level `offline` primary marker; preserve all five test bodies |
| `README.md` and/or `projects/spatial-pharma-dl/README.md` | Document the canonical `fast`, `notebook-smoke`, `network`, and `full-cohort` commands and direct debug commands |

No source module needs a behavioral change for TEST-01. In particular, do not modify `patches.py`, `models.py`, `train.py`, or `labels.py` to solve their later requirements from this phase.

## Existing Test Idiom to Preserve

Both current modules establish the uninstalled nested package with the same local path contract:

```python
PHARMA = Path(__file__).resolve().parents[1]
ROOT = PHARMA.parents[1]
sys.path[:0] = [str(PHARMA), str(ROOT)]
```

Centralize this in `conftest.py` only if both current imports remain identical. The suite uses plain pytest functions, standard `assert`, `np.testing`, small Torch modules, and explicit local RNGs. The best fixture pattern is already in `test_foundation.py`:

```python
rng = np.random.default_rng(12)
embeddings = np.eye(3)[y] + rng.normal(scale=0.05, size=(len(y), 3))
```

The cheapest model doubles are reusable patterns, not shared mutable instances:

```python
class MeanEncoder(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(2, 3))

class _TinyCamModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Factories should return a fresh object per call and use `np.random.default_rng(fixed_seed)` rather than global NumPy state. Every filesystem fixture should accept/use `tmp_path`.

## Synthetic AnnData Contract

Build one small real `anndata.AnnData` whose axes and spatial metadata match the APIs already consumed by production code:

```python
adata.obs_names = ["spot_00", ...]
adata.var_names = ["MT-CO1", "EPCAM", "COL1A1", ...]
adata.obs["slide_id"] = slide_id
adata.obsm["spatial"] = coordinates  # full-resolution x/y centers
adata.uns["spatial"] = {
    library_id: {
        "images": {"hires": rgb_uint8_image},
        "scalefactors": {
            "tissue_hires_scalef": 0.5,
            "spot_diameter_fullres": 12.0,
        },
    }
}
```

This mirrors the exact lookup chain in `utils/st_helpers.py`:

```python
lib = get_library_id(adata)
images = adata.uns["spatial"][lib]["images"]
return images[res]
```

and the coordinate transform in `src/patches.py`:

```python
def coords_hires(adata) -> np.ndarray:
    sf = st.get_scalefactors(adata)
    return adata.obsm["spatial"] * sf["tissue_hires_scalef"]
```

Use center and border coordinates and a nonuniform RGB image. Keep counts integral before serialization. Factory variants should cover null/duplicate/cross-slide IDs, unmatched labels/patches, empty/one-slide cohorts, single-class folds, grayscale/wrong-channel and all-white images, object-valued NPZ, missing keys, wrong shapes/dtypes, and corrupt JSON bytes. These variants establish reusable inputs; they must not assert the later production fixes yet.

## Artifact Round-Trip Patterns

The closest safe NPZ analog is the foundation embedding cache, which deliberately avoids object dtype:

```python
spot_ids = np.asarray(aligned_labels["spot_id"].astype(str), dtype=np.str_)
np.savez_compressed(
    cache_path,
    embeddings=embeddings.astype(np.float32),
    spot_ids=spot_ids,
)
with np.load(cache_path, allow_pickle=False) as cached:
    cached_spots = cached["spot_ids"].astype(str)
    cached_embeddings = cached["embeddings"]
```

Use that pattern for fixture-format proof. Add Parquet round trips via `DataFrame.to_parquet(..., index=False)`/`pd.read_parquet`, JSON containing primitives only, and H5AD via `adata.write_h5ad(tmp_path / "fixture.h5ad")`/`anndata.read_h5ad`. Assert keys, dtypes, shapes, row order, spot IDs, and spatial metadata after reading.

Explicitly exclude the current patch-cache loader from safe evidence:

```python
np.savez_compressed(path, patches=patches, meta=meta.to_dict("list"))
with np.load(path, allow_pickle=True) as data:
    meta = pd.DataFrame(data["meta"].item())
```

Likewise, do not call `load_model_from_checkpoint()` as a safe checkpoint test because it currently uses `torch.load(..., weights_only=False)`. Those migrations belong to Phase 5. An object-array NPZ fixture should instead demonstrate that `allow_pickle=False` rejects access to the object payload.

## Model and Fold Smoke Seams

The public no-download model seam is:

```python
def build_model(
    n_classes: int,
    n_genes: int,
    model_name: str | None = None,
    pretrained: bool = True,
) -> MultiTaskImageModel: ...
```

Always pass `pretrained=False`; monkeypatch torchvision weight-loading entry points if useful so a default regression fails clearly before network access. One public ResNet18 forward-shape check is enough. Use a tiny local model for the backward/optimizer step to keep the default tier fast.

Fold construction is already pure and deterministic:

```python
def loso_folds(slide_ids: list[str]) -> list[tuple[list[str], str]]:
    return [([s for s in slide_ids if s != v], v) for v in slide_ids]
```

For orchestration, monkeypatch `src.train.train_one_fold` and call:

```python
def train_loso(
    slide_ids: list[str],
    labels: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]: ...
```

Assert that each held-out slide appears exactly once, fold indices are stable, and no output path is touched. Do not execute real `train_one_fold`: it writes checkpoints through `pharma_outputs_dir()` and uses the outer slide for early stopping, which is Phase 7 work.

## Notebook Structure Pattern

Discovery should be repository-relative and assert the established public sequences:

- root: `00_*.ipynb` through `12_*.ipynb` (13 notebooks)
- pharma: `projects/spatial-pharma-dl/notebooks/01_*.ipynb` through `07_*.ipynb` (7 notebooks)

Parse with `nbformat.read` and call `nbformat.validate`. Assert nonempty notebooks, allowed cell types, text code sources, and established kernel families. The builders show the intended pharma metadata:

```python
KERNEL = {"display_name": "spatial-tx", "language": "python", "name": "spatial-tx"}
```

Do not require execution, cleared outputs, exact cell counts, or cell IDs. Some generated notebooks predate mandatory IDs, and execution can trigger downloads or depend on earlier caches.

## Tier and Network Enforcement

Declare four mutually exclusive primary markers in `pyproject.toml`: `offline`, `notebook_smoke`, `network`, and `full_cohort`. A `pytest_collection_modifyitems` hook should fail collection when a test has zero or more than one of these markers. Optional descriptive markers may coexist. Applying `pytestmark = pytest.mark.offline` at module scope is the smallest compatible update for the two existing modules.

During an offline session, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and deny socket creation/connection with an actionable exception. Enable external access only when the selected primary tier is `network` or `full_cohort`; do not provide an individual offline-test escape hatch. The guard should cover at least `socket.create_connection` and `socket.socket.connect` and identify the test/tier in its error.

The runner should use argument lists and `subprocess.run(..., check=True)`, exposing command construction separately enough for unit tests. Recommended selections:

```text
fast            -> ruff, then pytest -m offline
notebook-smoke  -> pytest -m notebook_smoke
network         -> pytest -m network
full-cohort     -> pytest -m full_cohort
```

Do not encode an always-on marker exclusion in global pytest `addopts`; that makes deliberate opt-in selection opaque. Empty opt-in tiers should be reported distinctly rather than silently presented as evidence.

## CI and Documentation Integration

The required GitHub Actions job should run on pull requests and pushes to `main`, use Ubuntu/Python 3.11 CPU, install the existing base plus pharma declarations and explicit verification tools, set Hugging Face/Transformers offline flags, and invoke exactly:

```bash
python scripts/verify.py fast
```

Opt-in tier jobs should be separately named and gated by `workflow_dispatch` inputs and/or schedules. Do not cache repository `data/`, `outputs/`, pretrained weights, or scientific artifacts. Phase 10 owns dependency reconciliation/locking, so Phase 1 should not use CI setup to rewrite environment policy.

Document the canonical commands near existing setup/run instructions. Preserve current notebook ordering, the pharma pipeline CLI, config keys, output names, and lazy package exports.

## Implementation Guardrails

- Keep `scripts/verify.py` import-safe: no Scanpy, Squidpy, Torchvision model, dataset, or foundation-backend imports at module import time.
- The default tier must not call `load_dataset`, `load_visium_sample`, `load_frozen_encoder`, a pretrained torchvision model, or repository cache/output helpers.
- Use `tmp_path` for NPZ, Parquet, JSON, H5AD, and any checkpoint-shaped fixture.
- Preserve the current eight regression tests verbatim except for tier classification/import centralization.
- Assert behavior and schemas, not exact random floating-point bit patterns across platforms.
- Keep notebook smoke execution separate from notebook structural validation.
- Do not bless unsafe `allow_pickle=True` or `weights_only=False` paths in Phase 1.

## Expected Verification Flow

1. Pytest collects every test and validates exactly one primary tier.
2. The offline session installs network denial and offline model-hub flags.
3. Deterministic factories create fresh valid/adversarial data in memory or `tmp_path`.
4. Unit and artifact tests verify primitive schemas and safe round trips.
5. Synthetic AnnData flows through existing image/scale/coordinate/patch seams.
6. Tiny CPU model and monkeypatched LOSO orchestration verify tensor/fold contracts.
7. All 20 notebooks receive structural validation without execution.
8. `scripts/verify.py fast` runs Ruff first, then the complete offline tier; CI calls the same command.

---

*Pattern mapping complete for Phase 01 planning.*
