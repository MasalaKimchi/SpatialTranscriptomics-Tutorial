"""Shared contracts for the repository's mutually exclusive evidence tiers."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

PHARMA = Path(__file__).resolve().parents[1]
ROOT = PHARMA.parents[1]
sys.path[:0] = [str(PHARMA), str(ROOT)]

PRIMARY_TIERS = frozenset({"offline", "notebook_smoke", "network", "full_cohort"})
EXTERNAL_TIERS = frozenset({"network", "full_cohort"})

_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


class OfflineNetworkError(RuntimeError):
    """Raised when offline evidence attempts to reach an external service."""


def _validate_primary_marker_names(marker_names: Iterable[str]) -> str:
    """Return the sole primary tier or reject an ambiguous classification."""
    selected = sorted(PRIMARY_TIERS.intersection(marker_names))
    if len(selected) != 1:
        rendered = ", ".join(selected) if selected else "none"
        raise pytest.UsageError(
            "Every test must declare exactly one primary evidence tier "
            f"({', '.join(sorted(PRIMARY_TIERS))}); found: {rendered}."
        )
    return selected[0]


def _selected_mark_expression(config: pytest.Config) -> str:
    return str(config.option.markexpr or "").strip()


def _external_tier_selected(config: pytest.Config) -> bool:
    """Allow sockets only for an explicitly selected external primary tier."""
    return _selected_mark_expression(config) in EXTERNAL_TIERS


def _deny_socket(*_args: object, **_kwargs: object) -> None:
    raise OfflineNetworkError(
        "Network access is disabled for the offline evidence tier; "
        "select the network or full_cohort tier explicitly."
    )


def pytest_configure(config: pytest.Config) -> None:
    if _external_tier_selected(config):
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.socket.connect = _deny_socket  # type: ignore[method-assign]
    socket.create_connection = _deny_socket


def pytest_unconfigure(config: pytest.Config) -> None:
    if _external_tier_selected(config):
        return
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT  # type: ignore[method-assign]
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    errors: list[str] = []
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        try:
            _validate_primary_marker_names(marker_names)
        except pytest.UsageError as exc:
            errors.append(f"{item.nodeid}: {exc}")
    if errors:
        raise pytest.UsageError("\n".join(errors))


@pytest.fixture
def synthetic_anndata_factory() -> Callable[..., Any]:
    """Build a fresh, small Visium-shaped AnnData with deterministic contents."""

    def build(
        *, slide_id: str = "slide_a", seed: int = 101, n_spots: int = 12
    ) -> Any:
        import anndata as ad

        rng = np.random.default_rng(seed)
        genes = [
            "MT-CO1",
            "EPCAM",
            "COL1A1",
            "CD3D",
            "MS4A1",
            "VIM",
            "MKI67",
            "CXCL9",
            "GAPDH",
            "ACTB",
        ]
        counts = rng.poisson(4.0, size=(n_spots, len(genes))).astype(np.int32)
        obs_names = [f"{slide_id}_spot_{index:02d}" for index in range(n_spots)]
        obs = pd.DataFrame({"slide_id": slide_id}, index=obs_names)
        var = pd.DataFrame(index=genes)
        adata = ad.AnnData(X=counts, obs=obs, var=var)

        base_coords = np.array(
            [[1.0, 1.0], [126.0, 1.0], [1.0, 126.0], [126.0, 126.0]]
        )
        interior = rng.integers(16, 112, size=(max(n_spots - 4, 0), 2)).astype(float)
        adata.obsm["spatial"] = np.vstack([base_coords[:n_spots], interior])

        image_rng = np.random.default_rng(seed + 1)
        image = image_rng.integers(35, 225, size=(64, 64, 3), dtype=np.uint8)
        library_id = f"library_{slide_id}"
        adata.uns["spatial"] = {
            library_id: {
                "images": {"hires": image},
                "scalefactors": {
                    "tissue_hires_scalef": 0.5,
                    "spot_diameter_fullres": 12.0,
                },
            }
        }
        return adata

    return build


@pytest.fixture
def cohort_factory() -> Callable[..., dict[str, Any]]:
    """Build a stable three-slide cohort with viable class support."""

    def build(*, seed: int = 211) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        slide_ids = ["slide_a", "slide_b", "slide_c"]
        rows: list[dict[str, Any]] = []
        for slide_index, slide_id in enumerate(slide_ids):
            for spot_index in range(4):
                rows.append(
                    {
                        "slide_id": slide_id,
                        "spot_id": f"{slide_id}_spot_{spot_index:02d}",
                        "tme_class_id": spot_index % 2,
                        "module_signal": float(
                            slide_index + rng.normal(loc=spot_index, scale=0.05)
                        ),
                    }
                )
        labels = pd.DataFrame(rows)
        patch_index = labels[["slide_id", "spot_id"]].copy()
        folds = [
            ([candidate for candidate in slide_ids if candidate != held_out], held_out)
            for held_out in slide_ids
        ]
        return {
            "slide_ids": slide_ids,
            "labels": labels,
            "patch_index": patch_index,
            "folds": folds,
        }

    return build


@pytest.fixture
def key_adversary_factory(
    cohort_factory: Callable[..., dict[str, Any]],
) -> Callable[[], dict[str, dict[str, pd.DataFrame]]]:
    """Build malformed key tables used by later alignment validation."""

    def build() -> dict[str, dict[str, pd.DataFrame]]:
        valid = cohort_factory()
        labels = valid["labels"]
        patch_index = valid["patch_index"]

        null_labels = labels.copy(deep=True)
        null_labels.loc[0, "spot_id"] = None
        duplicate_labels = labels.copy(deep=True)
        duplicate_labels.loc[1, ["slide_id", "spot_id"]] = duplicate_labels.loc[
            0, ["slide_id", "spot_id"]
        ].to_numpy()
        unmatched_labels = labels.copy(deep=True)
        unmatched_labels.loc[0, "spot_id"] = "label_only_spot"
        unmatched_patches = patch_index.copy(deep=True)
        unmatched_patches.loc[0, "spot_id"] = "patch_only_spot"
        cross_slide = labels.copy(deep=True)
        cross_slide.loc[4, "spot_id"] = cross_slide.loc[0, "spot_id"]

        return {
            "null": {"labels": null_labels, "patch_index": patch_index.copy()},
            "duplicate": {
                "labels": duplicate_labels,
                "patch_index": patch_index.copy(),
            },
            "unmatched_label": {
                "labels": unmatched_labels,
                "patch_index": patch_index.copy(),
            },
            "unmatched_patch": {
                "labels": labels.copy(),
                "patch_index": unmatched_patches,
            },
            "cross_slide": {
                "labels": cross_slide,
                "patch_index": patch_index.copy(),
            },
        }

    return build


@pytest.fixture
def fold_adversary_factory() -> Callable[[], dict[str, dict[str, Any]]]:
    """Build empty and unsupported LOSO cohort/fold shapes."""

    def frame(slides: list[str], classes: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "slide_id": slides,
                "spot_id": [f"spot_{index:02d}" for index in range(len(slides))],
                "tme_class_id": classes,
                "module_signal": np.arange(len(slides), dtype=float),
            }
        )

    def build() -> dict[str, dict[str, Any]]:
        return {
            "empty": {"slide_ids": [], "labels": frame([], [])},
            "one_slide": {
                "slide_ids": ["slide_a"],
                "labels": frame(["slide_a", "slide_a"], [0, 1]),
            },
            "single_class": {
                "slide_ids": ["slide_a", "slide_b", "slide_c"],
                "labels": frame(
                    ["slide_a", "slide_a", "slide_b", "slide_b"], [0, 0, 0, 0]
                ),
            },
            "unseen_held_out_class": {
                "slide_ids": ["slide_a", "slide_b", "slide_c"],
                "labels": frame(
                    ["slide_a", "slide_a", "slide_b", "slide_b", "slide_c"],
                    [0, 1, 0, 1, 2],
                ),
                "held_out": "slide_c",
            },
        }

    return build


@pytest.fixture
def image_adversary_factory() -> Callable[..., dict[str, Any]]:
    """Build malformed and edge-position image inputs deterministically."""

    def build(*, seed: int = 307) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        rank_one = np.full((16, 16, 3), 127, dtype=np.uint8)
        rank_one[..., 1] = rank_one[..., 0]
        rank_one[..., 2] = rank_one[..., 0]
        return {
            "grayscale": rng.integers(0, 256, size=(16, 16), dtype=np.uint8),
            "wrong_channel": rng.integers(
                0, 256, size=(16, 16, 4), dtype=np.uint8
            ),
            "invalid_range": rng.uniform(-2.0, 2.0, size=(16, 16, 3)).astype(
                np.float32
            ),
            "all_white": np.full((16, 16, 3), 255, dtype=np.uint8),
            "rank_deficient": rank_one,
            "border": {
                "image": rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8),
                "coordinates": np.array([[0.0, 0.0], [15.0, 15.0]]),
            },
        }

    return build


@pytest.fixture
def artifact_adversary_factory(
    tmp_path: Path,
) -> Callable[[], dict[str, dict[str, Any]]]:
    """Write only tmp-path adversarial artifacts without deserializing objects."""

    def build() -> dict[str, dict[str, Any]]:
        paths = {
            "missing_key": tmp_path / "missing_key.npz",
            "wrong_shape_dtype": tmp_path / "wrong_shape_dtype.npz",
            "object_npz": tmp_path / "object_payload.npz",
            "corrupt_json": tmp_path / "corrupt.json",
            "row_mismatch": tmp_path / "row_mismatch.npz",
            "corrupt_bytes": tmp_path / "corrupt.bin",
        }
        np.savez_compressed(paths["missing_key"], patches=np.zeros((2, 3, 4, 4)))
        np.savez_compressed(
            paths["wrong_shape_dtype"],
            patches=np.ones((2, 4, 4), dtype=np.int16),
            spot_ids=np.asarray(["a", "b"], dtype=np.str_),
        )
        np.savez_compressed(
            paths["object_npz"], payload=np.asarray([{"unsafe": True}], dtype=object)
        )
        paths["corrupt_json"].write_text('{"incomplete":', encoding="utf-8")
        np.savez_compressed(
            paths["row_mismatch"],
            patches=np.zeros((3, 3, 4, 4), dtype=np.float32),
            spot_ids=np.asarray(["a", "b"], dtype=np.str_),
        )
        paths["corrupt_bytes"].write_bytes(b"\x00\xffnot-an-artifact")
        return {
            name: {"path": path, "kind": name, "expected_safe": False}
            for name, path in paths.items()
        }

    return build
