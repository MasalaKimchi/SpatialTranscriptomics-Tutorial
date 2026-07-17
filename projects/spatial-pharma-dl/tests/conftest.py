"""Shared contracts for the repository's mutually exclusive evidence tiers."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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
OFFLINE_GUARD_PATH = Path(__file__).with_name("offline_guard")
_MISSING = object()
_OFFLINE_STATES: dict[int, "_OfflineState"] = {}


@dataclass(frozen=True)
class _OfflineState:
    """Exact process state to restore after an embedded pytest session."""

    environment: dict[str, object]
    socket_functions: dict[str, Callable[..., Any]]


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


def _prepend_pythonpath(path: Path) -> None:
    """Propagate the Python socket guard to child Python interpreters."""
    entries = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    rendered = str(path)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [rendered, *(entry for entry in entries if entry != rendered)]
    )


def pytest_configure(config: pytest.Config) -> None:
    if _external_tier_selected(config):
        return
    environment_keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "SPATIAL_TX_OFFLINE_GUARD",
        "PYTHONPATH",
    )
    socket_functions = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "gethostbyname": socket.gethostbyname,
        "gethostbyname_ex": socket.gethostbyname_ex,
        "gethostbyaddr": socket.gethostbyaddr,
    }
    _OFFLINE_STATES[id(config)] = _OfflineState(
        environment={key: os.environ.get(key, _MISSING) for key in environment_keys},
        socket_functions=socket_functions,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SPATIAL_TX_OFFLINE_GUARD"] = "1"
    _prepend_pythonpath(OFFLINE_GUARD_PATH)
    socket.socket.connect = _deny_socket  # type: ignore[method-assign]
    socket.socket.connect_ex = _deny_socket  # type: ignore[method-assign]
    socket.create_connection = _deny_socket
    socket.getaddrinfo = _deny_socket
    socket.gethostbyname = _deny_socket
    socket.gethostbyname_ex = _deny_socket
    socket.gethostbyaddr = _deny_socket


def pytest_unconfigure(config: pytest.Config) -> None:
    if _external_tier_selected(config):
        return
    state = _OFFLINE_STATES.pop(id(config), None)
    if state is None:
        return
    socket.socket.connect = state.socket_functions[  # type: ignore[method-assign]
        "connect"
    ]
    socket.socket.connect_ex = state.socket_functions[  # type: ignore[method-assign]
        "connect_ex"
    ]
    socket.create_connection = state.socket_functions["create_connection"]
    socket.getaddrinfo = state.socket_functions["getaddrinfo"]
    socket.gethostbyname = state.socket_functions["gethostbyname"]
    socket.gethostbyname_ex = state.socket_functions["gethostbyname_ex"]
    socket.gethostbyaddr = state.socket_functions["gethostbyaddr"]
    for key, value in state.environment.items():
        if value is _MISSING:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


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
        *,
        slide_id: str = "slide_a",
        seed: int = 101,
        n_spots: int = 12,
        n_genes: int = 10,
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
        if not 0 <= n_genes <= len(genes):
            raise ValueError(f"n_genes must be between 0 and {len(genes)}")
        genes = genes[:n_genes]
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
) -> Callable[[], dict[str, dict[str, Any]]]:
    """Build malformed key tables used by later alignment validation."""

    class HostileString(str):
        """A string subclass whose data-model hooks must never be executed."""

        def _raise(self, operation: str):
            raise AssertionError(f"identity validation executed hostile {operation}")

        def strip(self, *_args: object, **_kwargs: object):
            return self._raise("strip")

        def __repr__(self):
            return self._raise("repr")

        def __str__(self):
            return self._raise("str")

        def __hash__(self):
            return self._raise("hash")

        def __eq__(self, _other: object):
            return self._raise("equality")

        def __lt__(self, _other: object):
            return self._raise("comparison")

        def __iter__(self):
            return self._raise("iteration")

    def build() -> dict[str, dict[str, Any]]:
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

        shuffled_labels = labels.sample(frac=1.0, random_state=17).reset_index(drop=True)
        shuffled_metadata = patch_index.sample(
            frac=1.0, random_state=23
        ).reset_index(drop=True)
        missing_label_slide = labels.drop(columns="slide_id")
        missing_metadata_spot = patch_index.drop(columns="spot_id")
        blank_labels = labels.copy(deep=True)
        blank_labels.loc[0, "spot_id"] = "   "
        wrong_type_metadata = patch_index.copy(deep=True)
        wrong_type_metadata["spot_id"] = wrong_type_metadata["spot_id"].astype(object)
        wrong_type_metadata.loc[0, "spot_id"] = 17
        hostile_labels = labels.copy(deep=True)
        hostile_labels["spot_id"] = hostile_labels["spot_id"].astype(object)
        hostile_labels.loc[0, "spot_id"] = HostileString("hostile")
        hostile_metadata = patch_index.copy(deep=True)
        hostile_metadata["slide_id"] = hostile_metadata["slide_id"].astype(object)
        hostile_metadata.loc[0, "slide_id"] = HostileString("slide_a")
        duplicate_metadata = patch_index.copy(deep=True)
        duplicate_metadata.loc[1, ["slide_id", "spot_id"]] = (
            duplicate_metadata.loc[0, ["slide_id", "spot_id"]].to_numpy()
        )
        wrong_slide_metadata = patch_index.loc[
            patch_index["slide_id"] == "slide_a"
        ].copy()
        wrong_slide_metadata.loc[0, "slide_id"] = "slide_b"
        reserved_labels = labels.copy(deep=True)
        reserved_labels["_label_source_row"] = range(len(reserved_labels))
        reserved_metadata = patch_index.copy(deep=True)
        reserved_metadata["_patch_source_row"] = range(len(reserved_metadata))

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
            "shuffled_complete": {
                "labels": shuffled_labels,
                "patch_index": shuffled_metadata,
            },
            "missing_label_slide": {
                "labels": missing_label_slide,
                "patch_index": patch_index.copy(),
            },
            "missing_metadata_spot": {
                "labels": labels.copy(),
                "patch_index": missing_metadata_spot,
            },
            "blank": {
                "labels": blank_labels,
                "patch_index": patch_index.copy(),
            },
            "wrong_type": {
                "labels": labels.copy(),
                "patch_index": wrong_type_metadata,
            },
            "hostile_label": {
                "labels": hostile_labels,
                "patch_index": patch_index.copy(),
            },
            "hostile_metadata": {
                "labels": labels.copy(),
                "patch_index": hostile_metadata,
            },
            "duplicate_metadata": {
                "labels": labels.copy(),
                "patch_index": duplicate_metadata,
            },
            "wrong_slide_metadata": {
                "labels": labels.copy(),
                "patch_index": wrong_slide_metadata,
            },
            "value_row_mismatch": {
                "labels": labels.copy(),
                "patch_index": patch_index.copy(),
                "value_row_count": len(patch_index) - 1,
            },
            "reserved_label": {
                "labels": reserved_labels,
                "patch_index": patch_index.copy(),
            },
            "reserved_metadata": {
                "labels": labels.copy(),
                "patch_index": reserved_metadata,
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


@pytest.fixture
def artifact_contract_factory(tmp_path: Path) -> Callable[[], dict[str, Any]]:
    """Build fresh byte generations and inert fault logs for artifact tests."""

    def build() -> dict[str, Any]:
        generations = {
            "old": {
                "payload": b"trusted-old-generation\n",
                "schema": {"generation": "old", "encoding": "utf-8"},
            },
            "new": {
                "payload": b"trusted-new-generation\n",
                "schema": {"generation": "new", "encoding": "utf-8"},
            },
        }
        malformed = {
            "zero": b"",
            "invalid_utf8": b"\xff",
            "invalid_json": b'{"incomplete":',
            "duplicate": b'{"schema_version":1,"schema_version":2}',
            "oversized": b"{" + (b" " * 70_000) + b"}",
        }
        return {
            "root": tmp_path,
            "final_path": tmp_path / "artifact.bin",
            "generations": generations,
            "malformed": malformed,
            "operation_log": [],
            "faults": (
                "write_payload",
                "fsync_payload",
                "write_manifest",
                "fsync_manifest",
                "validate",
                "replace_payload",
                "fsync_directory_first",
                "replace_manifest",
                "fsync_directory_final",
            ),
        }

    return build
