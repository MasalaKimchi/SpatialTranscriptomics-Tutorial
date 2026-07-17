"""Shared contracts for the repository's mutually exclusive evidence tiers."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterable
from pathlib import Path

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
