"""Executable contracts for evidence-tier selection and offline isolation."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import OfflineNetworkError, _validate_primary_marker_names

pytestmark = pytest.mark.offline

PHARMA = Path(__file__).resolve().parents[1]
ROOT = PHARMA.parents[1]
TESTS = PHARMA / "tests"


def test_primary_tier_requires_exactly_one_marker() -> None:
    with pytest.raises(pytest.UsageError, match="exactly one"):
        _validate_primary_marker_names(set())
    with pytest.raises(pytest.UsageError, match="network, offline"):
        _validate_primary_marker_names({"offline", "network"})
    assert _validate_primary_marker_names({"offline"}) == "offline"


def test_offline_environment_disables_model_hubs() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_socket_apis_fail_closed() -> None:
    with pytest.raises(OfflineNetworkError, match="offline"):
        socket.create_connection(("192.0.2.1", 443))
    with pytest.raises(OfflineNetworkError, match="offline"):
        socket.socket().connect(("192.0.2.1", 443))


def test_bare_pytest_defaults_to_offline(tmp_path: Path) -> None:
    sentinel = tmp_path / "test_tier_sentinels.py"
    sentinel.write_text(
        """\
import pytest

@pytest.mark.offline
def test_offline_sentinel():
    assert True

@pytest.mark.notebook_smoke
def test_notebook_smoke_must_not_execute():
    raise AssertionError("notebook-smoke tier executed")

@pytest.mark.network
def test_network_must_not_execute():
    raise AssertionError("network tier executed")

@pytest.mark.full_cohort
def test_full_cohort_must_not_execute():
    raise AssertionError("full-cohort tier executed")
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(ROOT / "pyproject.toml"),
            "-p",
            "conftest",
            str(sentinel),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
    assert "3 deselected" in output
