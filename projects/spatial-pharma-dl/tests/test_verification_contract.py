"""Executable contracts for evidence-tier selection and offline isolation."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from conftest import (
    OfflineNetworkError,
    _validate_primary_marker_names,
    pytest_configure,
    pytest_unconfigure,
)

pytestmark = pytest.mark.offline

PHARMA = Path(__file__).resolve().parents[1]
ROOT = PHARMA.parents[1]
TESTS = PHARMA / "tests"


def _load_verify_module() -> ModuleType:
    path = ROOT / "scripts" / "verify.py"
    spec = spec_from_file_location("repository_verify", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    with pytest.raises(OfflineNetworkError, match="offline"):
        socket.socket().connect_ex(("192.0.2.1", 443))
    with pytest.raises(OfflineNetworkError, match="offline"):
        socket.getaddrinfo("example.invalid", 443)
    with pytest.raises(OfflineNetworkError, match="offline"):
        socket.gethostbyname("example.invalid")


def test_offline_guard_propagates_to_python_subprocesses() -> None:
    probe = """\
import socket

attempts = (
    lambda: socket.getaddrinfo("example.invalid", 443),
    lambda: socket.socket().connect_ex(("192.0.2.1", 443)),
)
for attempt in attempts:
    try:
        attempt()
    except RuntimeError as exc:
        assert "offline" in str(exc).lower()
    else:
        raise AssertionError("offline child network guard was bypassed")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_environment_is_restored_after_embedded_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "preserve-hf-value")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(markexpr="offline"))

    pytest_configure(config)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    pytest_unconfigure(config)

    assert os.environ["HF_HUB_OFFLINE"] == "preserve-hf-value"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


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


def test_verification_command_contract() -> None:
    verify = _load_verify_module()
    commands = verify.build_commands("fast")

    assert commands[0] == [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "utils",
        "scripts",
        "projects/spatial-pharma-dl/src",
        "projects/spatial-pharma-dl/scripts",
        "projects/spatial-pharma-dl/tests",
    ]
    assert commands[1] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--strict-markers",
        "-m",
        "offline",
        "projects/spatial-pharma-dl/tests",
    ]

    expected = {
        "notebook-smoke": "notebook_smoke",
        "network": "network",
        "full-cohort": "full_cohort",
    }
    for tier, marker in expected.items():
        command = verify.build_commands(tier)[0]
        selector_index = command.index("-m", 3)
        assert command[selector_index + 1] == marker
        assert "offline" not in command

    with pytest.raises(ValueError, match="unknown verification tier"):
        verify.build_commands("unknown")


def test_verification_failure_propagates_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    verify = _load_verify_module()
    calls: list[list[str]] = []

    def fail_first(command: list[str], *, check: bool) -> None:
        calls.append(command)
        assert check is True
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(verify.subprocess, "run", fail_first)
    assert verify.run_tier("fast") == 17
    assert calls == [verify.build_commands("fast")[0]]


def test_empty_opt_in_tier_is_explicit_non_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verify = _load_verify_module()

    def no_tests(command: list[str], *, check: bool) -> None:
        raise subprocess.CalledProcessError(5, command)

    monkeypatch.setattr(verify.subprocess, "run", no_tests)
    assert verify.run_tier("network") == 5
    output = capsys.readouterr().out
    assert "no tests defined for this opt-in tier" in output
    assert "no evidence was produced" in output


def test_verify_module_has_no_scientific_imports() -> None:
    source = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
    forbidden = (
        "scanpy",
        "squidpy",
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "datasets",
        "src.",
    )
    assert not any(name in source.lower() for name in forbidden)


def _run_commands(job: dict[str, object]) -> list[str]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return [
        str(step["run"])
        for step in steps
        if isinstance(step, dict) and "run" in step
    ]


def test_verification_workflow_contract() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "verify.yml"
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    triggers = workflow["on"]
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(dispatch_inputs) == {
        "run_notebook_smoke",
        "run_network",
        "run_full_cohort",
    }

    jobs = workflow["jobs"]
    fast = jobs["fast"]
    assert fast["runs-on"] == "ubuntu-latest"
    assert fast["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert "needs" not in fast
    assert _run_commands(fast)[-1] == "python scripts/verify.py fast"

    fast_setup = next(
        step for step in fast["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert fast_setup["with"]["python-version"] == "3.11"

    opt_in_jobs = {
        "notebook-smoke": ("run_notebook_smoke", "notebook-smoke"),
        "network": ("run_network", "network"),
        "full-cohort": ("run_full_cohort", "full-cohort"),
    }
    for job_name, (input_name, tier) in opt_in_jobs.items():
        job = jobs[job_name]
        gate = job.get("if", "")
        assert "workflow_dispatch" in gate
        assert input_name in gate
        assert _run_commands(job)[-1] == f"python scripts/verify.py {tier}"
        assert "needs" not in job
        assert job.get("continue-on-error") not in (True, "true")

    for job in jobs.values():
        for step in job["steps"]:
            cache_path = str(step.get("with", {}).get("path", "")).lower()
            assert not any(
                forbidden in cache_path
                for forbidden in (
                    "data",
                    "outputs",
                    "weight",
                    "model",
                    "artifact",
                )
            )


def test_verification_documentation_contract() -> None:
    commands = (
        "python scripts/verify.py fast",
        "python scripts/verify.py notebook-smoke",
        "python scripts/verify.py network",
        "python scripts/verify.py full-cohort",
        "python -m ruff check utils scripts projects/spatial-pharma-dl/src "
        "projects/spatial-pharma-dl/scripts projects/spatial-pharma-dl/tests",
        "python -m pytest -q --strict-markers -m offline "
        "projects/spatial-pharma-dl/tests",
    )
    for path in (ROOT / "README.md", PHARMA / "README.md"):
        documentation = path.read_text(encoding="utf-8").lower()
        assert all(command in documentation for command in commands)
        assert "required fast tier is the default cpu/offline" in documentation
        assert "explicit opt-ins" in documentation
        assert (
            "safe fixture round trips do not certify production "
            "cache/checkpoint migration"
        ) in documentation
        assert "later phases" in documentation
        assert "same marker/fixture/runner convention" in documentation
        assert "child python interpreters" in documentation
        assert "not an operating-system sandbox" in documentation
