#!/usr/bin/env python3
"""Run the repository's canonical verification tiers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

TIERS = ("fast", "notebook-smoke", "network", "full-cohort")
TEST_ROOT = "projects/spatial-pharma-dl/tests"
RUFF_PATHS = (
    "utils",
    "scripts",
    "projects/spatial-pharma-dl/src",
    "projects/spatial-pharma-dl/scripts",
    TEST_ROOT,
)
MARKERS = {
    "notebook-smoke": "notebook_smoke",
    "network": "network",
    "full-cohort": "full_cohort",
}


def _pytest_command(marker: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--strict-markers",
        "-m",
        marker,
        TEST_ROOT,
    ]


def build_commands(tier: str) -> list[list[str]]:
    """Return deterministic subprocess argument lists for ``tier``."""
    if tier == "fast":
        return [
            [sys.executable, "-m", "ruff", "check", *RUFF_PATHS],
            _pytest_command("offline"),
        ]
    if tier in MARKERS:
        return [_pytest_command(MARKERS[tier])]
    raise ValueError(f"unknown verification tier: {tier}")


def run_tier(tier: str) -> int:
    """Run one tier, stopping immediately when a subprocess fails."""
    print(f"Verification tier: {tier}", flush=True)
    for command in build_commands(tier):
        print("+ " + " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            if tier != "fast" and exc.returncode == 5:
                print(
                    f"{tier}: no tests defined for this opt-in tier; "
                    "no evidence was produced.",
                    flush=True,
                )
                return 5
            return int(exc.returncode or 1)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=TIERS, help="verification tier to run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and return a process status."""
    args = _parser().parse_args(argv)
    return run_tier(args.tier)


if __name__ == "__main__":
    raise SystemExit(main())
