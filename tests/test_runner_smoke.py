from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_scenarios_runner_smoke() -> None:
    from scenarios import run_all as scenarios_run_all

    rc = scenarios_run_all.main(["--fast", "--no-env", "--only", "Scenario 01 — M/M/1 basics"])
    assert rc == 0


def test_queueing_experiments_runner_smoke() -> None:
    """Contract: experiment runners execute without external path hacks."""

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "experiments" / "queueing" / "run_all.py"),
        "--fast",
        "--no-env",
        "--only",
        "Scenario 01 — M/M/1 basics",
    ]

    subprocess.run(cmd, cwd=str(repo_root), check=True)
