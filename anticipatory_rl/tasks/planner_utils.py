"""Shared helpers for invoking the symbolic planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_planner(
    planner: Path, domain: Path, problem: Path, search: str, workdir: Path
) -> Path:
    """Invoke Fast Downward (or compatible planner) and return the sas_plan path."""
    cmd = [
        sys.executable,
        str(planner),
        str(domain),
        str(problem.resolve()),
        "--search",
        search,
    ]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planner failed (code {proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    plan_candidates = sorted(workdir.glob("sas_plan*"))
    if not plan_candidates:
        raise FileNotFoundError(
            "Planner succeeded but produced no sas_plan* output."
        )
    return plan_candidates[0]


def plan_cost(plan_path: Path) -> int:
    """Count the number of actions in a sas_plan file (unit action costs)."""
    cost = 0
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("("):
            cost += 1
    return cost
