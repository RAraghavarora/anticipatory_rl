"""Shared helpers for invoking the symbolic planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_planner(
    planner: Path,
    domain: Path,
    problem: Path,
    workdir: Path,
    *,
    alias: str = "seq-sat-lama-2011",
    initial_search_time_limit: float = 10.0,
    max_search_time_limit: float = 320.0,
) -> Path:
    """Invoke Fast Downward with a portfolio alias and doubling search-time-limit.

    Starts with *initial_search_time_limit* seconds. If no plan is found, doubles
    the limit and retries, up to *max_search_time_limit* seconds total.
    Returns the path to the first sas_plan* file produced.
    """
    time_limit = initial_search_time_limit
    last_stderr = ""
    while time_limit <= max_search_time_limit:
        cmd = [
            sys.executable,
            str(planner.resolve()),
            "--alias",
            alias,
            "--search-time-limit",
            f"{int(time_limit)}s",
            str(domain.resolve()),
            str(problem.resolve()),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
                timeout=time_limit + 60,
            )
            last_stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            last_stderr = str(exc)
        plan_candidates = sorted(workdir.glob("sas_plan*"))
        if plan_candidates:
            return plan_candidates[0]
        time_limit *= 2
    raise RuntimeError(
        f"Planner found no plan within {max_search_time_limit}s.\nSTDERR:\n{last_stderr}"
    )


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
