"""Shared helpers for invoking the symbolic planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_planner(
    planner: Path,
    domain: Path,
    problem: Path,
    workdir: Path,
    *,
    alias: str = "seq-sat-lama-2011",
    search: Optional[str] = None,
    initial_search_time_limit: float = 10.0,
    max_search_time_limit: float = 320.0,
) -> Path:
    """Invoke Fast Downward and double the search-time-limit on retry.

    Starts with *initial_search_time_limit* seconds. If no plan is found, doubles
    the limit and retries, up to *max_search_time_limit* seconds total.
    Returns the path to the first sas_plan* file produced.

    *alias* selects a portfolio alias (default: ``seq-sat-lama-2011``). When
    *search* is provided, it is passed to Fast Downward via ``--search`` and the
    portfolio alias is bypassed.
    """
    time_limit = initial_search_time_limit
    last_stderr = ""
    while time_limit <= max_search_time_limit:
        # Driver options come before the input files; component options
        # (`--search "..."`) come after them. `--alias` is a driver option,
        # so it is consistent with the previous wrapper behaviour.
        cmd = [sys.executable, str(planner.resolve())]
        if search is None:
            cmd.extend(["--alias", alias])
        cmd.extend([
            "--search-time-limit",
            f"{int(time_limit)}s",
            str(domain.resolve()),
            str(problem.resolve()),
        ])
        if search is not None:
            cmd.extend(["--search", search])
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
            # LAMA outputs sas_plan.1, sas_plan.2, etc. where higher numbers are better plans.
            # We want the last (best) plan found before the timeout.
            def get_plan_idx(p: Path) -> int:
                try:
                    return int(p.suffix.strip('.'))
                except ValueError:
                    return 0
            plan_candidates.sort(key=get_plan_idx)
            return plan_candidates[-1]
        time_limit *= 2
    raise RuntimeError(
        f"Planner found no plan within {max_search_time_limit}s.\nSTDERR:\n{last_stderr}"
    )
