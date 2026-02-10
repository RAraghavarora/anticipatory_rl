"""Benchmark classical planner costs over sampled tasks."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from anticipatory_rl.tasks.build_problem_from_task import (
    build_problem_text_for_task,
    load_tasks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fast Downward over sampled tasks and log plan costs."
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("runs") / "tasks_200.json",
        help="JSON file containing sampled tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to evaluate (with replacement).",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=0,
        help="Seed for reproducible task sampling order.",
    )
    parser.add_argument(
        "--domain",
        type=Path,
        default=Path("pddl") / "gridworld_domain.pddl",
        help="Path to the domain PDDL file.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("pddl") / "gridworld_problem.pddl",
        help="Problem template describing static facts.",
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=Path("downward") / "fast-downward.py",
        help="Fast Downward entry point.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="astar(lmcut())",
        help="Fast Downward search configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "planner_costs.json",
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def _run_planner(
    planner: Path, domain: Path, problem: Path, search: str, workdir: Path
) -> Path:
    cmd = [
        sys.executable,
        str(planner),
        str(domain),
        str(problem.resolve()),
        "--search",
        search,
    ]
    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, check=False
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


def _plan_cost(plan_path: Path) -> int:
    cost = 0
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("("):
            cost += 1
    return cost


def evaluate_tasks(args: argparse.Namespace) -> Dict[str, object]:
    tasks = load_tasks(args.tasks_file)
    rng = random.Random(args.task_seed)
    results: List[Dict[str, object]] = []

    for eval_id in range(args.num_tasks):
        task_index = rng.randrange(len(tasks))
        task = tasks[task_index]
        problem_name = f"task-{task_index}-eval-{eval_id}"
        problem_text = build_problem_text_for_task(
            task, args.template, problem_name
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            problem_path = tmp_path / "problem.pddl"
            problem_path.write_text(problem_text)
            try:
                plan_path = _run_planner(
                    args.planner, args.domain, problem_path, args.search, tmp_path
                )
                cost = _plan_cost(plan_path)
                status = "solved"
            except Exception as exc:
                cost = None
                status = f"error: {exc}"

        results.append(
            {
                "eval_id": eval_id,
                "task_index": task_index,
                "task_type": task["task_type"],
                "cost": cost,
                "status": status,
            }
        )

    solved_costs = [r["cost"] for r in results if isinstance(r["cost"], int)]
    summary = {
        "tasks_file": str(args.tasks_file),
        "task_seed": args.task_seed,
        "num_tasks": args.num_tasks,
        "solved": len(solved_costs),
        "mean_cost": float(sum(solved_costs) / len(solved_costs))
        if solved_costs
        else None,
        "results": results,
    }
    return summary


def main() -> None:
    args = _parse_args()
    args.domain = args.domain.resolve()
    args.template = args.template.resolve()
    args.planner = args.planner.resolve()
    summary = evaluate_tasks(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(
        f"Wrote planner cost summary for {summary['num_tasks']} tasks to {args.output}"
    )


if __name__ == "__main__":
    main()
