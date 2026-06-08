#!/usr/bin/env python3
"""Compare two FD heuristics on the same toy-restaurant task sequence.

The task list is sampled once. Both runs start from the same initial state.
After each task the state evolves according to the computed plan — so the
two runs will diverge if the heuristics produce different plans.

Usage:
    python scripts/restaurant/toy_heuristic_compare.py \
        --config-path configs/restaurant/toy_restaurant.yaml \
        --domain-path pddl/toy_restaurant_domain.pddl \
        --planner-path downward/fast-downward.py \
        --num-tasks 40 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantState
from anticipatory_rl.tasks.restaurant.restaurant_planner import (
    RestaurantPlannerState,
    apply_plan,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.tasks.restaurant.restaurant_utils import sample_task


def _clone_planner_state(state: RestaurantPlannerState) -> RestaurantPlannerState:
    return state.copy()


def _run_sequence(
    *,
    env: RestaurantSymbolicEnv,
    planner_state: RestaurantPlannerState,
    tasks: List[Any],
    planner_path: Path,
    domain_path: Path,
    search: str,
    timeout_s: float,
    label: str,
) -> Dict[str, Any]:
    state = _clone_planner_state(planner_state)
    records: List[Dict[str, Any]] = []
    failures = 0
    total_wall = 0.0

    for t_idx, task in enumerate(tasks):
        t0 = time.perf_counter()
        result = solve_restaurant_task_with_fd(
            env, state, task,
            planner_path=planner_path,
            domain_path=domain_path,
            search=search,
            timeout_s=timeout_s,
        )
        elapsed = float(time.perf_counter() - t0)
        total_wall += elapsed

        steps = int(len(result.plan_actions))
        paper2_cost = float(
            planner_actions_paper2_cost(result.plan_actions, env)
            if result.success
            else float("inf")
        )
        uniform_cost = float(steps * 25)

        if result.success:
            state = apply_plan(state, result.plan_actions)
        else:
            failures += 1

        records.append({
            "task_idx": int(t_idx),
            "task_type": str(task.task_type),
            "success": bool(result.success),
            "steps": steps,
            "paper2_cost": paper2_cost,
            "uniform_cost": uniform_cost,
            "planner_solve_time_s": float(result.solve_time_s),
            "task_wall_time_s": elapsed,
            "error": result.error,
        })

    success_rates = [1.0 if r["success"] else 0.0 for r in records]
    return {
        "label": label,
        "search": search,
        "failures": int(failures),
        "success_rate": float(np.mean(success_rates)) if records else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in records])) if records else 0.0,
        "avg_paper2_cost": float(np.mean([r["paper2_cost"] for r in records if r["paper2_cost"] != float("inf")])) if records else float("nan"),
        "avg_uniform_cost": float(np.mean([r["uniform_cost"] for r in records])) if records else 0.0,
        "avg_planner_time_s": float(np.mean([r["planner_solve_time_s"] for r in records])) if records else 0.0,
        "total_wall_time_s": total_wall,
        "tasks": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two FD heuristics on toy restaurant.")
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_restaurant.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--search-a", type=str, default="astar(ff())")
    parser.add_argument("--search-b", type=str, default="lazy_greedy([ff()], preferred=[ff()])")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--num-tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_heuristic_compare/comparison.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=args.seed)
    env.reset(seed=args.seed)
    initial_state = RestaurantPlannerState.from_env(env)

    tasks = [sample_task(env, uniform_task_type_prob=0.5) for _ in range(args.num_tasks)]

    # Re-sync env for pick_place filtering
    _env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=args.seed)
    _env.reset(seed=args.seed)
    _s = RestaurantPlannerState.from_env(_env)
    safe_tasks: List[Any] = []
    for task in tasks:
        for _ in range(20):
            if task.task_type != "pick_place":
                break
            obj_name = task.object_name
            if obj_name is None:
                break
            obj = _env.state.objects.get(obj_name)
            if obj is None:
                break
            if obj.kind in {"water", "coffeegrinds"} or obj.contained_in is not None:
                task = sample_task(_env, uniform_task_type_prob=0.5)
                continue
            break
        safe_tasks.append(task)

    print(f"=== Run A: {args.search_a} ===")
    t0 = time.perf_counter()
    result_a = _run_sequence(
        env=env,
        planner_state=initial_state,
        tasks=safe_tasks,
        planner_path=args.planner_path,
        domain_path=args.domain_path,
        search=args.search_a,
        timeout_s=args.timeout_s,
        label="A",
    )
    print(f"  success={result_a['success_rate']:.2f}  failures={result_a['failures']}  "
          f"avg_steps={result_a['avg_steps']:.1f}  avg_cost={result_a['avg_paper2_cost']:.0f}  "
          f"wall={time.perf_counter()-t0:.1f}s")

    print(f"\n=== Run B: {args.search_b} ===")
    t0 = time.perf_counter()
    result_b = _run_sequence(
        env=env,
        planner_state=initial_state,
        tasks=safe_tasks,
        planner_path=args.planner_path,
        domain_path=args.domain_path,
        search=args.search_b,
        timeout_s=args.timeout_s,
        label="B",
    )
    print(f"  success={result_b['success_rate']:.2f}  failures={result_b['failures']}  "
          f"avg_steps={result_b['avg_steps']:.1f}  avg_cost={result_b['avg_paper2_cost']:.0f}  "
          f"wall={time.perf_counter()-t0:.1f}s")

    diff = {
        "success_rate": float(result_b["success_rate"] - result_a["success_rate"]),
        "avg_steps": float(result_b["avg_steps"] - result_a["avg_steps"]),
        "avg_paper2_cost": float(result_b["avg_paper2_cost"] - result_a["avg_paper2_cost"]),
        "avg_uniform_cost": float(result_b["avg_uniform_cost"] - result_a["avg_uniform_cost"]),
        "avg_planner_time_s": float(result_b["avg_planner_time_s"] - result_a["avg_planner_time_s"]),
        "total_wall_time_s": float(result_b["total_wall_time_s"] - result_a["total_wall_time_s"]),
    }

    print(f"\n=== Delta (B - A) ===")
    print(json.dumps(diff, indent=2))

    div = _divergence_count(result_a["tasks"], result_b["tasks"])
    print(f"\nPlan divergences: {div} / {args.num_tasks} tasks (different step counts)")

    comparison = {
        "seed": int(args.seed),
        "num_tasks": int(args.num_tasks),
        "search_a": args.search_a,
        "search_b": args.search_b,
        "result_a": result_a,
        "result_b": result_b,
        "delta": diff,
        "plan_divergences": int(div),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    print(f"\nWrote comparison -> {args.output_path}")


def _divergence_count(
    tasks_a: List[Dict[str, Any]],
    tasks_b: List[Dict[str, Any]],
) -> int:
    diverged = 0
    for i in range(min(len(tasks_a), len(tasks_b))):
        if tasks_a[i]["steps"] != tasks_b[i]["steps"]:
            diverged += 1
    return diverged


if __name__ == "__main__":
    main()
