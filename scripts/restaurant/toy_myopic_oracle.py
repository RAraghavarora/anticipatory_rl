#!/usr/bin/env python3
"""Myopic Fast Downward oracle baseline for the toy restaurant env.

Solves a sequence of tasks myopically: for each task, calls FD to get the
optimal plan from the current state, applies the plan, and continues.

Usage:
    python scripts/restaurant/toy_myopic_oracle.py \
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
from typing import Any, Dict, List

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_plan,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)


def run_myopic_oracle(
    *,
    config_path: Path,
    domain_path: Path,
    planner_path: Path,
    num_tasks: int,
    tasks_per_reset: int,
    search: str,
    timeout_s: float,
    seed: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed)
    obs, info = env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    records: List[Dict[str, Any]] = []
    failures = 0
    total_wall = 0.0
    tasks_since_reset = 0
    episode_index = 0

    for t_idx in range(num_tasks):
        task = env.task
        task_type = str(task.task_type)
        auto_satisfied = bool(env._pending_auto_success)

        if auto_satisfied:
            result_success = True
            plan_actions = []
            planner_solve_time_s = 0.0
            elapsed = 0.0
            error = None
        else:
            t0 = time.perf_counter()
            result = solve_restaurant_task_with_fd(
                env,
                state,
                task,
                planner_path=planner_path,
                domain_path=domain_path,
                search=search,
                timeout_s=timeout_s,
            )
            elapsed = float(time.perf_counter() - t0)
            total_wall += elapsed
            result_success = bool(result.success)
            plan_actions = result.plan_actions
            planner_solve_time_s = float(result.solve_time_s)
            error = result.error

        steps = int(len(plan_actions))
        paper2_cost = float(planner_actions_paper2_cost(plan_actions, env) if result_success else float("inf"))
        uniform_cost = float(steps * 25)

        if result_success:
            state = apply_plan(state, plan_actions)
            consume_delivery_from_state(state, task_type, task.target_location)
            env.state.agent_location = state.agent_location
            env.state.holding = state.holding
            env.state.bread_spread = state.bread_spread
            for name, obj in state.objects.items():
                env.state.objects[name].location = obj.location
                env.state.objects[name].dirty = obj.dirty
                env.state.objects[name].filled_with = obj.filled_with
                env.state.objects[name].contained_in = obj.contained_in
        else:
            failures += 1

        records.append({
            "task_idx": int(t_idx),
            "task_type": task_type,
            "target_location": task.target_location,
            "target_kind": task.target_kind,
            "object_name": task.object_name,
            "success": bool(result_success),
            "auto_satisfied": bool(auto_satisfied),
            "steps": steps,
            "paper2_cost": paper2_cost,
            "uniform_cost": uniform_cost,
            "planner_solve_time_s": planner_solve_time_s,
            "task_wall_time_s": elapsed,
            "error": error,
        })

        if result_success and steps > 0:
            for i, (name, args) in enumerate(plan_actions):
                print(f"    [{i}] {name} {' '.join(args)}")

        if result_success:
            tasks_since_reset += 1
        if tasks_per_reset > 0 and tasks_since_reset >= tasks_per_reset:
            episode_index += 1
            env.reset(seed=seed + 100_003 * episode_index)
            state = RestaurantPlannerState.from_env(env)
            tasks_since_reset = 0
        else:
            env._resample_task()

    stats = {
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "tasks_per_reset": int(tasks_per_reset),
        "failures": int(failures),
        "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in records])) if records else 0.0,
        "auto_rate": float(np.mean([1.0 if r["auto_satisfied"] else 0.0 for r in records])) if records else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in records])) if records else 0.0,
        "avg_paper2_cost": float(np.mean([r["paper2_cost"] for r in records if r["paper2_cost"] != float("inf")])) if records else 0.0,
        "avg_uniform_cost": float(np.mean([r["uniform_cost"] for r in records])) if records else 0.0,
        "avg_planner_time_s": float(np.mean([r["planner_solve_time_s"] for r in records])) if records else 0.0,
        "total_wall_time_s": total_wall,
    }
    return {"stats": stats, "tasks": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Myopic FD oracle baseline for toy restaurant.")
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_restaurant.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--search", type=str, default="astar(ff())")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--num-tasks", type=int, default=40)
    parser.add_argument("--tasks-per-reset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_myopic_oracle/results.json"))
    args = parser.parse_args()

    result = run_myopic_oracle(
        config_path=args.config_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        num_tasks=args.num_tasks,
        tasks_per_reset=args.tasks_per_reset,
        search=args.search,
        timeout_s=args.timeout_s,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
