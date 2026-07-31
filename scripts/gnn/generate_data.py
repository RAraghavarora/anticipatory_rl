#!/usr/bin/env python3
"""Generate training data for GNN: (graph, V_A.P.) pairs.

For each state:
1. Sample task from P(τ)
2. Solve with FD, apply plan to get terminal state
3. Compute V_A.P. = Σ P(τ') · plan_cost(state, τ')
4. Convert state to graph
5. Save (graph, v_ap) pair

Usage:
    python scripts/gnn/generate_data.py \
        --config-path configs/restaurant/toy_level_3.yaml \
        --planner-path downward/fast-downward.py \
        --domain-path pddl/toy_restaurant_domain.pddl \
        --num-states 100 \
        --output-path runs/train_data.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_plan,
    consume_delivery_from_state,
    solve_restaurant_task_with_fd,
)
from gnn.graph_encoder import state_to_graph


def _compute_v_ap(
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    tasks: list[tuple],
    planner_path: Path,
    domain_path: Path,
    timeout_s: float,
) -> tuple[float, int]:
    v_ap = 0.0
    n_calls = 0
    n_fail = 0
    for task, prob in tasks:
        result = solve_restaurant_task_with_fd(
            env=env,
            state=state,
            task=task,
            domain_path=domain_path,
            planner_path=planner_path,
            timeout_s=timeout_s,
            search="astar(ff())",
        )
        n_calls += 1
        if not result.success:
            n_fail += 1
        cost = result.plan_cost if result.success else 1e6
        v_ap += prob * cost
    return v_ap, n_calls, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GNN training data")
    ap.add_argument("--config-path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner-path", type=Path, required=True)
    ap.add_argument("--domain-path", type=Path, required=True)
    ap.add_argument("--num-states", type=int, default=2000)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()

    env = RestaurantSymbolicEnv(config_path=args.config_path)
    env.reset(seed=args.seed)

    tasks = env.enumerate_task_distribution()
    print(f"Task distribution: {len(tasks)} tasks", flush=True)

    dataset = []
    v_aps = []
    state = RestaurantPlannerState.from_env(env)
    total_fd_calls = 0
    total_fail = 0
    n_skipped = 0

    print(f"Generating {args.num_states} (graph, V_A.P.) pairs...", flush=True)
    t0 = time.time()

    for i in range(args.num_states):
        task, _ = tasks[env._task_rng.integers(len(tasks))]

        result = solve_restaurant_task_with_fd(
            env=env,
            state=state,
            task=task,
            domain_path=args.domain_path,
            planner_path=args.planner_path,
            timeout_s=args.timeout_s,
            search="astar(ff())",
        )
        total_fd_calls += 1

        if not result.success:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  [FAIL solve] {task.task_type} — {result.error}", flush=True)
            continue

        state = apply_plan(state, result.plan_actions)
        consume_delivery_from_state(state, task.task_type, task.target_location)

        v_ap, n_calls, n_fail = _compute_v_ap(
            state, env, tasks, args.planner_path, args.domain_path, args.timeout_s,
        )
        total_fd_calls += n_calls
        total_fail += n_fail
        graph = state_to_graph(state, env)
        graph.y = torch.tensor(v_ap, dtype=torch.float32)

        dataset.append({"graph": graph, "v_ap": v_ap})
        v_aps.append(v_ap)

        if (i + 1) % args.log_interval == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - n_skipped) / max(elapsed, 0.1)
            eta = (args.num_states - i - 1) / max(rate, 0.01) / 60
            print(
                f"[{i+1}/{args.num_states}] "
                f"v_ap={v_ap:.1f}  "
                f"elapsed={elapsed:.0f}s  "
                f"rate={rate:.1f}/s  "
                f"eta={eta:.0f}m  "
                f"fd={total_fd_calls} fail={total_fail}",
                flush=True,
            )

    elapsed = time.time() - t0
    n_generated = len(dataset)
    print(f"\nGenerated {n_generated} samples in {elapsed:.1f}s", flush=True)
    print(f"Total FD calls: {total_fd_calls} ({total_fd_calls / max(n_generated, 1):.1f} per state)", flush=True)
    print(f"Total FD failures: {total_fail}", flush=True)
    print(f"Skipped solves: {n_skipped}", flush=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = args.output_path.with_suffix(".pt")
    torch.save(dataset, pt_path)
    print(f"Saved {n_generated} samples to {pt_path}", flush=True)

    if v_aps:
        npz_path = args.output_path.with_suffix(".npz")
        np.savez(npz_path, v_ap=v_aps)
        print(f"Saved v_ap array ({len(v_aps)},) to {npz_path}", flush=True)
        print(f"V_A.P. stats: min={min(v_aps):.2f} max={max(v_aps):.2f} mean={np.mean(v_aps):.2f}", flush=True)


if __name__ == "__main__":
    main()
