#!/usr/bin/env python3
"""Probe V_A.P. for a single random state under toy_level_3.

Enumerates every task in the environment's task distribution, solves each
with Fast Downward (astar+ff for optimal cost), and computes the weighted
anticipatory planning cost:

    V_A.P.(s) = Σ_τ  P(τ) · plan_cost(s, τ)

Usage
-----
    python scripts/gnn/probe_v_ap.py \
        --config-path configs/restaurant/toy_level_3.yaml \
        --planner-path /path/to/fast-downward/fast-downward.py \
        --domain-path   pddl/toy_restaurant_domain.pddl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from gnn.graph_encoder import state_to_graph
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    solve_restaurant_task_with_fd,
)


def _build_env(config_path: Path) -> RestaurantSymbolicEnv:
    return RestaurantSymbolicEnv(config_path=config_path)


def _task_label(task) -> str:
    parts = [task.task_type]
    if task.target_location:
        parts.append(f"@{task.target_location}")
    if task.target_kind:
        parts.append(f"kind={task.target_kind}")
    if task.object_name:
        parts.append(f"obj={task.object_name}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe V_A.P. for one state")
    ap.add_argument("--config-path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner-path", type=Path, required=True)
    ap.add_argument("--domain-path", type=Path, required=True)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--output-path", type=Path, default=None)
    args = ap.parse_args()

    env = _build_env(args.config_path)
    env.reset(seed=args.seed)

    state = RestaurantPlannerState.from_env(env)
    tasks = env.enumerate_task_distribution()

    print(f"State seed={args.seed}  |  {len(tasks)} tasks in distribution\n")
    print(f"{'Task':<40s} {'P(τ)':>8s} {'Cost':>10s} {'P·Cost':>12s} {'Status'}")
    print("-" * 90)

    v_ap = 0.0
    n_fail = 0
    t0 = time.time()

    for task, prob in tasks:
        result = solve_restaurant_task_with_fd(
            env=env,
            state=state,
            task=task,
            domain_path=args.domain_path,
            planner_path=args.planner_path,
            timeout_s=args.timeout_s,
            search="astar(ff())",
        )
        if result.success:
            cost = result.plan_cost
            status = "ok"
        else:
            cost = 1e6
            status = "FAIL"
            n_fail += 1

        weighted = prob * cost
        v_ap += weighted
        print(f"{_task_label(task):<40s} {prob:>8.4f} {cost:>10.2f} {weighted:>12.4f} {status}")

    elapsed = time.time() - t0
    print("-" * 90)
    print(f"V_A.P.(s) = {v_ap:.4f}")
    print(f"Failed: {n_fail}/{len(tasks)}  |  Elapsed: {elapsed:.1f}s")

    graph = state_to_graph(state, env)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, {graph.x.shape[1]}-dim features")

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graph": graph, "v_ap": v_ap, "seed": args.seed}, args.output_path)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
