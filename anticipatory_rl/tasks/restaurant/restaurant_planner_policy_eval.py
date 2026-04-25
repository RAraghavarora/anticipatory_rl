"""Evaluate myopic vs anticipatory restaurant planner policies (FD + GNN)."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from anticipatory_rl.agents.train_restaurant_apcost_gnn import APCostEstimator
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.tasks.restaurant_graph import build_restaurant_graph
from anticipatory_rl.tasks.restaurant_planner import (
    PlannerResult,
    RestaurantPlannerState,
    apply_plan,
    solve_restaurant_task_with_fd,
    task_goal_clauses,
)
from anticipatory_rl.tasks.restaurant.restaurant_utils import sample_task


@dataclass
class EpisodeSpec:
    layout: Dict[str, Any]
    initial_state: RestaurantPlannerState
    tasks: List[RestaurantTask]


def _load_layouts(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("layouts"), list):
        return [x for x in payload["layouts"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError(f"Unsupported layout corpus format: {path}")


def _build_episode_specs(
    *,
    env: RestaurantSymbolicEnv,
    layouts: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    eval_layout_count: int,
    task_sequence_length: int,
    sample_layout_per_reset: bool,
    use_layout_task_library: bool,
    seed_base: int,
) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for i in range(eval_layout_count):
        if sample_layout_per_reset:
            layout = layouts[int(rng.integers(0, len(layouts)))]
        else:
            layout = layouts[i % len(layouts)]
        options: Dict[str, Any] = {"layout": layout}
        if use_layout_task_library and isinstance(layout.get("task_library"), list):
            options["task_library"] = layout.get("task_library")
        env.reset(seed=int(seed_base + i), options=options)
        initial_state = RestaurantPlannerState.from_env(env)
        tasks: List[RestaurantTask] = []
        for _ in range(task_sequence_length):
            tasks.append(sample_task(env))
        specs.append(EpisodeSpec(layout=layout, initial_state=initial_state, tasks=tasks))
    return specs


def _predict_ap_cost(
    model: APCostEstimator,
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    device: torch.device,
) -> float:
    graph = build_restaurant_graph(state, locations=env.locations, location_coords=env.location_coords)
    with torch.no_grad():
        pred = model(
            torch.tensor(graph.node_features, dtype=torch.float32, device=device),
            torch.tensor(graph.edge_index, dtype=torch.int64, device=device),
        )
    return float(pred.item())


def _pick_anticipatory_plan(
    *,
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    model: APCostEstimator,
    planner_path: Path,
    domain_path: Path,
    search: str,
    followup_candidates: Sequence[RestaurantTask],
    timeout_s: float,
    device: torch.device,
) -> PlannerResult:
    best: PlannerResult | None = None
    best_score = float("inf")
    for fut in followup_candidates:
        extra = task_goal_clauses(
            state,
            fut,
            service_locations=env.service_locations,
            wash_ready_locations=env.wash_ready_locations,
        )
        result = solve_restaurant_task_with_fd(
            env,
            state,
            task,
            planner_path=planner_path,
            domain_path=domain_path,
            search=search,
            extra_goal_clauses=extra,
            timeout_s=timeout_s,
        )
        if not result.success:
            continue
        terminal = apply_plan(state, result.plan_actions)
        ap_pred = _predict_ap_cost(model, terminal, env, device)
        score = float(result.plan_cost + ap_pred)
        if score < best_score:
            best_score = score
            best = result
    if best is not None:
        return best
    return solve_restaurant_task_with_fd(
        env,
        state,
        task,
        planner_path=planner_path,
        domain_path=domain_path,
        search=search,
        timeout_s=timeout_s,
    )


def evaluate_policy(
    *,
    policy: str,
    env: RestaurantSymbolicEnv,
    episodes: Sequence[EpisodeSpec],
    planner_path: Path,
    domain_path: Path,
    search: str,
    timeout_s: float,
    model: APCostEstimator | None,
    device: torch.device,
    anticipatory_followups: int,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    failures = 0
    for epi_idx, epi in enumerate(episodes):
        state = epi.initial_state.copy()
        for t_idx, task in enumerate(epi.tasks):
            t0 = time.perf_counter()
            if policy == "myopic":
                result = solve_restaurant_task_with_fd(
                    env,
                    state,
                    task,
                    planner_path=planner_path,
                    domain_path=domain_path,
                    search=search,
                    timeout_s=timeout_s,
                )
            else:
                assert model is not None
                fut = [sample_task(env) for _ in range(anticipatory_followups)]
                result = _pick_anticipatory_plan(
                    env=env,
                    state=state,
                    task=task,
                    model=model,
                    planner_path=planner_path,
                    domain_path=domain_path,
                    search=search,
                    followup_candidates=fut,
                    timeout_s=timeout_s,
                    device=device,
                )
            if result.success:
                state = apply_plan(state, result.plan_actions)
            else:
                failures += 1
            elapsed = float(time.perf_counter() - t0)
            records.append(
                {
                    "episode_idx": int(epi_idx),
                    "task_idx": int(t_idx),
                    "task_type": task.task_type,
                    "target_location": task.target_location,
                    "target_kind": task.target_kind,
                    "success": bool(result.success),
                    "steps": int(len(result.plan_actions)),
                    "paper2_cost": float(result.plan_cost),
                    "planner_solve_time_s": float(result.solve_time_s),
                    "task_wall_time_s": elapsed,
                    "error": result.error,
                }
            )
    stats = {
        "tasks_attempted": int(len(records)),
        "failures": int(failures),
        "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in records])) if records else 0.0,
        "avg_steps_per_task": float(np.mean([r["steps"] for r in records])) if records else 0.0,
        "avg_paper2_cost": float(np.mean([r["paper2_cost"] for r in records])) if records else 0.0,
        "avg_task_wall_time_s": float(np.mean([r["task_wall_time_s"] for r in records])) if records else 0.0,
        "total_wall_time_s": float(np.sum([r["task_wall_time_s"] for r in records])) if records else 0.0,
        "avg_planner_solve_time_s": float(np.mean([r["planner_solve_time_s"] for r in records])) if records else 0.0,
    }
    return {"policy": policy, "stats": stats, "tasks": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate restaurant planner myopic vs anticipatory (FD + GNN).")
    parser.add_argument("--layout-corpus", type=Path, default=Path("data/restaurant_layouts/paper2_scale_layouts.json"))
    parser.add_argument("--config-path", type=Path, default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/restaurant_domain.pddl"))
    parser.add_argument("--search", type=str, default="astar(lmcut())")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--eval-layout-count", type=int, default=10)
    parser.add_argument("--task-sequence-length", type=int, default=40)
    parser.add_argument("--sample-layout-per-reset", action="store_true")
    parser.add_argument("--task-library-per-layout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anticipatory-followups", type=int, default=8)
    parser.add_argument("--apcost-weights", type=Path, default=Path("runs/restaurant_apcost_gnn/apcost_estimator.pt"))
    parser.add_argument("--apcost-hidden-dim", type=int, default=128)
    parser.add_argument("--apcost-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("runs/paper2_planner_compare/planner_compare.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layouts = _load_layouts(args.layout_corpus)
    env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=args.seed)
    rng = np.random.default_rng(args.seed + 7)
    episodes = _build_episode_specs(
        env=env,
        layouts=layouts,
        rng=rng,
        eval_layout_count=args.eval_layout_count,
        task_sequence_length=args.task_sequence_length,
        sample_layout_per_reset=bool(args.sample_layout_per_reset),
        use_layout_task_library=bool(args.task_library_per_layout),
        seed_base=args.seed + 100_000,
    )

    model = APCostEstimator(in_dim=13, hidden_dim=args.apcost_hidden_dim, layers=args.apcost_layers).to(device)
    model.load_state_dict(torch.load(args.apcost_weights, map_location=device))
    model.eval()

    myopic = evaluate_policy(
        policy="myopic",
        env=env,
        episodes=episodes,
        planner_path=args.planner_path,
        domain_path=args.domain_path,
        search=args.search,
        timeout_s=args.timeout_s,
        model=None,
        device=device,
        anticipatory_followups=args.anticipatory_followups,
    )
    anticipatory = evaluate_policy(
        policy="anticipatory",
        env=env,
        episodes=episodes,
        planner_path=args.planner_path,
        domain_path=args.domain_path,
        search=args.search,
        timeout_s=args.timeout_s,
        model=model,
        device=device,
        anticipatory_followups=args.anticipatory_followups,
    )
    comparison = {
        "seed": int(args.seed),
        "eval_layout_count": int(args.eval_layout_count),
        "task_sequence_length": int(args.task_sequence_length),
        "myopic": myopic,
        "anticipatory": anticipatory,
        "delta": {
            "avg_paper2_cost": float(anticipatory["stats"]["avg_paper2_cost"] - myopic["stats"]["avg_paper2_cost"]),
            "avg_steps_per_task": float(anticipatory["stats"]["avg_steps_per_task"] - myopic["stats"]["avg_steps_per_task"]),
            "avg_task_wall_time_s": float(anticipatory["stats"]["avg_task_wall_time_s"] - myopic["stats"]["avg_task_wall_time_s"]),
            "success_rate": float(anticipatory["stats"]["success_rate"] - myopic["stats"]["success_rate"]),
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    print(f"Wrote planner comparison -> {args.output_path}")
    print(json.dumps(comparison["delta"], indent=2))


if __name__ == "__main__":
    main()
