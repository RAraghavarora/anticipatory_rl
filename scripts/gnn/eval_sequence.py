#!/usr/bin/env python3
"""Evaluate trained GNN (APCostEstimator) on a persistent task sequence.

Anticipatory planning: for each task, evaluates myopic + augmented FD plans,
scores each via prefix_cost + gamma * V_AP(post), and selects the best.

Usage:
    python scripts/gnn/eval_sequence.py \
        --sequence-path experiments/sequences/iid-eval-seq-00.json \
        --gnn-model runs/gnn_train/best_model.pt \
        --max-tasks 3 \
        --max-augs 5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "restaurant"))
import toy_anticipatory_oracle as tao

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    PlannerResult,
    RestaurantPlannerState,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from gnn.graph_encoder import BINARY_ATTRS, NODE_TYPES, SBERT_DIM, state_to_graph
from gnn.model import APCostEstimator

INPUT_DIM = SBERT_DIM + len(NODE_TYPES) + 2 + len(BINARY_ATTRS)  # 399


# ---------------------------------------------------------------------------
# GNN helpers
# ---------------------------------------------------------------------------


def load_gnn_model(checkpoint_path: Path, hidden_dim: int, device: torch.device) -> APCostEstimator:
    """Load saved state_dict, return eval-mode model."""
    model = APCostEstimator(INPUT_DIM, hidden_dim=hidden_dim)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_v_ap(
    state: RestaurantPlannerState,
    model: APCostEstimator,
    env: RestaurantSymbolicEnv,
    device: torch.device,
) -> float:
    """state_to_graph → GNN forward → float. Returns scalar."""
    graph = state_to_graph(state, env)
    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        out = model(x, edge_index, batch)
    return float(out.item())


# ---------------------------------------------------------------------------
# Augmented-clause generation
# ---------------------------------------------------------------------------


def extract_encountered_objects(
    prefix: list[tuple[str, list[str]]],
    state: RestaurantPlannerState,
) -> tuple[set[str], set[str]]:
    """Extract object names and visited locations from prefix action arguments.

    Returns (encountered_objects, visited_locations).
    """
    encountered: set[str] = set()
    visited_locations: set[str] = set()

    for name, args in prefix:
        if name == "move":
            if len(args) >= 2:
                visited_locations.add(args[1])
        for arg in args:
            if arg in state.objects:
                encountered.add(arg)

    for obj_name, obj in state.objects.items():
        if obj.location in visited_locations:
            encountered.add(obj_name)

    return encountered, visited_locations


def generate_augmented_clauses(
    encountered: set[str],
    visited_locations: set[str],
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
) -> list[str]:
    """Generate p_add predicate clauses per object kind.

    fountain-gated rules only trigger when fountain is in visited_locations.
    coffeemachine-gated rules only trigger when coffeemachine is visited.
    """
    coffeemachine_locs = {
        loc for loc in env.locations if env._is_location(loc, "coffeemachine")
    }
    fountain_locs = {
        loc for loc in env.locations if env._is_location(loc, "fountain")
    }
    fountain_visited = bool(visited_locations & fountain_locs)
    coffee_visited = bool(visited_locations & coffeemachine_locs)

    clauses: set[str] = set()

    for obj_name in encountered:
        if obj_name not in state.objects:
            continue
        obj = state.objects[obj_name]

        if obj.kind == "water" and obj_name != "water_fountain":
            if obj.location is None:
                clauses.add(f"(is-at water {env.station_coffee})")

        if obj.kind in {"cup", "mug", "jar", "bowl", "knife", "plate"}:
            if obj.dirty:
                clauses.add(f"(not (is-dirty {obj_name}))")

        if obj.kind in {"cup", "mug", "jar"}:
            if obj.filled_with is None and fountain_visited:
                clauses.add(f"(filled-with water {obj_name})")

        if obj.kind == "jar":
            if obj.location not in coffeemachine_locs and coffee_visited:
                clauses.add(f"(is-at {obj_name} {env.station_coffee})")

    return list(clauses)


# ---------------------------------------------------------------------------
# Candidate plan scoring
# ---------------------------------------------------------------------------

_Candidate = Dict[str, Any]


def _evaluate_plan(
    state: RestaurantPlannerState,
    plan_actions: list[tuple[str, list[str]]],
    task: RestaurantTask,
    env: RestaurantSymbolicEnv,
    model: APCostEstimator,
    device: torch.device,
    gamma: float,
    strategy: str,
) -> _Candidate:
    """Score a plan: apply until task satisfied → post → GNN → score."""
    terminal, prefix = tao.apply_plan_until_first_task_satisfied(
        state, plan_actions, task, env,
    )
    if not tao._task_is_auto_satisfied(terminal, task, env):
        raise ValueError(
            f"Terminal state does not satisfy task {task.task_type}"
        )

    prefix_cost = planner_actions_paper2_cost(prefix, env)

    post = terminal.copy()
    consume_delivery_from_state(post, task.task_type, task.target_location)

    assert post.agent_location in env.location_index, (
        f"Agent at invalid location {post.agent_location}"
    )

    v_ap = predict_v_ap(post, model, env, device)

    discount = gamma ** len(prefix)
    score = prefix_cost + discount * v_ap

    return {
        "prefix": prefix,
        "prefix_cost": float(prefix_cost),
        "post": post,
        "strategy": strategy,
        "v_ap": v_ap,
        "score": float(score),
        "actions": len(prefix),
    }


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------


def run_sequence(args: argparse.Namespace) -> dict:
    """Main sequence runner. Returns {summary: {...}, tasks: [...]}."""

    # Set up device
    device = torch.device("cpu")

    # Load env and initial state
    env = RestaurantSymbolicEnv(config_path=args.config_path)
    obs, info = env.reset(seed=args.seed)
    state = RestaurantPlannerState.from_env(env)

    # Load task sequence
    with open(args.sequence_path, "r", encoding="utf-8") as fh:
        seq_data = json.load(fh)
    all_tasks_raw: list[dict] = seq_data["tasks"]
    if args.max_tasks is not None and args.max_tasks > 0:
        all_tasks_raw = all_tasks_raw[: args.max_tasks]

    # Load GNN model
    model = load_gnn_model(args.gnn_model, args.hidden_dim, device)

    # Build RestaurantTask objects
    tasks: list[RestaurantTask] = []
    for raw in all_tasks_raw:
        tasks.append(RestaurantTask(
            task_type=raw["task_type"],
            target_location=raw.get("target_location"),
            target_kind=raw.get("target_kind"),
            object_name=raw.get("object_name"),
        ))

    t_start = time.perf_counter()
    records: list[dict] = []

    total_fd_calls = 0
    total_gnn_calls = 0
    completed = 0
    auto_count = 0
    myopic_total_cost = 0.0
    total_cost = 0.0
    strategy_counts: dict[str, int] = {}
    search = args.search

    for idx, task in enumerate(tasks):
        task_type = str(task.task_type)
        fd_calls = 0
        gnn_calls = 0

        # --- Step 1: auto-satisfied ---
        if tao._task_is_auto_satisfied(state, task, env):
            consume_delivery_from_state(state, task_type, task.target_location)
            records.append({
                "index": idx,
                "task_type": task_type,
                "auto": True,
                "success": True,
                "cost": 0.0,
                "actions": 0,
                "strategy": "auto",
                "v_ap": 0.0,
                "fd_calls": 0,
                "gnn_calls": 0,
                "trace": [],
                "augments_tried": 0,
                "augments_accepted": 0,
            })
            auto_count += 1
            completed += 1
            strategy_counts["auto"] = strategy_counts.get("auto", 0) + 1
            continue

        # --- Step 2: myopic candidate ---
        result: PlannerResult = solve_restaurant_task_with_fd(
            env=env,
            state=state,
            task=task,
            planner_path=args.planner_path,
            domain_path=args.domain_path,
            search=search,
            timeout_s=args.fd_timeout_s,
        )
        fd_calls += 1

        if not result.success:
            records.append({
                "index": idx,
                "task_type": task_type,
                "auto": False,
                "success": False,
                "cost": float("inf"),
                "actions": 0,
                "strategy": "failed_myopic",
                "v_ap": 0.0,
                "fd_calls": fd_calls,
                "gnn_calls": 0,
                "trace": [],
                "augments_tried": 0,
                "augments_accepted": 0,
            })
            total_fd_calls += fd_calls
            continue

        try:
            myopic_candidate = _evaluate_plan(
                state, result.plan_actions, task, env,
                model, device, args.gamma, "myopic",
            )
        except ValueError:
            records.append({
                "index": idx, "task_type": task_type, "auto": False,
                "success": False, "cost": float("inf"), "actions": 0,
                "strategy": "failed_myopic", "v_ap": 0.0,
                "fd_calls": fd_calls, "gnn_calls": 0, "trace": [],
                "augments_tried": 0, "augments_accepted": 0,
            })
            total_fd_calls += fd_calls
            continue
        gnn_calls += 1

        best = myopic_candidate
        best_score = best["score"]
        candidates = [myopic_candidate]

        # --- Step 3: augmented candidates ---
        prefix = best["prefix"]
        encountered, visited_locs = extract_encountered_objects(prefix, state)
        p_add_list = generate_augmented_clauses(encountered, visited_locs, state, env)

        augments_tried = 0
        augments_accepted = 0
        for p_add in p_add_list[: args.max_augs]:
            aug_result = solve_restaurant_task_with_fd(
                env=env,
                state=state,
                task=task,
                planner_path=args.planner_path,
                domain_path=args.domain_path,
                search=search,
                extra_goal_clauses=[p_add],
                timeout_s=args.fd_timeout_s,
            )
            fd_calls += 1
            augments_tried += 1

            if not aug_result.success:
                continue

            try:
                aug_candidate = _evaluate_plan(
                    state, aug_result.plan_actions, task, env,
                    model, device, args.gamma, f"aug+{p_add}",
                )
            except ValueError:
                continue
            gnn_calls += 1

            if aug_candidate["score"] < best_score:
                best = aug_candidate
                best_score = aug_candidate["score"]
                augments_accepted += 1
            candidates.append(aug_candidate)

        # --- Step 4: execute best ---
        state = best["post"]

        records.append({
            "index": idx,
            "task_type": task_type,
            "auto": False,
            "success": True,
            "cost": best["prefix_cost"],
            "actions": best["actions"],
            "strategy": best["strategy"],
            "v_ap": best["v_ap"],
            "fd_calls": fd_calls,
            "gnn_calls": gnn_calls,
            "trace": [f"{name}({', '.join(args)})" for name, args in best["prefix"]],
            "augments_tried": augments_tried,
            "augments_accepted": augments_accepted,
        })
        completed += 1
        total_cost += best["prefix_cost"]
        myopic_total_cost += myopic_candidate["prefix_cost"]
        total_fd_calls += fd_calls
        total_gnn_calls += gnn_calls
        strategy_counts[best["strategy"]] = strategy_counts.get(best["strategy"], 0) + 1

    wall_seconds = time.perf_counter() - t_start
    attempted = len(tasks)
    mean_cost = total_cost / completed if completed > 0 else float("inf")
    cost_delta = total_cost - myopic_total_cost if completed > 0 else 0.0

    summary = {
        "sequence_path": str(args.sequence_path),
        "policy": "gnn_anticipatory",
        "gnn_model": str(args.gnn_model),
        "gamma": args.gamma,
        "attempted": attempted,
        "completed": completed,
        "auto_count": auto_count,
        "total_cost": float(total_cost),
        "mean_cost": float(mean_cost),
        "myopic_cost": float(myopic_total_cost),
        "cost_delta": float(cost_delta),
        "total_fd_calls": total_fd_calls,
        "total_gnn_calls": total_gnn_calls,
        "strategy_distribution": strategy_counts,
        "wall_seconds": wall_seconds,
    }

    result = {"summary": summary, "tasks": records}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN anticipatory planning on a task sequence."
    )
    parser.add_argument("--sequence-path", type=Path, required=True,
                        help="JSON task sequence")
    parser.add_argument("--gnn-model", type=Path, required=True,
                        help="Trained GNN checkpoint")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Optional JSON output file")
    parser.add_argument("--config-path", type=Path,
                        default=Path("configs/restaurant/toy_level_3.yaml"),
                        help="Env config")
    parser.add_argument("--domain-path", type=Path,
                        default=Path("pddl/toy_restaurant_domain.pddl"),
                        help="PDDL domain")
    parser.add_argument("--planner-path", type=Path,
                        default=Path("downward/fast-downward.py"),
                        help="FD binary")
    parser.add_argument("--search", type=str, default="astar(ff())",
                        help="FD --search string")
    parser.add_argument("--fd-timeout-s", type=float, default=20.0,
                        help="Per-FD-call timeout")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount (paper: 1.0 = no discount)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GNN hidden dim")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Truncate first N tasks")
    parser.add_argument("--max-augs", type=int, default=10,
                        help="Max augmented candidates per task")
    args = parser.parse_args()

    result = run_sequence(args)

    # Always print to stdout
    print(json.dumps(result["summary"], indent=2))

    # Optionally write to file
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()
