#!/usr/bin/env python3
"""Anticipatory Fast Downward oracle baseline for the toy restaurant env.

Like the myopic oracle, but instead of minimizing only the current task's
plan cost, it minimizes:

    plan_cost + gamma * V_AP(terminal_state)

where V_AP(s) = E_tau[cost(s, tau)] is the exact expected FD cost over all
possible future tasks from state s. Candidate plans include the myopic
optimal plan plus joint plans that achieve the current task goal AND a
future task type's goal simultaneously (preparation).

Usage:
    python scripts/restaurant/toy_anticipatory_oracle.py \
        --config-path configs/restaurant/toy_level_2_2.yaml \
        --domain-path pddl/toy_restaurant_domain.pddl \
        --planner-path downward/fast-downward.py \
        --num-tasks 200 --seed 0 --gamma 0.99
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_planner_action,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
    task_goal_clauses,
)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _state_signature(state: RestaurantPlannerState) -> Tuple:
    """Hashable cache key uniquely identifying a planner state."""
    parts: List[Any] = [state.agent_location, state.holding or "__none__"]
    for name in sorted(state.objects.keys()):
        obj = state.objects[name]
        parts.append((
            name,
            obj.location or "__held__",
            bool(obj.dirty),
            obj.filled_with or "__none__",
            obj.contained_in or "__none__",
        ))
    return tuple(parts)


def apply_plan_until_first_task_satisfied(
    state: RestaurantPlannerState,
    plan_actions: Sequence[Tuple[str, List[str]]],
    task: RestaurantTask,
    env: RestaurantSymbolicEnv,
) -> Tuple[RestaurantPlannerState, List[Tuple[str, List[str]]]]:
    """Apply plan actions one by one, stopping at the first action that
    satisfies the current task. Returns (terminal_state, executed_prefix).

    In the real env the task ends immediately upon satisfaction, so any
    post-completion actions in a joint plan would never execute. This
    function correctly truncates the plan at the task-completion boundary.
    """
    new_state = state.copy()
    executed: List[Tuple[str, List[str]]] = []
    for action in plan_actions:
        apply_planner_action(new_state, action)
        executed.append(action)
        if _task_is_auto_satisfied(new_state, task, env):
            return new_state, executed
    return new_state, executed


def _task_is_auto_satisfied(
    state: RestaurantPlannerState,
    task: RestaurantTask,
    env: RestaurantSymbolicEnv,
) -> bool:
    """Check if task is already satisfied in the given planner state."""
    if task.task_type == "serve_water":
        assert task.target_location is not None
        return any(
            o.location == task.target_location
            and o.kind in {"cup", "mug"}
            and o.filled_with == "water"
            for o in state.objects.values()
        )
    if task.task_type == "make_coffee":
        assert task.target_location is not None
        return any(
            o.location == task.target_location
            and o.kind in {"cup", "mug"}
            and o.filled_with == "coffee"
            for o in state.objects.values()
        )
    if task.task_type == "make_fruit_bowl":
        assert task.target_location is not None
        bowls = [
            n for n, o in state.objects.items()
            if o.kind == "bowl" and o.location == task.target_location
        ]
        if not bowls:
            return False
        return any(
            o.kind == "apple" and o.contained_in in bowls
            for o in state.objects.values()
        )
    if task.task_type == "clear_containers":
        assert task.target_location is not None
        return not any(
            o.location == task.target_location for o in state.objects.values()
        )
    if task.task_type == "wash_objects":
        assert task.target_kind is not None
        return any(
            o.kind == task.target_kind
            and not o.dirty
            and o.filled_with is None
            and o.location in env.wash_ready_locations
            and o.contained_in is None
            for o in state.objects.values()
        )
    if task.task_type == "pick_place":
        assert task.object_name is not None and task.target_location is not None
        obj = state.objects.get(task.object_name)
        return (
            obj is not None
            and obj.location == task.target_location
            and state.holding is None
        )
    raise ValueError(f"Unsupported task type: {task.task_type}")


# ---------------------------------------------------------------------------
# Task enumeration
# ---------------------------------------------------------------------------

def _enumerate_future_tasks(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
) -> List[Tuple[RestaurantTask, float]]:
    """Enumerate all possible future tasks with their probabilities."""
    tasks: List[Tuple[RestaurantTask, float]] = []
    dist = env.task_distribution
    total_weight = sum(dist.values())
    if total_weight <= 0:
        return tasks

    for ttype, weight in dist.items():
        p_type = weight / total_weight
        if p_type <= 0:
            continue

        if ttype in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
            loc_dist = env.service_location_distribution
            loc_total = sum(loc_dist.values())
            if loc_total <= 0:
                continue
            for loc, loc_w in loc_dist.items():
                p = p_type * (loc_w / loc_total)
                if p > 0:
                    tasks.append((RestaurantTask(task_type=ttype, target_location=loc), p))

        elif ttype == "pick_place":
            valid_objects = [
                name for name, obj in state.objects.items()
                if obj.kind not in {"water", "coffeegrinds"} and obj.contained_in is None
            ]
            locations = list(env.locations)
            if not valid_objects or not locations:
                continue
            p_each = p_type / len(valid_objects) / len(locations)
            for obj_name in valid_objects:
                for loc in locations:
                    tasks.append((RestaurantTask(
                        task_type=ttype, target_location=loc, object_name=obj_name,
                    ), p_each))

        elif ttype == "wash_objects":
            kind_dist = env.wash_kind_distribution
            kind_total = sum(kind_dist.values())
            if kind_total <= 0:
                continue
            for kind, kind_w in kind_dist.items():
                p = p_type * (kind_w / kind_total)
                if p > 0:
                    tasks.append((RestaurantTask(task_type=ttype, target_kind=kind), p))

    return tasks


def _future_candidate_tasks(
    env: RestaurantSymbolicEnv,
) -> List[RestaurantTask]:
    """One representative task per non-pick_place future task type.

    These are used to generate joint plans (current task + future goal).
    pick_place is skipped because its random target makes it a poor
    preparation target.
    """
    candidates: List[RestaurantTask] = []
    loc_dist = env.service_location_distribution
    loc_total = sum(loc_dist.values())
    for ttype in env.task_types:
        if ttype == "pick_place":
            continue
        if ttype in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
            best_loc = max(loc_dist, key=loc_dist.get) if loc_total > 0 else env.service_locations[0]
            candidates.append(RestaurantTask(task_type=ttype, target_location=best_loc))
        elif ttype == "wash_objects":
            kind_dist = env.wash_kind_distribution
            kind_total = sum(kind_dist.values())
            best_kind = max(kind_dist, key=kind_dist.get) if kind_total > 0 else env.object_kinds[0]
            candidates.append(RestaurantTask(task_type=ttype, target_kind=best_kind))
    return candidates


# ---------------------------------------------------------------------------
# V_AP computation
# ---------------------------------------------------------------------------

def _compute_v_ap(
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    future_tasks: Sequence[Tuple[RestaurantTask, float]],
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    timeout_s: float,
    cache: Dict[Tuple, float],
) -> float:
    """E_tau[cost(state, tau)] via FD, with caching by state signature."""
    key = _state_signature(state)
    cached = cache.get(key)
    if cached is not None:
        return cached

    total = 0.0
    for task, prob in future_tasks:
        if prob <= 0:
            continue
        if _task_is_auto_satisfied(state, task, env):
            cost = 0.0
        else:
            result = solve_restaurant_task_with_fd(
                env, state, task,
                planner_path=planner_path,
                domain_path=domain_path,
                alias=alias,
                timeout_s=timeout_s,
            )
            cost = result.plan_cost if result.success else 1e6
        total += prob * cost

    cache[key] = total
    return total


# ---------------------------------------------------------------------------
# Anticipatory planning
# ---------------------------------------------------------------------------

@dataclass
class AnticipatoryPlan:
    prefix_actions: List[Tuple[str, List[str]]]
    prefix_cost: float
    terminal_state: RestaurantPlannerState
    strategy: str
    v_ap: float
    full_plan_steps: int


def _solve_anticipatory_task(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    future_candidates: Sequence[RestaurantTask],
    future_tasks: Sequence[Tuple[RestaurantTask, float]],
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    timeout_s: float,
    gamma: float,
    v_ap_cache: Dict[Tuple, float],
) -> AnticipatoryPlan | None:
    """Generate candidate plans and pick the one minimizing
    prefix_cost + (gamma ** len(prefix)) * V_AP(terminal).

    Only the executed prefix (up to first task satisfaction) is scored
    and returned, matching env semantics where the task ends immediately
    upon completion.
    """
    best: AnticipatoryPlan | None = None
    best_score = float("inf")

    # Candidate 0: myopic optimal (no extra goals)
    myopic_result = solve_restaurant_task_with_fd(
        env, state, task,
        planner_path=planner_path,
        domain_path=domain_path,
        alias=alias,
        timeout_s=timeout_s,
    )
    if myopic_result.success:
        terminal, prefix = apply_plan_until_first_task_satisfied(
            state, myopic_result.plan_actions, task, env,
        )
        prefix_cost = planner_actions_paper2_cost(prefix, env)
        v_ap = _compute_v_ap(
            terminal, env, future_tasks,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
            cache=v_ap_cache,
        )
        score = prefix_cost + (gamma ** len(prefix)) * v_ap
        best = AnticipatoryPlan(
            prefix_actions=prefix,
            prefix_cost=float(prefix_cost),
            terminal_state=terminal,
            strategy="myopic",
            v_ap=float(v_ap),
            full_plan_steps=len(myopic_result.plan_actions),
        )
        best_score = score

    # Candidates 1..N: joint plans (current task + future task type goal)
    for fut in future_candidates:
        extra = task_goal_clauses(
            state, fut,
            service_locations=env.service_locations,
            wash_ready_locations=env.wash_ready_locations,
        )
        result = solve_restaurant_task_with_fd(
            env, state, task,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            extra_goal_clauses=extra,
            timeout_s=timeout_s,
        )
        if not result.success:
            continue
        terminal, prefix = apply_plan_until_first_task_satisfied(
            state, result.plan_actions, task, env,
        )
        prefix_cost = planner_actions_paper2_cost(prefix, env)
        v_ap = _compute_v_ap(
            terminal, env, future_tasks,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
            cache=v_ap_cache,
        )
        score = prefix_cost + (gamma ** len(prefix)) * v_ap
        if score < best_score:
            best = AnticipatoryPlan(
                prefix_actions=prefix,
                prefix_cost=float(prefix_cost),
                terminal_state=terminal,
                strategy=f"joint+{fut.task_type}",
                v_ap=float(v_ap),
                full_plan_steps=len(result.plan_actions),
            )
            best_score = score

    return best


# ---------------------------------------------------------------------------
# Main oracle loop
# ---------------------------------------------------------------------------

def run_anticipatory_oracle(
    *,
    config_path: Path,
    domain_path: Path,
    planner_path: Path,
    num_tasks: int,
    tasks_per_reset: int,
    alias: str,
    timeout_s: float,
    seed: int,
    gamma: float,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed)
    obs, info = env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    future_candidates = _future_candidate_tasks(env)
    v_ap_cache: Dict[Tuple, float] = {}

    records: List[Dict[str, Any]] = []
    failures = 0
    total_wall = 0.0
    tasks_since_reset = 0
    episode_index = 0
    strategy_counts: Dict[str, int] = {}

    for t_idx in range(num_tasks):
        task = env.task
        task_type = str(task.task_type)
        auto_satisfied = bool(env._pending_auto_success)
        plan = None

        if auto_satisfied:
            result_success = True
            prefix_actions: List = []
            prefix_cost = 0.0
            planner_solve_time_s = 0.0
            elapsed = 0.0
            error = None
            strategy = "auto"
            v_ap_val = 0.0
            full_plan_steps = 0
        else:
            t0 = time.perf_counter()
            plan = _solve_anticipatory_task(
                env, state, task,
                future_candidates=future_candidates,
                future_tasks=_enumerate_future_tasks(env, state),
                planner_path=planner_path,
                domain_path=domain_path,
                alias=alias,
                timeout_s=timeout_s,
                gamma=gamma,
                v_ap_cache=v_ap_cache,
            )
            elapsed = float(time.perf_counter() - t0)
            total_wall += elapsed
            planner_solve_time_s = elapsed
            if plan is not None:
                result_success = True
                prefix_actions = plan.prefix_actions
                prefix_cost = plan.prefix_cost
                strategy = plan.strategy
                v_ap_val = plan.v_ap
                full_plan_steps = plan.full_plan_steps
                error = None
            else:
                result_success = False
                prefix_actions = []
                prefix_cost = float("inf")
                strategy = "failed"
                v_ap_val = 0.0
                full_plan_steps = 0
                error = "no feasible plan"

        steps = int(len(prefix_actions))
        paper2_cost = float(prefix_cost) if result_success else float("inf")
        uniform_cost = float(steps * 25)

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if result_success:
            if plan is not None:
                state = plan.terminal_state
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
            "strategy": strategy,
            "v_ap": float(v_ap_val),
            "full_plan_steps": int(full_plan_steps),
            "error": error,
        })

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
        "gamma": float(gamma),
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
        "strategy_counts": strategy_counts,
        "v_ap_cache_size": len(v_ap_cache),
    }
    return {"stats": stats, "tasks": records}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anticipatory FD oracle baseline for toy restaurant."
    )
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_restaurant.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--num-tasks", type=int, default=40)
    parser.add_argument("--tasks-per-reset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_anticipatory_oracle/results.json"))
    args = parser.parse_args()

    result = run_anticipatory_oracle(
        config_path=args.config_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        num_tasks=args.num_tasks,
        tasks_per_reset=args.tasks_per_reset,
        alias=args.alias,
        timeout_s=args.timeout_s,
        seed=args.seed,
        gamma=args.gamma,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
