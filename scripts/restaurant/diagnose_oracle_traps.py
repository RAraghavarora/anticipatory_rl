#!/usr/bin/env python3
"""Diagnose whether myopic oracle plans leave future-cost traps.

This is intentionally a diagnostic script, not a benchmark.  It logs enough
state to answer: after a locally optimal task plan, are there plausible next
tasks that become expensive because of the resulting world state?
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_plan,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.envs.restaurant.task_sampling import sample_task


def _state_to_env(env: RestaurantSymbolicEnv, state: RestaurantPlannerState) -> None:
    env.state.agent_location = state.agent_location
    env.state.holding = state.holding
    env.state.bread_spread = state.bread_spread
    for name, obj in state.objects.items():
        env.state.objects[name].location = obj.location
        env.state.objects[name].dirty = obj.dirty
        env.state.objects[name].filled_with = obj.filled_with
        env.state.objects[name].contained_in = obj.contained_in


def _state_summary(state: RestaurantPlannerState) -> Dict[str, Any]:
    return {
        "agent_location": state.agent_location,
        "holding": state.holding,
        "bread_spread": state.bread_spread,
        "objects": {
            name: {
                "kind": obj.kind,
                "location": obj.location,
                "dirty": bool(obj.dirty),
                "filled_with": obj.filled_with,
                "contained_in": obj.contained_in,
            }
            for name, obj in sorted(state.objects.items())
        },
    }


def _task_summary(task: RestaurantTask) -> Dict[str, Any]:
    return {
        "task_type": task.task_type,
        "target_location": task.target_location,
        "target_kind": task.target_kind,
        "object_name": task.object_name,
    }


def _suspicious_leftovers(env: RestaurantSymbolicEnv, state: RestaurantPlannerState) -> List[Dict[str, Any]]:
    """Heuristic flags for locally cheap states that may be future-costly."""
    flags: List[Dict[str, Any]] = []
    service_locations = set(env.service_locations)
    wash_ready = set(env.wash_ready_locations)
    dirty_drop = set(env.dirty_drop_locations)
    prep_locations = {env.countertop_location, env.station_coffee, env.station_water, env.station_fruit}

    for name, obj in sorted(state.objects.items()):
        if obj.kind in {"cup", "mug", "bowl", "knife"} and obj.dirty:
            flags.append(
                {
                    "kind": "dirty_reusable_object",
                    "object": name,
                    "object_kind": obj.kind,
                    "location": obj.location,
                    "reason": "Dirty reusable object may force wash/fetch before future service or prep tasks.",
                }
            )
        if obj.kind in {"cup", "mug", "bowl"} and obj.location in service_locations:
            flags.append(
                {
                    "kind": "service_object_left_at_service_location",
                    "object": name,
                    "object_kind": obj.kind,
                    "location": obj.location,
                    "reason": "Can help repeated service tasks but hurts future clear_containers at the service location.",
                }
            )
        if obj.kind in {"cup", "mug", "bowl", "knife", "apple"} and obj.location in dirty_drop:
            flags.append(
                {
                    "kind": "object_in_dirty_drop_location",
                    "object": name,
                    "object_kind": obj.kind,
                    "location": obj.location,
                    "reason": "Object in sink/bus_tub may be cheap to dump but costly to retrieve for prep tasks.",
                }
            )
        if obj.kind in {"apple", "bowl", "knife"} and obj.location is not None and obj.location not in prep_locations:
            flags.append(
                {
                    "kind": "prep_object_away_from_prep",
                    "object": name,
                    "object_kind": obj.kind,
                    "location": obj.location,
                    "reason": "Future make_fruit_bowl may require fetching this object back to the counter.",
                }
            )
        if obj.kind in {"cup", "mug"} and obj.filled_with is not None and obj.location not in service_locations:
            flags.append(
                {
                    "kind": "filled_cup_away_from_service",
                    "object": name,
                    "object_kind": obj.kind,
                    "location": obj.location,
                    "filled_with": obj.filled_with,
                    "reason": "Future service may need this cup moved or another clean empty cup found.",
                }
            )
    return flags


def _solve_task_cost(
    *,
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    timeout_s: float,
) -> Dict[str, Any]:
    _state_to_env(env, state)
    env.set_task(
        task.task_type,
        target_location=task.target_location,
        target_kind=task.target_kind,
        object_name=task.object_name,
        task_source="diagnostic",
    )
    auto_satisfied = bool(env._pending_auto_success)
    if auto_satisfied:
        return {
            "task": _task_summary(task),
            "success": True,
            "auto_satisfied": True,
            "steps": 0,
            "paper2_cost": 0.0,
            "plan_actions": [],
            "solve_time_s": 0.0,
            "error": None,
        }

    result = solve_restaurant_task_with_fd(
        env,
        state,
        task,
        planner_path=planner_path,
        domain_path=domain_path,
        alias=alias,
        timeout_s=timeout_s,
    )
    return {
        "task": _task_summary(task),
        "success": bool(result.success),
        "auto_satisfied": False,
        "steps": int(len(result.plan_actions)),
        "paper2_cost": float(planner_actions_paper2_cost(result.plan_actions, env) if result.success else float("inf")),
        "plan_actions": result.plan_actions,
        "solve_time_s": float(result.solve_time_s),
        "error": result.error,
    }


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_probe_results(probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    non_auto = [p for p in probes if not p["auto_satisfied"]]
    finite_costs = [float(p["paper2_cost"]) for p in probes if p["success"] and np.isfinite(float(p["paper2_cost"]))]
    worst = sorted(probes, key=lambda p: float(p["paper2_cost"]) if p["success"] else float("inf"), reverse=True)[:5]
    return {
        "n": int(len(probes)),
        "auto_rate": _mean([1.0 if p["auto_satisfied"] else 0.0 for p in probes]),
        "avg_steps": _mean([float(p["steps"]) for p in probes]),
        "avg_paper2_cost": _mean(finite_costs),
        "non_auto_avg_steps": _mean([float(p["steps"]) for p in non_auto]),
        "non_auto_avg_paper2_cost": _mean(
            [float(p["paper2_cost"]) for p in non_auto if p["success"] and np.isfinite(float(p["paper2_cost"]))]
        ),
        "worst_probes": worst,
    }


def _sample_probe_tasks(
    *,
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    n_probe_tasks: int,
    probe_seed: int,
) -> List[RestaurantTask]:
    py_random_state = random.getstate()
    try:
        random.seed(probe_seed)
        probe_env = RestaurantSymbolicEnv(config_path=env.config_path, rng_seed=probe_seed)
        probe_env.reset(seed=probe_seed)
        _state_to_env(probe_env, state)
        return [copy.deepcopy(sample_task(probe_env)) for _ in range(n_probe_tasks)]
    finally:
        random.setstate(py_random_state)


def _paired_future_probe(
    *,
    env: RestaurantSymbolicEnv,
    pre_state: RestaurantPlannerState,
    post_state: RestaurantPlannerState,
    n_probe_tasks: int,
    probe_seed: int,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    timeout_s: float,
    log_probes: bool,
) -> Dict[str, Any]:
    """Probe the same sampled future tasks from pre- and post-action states."""
    tasks = _sample_probe_tasks(env=env, state=post_state, n_probe_tasks=n_probe_tasks, probe_seed=probe_seed)
    pre_env = RestaurantSymbolicEnv(config_path=env.config_path, rng_seed=probe_seed)
    post_env = RestaurantSymbolicEnv(config_path=env.config_path, rng_seed=probe_seed)
    pre_env.reset(seed=probe_seed)
    post_env.reset(seed=probe_seed)

    paired: List[Dict[str, Any]] = []
    pre_results: List[Dict[str, Any]] = []
    post_results: List[Dict[str, Any]] = []
    for task in tasks:
        pre = _solve_task_cost(
            env=pre_env,
            state=pre_state,
            task=task,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
        )
        post = _solve_task_cost(
            env=post_env,
            state=post_state,
            task=task,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
        )
        pre_results.append(pre)
        post_results.append(post)

        pre_cost = float(pre["paper2_cost"])
        post_cost = float(post["paper2_cost"])
        finite_delta = np.isfinite(pre_cost) and np.isfinite(post_cost)
        paired.append(
            {
                "task": _task_summary(task),
                "pre": pre,
                "post": post,
                "delta_steps": int(post["steps"]) - int(pre["steps"]),
                "delta_paper2_cost": float(post_cost - pre_cost) if finite_delta else None,
                "auto_lost": bool(pre["auto_satisfied"] and not post["auto_satisfied"]),
                "auto_gained": bool(post["auto_satisfied"] and not pre["auto_satisfied"]),
            }
        )

    finite_delta_entries = [p for p in paired if p["delta_paper2_cost"] is not None]
    deltas = [float(p["delta_paper2_cost"]) for p in finite_delta_entries]
    step_deltas = [float(p["delta_steps"]) for p in paired]
    worst_regressions = sorted(finite_delta_entries, key=lambda p: float(p["delta_paper2_cost"]), reverse=True)[:5]
    largest_improvements = sorted(finite_delta_entries, key=lambda p: float(p["delta_paper2_cost"]))[:5]
    result = {
        "pre": _summarize_probe_results(pre_results),
        "post": _summarize_probe_results(post_results),
        "delta": {
            "n": int(len(paired)),
            "finite_delta_n": int(len(finite_delta_entries)),
            "avg_delta_paper2_cost": _mean(deltas),
            "avg_delta_steps": _mean(step_deltas),
            "worsened_rate": _mean([1.0 if float(p["delta_paper2_cost"]) > 0.0 else 0.0 for p in finite_delta_entries]),
            "improved_rate": _mean([1.0 if float(p["delta_paper2_cost"]) < 0.0 else 0.0 for p in finite_delta_entries]),
            "auto_lost_rate": _mean([1.0 if p["auto_lost"] else 0.0 for p in paired]),
            "auto_gained_rate": _mean([1.0 if p["auto_gained"] else 0.0 for p in paired]),
            "max_delta_paper2_cost": float(max(deltas)) if deltas else 0.0,
            "min_delta_paper2_cost": float(min(deltas)) if deltas else 0.0,
            "max_delta_steps": int(max(step_deltas)) if step_deltas else 0,
            "min_delta_steps": int(min(step_deltas)) if step_deltas else 0,
            "worst_regressions": worst_regressions,
            "largest_improvements": largest_improvements,
        },
        "highest_step_post_probes": sorted(post_results, key=lambda p: int(p["steps"]), reverse=True)[:5],
        "total_solve_time_s": float(
            sum(float(p["solve_time_s"]) for p in pre_results) + sum(float(p["solve_time_s"]) for p in post_results)
        ),
    }
    if log_probes:
        result["paired_probes"] = paired
    return result


def _plan_record(task_idx: int, task: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "task_idx": int(task_idx),
        "task": dict(task),
        "steps": int(result["steps"]),
        "paper2_cost": float(result["paper2_cost"]),
        "auto_satisfied": bool(result["auto_satisfied"]),
        "plan_actions": result["plan_actions"],
    }


def _update_highest_plan(
    best: Dict[str, Dict[str, Any]],
    *,
    task_idx: int,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    task_type = str(task["task_type"])
    current = best.get(task_type)
    if current is None or int(result["steps"]) > int(current["steps"]):
        best[task_type] = _plan_record(task_idx, task, result)


def _summarize_records(
    records: List[Dict[str, Any]],
    highest_current_plan_by_task: Dict[str, Dict[str, Any]],
    highest_probe_plan_by_task: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    task_types = sorted({str(r["task"]["task_type"]) for r in records})
    by_task: Dict[str, Any] = {}
    for task_type in task_types:
        task_records = [r for r in records if r["task"]["task_type"] == task_type]
        by_task[task_type] = {
            "n": int(len(task_records)),
            "auto_rate": _mean([1.0 if r["current"]["auto_satisfied"] else 0.0 for r in task_records]),
            "avg_steps": _mean([float(r["current"]["steps"]) for r in task_records]),
            "avg_pre_future_cost": _mean([float(r["paired_future_probe"]["pre"]["avg_paper2_cost"]) for r in task_records]),
            "avg_post_future_cost": _mean([float(r["paired_future_probe"]["post"]["avg_paper2_cost"]) for r in task_records]),
            "avg_future_delta_cost": _mean(
                [float(r["paired_future_probe"]["delta"]["avg_delta_paper2_cost"]) for r in task_records]
            ),
            "avg_future_delta_steps": _mean(
                [float(r["paired_future_probe"]["delta"]["avg_delta_steps"]) for r in task_records]
            ),
            "future_worsened_rate": _mean(
                [float(r["paired_future_probe"]["delta"]["worsened_rate"]) for r in task_records]
            ),
            "future_auto_lost_rate": _mean(
                [float(r["paired_future_probe"]["delta"]["auto_lost_rate"]) for r in task_records]
            ),
            "future_auto_gained_rate": _mean(
                [float(r["paired_future_probe"]["delta"]["auto_gained_rate"]) for r in task_records]
            ),
        }

    top_future_regression_states = sorted(
        records,
        key=lambda r: float(r["paired_future_probe"]["delta"]["max_delta_paper2_cost"]),
        reverse=True,
    )[:10]
    return {
        "by_current_task_type": by_task,
        "highest_step_current_plan_by_task": highest_current_plan_by_task,
        "highest_step_future_probe_plan_by_task": highest_probe_plan_by_task,
        "top_future_regression_states": [
            {
                "task_idx": int(r["task_idx"]),
                "task": r["task"],
                "current_steps": int(r["current"]["steps"]),
                "current_auto_satisfied": bool(r["current"]["auto_satisfied"]),
                "avg_delta_paper2_cost": float(r["paired_future_probe"]["delta"]["avg_delta_paper2_cost"]),
                "max_delta_paper2_cost": float(r["paired_future_probe"]["delta"]["max_delta_paper2_cost"]),
                "avg_delta_steps": float(r["paired_future_probe"]["delta"]["avg_delta_steps"]),
                "max_delta_steps": int(r["paired_future_probe"]["delta"]["max_delta_steps"]),
                "pre_future_auto_rate": float(r["paired_future_probe"]["pre"]["auto_rate"]),
                "post_future_auto_rate": float(r["paired_future_probe"]["post"]["auto_rate"]),
                "worst_regressions": r["paired_future_probe"]["delta"]["worst_regressions"],
            }
            for r in top_future_regression_states
        ],
    }


def run_diagnostic(
    *,
    config_path: Path,
    domain_path: Path,
    planner_path: Path,
    num_tasks: int,
    tasks_per_reset: int,
    probe_tasks: int,
    alias: str,
    timeout_s: float,
    seed: int,
    log_all: bool,
    log_probes: bool,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed)
    env.config_path = config_path
    env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    records: List[Dict[str, Any]] = []
    tasks_since_reset = 0
    episode_index = 0
    total_current_solve_time = 0.0
    total_probe_solve_time = 0.0
    highest_current_plan_by_task: Dict[str, Dict[str, Any]] = {}
    highest_probe_plan_by_task: Dict[str, Dict[str, Any]] = {}

    for task_idx in range(num_tasks):
        task = copy.deepcopy(env.task)
        task_info = _task_summary(task)
        pre_state = state.copy()
        current = _solve_task_cost(
            env=env,
            state=pre_state,
            task=task,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
        )
        total_current_solve_time += float(current["solve_time_s"])
        _update_highest_plan(highest_current_plan_by_task, task_idx=task_idx, task=task_info, result=current)
        post_state = apply_plan(pre_state, current["plan_actions"]) if current["success"] else pre_state.copy()
        _state_to_env(env, post_state)

        suspicious = _suspicious_leftovers(env, post_state)
        paired_future_probe = _paired_future_probe(
            env=env,
            pre_state=pre_state,
            post_state=post_state,
            n_probe_tasks=probe_tasks,
            probe_seed=seed + 1_000_003 * task_idx,
            planner_path=planner_path,
            domain_path=domain_path,
            alias=alias,
            timeout_s=timeout_s,
            log_probes=log_probes,
        )
        total_probe_solve_time += float(paired_future_probe["total_solve_time_s"])
        for probe in paired_future_probe["highest_step_post_probes"]:
            _update_highest_plan(highest_probe_plan_by_task, task_idx=task_idx, task=probe["task"], result=probe)

        is_interesting = (
            bool(suspicious)
            or paired_future_probe["post"]["non_auto_avg_steps"] >= 5.0
            or paired_future_probe["delta"]["max_delta_paper2_cost"] > 0.0
        )

        record = {
            "task_idx": int(task_idx),
            "task": task_info,
            "current": current,
            "suspicious_leftovers": suspicious,
            "future_probe": paired_future_probe["post"],
            "paired_future_probe": paired_future_probe,
        }
        if log_all or is_interesting:
            record["pre_state"] = _state_summary(pre_state)
            record["post_state"] = _state_summary(post_state)
        records.append(record)

        state = post_state
        if current["success"]:
            tasks_since_reset += 1
        if tasks_per_reset > 0 and tasks_since_reset >= tasks_per_reset:
            episode_index += 1
            env.reset(seed=seed + 100_003 * episode_index)
            state = RestaurantPlannerState.from_env(env)
            tasks_since_reset = 0
        else:
            env._resample_task()

    suspicious_records = [r for r in records if r["suspicious_leftovers"]]
    stats = {
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "tasks_per_reset": int(tasks_per_reset),
        "probe_tasks_per_state": int(probe_tasks),
        "success_rate": _mean([1.0 if r["current"]["success"] else 0.0 for r in records]),
        "auto_rate": _mean([1.0 if r["current"]["auto_satisfied"] else 0.0 for r in records]),
        "avg_steps": _mean([float(r["current"]["steps"]) for r in records]),
        "avg_pre_future_probe_cost": _mean(
            [float(r["paired_future_probe"]["pre"]["avg_paper2_cost"]) for r in records]
        ),
        "avg_post_future_probe_cost": _mean(
            [float(r["paired_future_probe"]["post"]["avg_paper2_cost"]) for r in records]
        ),
        "avg_future_probe_cost": _mean(
            [float(r["paired_future_probe"]["post"]["avg_paper2_cost"]) for r in records]
        ),
        "avg_future_delta_cost": _mean(
            [float(r["paired_future_probe"]["delta"]["avg_delta_paper2_cost"]) for r in records]
        ),
        "avg_future_delta_steps": _mean(
            [float(r["paired_future_probe"]["delta"]["avg_delta_steps"]) for r in records]
        ),
        "avg_future_probe_auto_rate": _mean([float(r["paired_future_probe"]["post"]["auto_rate"]) for r in records]),
        "avg_future_auto_lost_rate": _mean(
            [float(r["paired_future_probe"]["delta"]["auto_lost_rate"]) for r in records]
        ),
        "avg_future_auto_gained_rate": _mean(
            [float(r["paired_future_probe"]["delta"]["auto_gained_rate"]) for r in records]
        ),
        "future_worsened_state_rate": _mean(
            [1.0 if float(r["paired_future_probe"]["delta"]["avg_delta_paper2_cost"]) > 0.0 else 0.0 for r in records]
        ),
        "suspicious_state_rate": float(len(suspicious_records) / max(1, len(records))),
        "current_task_planner_solve_time_s": float(total_current_solve_time),
        "future_probe_planner_solve_time_s": float(total_probe_solve_time),
        "total_planner_solve_time_s": float(total_current_solve_time + total_probe_solve_time),
    }
    summary = _summarize_records(records, highest_current_plan_by_task, highest_probe_plan_by_task)
    return {"stats": stats, "summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose myopic oracle trap states with future-task probes.")
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_restaurant.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--num-tasks", type=int, default=25)
    parser.add_argument("--tasks-per-reset", type=int, default=200)
    parser.add_argument("--probe-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-all", action="store_true")
    parser.add_argument("--log-probes", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_oracle_trap_diagnostic/results.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_diagnostic(
        config_path=args.config_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        num_tasks=args.num_tasks,
        tasks_per_reset=args.tasks_per_reset,
        probe_tasks=args.probe_tasks,
        alias=args.alias,
        timeout_s=args.timeout_s,
        seed=args.seed,
        log_all=args.log_all,
        log_probes=args.log_probes,
    )
    result["stats"]["wall_time_s"] = float(time.perf_counter() - t0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
