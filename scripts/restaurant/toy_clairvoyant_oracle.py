#!/usr/bin/env python3
"""Rolling finite-K clairvoyant satisficing planner for a fixed task sequence.

This finite-K clairvoyant benchmark knows the next K concrete tasks and uses a
satisficing planner for their undiscounted PDDL physical cost. Window costs are
plan costs, not lower bounds or optima. It is not V_AP, non-clairvoyant
anticipation, or discounted RL return.

Sequence JSON schema:
    {
      "sequence_id": "example-v1",
      "tasks": [
        {"task_type": "make_coffee", "target_location": "servingtable"},
        {"task_type": "wash_objects", "target_kind": "cup"},
        {"task_type": "pick_place", "target_location": "shelf",
         "object_name": "plate_0"}
      ]
    }

Every pick_place task must name its concrete object in this fixed input.
At each index the oracle solves tasks[i:i+K], replays only the first parsed
segment, applies its completion consumption once, then shifts and replans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_planner_action,
    consume_delivery_from_state,
    solve_restaurant_sequence_with_fd,
)


def _load_sequence(path: Path) -> tuple[str, str, list[RestaurantTask]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("Sequence JSON must be an object containing a 'tasks' list.")
    tasks = [RestaurantTask(**item) for item in payload["tasks"]]
    if not tasks:
        raise ValueError("Sequence must contain at least one task.")
    sequence_id = str(payload.get("sequence_id") or path.stem)
    return sequence_id, hashlib.sha256(raw).hexdigest(), tasks


def run_oracle(
    *,
    config_path: Path,
    sequence_path: Path,
    domain_path: Path,
    planner_path: Path,
    output_path: Path,
    k: int,
    timeout_s: float,
    seed: int,
    alias: str = "seq-sat-lama-2011",
    search: str | None = None,
) -> dict[str, Any]:
    if k < 1:
        raise ValueError("K must be at least 1.")

    sequence_id, sequence_sha256, tasks = _load_sequence(sequence_path)
    env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed)
    env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    windows: list[dict[str, Any]] = []
    total_cost = 0.0
    total_actions = 0
    completions = 0
    failure_index: int | None = None
    failure_error: str | None = None
    for index in range(len(tasks)):
        window = tasks[index:index + k]
        try:
            result = solve_restaurant_sequence_with_fd(
                env,
                state,
                window,
                planner_path=planner_path,
                domain_path=domain_path,
                alias=alias,
                search=search,
                timeout_s=timeout_s,
            )
            success = result.success
            error = result.error
        except Exception as exc:
            result = None
            success = False
            error = str(exc)

        record: dict[str, Any] = {
            "index": index,
            "window_size": len(window),
            "selected_search": result.selected_search if result else None,
            "solve_time_s": result.solve_time_s if result else 0.0,
            "success": success,
            "error": error,
            "window_plan_cost": result.physical_cost if success and result else None,
            "window_completion_count": result.completion_count if result else 0,
        }

        if not success or result is None:
            failure_index = index
            failure_error = error or "solver returned an unsuccessful result"
            record["error"] = failure_error
            windows.append(record)
            break

        next_state = state.copy()
        try:
            segment = result.task_segments[0]
            for action in segment.physical_actions:
                apply_planner_action(next_state, action)
            consume_delivery_from_state(
                next_state, segment.task.task_type, segment.task.target_location,
            )
        except Exception as exc:
            failure_index = index
            failure_error = f"first-segment replay failed: {exc}"
            record.update(success=False, error=failure_error)
            windows.append(record)
            break

        state = next_state
        action_count = len(segment.physical_actions)
        total_cost += segment.paper2_cost
        total_actions += action_count
        completions += 1
        record.update(
            first_segment_cost=segment.paper2_cost,
            first_segment_actions=action_count,
            first_segment_plan=segment.physical_actions,
            auto_success=segment.auto_success,
        )
        windows.append(record)

    requested_task_count = len(tasks)
    sequence_complete = completions == requested_task_count
    prefix_mean = total_cost / completions if completions else None
    output = {
        # An optimal --search (e.g. astar(hmax())) makes each window cost a true optimum,
        # so the "not optima" caveat must not be recorded for those runs.
        "research_object": (
            "finite-K clairvoyant OPTIMAL planner benchmark over undiscounted PDDL "
            "physical cost; each window cost is that window's exact optimum, so a "
            "refusal to act is a proof rather than a search failure; still not V_AP, "
            "non-clairvoyant anticipation, or RL return"
            if search else
            "finite-K clairvoyant satisficing planner benchmark over undiscounted "
            "PDDL physical cost; window costs are plan costs, not lower bounds "
            "or optima; not V_AP, non-clairvoyant anticipation, or RL return"
        ),
        "planner": f"search:{search}" if search else f"alias:{alias}",
        "config": str(config_path),
        "seed": seed,
        "sequence_id": sequence_id,
        "sequence_path": str(sequence_path),
        "sequence_sha256": sequence_sha256,
        "K": k,
        "requested_task_count": requested_task_count,
        "attempted_window_count": len(windows),
        "valid_prefix_completion_count": completions,
        "valid_prefix_completion_rate": completions / requested_task_count,
        "sequence_complete": sequence_complete,
        "failure_index": failure_index,
        "failure_error": failure_error,
        "valid_prefix_physical_cost": total_cost,
        "valid_prefix_mean_physical_cost_per_completion": prefix_mean,
        "whole_sequence_physical_cost": total_cost if sequence_complete else None,
        "whole_sequence_mean_physical_cost_per_completion": prefix_mean if sequence_complete else None,
        "valid_prefix_physical_action_count": total_actions,
        "windows": windows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finite-K clairvoyant satisficing planner benchmark over known-sequence "
            "PDDL physical cost; window costs are plan costs, not lower bounds or "
            "optima (and not V_AP or RL return)."
        )
    )
    parser.add_argument("--sequence-path", type=Path, required=True)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_sequence_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011",
                        help="FD alias. Satisficing (seq-sat-*) plan cost depends on the "
                             "search budget, which confounds comparisons across K; an "
                             "optimal alias (seq-opt-*) removes that dependence.")
    parser.add_argument("--search", type=str, default=None,
                        help="FD --search string; bypasses --alias. The seq-opt-* aliases all "
                             "reject this domain's axioms, but astar(hmax()) is optimal, "
                             "axiom-safe, and far faster than satisficing lama here.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_clairvoyant_oracle/results.json"))
    args = parser.parse_args()

    output = run_oracle(
        config_path=args.config_path,
        sequence_path=args.sequence_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        output_path=args.output_path,
        k=args.K,
        timeout_s=args.timeout_s,
        seed=args.seed,
        alias=args.alias,
        search=args.search,
    )
    print(json.dumps({key: value for key, value in output.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
