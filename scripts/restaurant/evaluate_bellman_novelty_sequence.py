#!/usr/bin/env python3
"""Thin paired persistent-sequence evaluation harness for myopic vs cost_bounded.

Reuses `search_task` from `toy_bellman_novelty_planner.py`; loads an explicit
fixed task sequence JSON and evaluates one or both policies on the same
deterministic world, reporting physical PDDL cost (not RL cost).

Usage:
    python scripts/restaurant/evaluate_bellman_novelty_sequence.py \
        --policy both --sequence-path experiments/sequences/iid-eval-seq-00-sub10.json \
        --q-weights runs/<checkpoint>/restaurant_dqn.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.utils import select_device
from toy_anticipatory_oracle import (  # noqa: E402
    _task_is_auto_satisfied,
    apply_plan_until_first_task_satisfied,
)
import toy_bellman_novelty_planner as bnp  # noqa: E402


# ---------------------------------------------------------------------------
# Task sequence
# ---------------------------------------------------------------------------

def _load_sequence(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"Expected JSON object with 'tasks' key, got: {type(data)}")
    return list(data["tasks"])


def _task_from_dict(d: Dict[str, Any]) -> RestaurantTask:
    return RestaurantTask(
        task_type=str(d["task_type"]),
        target_location=d.get("target_location"),
        target_kind=d.get("target_kind"),
        object_name=d.get("object_name"),
    )


# ---------------------------------------------------------------------------
# Per-policy result dataclass
# ---------------------------------------------------------------------------

@dataclass
class _MyopicResult:
    success: bool
    cost: float
    num_actions: int
    trace: List[str]
    error: Optional[str]
    next_state: RestaurantPlannerState


@dataclass
class _CostBoundedResult:
    success: bool
    cost: float
    num_actions: int
    trace: List[str]
    error: Optional[str]
    next_state: RestaurantPlannerState
    v_ap: float
    reference_cost: Optional[float]
    cost_budget: Optional[float]
    eligible_count: int
    expansions: int
    source: str = "none"


# ---------------------------------------------------------------------------
# Per-policy runners
# ---------------------------------------------------------------------------

def _run_myopic(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
) -> _MyopicResult:
    result = solve_restaurant_task_with_fd(
        env, init_state.copy(), task,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, timeout_s=fd_timeout_s,
    )
    if not result.success:
        return _MyopicResult(
            success=False, cost=0.0, num_actions=0, trace=[],
            error=f"FD plan failure: {result.error}", next_state=init_state,
        )

    terminal, prefix = apply_plan_until_first_task_satisfied(
        init_state.copy(), result.plan_actions, task, env,
    )
    consume_delivery_from_state(terminal, task.task_type, task.target_location)
    cost = planner_actions_paper2_cost(prefix, env)
    trace = [f"{a}({', '.join(args)})" for a, args in prefix]
    return _MyopicResult(
        success=True, cost=cost, num_actions=len(prefix), trace=trace,
        error=None, next_state=terminal,
    )


def _run_cost_bounded(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    model: RestaurantQNetwork,
    device: torch.device,
    gamma: float,
    success_reward: float,
    max_depth: int,
    max_expansions: int,
    cost_ratio: float,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
) -> _CostBoundedResult:
    result = bnp.search_task(
        env=env, init_state=init_state.copy(), task=task,
        model=model, device=device,
        gamma=gamma, success_reward=success_reward,
        max_depth=max_depth, max_expansions=max_expansions,
        scoring_mode="cost_bounded", cost_ratio=cost_ratio,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, fd_timeout_s=fd_timeout_s,
    )
    if result.selected is None:
        return _CostBoundedResult(
            success=False, cost=0.0, num_actions=0, trace=[],
            error="no feasible candidate", next_state=init_state,
            v_ap=0.0, reference_cost=result.reference_cost,
            cost_budget=result.cost_budget,
            eligible_count=getattr(result, "eligible_terminal_count", 0) or 0,
            expansions=result.expansions, source="none",
        )

    sel = result.selected
    cost = planner_actions_paper2_cost(sel.prefix, env)
    trace = [f"{a}({', '.join(args)})" for a, args in sel.prefix]
    return _CostBoundedResult(
        success=True, cost=cost, num_actions=sel.depth, trace=trace,
        error=None, next_state=sel.state,
        v_ap=sel.v_ap, reference_cost=result.reference_cost,
        cost_budget=result.cost_budget,
        eligible_count=getattr(result, "eligible_terminal_count", 0) or 0,
        expansions=result.expansions, source=result.selected_source,
    )


# ---------------------------------------------------------------------------
# Main sequence runner
# ---------------------------------------------------------------------------

def run_sequence(
    policy: str,
    *,
    sequence_path: Path,
    config_path: Path,
    domain_path: Path,
    planner_path: Path,
    alias: str,
    fd_timeout_s: float,
    seed: int,
    gamma: float,
    success_reward: float,
    hidden_dim: int,
    max_depth: int,
    max_expansions: int,
    cost_ratio: float,
    q_weights: Optional[Path] = None,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    assert policy in ("myopic", "cost_bounded")

    np.random.seed(seed)

    tasks = _load_sequence(sequence_path)
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    env = RestaurantSymbolicEnv(
        config_path=config_path, rng_seed=seed,
        success_reward=success_reward,
    )
    obs, _info = env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    device: Optional[torch.device] = None
    model: Optional[RestaurantQNetwork] = None
    if policy == "cost_bounded":
        if q_weights is None:
            raise ValueError("q_weights is required for cost_bounded policy")
        device = select_device()
        model = RestaurantQNetwork(
            input_dim=int(np.asarray(obs).shape[0]),
            action_type_dim=int(env.action_space["action_type"].n),
            object_dim=int(env.action_space["object1"].n),
            location_dim=int(env.action_space["location"].n),
            hidden_dim=hidden_dim,
        ).to(device)
        model.load_state_dict(torch.load(
            q_weights.expanduser().resolve(), map_location=device, weights_only=True,
        ))
        model.eval()

    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    auto_count = 0
    jar_ready_count = 0
    total_cost = 0.0
    total_actions = 0
    completed = 0

    for idx, task_dict in enumerate(tasks):
        task = _task_from_dict(task_dict)
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )

        auto = _task_is_auto_satisfied(state, task, env)
        rec: Dict[str, Any] = {
            "index": idx,
            "task_type": task.task_type,
            "target_location": task.target_location,
            "target_kind": task.target_kind,
            "object_name": task.object_name,
            "auto": bool(auto),
        }

        if auto:
            consume_delivery_from_state(state, task.task_type, task.target_location)
            rec.update(
                success=True, cost=0.0, actions=0, trace=[], error=None,
                v_ap=None, source=None, reference_cost=None,
                budget=None, eligible=None, expansions=None,
            )
            auto_count += 1
            completed += 1
        else:
            if policy == "myopic":
                r = _run_myopic(
                    env, state, task,
                    planner_path=planner_path, domain_path=domain_path,
                    alias=alias, fd_timeout_s=fd_timeout_s,
                )
                rec.update(
                    success=r.success, cost=r.cost, actions=r.num_actions,
                    trace=r.trace, error=r.error,
                    v_ap=None, source=None, reference_cost=None,
                    budget=None, eligible=None, expansions=None,
                )
                next_state = r.next_state
            else:
                assert model is not None and device is not None
                r = _run_cost_bounded(
                    env, state, task,
                    model=model, device=device,
                    gamma=gamma, success_reward=success_reward,
                    max_depth=max_depth, max_expansions=max_expansions,
                    cost_ratio=cost_ratio,
                    planner_path=planner_path, domain_path=domain_path,
                    alias=alias, fd_timeout_s=fd_timeout_s,
                )
                rec.update(
                    success=r.success, cost=r.cost, actions=r.num_actions,
                    trace=r.trace, error=r.error,
                    v_ap=round(r.v_ap, 6) if r.v_ap is not None else None,
                    source=r.source,
                    reference_cost=round(r.reference_cost, 6) if r.reference_cost is not None else None,
                    budget=round(r.cost_budget, 6) if r.cost_budget is not None else None,
                    eligible=r.eligible_count,
                    expansions=r.expansions,
                )
                next_state = r.next_state

            if r.success:
                state = next_state
                total_cost += r.cost
                total_actions += r.num_actions
                completed += 1

        rec["jar_ready"] = bnp._jar_ready(state)
        if rec["jar_ready"]:
            jar_ready_count += 1
        records.append(rec)

    wall = time.perf_counter() - t0

    attempted = len(records)
    summary: Dict[str, Any] = {
        "sequence_path": str(sequence_path),
        "policy": policy,
        "attempted": attempted,
        "completed": completed,
        "total_pddl_cost": round(total_cost, 6),
        "mean_pddl_cost": round(total_cost / completed, 6) if completed > 0 else 0.0,
        "total_actions": total_actions,
        "auto_count": auto_count,
        "auto_rate": round(auto_count / attempted, 4) if attempted else 0.0,
        "jar_ready_task_count": jar_ready_count,
        "wall_seconds": round(wall, 4),
    }
    if policy == "cost_bounded":
        summary["cost_ratio"] = cost_ratio

    return {"summary": summary, "tasks": records}


# ---------------------------------------------------------------------------
# Paired output
# ---------------------------------------------------------------------------

def _pair_results(
    myopic: Dict[str, Any],
    guided: Dict[str, Any],
) -> Dict[str, Any]:
    mc = myopic["summary"]["total_pddl_cost"]
    gc = guided["summary"]["total_pddl_cost"]
    return {
        "myopic": myopic,
        "guided": guided,
        "paired_cost_delta": round(gc - mc, 6),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired persistent-sequence evaluation: myopic vs cost_bounded."
    )
    parser.add_argument("--policy", type=str, required=True,
                        choices=["myopic", "cost_bounded", "both"])
    parser.add_argument("--sequence-path", type=Path, required=True)
    parser.add_argument("--q-weights", type=Path, default=None)
    parser.add_argument("--config-path", type=Path,
                        default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--domain-path", type=Path,
                        default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path,
                        default=Path("downward/fast-downward.py"))
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--fd-timeout-s", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--success-reward", type=float, default=81.06943684690286)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--cost-ratio", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    if args.policy in ("cost_bounded", "both") and args.q_weights is None:
        parser.error("--q-weights is required for cost_bounded/both policies")

    def _run(policy: str) -> Dict[str, Any]:
        return run_sequence(
            policy=policy,
            sequence_path=args.sequence_path,
            config_path=args.config_path,
            domain_path=args.domain_path,
            planner_path=args.planner_path,
            alias=args.alias,
            fd_timeout_s=args.fd_timeout_s,
            seed=args.seed,
            gamma=args.gamma,
            success_reward=args.success_reward,
            hidden_dim=args.hidden_dim,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            cost_ratio=args.cost_ratio,
            # myopic ignores q_weights; cost_bounded raises inside run_sequence.
            q_weights=args.q_weights if args.policy != "myopic" else None,
            max_tasks=args.max_tasks,
        )

    if args.policy == "both":
        myopic_result = _run("myopic")
        guided_result = _run("cost_bounded")
        output = _pair_results(myopic_result, guided_result)
    else:
        output = _run(args.policy)

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
