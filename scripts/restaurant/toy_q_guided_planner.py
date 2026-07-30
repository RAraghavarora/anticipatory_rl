#!/usr/bin/env python3
"""Q-guided anticipatory planner for the toy restaurant env.

Uses Fast Downward to generate candidate plans for the current task, then
scores each candidate terminal state with the trained anticipatory DQN's
learned future-return value:

    score = prefix_cost - beta * gamma^N * V_Q_AP(terminal)

where V_Q_AP(s) = E_tau [ V_Q(s, tau) ] and V_Q(s, tau) = max_a Q(s, tau, a).

For auto-satisfied future tasks, the value is Q(s, tau, auto_complete) = V(s, tau),
the preparedness-event value the DQN learned, not an arbitrary domain action.

Usage:
    python scripts/restaurant/toy_q_guided_planner.py \
        --config-path configs/restaurant/toy_level_2_2.yaml \
        --domain-path pddl/toy_restaurant_domain.pddl \
        --planner-path downward/fast-downward.py \
        --q-weights runs/<checkpoint>/restaurant_dqn.pt \
        --beta 1.0 --gamma 0.99 \
        --num-tasks 5000 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
    task_goal_clauses,
)
from anticipatory_rl.utils import select_device

# Support both `python scripts/restaurant/toy_q_guided_planner.py` and
# `python -m scripts.restaurant.toy_q_guided_planner` when importing shared
# helpers from the sibling toy_anticipatory_oracle.py script.
import sys  # noqa: E402
_THIS_DIR = Path(__file__).parent.resolve()  # noqa: E402
if str(_THIS_DIR) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_THIS_DIR))  # noqa: E402
from toy_anticipatory_oracle import (  # noqa: E402
    AnticipatoryPlan,
    _enumerate_future_tasks,
    _future_candidate_tasks,
    _state_signature,
    _task_is_auto_satisfied,
    apply_plan_until_first_task_satisfied,
)


# ---------------------------------------------------------------------------
# Env / mask helpers
# ---------------------------------------------------------------------------

def _sync_env_from_planner_state(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
) -> None:
    """Copy a planner state into env.state so env._obs() and masks work."""
    env.state.agent_location = state.agent_location
    env.state.holding = state.holding
    env.state.bread_spread = state.bread_spread
    for name, obj in state.objects.items():
        env.state.objects[name].location = obj.location
        env.state.objects[name].dirty = obj.dirty
        env.state.objects[name].filled_with = obj.filled_with
        env.state.objects[name].contained_in = obj.contained_in


def _build_auto_complete_masks(env: RestaurantSymbolicEnv) -> Dict[str, np.ndarray]:
    """Replay-style masks where only the internal auto_complete event is valid."""
    auto_idx = env.action_type_index["auto_complete"]
    none_object = env.none_object_index
    none_location = env.none_location_index
    n_action_types = len(env.action_type_index)
    n_objects = env.num_objects + 1
    n_locations = env.num_locations + 1

    masks = {
        "valid_action_type_mask": np.zeros((n_action_types,), dtype=np.float32),
        "valid_object1_mask": np.zeros((n_action_types, n_objects), dtype=np.float32),
        "valid_location_mask": np.zeros((n_action_types, n_locations), dtype=np.float32),
        "valid_object2_mask": np.zeros((n_action_types, n_objects, n_objects), dtype=np.float32),
    }
    masks["valid_action_type_mask"][auto_idx] = 1.0
    masks["valid_object1_mask"][auto_idx, none_object] = 1.0
    masks["valid_location_mask"][auto_idx, none_location] = 1.0
    masks["valid_object2_mask"][auto_idx, none_object, none_object] = 1.0
    return masks


# ---------------------------------------------------------------------------
# V_Q_AP computation
# ---------------------------------------------------------------------------

def _compute_v_q_ap(
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    future_tasks: Sequence[Tuple[RestaurantTask, float]],
    *,
    model: RestaurantQNetwork,
    device: torch.device,
    cache: Dict[Tuple, float],
) -> float:
    """E_tau[V_Q(s, tau)] via the trained DQN, with caching by state signature.

    Cache key is just the state signature; future_tasks is fixed by the config's
    state-independent task distribution.

    Saves and restores env.state/env.task so the temporary syncs to candidate
    terminal states do not leak into the caller.
    """
    key = _state_signature(state)
    cached = cache.get(key)
    if cached is not None:
        return cached

    saved_state = copy.deepcopy(env.state)
    saved_task = env.task
    try:
        _sync_env_from_planner_state(env, state)

        total = 0.0
        for task, prob in future_tasks:
            if prob <= 0:
                continue

            # Set the future task so the observation and auto-satisfaction check
            # are task-conditioned.
            env.set_task(
                task.task_type,
                target_location=task.target_location,
                target_kind=task.target_kind,
                object_name=task.object_name,
            )
            obs = env._obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            auto = _task_is_auto_satisfied(state, task, env)
            if auto:
                masks = _build_auto_complete_masks(env)
                auto_idx = env.action_type_index["auto_complete"]
                none_object = env.none_object_index
                none_location = env.none_location_index
                action_type_t = torch.tensor([[auto_idx]], dtype=torch.int64, device=device)
                object1_t = torch.tensor([[none_object]], dtype=torch.int64, device=device)
                location_t = torch.tensor([[none_location]], dtype=torch.int64, device=device)
                object2_t = torch.tensor([[none_object]], dtype=torch.int64, device=device)
            else:
                masks = env._build_action_masks()
                action_type_mask_t = torch.tensor(
                    masks["valid_action_type_mask"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                object1_mask_t = torch.tensor(
                    masks["valid_object1_mask"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                location_mask_t = torch.tensor(
                    masks["valid_location_mask"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                object2_mask_t = torch.tensor(
                    masks["valid_object2_mask"], dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action_type_t, object1_t, location_t, object2_t = model(
                        obs_t,
                        action_type_masks=action_type_mask_t,
                        object1_masks=object1_mask_t,
                        location_masks=location_mask_t,
                        object2_masks=object2_mask_t,
                        decode_greedy=True,
                    )

            action_type_mask_t = torch.tensor(
                masks["valid_action_type_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            object1_mask_t = torch.tensor(
                masks["valid_object1_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            location_mask_t = torch.tensor(
                masks["valid_location_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            object2_mask_t = torch.tensor(
                masks["valid_object2_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                q_value = model(
                    obs_t,
                    action_types=action_type_t,
                    object1=object1_t,
                    location=location_t,
                    object2=object2_t,
                    action_type_masks=action_type_mask_t,
                    object1_masks=object1_mask_t,
                    location_masks=location_mask_t,
                    object2_masks=object2_mask_t,
                )
            total += prob * float(q_value.item())

        cache[key] = total
        return total
    finally:
        env.state = saved_state
        env.task = saved_task


# ---------------------------------------------------------------------------
# Q-guided planning
# ---------------------------------------------------------------------------

def _solve_q_guided_task(
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
    beta: float,
    model: RestaurantQNetwork,
    device: torch.device,
    v_ap_cache: Dict[Tuple, float],
) -> AnticipatoryPlan | None:
    """Generate candidate plans and pick the one minimizing
    prefix_cost - beta * gamma^N * V_Q_AP(terminal).

    Returns the executed prefix (up to first task satisfaction) and terminal state.
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
        consumed_terminal = terminal.copy()
        consume_delivery_from_state(consumed_terminal, task.task_type, task.target_location)
        prefix_cost = planner_actions_paper2_cost(prefix, env)
        v_q_ap = _compute_v_q_ap(
            consumed_terminal, env, future_tasks,
            model=model,
            device=device,
            cache=v_ap_cache,
        )
        score = prefix_cost - beta * (gamma ** len(prefix)) * v_q_ap
        best = AnticipatoryPlan(
            prefix_actions=prefix,
            prefix_cost=float(prefix_cost),
            terminal_state=consumed_terminal,
            strategy="myopic",
            v_ap=float(v_q_ap),
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
        consumed_terminal = terminal.copy()
        consume_delivery_from_state(consumed_terminal, task.task_type, task.target_location)
        prefix_cost = planner_actions_paper2_cost(prefix, env)
        v_q_ap = _compute_v_q_ap(
            consumed_terminal, env, future_tasks,
            model=model,
            device=device,
            cache=v_ap_cache,
        )
        score = prefix_cost - beta * (gamma ** len(prefix)) * v_q_ap
        if score < best_score:
            best = AnticipatoryPlan(
                prefix_actions=prefix,
                prefix_cost=float(prefix_cost),
                terminal_state=consumed_terminal,
                strategy=f"joint+{fut.task_type}",
                v_ap=float(v_q_ap),
                full_plan_steps=len(result.plan_actions),
            )
            best_score = score

    return best


# ---------------------------------------------------------------------------
# Main planner loop
# ---------------------------------------------------------------------------

def run_q_guided_planner(
    *,
    config_path: Path,
    domain_path: Path,
    planner_path: Path,
    q_weights: Path,
    num_tasks: int,
    tasks_per_reset: int,
    alias: str,
    timeout_s: float,
    seed: int,
    gamma: float,
    beta: float,
    hidden_dim: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed)
    obs, info = env.reset(seed=seed)
    state = RestaurantPlannerState.from_env(env)

    device = select_device()
    model = RestaurantQNetwork(
        input_dim=int(np.asarray(obs).shape[0]),
        action_type_dim=int(env.action_space["action_type"].n),
        object_dim=int(env.action_space["object1"].n),
        location_dim=int(env.action_space["location"].n),
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(q_weights.expanduser().resolve(), map_location=device))
    model.eval()

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
            plan = _solve_q_guided_task(
                env, state, task,
                future_candidates=future_candidates,
                future_tasks=_enumerate_future_tasks(env, state),
                planner_path=planner_path,
                domain_path=domain_path,
                alias=alias,
                timeout_s=timeout_s,
                gamma=gamma,
                beta=beta,
                model=model,
                device=device,
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
            else:
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
        "beta": float(beta),
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
        "v_ap_mode": "q",
    }
    return {"stats": stats, "tasks": records}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-guided anticipatory planner for toy restaurant."
    )
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_restaurant.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--q-weights", type=Path, required=True)
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--num-tasks", type=int, default=40)
    parser.add_argument("--tasks-per-reset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-path", type=Path, default=Path("runs/toy_q_guided_planner/results.json"))
    args = parser.parse_args()

    result = run_q_guided_planner(
        config_path=args.config_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        q_weights=args.q_weights,
        num_tasks=args.num_tasks,
        tasks_per_reset=args.tasks_per_reset,
        alias=args.alias,
        timeout_s=args.timeout_s,
        seed=args.seed,
        gamma=args.gamma,
        beta=args.beta,
        hidden_dim=args.hidden_dim,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
