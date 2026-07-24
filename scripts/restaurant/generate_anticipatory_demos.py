"""Generate anticipatory oracle demo transitions offline for DQN replay seeding.

Plans each task using the Clairvoyant Sequence Oracle (K>1) with Fast-Downward,
executes ONLY the first task's physical actions in the symbolic env, and stores
every (s, a, r, s') transition. Auto-success tasks are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from anticipatory_rl.agents.restaurant.dqn import _planner_action_to_env_action, _store_transition
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    solve_restaurant_sequence_with_fd,
)
from anticipatory_rl.utils import extract_masks


class _Collector:
    """Duck-type of a ReplayBuffer that just appends TensorDicts to a list."""

    def __init__(self) -> None:
        self.items: List[Any] = []

    def add(self, td: Any) -> None:
        self.items.append(td)


def _read_max_steps_from_config(config_path: Path) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "max_steps_per_task" in cfg:
        return int(cfg["max_steps_per_task"])
    return 64


def _seed_replay_with_anticipatory_oracle(
    replay: _Collector,
    env: RestaurantSymbolicEnv,
    *,
    config_path: Path,
    n_outcomes: int,
    K: int,
    max_steps: int,
    seed_base: int,
    planner_path: Path,
    domain_path: Path,
    alias: str = "seq-sat-lama-2011",
    timeout_s: float = 120.0,
    env_reset_tasks: int = 50,
) -> int:
    """Persistent-world anticipatory oracle roll-out."""
    stored = 0
    outcomes = 0
    world_index = 0

    total_tasks_needed = n_outcomes + K + (n_outcomes // env_reset_tasks) * K

    dummy_env = RestaurantSymbolicEnv(config_path=config_path, rng_seed=seed_base)
    tasks_pool: List[RestaurantTask] = []
    dummy_world_index = 0
    dummy_env.reset(seed=seed_base + 100_003 * dummy_world_index)
    dummy_world_index += 1

    for i in range(total_tasks_needed):
        if env_reset_tasks > 0 and i > 0 and i % env_reset_tasks == 0:
            dummy_env.reset(seed=seed_base + 100_003 * dummy_world_index)
            dummy_world_index += 1
        dummy_env._resample_task()
        tasks_pool.append(dummy_env.task)

    task_index = 0
    n_tasks = len(env.enumerate_task_distribution())

    obs, info = env.reset(seed=seed_base + 100_003 * world_index)
    world_index += 1

    while outcomes < n_outcomes:
        if env_reset_tasks > 0 and outcomes > 0 and outcomes % env_reset_tasks == 0:
            obs, info = env.reset(seed=seed_base + 100_003 * world_index)
            world_index += 1
            task_index = outcomes

        env.set_task(
            tasks_pool[task_index].task_type,
            target_location=tasks_pool[task_index].target_location,
            target_kind=tasks_pool[task_index].target_kind,
            object_name=tasks_pool[task_index].object_name,
            task_source="library",
        )
        env._task_steps = 0
        env._pending_auto_success = env._task_already_satisfied()
        obs, info = env._obs(), env._info(success=False)

        if env._pending_auto_success:
            obs, _, _, _, info = env.step(env.action_space.sample())
            outcomes += 1
            task_index += 1
            continue

        state = RestaurantPlannerState.from_env(env)

        current_task = env.task
        seg_end = ((outcomes // env_reset_tasks) + 1) * env_reset_tasks if env_reset_tasks > 0 else len(tasks_pool)
        future_tasks = tasks_pool[task_index + 1 : min(task_index + K, seg_end)]
        window = [current_task] + future_tasks

        result = solve_restaurant_sequence_with_fd(
            env, state, window,
            planner_path=planner_path, domain_path=domain_path, timeout_s=timeout_s,
        )

        if not result.success:
            print(f"Planner failed on task {outcomes}. Advancing without transition...")
            outcomes += 1
            task_index += 1
            continue

        first_segment = result.task_segments[0]
        if len(first_segment.physical_actions) > max_steps:
            print(f"Planner produced >max_steps on task {outcomes}. Advancing...")
            outcomes += 1
            task_index += 1
            continue

        for plan_action in first_segment.physical_actions:
            masks = extract_masks(info)
            action = _planner_action_to_env_action(env, plan_action)
            parsed = env._normalize_action(action)
            if not env._is_action_valid(parsed):
                raise RuntimeError(f"FD plan action invalid in env: {plan_action} → {action}")
            next_obs, reward, success, truncated, next_info = env.step(action)
            next_masks = extract_masks(next_info)

            task_boundary = bool(success)
            transition_done = False

            next_auto_mask = torch.zeros(n_tasks, dtype=torch.float32)
            if task_boundary:
                for k, (tau_k, _) in enumerate(env.enumerate_task_distribution()):
                    if env._task_already_satisfied(task=tau_k):
                        next_auto_mask[k] = 1.0

            _store_transition(
                replay, obs, action, reward, masks,
                next_obs, next_masks, transition_done, task_boundary,
                next_auto_satisfied_mask=next_auto_mask,
            )
            stored += 1
            obs, info = next_obs, next_info
            if success or truncated:
                break

        outcomes += 1
        task_index += 1
        print(f"Generated {outcomes}/{n_outcomes} outcomes ({stored} transitions so far)")

    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate anticipatory oracle demo transitions")
    parser.add_argument("--config-path", required=True, help="Env YAML config")
    parser.add_argument("--num-tasks", type=int, default=50,
                        help="Number of task outcomes to demonstrate")
    parser.add_argument("--K", type=int, default=3, help="Lookahead horizon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Output .pt path (default: demos/<config_stem>_anticipatoryK<K>_<num_tasks>_seed<seed>.pt)")
    parser.add_argument("--planner-path", default="downward/fast-downward.py")
    parser.add_argument("--domain-path", default="pddl/toy_restaurant_sequence_domain.pddl")
    parser.add_argument("--alias", default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--env-reset-tasks", type=int, default=50,
                        help="Episode horizon (tasks per episode)")
    parser.add_argument("--success-reward", type=float, default=95.31,
                        help="Task success reward (should match rl_costs.yaml R_star)")
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    max_steps = _read_max_steps_from_config(config_path)

    env = RestaurantSymbolicEnv(
        config_path=config_path,
        rng_seed=args.seed,
        max_steps_per_task=max_steps,
        success_reward=args.success_reward,
        invalid_action_penalty=args.invalid_action_penalty,
    )

    if args.output is None:
        stem = config_path.stem
        out_dir = Path("demos")
        output = out_dir / f"{stem}_anticipatoryK{args.K}_{args.num_tasks}_seed{args.seed}.pt"
    else:
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    collector = _Collector()
    stored = _seed_replay_with_anticipatory_oracle(
        collector, env,
        config_path=config_path,
        n_outcomes=args.num_tasks,
        K=args.K,
        max_steps=max_steps,
        seed_base=args.seed,
        planner_path=Path(args.planner_path),
        domain_path=Path(args.domain_path),
        alias=args.alias,
        timeout_s=args.timeout_s,
        env_reset_tasks=args.env_reset_tasks,
    )

    metadata: Dict[str, Any] = {
        "config_path": str(config_path.resolve()),
        "obs_dim": int(env.observation_space.shape[0]),
        "num_objects": len(env.object_names),
        "num_locations": len(env.locations),
        "num_action_types": len(env.action_type_index),
        "none_object_index": env.none_object_index,
        "none_location_index": env.none_location_index,
        "seed": args.seed,
        "num_tasks": args.num_tasks,
        "n_outcomes": args.num_tasks,
        "env_reset_tasks": args.env_reset_tasks,
        "stored": stored,
        "max_steps_per_task": max_steps,
        "credit_horizon": "anticipatory",
        "K": args.K,
        "success_reward": args.success_reward,
        "invalid_action_penalty": args.invalid_action_penalty,
        "travel_cost_scale": env.travel_cost_scale,
        "pick_cost": env.pick_cost,
        "place_cost": env.place_cost,
        "wash_cost": env.wash_cost,
        "fill_cost": env.fill_cost,
        "brew_cost": env.brew_cost,
        "fruit_cost": env.fruit_cost,
        "spread_cost": env.spread_cost,
        "pour_cost": env.pour_cost,
        "refill_cost": env.refill_cost,
        "drain_cost": env.drain_cost,
        "object_names": list(env.object_names),
        "locations": list(env.locations),
        "object_kinds": list(env.object_kinds),
        "contents": list(env.contents),
        "task_types": list(env.task_types),
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    torch.save({"metadata": metadata, "transitions": collector.items}, output)
    print(f"Saved {stored} transitions to {output}")


if __name__ == "__main__":
    main()
