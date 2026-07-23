"""Generate myopic oracle demo transitions offline for DQN replay seeding.

Plans each task with Fast-Downward (myopic, single-task horizon), executes the
plan in the symbolic env, and stores every (s, a, r, s') transition.  Auto-success
tasks are skipped (no transition stored — the env settles them before any action).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from anticipatory_rl.agents.restaurant.dqn import _seed_replay_with_oracle
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate myopic oracle demo transitions")
    parser.add_argument("--config-path", required=True, help="Env YAML config")
    parser.add_argument("--num-tasks", type=int, default=100,
                        help="Number of task outcomes to demonstrate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Output .pt path (default: demos/<config_stem>_myopic_<num_tasks>_seed<seed>.pt)")
    parser.add_argument("--planner-path", default="downward/fast-downward.py")
    parser.add_argument("--domain-path", default="pddl/toy_restaurant_domain.pddl")
    parser.add_argument("--alias", default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--success-reward", type=float, default=95.31)
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
        output = out_dir / f"{stem}_myopic_{args.num_tasks}_seed{args.seed}.pt"
    else:
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    collector = _Collector()
    stored = _seed_replay_with_oracle(
        collector, env,
        n_outcomes=args.num_tasks,
        max_steps=max_steps,
        seed_base=args.seed,
        planner_path=Path(args.planner_path),
        domain_path=Path(args.domain_path),
        alias=args.alias,
        timeout_s=args.timeout_s,
        n_tasks=len(env.enumerate_task_distribution()),
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
        "stored": stored,
        "max_steps_per_task": max_steps,
        "credit_horizon": "myopic",
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
