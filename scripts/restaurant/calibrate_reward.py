"""Calibrate the scalar success_reward from myopic oracle demonstrations.

Runs the persistent-world myopic oracle with success_reward=0, then sets
R_star so that every demonstrated optimal plan strictly beats deliberate timeout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import yaml

from anticipatory_rl.agents.restaurant.dqn import _persistent_oracle_rollout
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


def _read_max_steps_from_config(config_path: Path) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "max_steps_per_task" in cfg:
        return int(cfg["max_steps_per_task"])
    return 64


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate success_reward from myopic oracle demos")
    parser.add_argument("--config-path", required=True, help="Env YAML config")
    parser.add_argument("--n-outcomes", type=int, default=2000, help="Number of task outcomes to roll out")
    parser.add_argument("--env-reset-tasks", type=int, default=200, help="Physical env reset interval in tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--planner-path", default="downward/fast-downward.py")
    parser.add_argument("--domain-path", default="pddl/toy_restaurant_domain.pddl")
    parser.add_argument("--alias", default="seq-sat-lama-2011")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", default="calibrated_reward.json", help="Path to write JSON result")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    max_steps = _read_max_steps_from_config(config_path)

    env = RestaurantSymbolicEnv(
        config_path=config_path,
        max_steps_per_task=max_steps,
        success_reward=0.0,
        rng_seed=args.seed,
    )

    result = _persistent_oracle_rollout(
        env,
        n_outcomes=args.n_outcomes,
        max_steps=max_steps,
        seed_base=args.seed,
        planner_path=Path(args.planner_path),
        domain_path=Path(args.domain_path),
        alias=args.alias,
        timeout_s=args.timeout_s,
        transition_store=None,
        env_reset_tasks=args.env_reset_tasks,
    )

    successful_plan_rewards: List[List[float]] = result["successful_plan_rewards"]  # type: ignore[assignment]
    D_list = []
    for rewards in successful_plan_rewards:
        T = len(rewards)
        costs = [-r for r in rewards]
        D = sum(costs[t] * (args.gamma ** (t - (T - 1))) for t in range(T))
        D_list.append(D)

    max_D = max(D_list) if D_list else 0.0
    R_star = (1.0 + args.margin) * max_D

    print(f"R_star = {R_star}")

    output = {"gamma": args.gamma, "margin": args.margin, "max_D": max_D, "R_star": R_star,
              "n_outcomes": args.n_outcomes, "seed": args.seed}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote calibration result to {output_path}")


if __name__ == "__main__":
    main()
