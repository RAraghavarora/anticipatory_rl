"""Smoke-test the restaurant PDDL planner stack on one sampled task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anticipatory_rl.envs.restaurant_symbolic_env import RestaurantSymbolicEnv
from anticipatory_rl.tasks.restaurant_planner import RestaurantPlannerState, solve_restaurant_task_with_fd


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for restaurant FD planner stack.")
    parser.add_argument("--layout-corpus", type=Path, default=Path("data/restaurant_layouts/paper2_scale_layouts.json"))
    parser.add_argument("--config-path", type=Path, default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/restaurant_domain.pddl"))
    parser.add_argument("--search", type=str, default="astar(lmcut())")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.layout_corpus.read_text(encoding="utf-8"))
    layouts = payload["layouts"] if isinstance(payload, dict) else payload
    layout = layouts[args.seed % len(layouts)]
    env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=args.seed)
    _, info = env.reset(seed=args.seed + 100, options={"layout": layout, "task_library": layout.get("task_library", [])})
    task = env.task
    state = RestaurantPlannerState.from_env(env)

    result = solve_restaurant_task_with_fd(
        env,
        state,
        task,
        planner_path=args.planner_path,
        domain_path=args.domain_path,
        search=args.search,
    )
    print("Task:", info.get("task"))
    print("Success:", result.success)
    print("Actions:", len(result.plan_actions))
    print("paper2_cost:", result.plan_cost)
    print("Solve time (s):", result.solve_time_s)
    if result.error:
        print("Error:", result.error)


if __name__ == "__main__":
    main()
