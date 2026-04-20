from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anticipatory_rl.envs.restaurant.restaurant_symbolic_env import CONFIG_PATH, RestaurantSymbolicEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one restaurant env state or a short random rollout to PNG frames."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
        help="Restaurant YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "restaurant_viz",
        help="Directory for PNG frames and metadata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="Number of rollout steps after reset.")
    parser.add_argument(
        "--policy",
        choices=("random", "valid-random"),
        default="valid-random",
        help="Action policy for rollout frames.",
    )
    parser.add_argument("--max-task-steps", type=int, default=24)
    parser.add_argument("--success-reward", type=float, default=15.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--travel-cost-scale", type=float, default=1.0)
    parser.add_argument("--pick-cost", type=float, default=1.0)
    parser.add_argument("--place-cost", type=float, default=1.0)
    parser.add_argument("--wash-cost", type=float, default=2.0)
    parser.add_argument("--fill-cost", type=float, default=1.0)
    parser.add_argument("--brew-cost", type=float, default=2.0)
    parser.add_argument("--fruit-cost", type=float, default=2.0)
    return parser


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, frame)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _choose_action(env: RestaurantSymbolicEnv, *, policy: str, rng: np.random.Generator) -> int:
    if policy == "random":
        return int(rng.integers(env.action_space.n))
    info = env._info(success=False)
    valid = np.flatnonzero(np.asarray(info["valid_action_mask"], dtype=np.float32) > 0.0)
    if valid.size == 0:
        return int(rng.integers(env.action_space.n))
    return int(rng.choice(valid))


def main() -> None:
    args = build_parser().parse_args()
    env = RestaurantSymbolicEnv(
        config_path=args.config_path,
        render_mode="rgb_array",
        max_task_steps=args.max_task_steps,
        success_reward=args.success_reward,
        invalid_action_penalty=args.invalid_action_penalty,
        travel_cost_scale=args.travel_cost_scale,
        pick_cost=args.pick_cost,
        place_cost=args.place_cost,
        wash_cost=args.wash_cost,
        fill_cost=args.fill_cost,
        brew_cost=args.brew_cost,
        fruit_cost=args.fruit_cost,
        rng_seed=args.seed,
    )

    obs, info = env.reset(seed=args.seed)
    del obs
    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict[str, Any]] = []

    frame = env.render()
    frame_path = output_dir / "frame_000.png"
    _save_frame(frame_path, frame)
    metadata.append(
        {
            "frame": frame_path.name,
            "step": 0,
            "action": None,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": _jsonify(info),
        }
    )

    for step in range(1, args.steps + 1):
        action = _choose_action(env, policy=args.policy, rng=rng)
        _, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frame_path = output_dir / f"frame_{step:03d}.png"
        _save_frame(frame_path, frame)
        metadata.append(
            {
                "frame": frame_path.name,
                "step": step,
                "action": int(action),
                "action_name": env.get_action_meanings()[action],
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": _jsonify(info),
            }
        )
        if terminated or truncated:
            break

    summary_path = output_dir / "frames.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(metadata)} frame(s) to {output_dir}")
    print(f"Metadata -> {summary_path}")


if __name__ == "__main__":
    main()
