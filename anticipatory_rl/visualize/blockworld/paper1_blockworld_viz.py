from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import List

import numpy as np
from PIL import Image

from anticipatory_rl.envs.paper1_blockworld_image_env import Paper1BlockworldImageEnv


def obs_to_rgb(obs: np.ndarray) -> np.ndarray:
    rgb = np.clip(obs[:3], 0.0, 1.0)
    return np.rint(rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def describe_task(info: dict) -> str:
    assignments = info.get("task_assignments", ())
    return ", ".join(f"{block}->{region}" for block, region in assignments)


def save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to save.")
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / max(1, fps))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the reproduced paper1 2D Blockworld environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper1_blockworld_viz.png"),
        help="PNG or GIF output path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="If > 0, record a rollout GIF.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--task-library-size", type=int, default=20)
    parser.add_argument("--max-task-steps", type=int, default=64)
    parser.add_argument("--render-tile-px", type=int, default=24)
    parser.add_argument(
        "--policy",
        choices=("random",),
        default="random",
        help="Rollout policy when --steps > 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    env = Paper1BlockworldImageEnv(
        task_library_size=args.task_library_size,
        max_task_steps=args.max_task_steps,
        render_tile_px=args.render_tile_px,
        procedural_layout=True,
    )

    obs, info = env.reset(seed=args.seed)
    frame0 = obs_to_rgb(obs)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.steps <= 0:
        Image.fromarray(frame0).save(args.output)
        print(f"[paper1-blockworld-viz] wrote snapshot to {args.output.resolve()}")
        print("Current task:", describe_task(info))
        return

    frames = [frame0]
    for _ in range(args.steps):
        action = env.action_space.sample() if args.policy == "random" else env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(obs_to_rgb(obs))
        if terminated or truncated:
            obs, _ = env.reset(seed=rng.randint(0, 10**9))
            frames.append(obs_to_rgb(obs))

    if args.output.suffix.lower() != ".gif":
        args.output = args.output.with_suffix(".gif")
    save_gif(frames, args.output, fps=args.fps)
    print(f"[paper1-blockworld-viz] wrote GIF to {args.output.resolve()}")


if __name__ == "__main__":
    main()
