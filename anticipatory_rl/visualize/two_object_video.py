"""
Record a random rollout in a MiniWorld variant with exactly two objects and two
receptacles, saving the frames as a GIF.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Set, TypeVar, Protocol

import numpy as np
from PIL import Image

from anticipatory_rl.envs.miniworld_env import MiniWorldGridRearrange, OBJECT_SPECS, RECEPTACLES


class _HasName(Protocol):
    name: str


T = TypeVar("T", bound=_HasName)


def _select_specs(names: Set[str], specs: Sequence[T]) -> List[T]:
    selected = [spec for spec in specs if spec.name in names]
    if len(selected) != len(names):
        found = {spec.name for spec in selected}
        missing = names - found
        raise ValueError(f"Missing specs for: {', '.join(sorted(missing))}")
    return selected


def capture_top_frame(env: MiniWorldGridRearrange) -> np.ndarray:
    """Return a copy of the latest top-down RGB frame."""
    frame = env.render_top_view(env.vis_fb)
    return np.array(frame, copy=True)


def save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames recorded; cannot save GIF.")
    images = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / max(fps, 1))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def record_rollout(
    output_path: Path,
    steps: int,
    fps: int,
    seed: Optional[int],
) -> Path:
    target_receptacles = {"kitchen_counter", "dining_table"}
    target_objects = {"water_bottle", "apple"}

    receptacles = _select_specs(target_receptacles, RECEPTACLES)
    objects = _select_specs(target_objects, OBJECT_SPECS)

    env = MiniWorldGridRearrange(
        receptacles=receptacles,
        object_specs=objects,
        render_mode=None,
    )
    rng = random.Random(seed)
    frames: List[np.ndarray] = []

    env.reset(seed=rng.randint(0, 10**9))
    frames.append(capture_top_frame(env))

    for _ in range(steps):
        action = env.action_space.sample()
        env.step(action)
        frames.append(capture_top_frame(env))

    env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, output_path, fps=fps)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a GIF of a random policy in the 2-object MiniWorld variant."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/two_obj_two_receptacles.gif"),
        help="Where to save the GIF (default: runs/two_obj_two_receptacles.gif).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Number of environment steps to record.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Playback FPS for the GIF.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = record_rollout(args.output, args.steps, args.fps, args.seed)
    print(f"[two-object-video] wrote GIF to {output_path.resolve()}")


if __name__ == "__main__":
    main()
