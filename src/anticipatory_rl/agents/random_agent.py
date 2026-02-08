"""
Random-action baseline for the MiniWorld rearrangement environment.

Features:
- Samples a task from the local dataset.
- Runs a random policy until the task is solved (or until a user-specified cap).
- Optional live rendering (`--render`).
- Optional RGB video recording of the full run (`--record out.mp4`) using imageio.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from anticipatory_rl.envs.miniworld_env import MiniWorldGridRearrange
from anticipatory_rl.tasks.generator import Task, generate_task_sequence


@dataclass
class EpisodeResult:
    task: Task
    success: bool
    attempts: int
    steps: int
    frames: Optional[List[np.ndarray]] = None


class RandomAgent:
    """Uniform random policy over the environment's discrete action space."""

    def __init__(self, env: MiniWorldGridRearrange) -> None:
        self.env = env

    def select_action(self) -> int:
        return int(self.env.action_space.sample())


def task_completed(state: Dict[str, object], task: Task) -> bool:
    objects: Dict[str, Dict[str, object]] = state["objects"]
    payload = task.payload

    if task.task_type == "bring_single":
        obj = payload["objects"][0]
        target = payload["target"]
        return objects[obj]["region"] == target

    if task.task_type == "bring_pair":
        target = payload["target"]
        return all(objects[obj]["region"] == target for obj in payload["objects"])

    if task.task_type == "clear_receptacle":
        source = payload["source"]
        return all(info["region"] != source for info in objects.values())

    raise ValueError(f"Unknown task type: {task.task_type}")


def capture_top_frame(env: MiniWorldGridRearrange) -> np.ndarray:
    """Return a copy of the latest top-down RGB frame."""
    frame = env.render_top_view(env.vis_fb)
    return np.array(frame, copy=True)


def run_random_agent(
    task: Task,
    *,
    max_attempts: Optional[int] = None,
    max_steps_per_attempt: int = 400,
    seed: int | None = None,
    render: bool = False,
    record: bool = False,
) -> EpisodeResult:
    rng = random.Random(seed)
    env = MiniWorldGridRearrange(render_mode="human" if render else None)
    policy = RandomAgent(env)
    frames: Optional[List[np.ndarray]] = [] if record else None

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        obs, _ = env.reset(seed=rng.randint(0, 10**9))
        if render:
            env.render()
        if record and frames is not None:
            frames.append(capture_top_frame(env))

        for step in range(1, max_steps_per_attempt + 1):
            action = policy.select_action()
            obs, reward, termination, truncation, info = env.step(action)
            if render:
                env.render()
            if record and frames is not None:
                frames.append(capture_top_frame(env))

            if task_completed(info["state"], task):
                env.close()
                return EpisodeResult(task, True, attempt, step, frames)

            if termination or truncation:
                break

    env.close()
    return EpisodeResult(task, False, attempt, max_steps_per_attempt, frames)


def save_video(frames: List[np.ndarray], path: str, fps: int = 12) -> None:
    if not frames:
        print("No frames recorded; skipping video export.")
        return
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            "Recording requires imageio>=2.26. Install with `pip install imageio imageio-ffmpeg`."
        ) from exc

    iio.imwrite(path, frames, fps=fps)
    print(f"[recording] wrote {len(frames)} frames to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random MiniWorld agent runner.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for task/policy.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Stop after this many attempts (default: keep trying until success).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Max environment steps per attempt before resetting.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MiniWorld top-down window while the agent runs.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="VIDEO_PATH",
        help="Save an RGB video (e.g., run.mp4 or run.gif). Requires `pip install imageio imageio-ffmpeg`.",
    )
    parser.add_argument("--fps", type=int, default=12, help="Playback FPS when recording video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = generate_task_sequence(count=1, seed=args.seed)[0]
    result = run_random_agent(
        task,
        max_attempts=args.max_attempts,
        max_steps_per_attempt=args.max_steps,
        seed=args.seed,
        render=args.render,
        record=args.record is not None,
    )

    print(f"Task type: {task.task_type}")
    print(f"Instructions: {task.instructions}")
    print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Attempts: {result.attempts}, Steps in final attempt: {result.steps}")

    if args.record:
        if result.frames is None:
            print("Recording was requested but no frames were captured.")
        else:
            save_video(result.frames, args.record, fps=args.fps)


if __name__ == "__main__":
    main()
