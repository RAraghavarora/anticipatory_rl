"""Greedy inference utility for SimpleGrid image DQN checkpoints.

This script mirrors the network/env setup from the training script but runs a
single-environment rollout, saving the pixel channels of every visited state
to disk. Useful for debugging datasets and verifying that trained policies
behave as expected under the latest environment dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork
from anticipatory_rl.envs.simple_grid_image_env import (
    OBJECT_NAMES,
    SimpleGridImageEnv,
)


@dataclass
class SampledTask:
    task_type: str  # "move" or "clear"
    object_name: Optional[str]
    receptacle_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy DQN inference and save pixel channels.")
    parser.add_argument("--state-dict", type=Path, required=True, help="Checkpoint containing ConvQNetwork weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/image_dqn_pixels"), help="Where to store saved pixel arrays.")
    parser.add_argument("--total-steps", type=int, default=5_000, help="Number of primitive steps to execute.")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-objects", type=int, default=len(OBJECT_NAMES))
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to task/object/receptacle distribution YAML.",
    )
    parser.add_argument("--clear-task-prob", type=float, default=None, help="Override clear-task probability (defaults to config).")
    parser.add_argument("--tasks-per-reset", type=int, default=1_000, help="Force env reset after this many tasks (match training).")
    parser.add_argument(
        "--tasks-per-sequence",
        type=int,
        default=100,
        help="Number of sampled tasks to queue at a time (must be > 0).",
    )
    parser.add_argument(
        "--save-format",
        choices=("png", "npy"),
        default="png",
        help="How to serialize each observation (default: png).",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> SimpleGridImageEnv:
    return SimpleGridImageEnv(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_task_prob=args.clear_task_prob,
        config_path=args.config_path,
    )


def weighted_choice(
    distribution: Mapping[str, float],
    candidates: Sequence[str],
    rng: np.random.Generator,
) -> str:
    if not candidates:
        raise ValueError("Cannot sample from an empty candidate set.")
    weights = np.array([max(distribution.get(name, 0.0), 0.0) for name in candidates], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return str(rng.choice(candidates))
    probs = weights / total
    return str(rng.choice(candidates, p=probs))


def _eligible_receptacles(receptacles: Sequence[str], last_sampled: Optional[str]) -> List[str]:
    if last_sampled is None or len(receptacles) <= 1:
        return list(receptacles)
    filtered = [rec for rec in receptacles if rec != last_sampled]
    return filtered or list(receptacles)


def sample_task_sequence(
    env: SimpleGridImageEnv,
    num_tasks: int,
    rng: np.random.Generator,
) -> List[SampledTask]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be > 0 when sampling task sequences.")
    tasks: List[SampledTask] = []
    last_rec: Optional[str] = None
    active_objects = list(env.active_objects)
    receptacles = list(env.receptacle_names)
    object_distribution = getattr(env, "object_distribution", {})
    surface_distribution = getattr(env, "surface_distribution", {})
    object_source_distribution = getattr(env, "object_source_distribution", {})
    clear_prob = getattr(env, "clear_task_prob", 0.0)
    for _ in range(num_tasks):
        if clear_prob > 0.0 and rng.random() < clear_prob:
            rec_choices = _eligible_receptacles(receptacles, last_rec)
            rec = weighted_choice(surface_distribution, rec_choices, rng)
            tasks.append(SampledTask("clear", None, rec))
            last_rec = rec
            continue
        obj = weighted_choice(object_distribution, active_objects, rng)
        source_dist = object_source_distribution.get(obj, surface_distribution)
        rec_choices = _eligible_receptacles(receptacles, last_rec)
        rec = weighted_choice(source_dist, rec_choices, rng)
        tasks.append(SampledTask("move", obj, rec))
        last_rec = rec
    return tasks


def apply_sampled_task(env: SimpleGridImageEnv, task: SampledTask):
    env.task_type = task.task_type
    env.target_object = task.object_name
    env.target_receptacle = task.receptacle_name
    if hasattr(env, "_last_target_receptacle"):
        env._last_target_receptacle = task.receptacle_name  # noqa: SLF001
    if hasattr(env, "_task_steps"):
        env._task_steps = 0  # noqa: SLF001
    if hasattr(env, "_pending_auto_success") and hasattr(env, "_task_already_satisfied"):
        env._pending_auto_success = env._task_already_satisfied()  # noqa: SLF001
    return env._obs(), env._info()


def _describe_record(record: dict) -> str:
    if record.get("task_type") == "clear":
        return f"CLEAR {record.get('target_receptacle')}"
    obj = record.get("target_object")
    rec = record.get("target_receptacle")
    if obj is None:
        return f"MOVE (?) -> {rec}"
    return f"MOVE {obj} -> {rec}"


def save_observation(output_dir: Path, step: int, obs: np.ndarray, save_format: str) -> None:
    if save_format == "png":
        rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"frame_{step:06d}.png")
    else:
        np.save(output_dir / f"frame_{step:06d}.npy", obs)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    obs_shape = obs.shape
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    state_dict = torch.load(args.state_dict, map_location=device)
    q_net.load_state_dict(state_dict)
    q_net.eval()

    if args.tasks_per_sequence <= 0:
        raise ValueError("--tasks-per-sequence must be >= 1.")

    task_rng = np.random.default_rng(args.seed + 1)
    task_buffer: List[SampledTask] = []
    task_cursor = 0

    def dequeue_task() -> SampledTask:
        nonlocal task_buffer, task_cursor
        if not task_buffer or task_cursor >= len(task_buffer):
            task_buffer = sample_task_sequence(env, args.tasks_per_sequence, task_rng)
            task_cursor = 0
        task = task_buffer[task_cursor]
        task_cursor += 1
        return task

    current_task = dequeue_task()
    obs, info = apply_sampled_task(env, current_task)

    pick_action = SimpleGridImageEnv.PICK
    place_action = SimpleGridImageEnv.PLACE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_steps": 0,
        "tasks_attempted": 0,
        "successes": 0,
        "failures": 0,
    }
    tasks_since_reset = 0
    task_step_counter = 0
    task_return = 0.0
    task_records = []
    task_frame_ranges: List[Tuple[dict, int, int]] = []
    current_task_start = 0
    progress = tqdm(total=args.total_steps, desc="Inference rollout")

    for step in range(args.total_steps):
        save_observation(args.output_dir, step, obs, args.save_format)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            can_pick = info.get("can_pick", True)
            can_place = info.get("can_place", True)
            if not can_pick:
                q_values[0, pick_action] = float("-inf")
            if not can_place:
                q_values[0, place_action] = float("-inf")
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, success, horizon, info = env.step(action)
        progress.update(1)
        stats["total_steps"] += 1
        task_step_counter += 1
        task_return += float(reward)

        if success or horizon:
            stats["tasks_attempted"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            record = {
                "task_number": stats["tasks_attempted"],
                "task_type": current_task.task_type,
                "target_object": current_task.object_name,
                "target_receptacle": current_task.receptacle_name,
                "success": bool(success),
                "steps": task_step_counter,
                "return": task_return,
            }
            task_records.append(record)
            task_step_counter = 0
            task_return = 0.0
            tasks_since_reset += 1
            task_frame_ranges.append((record, current_task_start, step))
            current_task_start = step + 1
            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                obs, info = env.reset()
                tasks_since_reset = 0
            current_task = dequeue_task()
            obs, info = apply_sampled_task(env, current_task)

    progress.close()
    success_rate = stats["successes"] / max(1, stats["tasks_attempted"])
    stats["success_rate"] = success_rate
    task_steps = [rec["steps"] for rec in task_records]
    stats["avg_task_steps"] = float(np.mean(task_steps)) if task_steps else 0.0
    stats["median_task_steps"] = float(np.median(task_steps)) if task_steps else 0.0
    report = {
        "stats": stats,
        "tasks": task_records,
    }
    with (args.output_dir / "rollout_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    summary = (
        f"Saved {args.total_steps} frames/pixels to {args.output_dir.resolve()} "
        f"| success rate {success_rate:.1%} ({stats['successes']}/{stats['tasks_attempted']}) "
        f"| avg steps/task {stats['avg_task_steps']:.1f}"
    )
    if task_steps:
        summary += f" | median steps {stats['median_task_steps']:.1f}"
    print(summary)
    if task_step_counter > 0:
        print(
            f"Step budget expired with an unfinished task "
            f"({task_step_counter} steps so far); that attempt is not included in the stats."
        )
        pending_record = {
            "task_number": stats["tasks_attempted"] + 1,
            "task_type": current_task.task_type,
            "target_object": current_task.object_name,
            "target_receptacle": current_task.receptacle_name,
            "success": False,
            "steps": task_step_counter,
            "return": task_return,
        }
        task_frame_ranges.append(
            (pending_record, current_task_start, args.total_steps - 1)
        )
    if task_frame_ranges:
        print("Frame ranges per sampled task:")
        for record, start_idx, end_idx in task_frame_ranges:
            label = _describe_record(record)
            print(
                f"  Task {record['task_number']:>4}: frames {start_idx}–{end_idx} | {label}"
            )
    else:
        print("No completed tasks to report frame ranges for.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
