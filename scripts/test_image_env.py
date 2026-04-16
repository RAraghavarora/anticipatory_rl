import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork
from anticipatory_rl.envs.simple_grid_image_env import (
    OBJECT_DISTRIBUTION,
    OBJECT_NAMES,
    OBJECT_SOURCE_DISTRIBUTION,
    SURFACE_DISTRIBUTION,
    SimpleGridImageEnv,
)


@dataclass
class SampledTask:
    task_type: str  # "move" or "clear"
    object_name: Optional[str]
    receptacle_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render greedy rollouts from the image DQN.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/image_dqn_frames"),
        help="Directory where rendered frames are saved.",
    )
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=Path("runs/10_image_dqn/simple_grid_image_dqn.pt"),
        help="Checkpoint path for the ConvQNetwork weights.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size of the MLP head.")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size used by the env.")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=len(OBJECT_NAMES),
        help="Number of active objects.",
    )
    parser.add_argument("--success-reward", type=float, default=10.0, help="Success reward in env.")
    parser.add_argument(
        "--distance-reward-scale",
        type=float,
        default=1.0,
        help="Scaling for distance shaping.",
    )
    parser.add_argument(
        "--clear-task-prob",
        type=float,
        default=None,
        help="Override probability of requesting clear-receptacle tasks (default from configs/config.yaml).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Maximum primitive steps to capture before terminating.",
    )
    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=0.05,
        help="Probability of sampling a random action (epsilon).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed used for reset.",
    )
    parser.add_argument(
        "--render-upscale",
        type=int,
        default=6,
        help="How much to upscale the observation for saving (integer factor).",
    )
    parser.add_argument(
        "--tasks-per-sequence",
        type=int,
        default=10,
        help="Number of consecutive tasks to execute before exiting.",
    )
    parser.add_argument(
        "--tasks-per-reset",
        type=int,
        default=5,
        help="Number of completed tasks before forcing an env reset during evaluation.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int = 0) -> SimpleGridImageEnv:
    env = SimpleGridImageEnv(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_task_prob=args.clear_task_prob,
    )
    env.reset(seed=seed)
    return env


def load_policy(env: SimpleGridImageEnv, args: argparse.Namespace) -> ConvQNetwork:
    obs_shape = env.observation_space.shape
    policy = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n)
    state_dict = torch.load(args.state_dict, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def valid_action_mask(env: SimpleGridImageEnv) -> np.ndarray:
    mask = np.ones(env.action_space.n, dtype=bool)
    ax, ay = env.state.agent
    grid_max = env.grid_size - 1
    if ax <= 0:
        mask[SimpleGridImageEnv.MOVE_LEFT] = False
    if ax >= grid_max:
        mask[SimpleGridImageEnv.MOVE_RIGHT] = False
    if ay <= 0:
        mask[SimpleGridImageEnv.MOVE_UP] = False
    if ay >= grid_max:
        mask[SimpleGridImageEnv.MOVE_DOWN] = False
    on_receptacle = any(env.state.agent in tiles for tiles in env.receptacles.values())
    mask[SimpleGridImageEnv.PLACE] = bool(env.state.carrying) and on_receptacle
    can_pick = any(
        coord == env.state.agent
        and obj in env.active_objects
        and obj not in env.state.carrying
        for obj, coord in env.state.objects.items()
    )
    mask[SimpleGridImageEnv.PICK] = can_pick
    return mask


def render_observation(
    obs: np.ndarray,
    task_label: str,
    scale: int,
    header: Optional[str] = None,
) -> Image.Image:
    rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), resample=Image.NEAREST
    )
    banner_lines: List[str] = []
    if header:
        banner_lines.append(header)
    banner_lines.append(f"{task_label}  (yellow=target obj, red=goal)")
    # Compute banner height (12 px per line with small padding)
    banner_height = 12 * len(banner_lines)
    if img.width > 0:
        banner = Image.new("RGB", (img.width, banner_height), color=(30, 30, 30))
        combined = Image.new("RGB", (img.width, img.height + banner_height))
        combined.paste(banner, (0, 0))
        combined.paste(img, (0, banner_height))
        draw = ImageDraw.Draw(combined)
        for idx, line in enumerate(banner_lines):
            draw.text((2, 1 + idx * 12), line, fill=(255, 255, 255))
        return combined
    return img


def format_task_label(task: SampledTask) -> str:
    if task.task_type == "clear":
        return f"CLEAR {task.receptacle_name}"
    return f"{task.object_name} → {task.receptacle_name}"


def weighted_choice(
    distribution: Mapping[str, float],
    candidates: Sequence[str],
    rng: np.random.Generator,
) -> str:
    if not candidates:
        raise ValueError("Cannot sample from an empty candidate list.")
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
        return []
    tasks: List[SampledTask] = []
    last_rec: Optional[str] = None
    active_objects = list(env.active_objects)
    receptacles = list(env.receptacle_names)
    clear_prob = getattr(env, "clear_task_prob", 0.0)
    for _ in range(num_tasks):
        if clear_prob > 0.0 and rng.random() < clear_prob:
            rec_choices = _eligible_receptacles(receptacles, last_rec)
            rec = weighted_choice(SURFACE_DISTRIBUTION, rec_choices, rng)
            tasks.append(SampledTask("clear", None, rec))
            last_rec = rec
            continue
        obj = weighted_choice(OBJECT_DISTRIBUTION, active_objects, rng)
        source_dist = OBJECT_SOURCE_DISTRIBUTION.get(obj, SURFACE_DISTRIBUTION)
        rec_choices = _eligible_receptacles(receptacles, last_rec)
        rec = weighted_choice(source_dist, rec_choices, rng)
        tasks.append(SampledTask("move", obj, rec))
        last_rec = rec
    return tasks


def apply_sampled_task(env: SimpleGridImageEnv, task: SampledTask) -> Tuple[np.ndarray, dict]:
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


def main():
    args = parse_args()
    env = make_env(args, seed=args.seed)
    policy = load_policy(env, args)
    obs, info = env.reset(seed=args.seed)
    frames: List[Image.Image] = []
    action_rng = np.random.default_rng(42)
    task_rng = np.random.default_rng(args.seed + 1)
    sampled_tasks = sample_task_sequence(env, args.tasks_per_sequence, task_rng)
    if not sampled_tasks:
        print("No tasks sampled; set --tasks-per-sequence > 0 to evaluate the policy.")
        return

    obs, info = apply_sampled_task(env, sampled_tasks[0])
    tasks_since_reset = 0
    current_task_idx = 0
    current_task_steps = 0
    current_task_start = 0
    task_frame_ranges: List[Tuple[int, int, int]] = []
    task_history: List[SampledTask] = []
    task_results: List[dict] = []
    successes = 0

    progress = tqdm(total=args.max_steps, desc="Inference rollout")
    for step in range(args.max_steps):
        current_task = sampled_tasks[current_task_idx]
        label = format_task_label(current_task)
        header = f"Task {current_task_idx + 1}/{len(sampled_tasks)}"
        frames.append(
            render_observation(
                obs,
                label,
                args.render_upscale,
                header=header,
            )
        )
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy(obs_tensor).squeeze(0).numpy()
        mask = valid_action_mask(env)
        masked_q = q_values.copy()
        masked_q[~mask] = -np.inf
        greedy_action = int(masked_q.argmax()) if mask.any() else env.action_space.sample()
        if args.random_action_prob > 0.0 and action_rng.random() < args.random_action_prob:
            action = env.action_space.sample()
        else:
            action = greedy_action

        obs, reward, success, horizon, info = env.step(action)
        progress.update(1)
        current_task_steps += 1

        if success or horizon:
            outcome = "COMPLETED" if success else "FAILED"
            header = f"{outcome} • Task {current_task_idx + 1}/{len(sampled_tasks)}"
            frames.append(
                render_observation(
                    obs,
                    label,
                    args.render_upscale,
                    header=header,
                )
            )
            task_frame_ranges.append((current_task_idx + 1, current_task_start, len(frames) - 1))
            task_history.append(current_task)
            if success:
                successes += 1
            task_results.append(
                {
                    "task": current_task,
                    "task_type": current_task.task_type,
                    "success": bool(success),
                    "steps": current_task_steps,
                }
            )
            current_task_idx += 1
            tasks_since_reset += 1
            current_task_steps = 0
            current_task_start = len(frames)

            if current_task_idx >= len(sampled_tasks):
                break

            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                obs, info = env.reset()
                tasks_since_reset = 0

            obs, info = apply_sampled_task(env, sampled_tasks[current_task_idx])

    progress.close()

    attempted = len(task_results)
    if attempted < len(sampled_tasks):
        print(
            f"Step budget exhausted before attempting all sampled tasks "
            f"({attempted}/{len(sampled_tasks)} attempted). Increase --max-steps if needed."
        )
    if attempted > 0:
        success_rate = successes / attempted
        print(f"Policy success rate over sampled tasks: {successes}/{attempted} ({success_rate:.1%})")
    else:
        print("No sampled tasks were attempted within the allotted step budget.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(args.output_dir / f"image_frame_{idx:04d}.png")
    output_path = args.output_dir.resolve()
    print(
        f"Saved {len(frames)} frames to {output_path} "
        f"(attempted {attempted}/{len(sampled_tasks)} sampled tasks)"
    )
    for task_idx, start_idx, end_idx in task_frame_ranges:
        print(f"Task {task_idx}: frames {start_idx}–{end_idx}")
    if not task_history:
        print("No sampled task attempts recorded; distribution unavailable.")


if __name__ == "__main__":
    main()
