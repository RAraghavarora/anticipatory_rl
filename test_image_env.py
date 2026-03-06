import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from collections import Counter
from tqdm import tqdm

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork
from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv


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
    parser.add_argument("--num-objects", type=int, default=2, help="Number of active objects.")
    parser.add_argument("--success-reward", type=float, default=10.0, help="Success reward in env.")
    parser.add_argument(
        "--distance-reward-scale",
        type=float,
        default=1.0,
        help="Scaling for distance shaping.",
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
    carrying = env.state.carrying
    if carrying is None:
        mask[SimpleGridImageEnv.PLACE] = False
        can_pick = any(
            coord == env.state.agent and obj in env.active_objects
            for obj, coord in env.state.objects.items()
        )
        mask[SimpleGridImageEnv.PICK] = can_pick
    else:
        mask[SimpleGridImageEnv.PICK] = False
    return mask


def render_observation(obs: np.ndarray, task_label: str, scale: int) -> Image.Image:
    rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), resample=Image.NEAREST
    )
    banner_height = 12
    if img.width > 0:
        banner = Image.new("RGB", (img.width, banner_height), color=(30, 30, 30))
        combined = Image.new("RGB", (img.width, img.height + banner_height))
        combined.paste(banner, (0, 0))
        combined.paste(img, (0, banner_height))
        draw = ImageDraw.Draw(combined)
        draw.text((2, 1), f"{task_label}  (yellow=target obj, red=goal)", fill=(255, 255, 255))
        return combined
    return img


def main():
    args = parse_args()
    env = make_env(args, seed=args.seed)
    policy = load_policy(env, args)
    obs, info = env.reset()
    frames = []
    rng = np.random.default_rng(42)

    tasks_finished = 0
    task_frame_ranges: list[tuple[int, int, int]] = []
    current_task_start = 0
    tasks_since_reset = 0
    task_history: list[tuple[str, str]] = []
    progress = tqdm(total=args.max_steps, desc="Inference rollout")
    for step in range(args.max_steps):
        task_label = f"{env.target_object} → {env.target_receptacle}"
        frames.append(render_observation(obs, task_label, args.render_upscale))
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy(obs_tensor).squeeze(0).numpy()
        mask = valid_action_mask(env)
        masked_q = q_values.copy()
        masked_q[~mask] = -np.inf
        greedy_action = int(masked_q.argmax()) if mask.any() else env.action_space.sample()
        if args.random_action_prob > 0.0 and rng.random() < args.random_action_prob:
            action = env.action_space.sample()
        else:
            action = greedy_action
        prev_target_obj = env.target_object
        prev_target_rec = env.target_receptacle
        obs, reward, success, horizon, info = env.step(action)
        progress.update(1)
        if success:
            tasks_finished += 1
            tasks_since_reset += 1
            success_label = (
                f"COMPLETED {prev_target_obj} → {prev_target_rec} "
                f"({tasks_finished}/{args.tasks_per_sequence})"
            )
            frames.append(render_observation(obs, success_label, args.render_upscale))
            task_frame_ranges.append(
                (tasks_finished, current_task_start, len(frames) - 1)
            )
            task_history.append((prev_target_obj, prev_target_rec))
            current_task_start = len(frames)
            if tasks_finished >= args.tasks_per_sequence:
                break
            if tasks_since_reset >= args.tasks_per_reset:
                obs, info = env.reset()
                tasks_since_reset = 0
                current_task_start = len(frames)
        elif horizon:
            obs, info = env.reset()
            tasks_since_reset = 0
            current_task_start = len(frames)
    progress.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(args.output_dir / f"image_frame_{idx:04d}.png")
    output_path = args.output_dir.resolve()
    print(
        f"Saved {len(frames)} frames to {output_path} "
        f"(completed {tasks_finished}/{args.tasks_per_sequence} tasks)"
    )
    for task_idx, start_idx, end_idx in task_frame_ranges:
        print(f"Task {task_idx}: frames {start_idx}–{end_idx}")
    if task_history:
        pair_counts = Counter(task_history)
        total = sum(pair_counts.values())
        print("Estimated task distribution (top 5 object → receptacle pairs):")
        for (obj, rec), count in pair_counts.most_common(5):
            prob = count / total
            print(f"  {obj} → {rec}: {count} ({prob:.2%})")
        object_counts = Counter(obj for obj, _ in task_history)
        next_obj = object_counts.most_common(1)[0][0]
        rec_counts = Counter(rec for obj, rec in task_history if obj == next_obj)
        next_rec = rec_counts.most_common(1)[0][0]
        print(f"Anticipated next task: move {next_obj} to {next_rec}")
    else:
        print("No task successes recorded; distribution unavailable.")


if __name__ == "__main__":
    main()
