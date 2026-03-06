import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork
from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render raw observations seen by the image DQN.")
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=Path("runs/10_image_dqn/simple_grid_image_dqn.pt"),
        help="Checkpoint path for ConvQNetwork weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/image_dqn_frames_exact"),
        help="Directory where frames will be written.",
    )
    parser.add_argument("--grid-size", type=int, default=10, help="Environment grid size.")
    parser.add_argument("--num-objects", type=int, default=2, help="Active objects inside the env.")
    parser.add_argument("--success-reward", type=float, default=10.0, help="Success reward.")
    parser.add_argument(
        "--distance-reward-scale",
        type=float,
        default=1.0,
        help="Scaling applied to distance shaping.",
    )
    parser.add_argument(
        "--clear-task-prob",
        type=float,
        default=0.0,
        help="Probability of sampling clear tasks in the environment.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP head size.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum number of primitive steps to capture.",
    )
    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=0.05,
        help="Epsilon used during inference rollouts.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale factor when saving frames (1 = raw).",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> SimpleGridImageEnv:
    env = SimpleGridImageEnv(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_task_prob=args.clear_task_prob,
    )
    env.reset(seed=args.seed)
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


def render_raw_observation(obs: np.ndarray, scale: int) -> Image.Image:
    rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if scale > 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    return img


def main() -> None:
    args = parse_args()
    env = make_env(args)
    policy = load_policy(env, args)
    obs, info = env.reset(seed=args.seed)
    frames: list[Image.Image] = []
    rng = np.random.default_rng(0)

    for step in range(args.max_steps):
        frames.append(render_raw_observation(obs, args.scale))
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

        prev_task_type = getattr(env, "task_type", "move")
        prev_obj = env.target_object
        prev_rec = env.target_receptacle
        obs, reward, success, horizon, info = env.step(action)
        if success:
            frames.append(render_raw_observation(obs, args.scale))
            if prev_task_type == "clear":
                print(f"Cleared {prev_rec} at step {step}")
            else:
                print(f"Completed {prev_obj} → {prev_rec} at step {step}")
            break
        if horizon:
            obs, info = env.reset()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(args.output_dir / f"frame_{idx:04d}.png")
    print(f"Saved {len(frames)} frames to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
