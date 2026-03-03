import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv
from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork

OUTPUT_DIR = Path("runs/image_dqn_frames")
STATE_DICT = Path("runs/10_image_dqn/simple_grid_image_dqn.pt")
HIDDEN_DIM = 256
GRID_SIZE = 10
NUM_OBJECTS = 2
SUCCESS_REWARD = 10.0
DISTANCE_REWARD_SCALE = 1.0
MAX_STEPS = GRID_SIZE * GRID_SIZE * 6
RANDOM_ACTION_PROB = 0.05


def make_env(seed: int = 0) -> SimpleGridImageEnv:
    env = SimpleGridImageEnv(
        grid_size=GRID_SIZE,
        num_objects=NUM_OBJECTS,
        success_reward=SUCCESS_REWARD,
        distance_reward=True,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
    )
    env.reset(seed=seed)
    return env


def load_policy(env: SimpleGridImageEnv) -> ConvQNetwork:
    obs_shape = env.observation_space.shape
    policy = ConvQNetwork(obs_shape, hidden_dim=HIDDEN_DIM, num_actions=env.action_space.n)
    state_dict = torch.load(STATE_DICT, map_location="cpu")
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


def render_observation(obs: np.ndarray, task_label: str, scale: int = 32) -> Image.Image:
    rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), resample=Image.NEAREST
    )
    draw = ImageDraw.Draw(img)

    def draw_mask_outline(mask: np.ndarray, color: tuple[int, int, int], fill: bool = False):
        ys, xs = np.where(mask > 0.5)
        for x, y in zip(xs, ys):
            x0 = x * scale
            y0 = y * scale
            x1 = x0 + scale - 1
            y1 = y0 + scale - 1
            if fill:
                draw.rectangle([x0, y0, x1, y1], outline=color, fill=color + (60,), width=2)
            else:
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    draw_mask_outline(obs[3], (255, 255, 0))  # target object
    draw_mask_outline(obs[4], (255, 0, 0), fill=True)  # target receptacle area
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
    env = make_env(seed=0)
    policy = load_policy(env)
    obs, info = env.reset()
    frames = []
    rng = np.random.default_rng(42)

    for step in range(MAX_STEPS):
        task_label = f"{env.target_object} → {env.target_receptacle}"
        frames.append(render_observation(obs, task_label))
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy(obs_tensor).squeeze(0).numpy()
        mask = valid_action_mask(env)
        masked_q = q_values.copy()
        masked_q[~mask] = -np.inf
        greedy_action = int(masked_q.argmax()) if mask.any() else env.action_space.sample()
        if RANDOM_ACTION_PROB > 0.0 and rng.random() < RANDOM_ACTION_PROB:
            action = env.action_space.sample()
        else:
            action = greedy_action
        prev_target_obj = env.target_object
        prev_target_rec = env.target_receptacle
        obs, reward, success, horizon, info = env.step(action)
        if success:
            success_label = f"COMPLETED {prev_target_obj} → {prev_target_rec}"
            frames.append(render_observation(obs, success_label))
            break
        if horizon:
            obs, info = env.reset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(OUTPUT_DIR / f"image_frame_{idx:04d}.png")
    print(f"Saved {len(frames)} frames to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
