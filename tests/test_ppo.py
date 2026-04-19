"""PPO inference script mirroring test.py for SB3 models."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv, SimpleGridState


MODEL_PATH = Path("runs/simple_grid_ppo.zip")
OUTPUT_DIR = Path("runs/ppo_frames")
GRID_SIZE = 3
NUM_OBJECTS = 2
SUCCESS_REWARD = 10.0
DISTANCE_REWARD_SCALE = 1.0
TASKS_PER_EPISODE = 5
MAX_TASK_STEPS = 200
MAX_FRAMES = 150


def make_env(seed: int = 0) -> SimpleGridImageEnv:
    env = SimpleGridImageEnv(
        grid_size=GRID_SIZE,
        num_objects=NUM_OBJECTS,
        success_reward=SUCCESS_REWARD,
        correct_pick_bonus=1.0,
        distance_reward=True,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        max_task_steps=MAX_TASK_STEPS,
    )
    env.reset(seed=seed)
    return env


def render_frame(env: SimpleGridImageEnv, *, target_obj: str | None = None, target_rec: str | None = None) -> Image.Image:
    grid = env.grid_size
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_facecolor("white")
    ax.invert_yaxis()

    task_type = getattr(env, "task_type", "move")
    task_obj = target_obj if target_obj is not None else env.target_object
    task_rec = target_rec if target_rec is not None else env.target_receptacle

    for name, tiles in env.receptacles.items():
        color = "lightgreen" if name == task_rec else "lightgray"
        for coord in tiles:
            rect = plt.Rectangle((coord[0] - 0.5, coord[1] - 0.5), 1, 1, color=color, alpha=0.25)
            ax.add_patch(rect)
        centroid_x = sum(tile[0] for tile in tiles) / len(tiles)
        centroid_y = sum(tile[1] for tile in tiles) / len(tiles)
        ax.text(centroid_x, centroid_y, name, ha="center", va="center", fontsize=6, color="darkgreen")

    for obj, coord in env.state.objects.items():
        base_color = "orange"
        if task_type == "move" and obj == task_obj:
            base_color = "red"
        ax.scatter(coord[0], coord[1], s=300, c=base_color, marker="o", edgecolors="black")
        ax.text(coord[0], coord[1], obj, color="white", ha="center", va="center", fontsize=6)

    ax.scatter(env.state.agent[0], env.state.agent[1], c="blue", s=300, marker="*", edgecolors="black")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    image = np.asarray(renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(image)


def reset_env(env: SimpleGridImageEnv, *, seed: int | None = None) -> Tuple[np.ndarray, dict]:
    return env.reset(seed=seed)


def main() -> None:
    env = make_env()
    model = PPO.load(str(MODEL_PATH), device="cpu")
    obs, info = reset_env(env)
    import pdb; pdb.set_trace()
    frames: list[Image.Image] = []
    tasks_completed = 0

    for step in range(MAX_FRAMES):
        frames.append(render_frame(env))
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step}: Action taken: {action}")
        old_task_type = env.task_type
        old_target_obj = env.target_object
        old_target_rec = env.target_receptacle
        obs, reward, success, horizon, info = env.step(int(action))

        if success:
            tasks_completed += 1
            frames.append(
                render_frame(
                    env,
                    target_obj=old_target_obj,
                    target_rec=old_target_rec,
                )
            )
            if old_task_type == "clear":
                print(f"Task {tasks_completed}: cleared {old_target_rec} at step {step}")
            else:
                print(f"Task {tasks_completed}: moved {old_target_obj} to {old_target_rec} at step {step}")
            if tasks_completed >= TASKS_PER_EPISODE:
                break
        if horizon:
            print("Task horizon reached. Resetting task.")
            obs, info = reset_env(env)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_path = OUTPUT_DIR / f"frame_{idx:04d}.png"
        frame.save(frame_path)
    print(f"Saved {len(frames)} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
