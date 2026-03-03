import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from anticipatory_rl.envs.simple_grid_env import SimpleGridEnv
from anticipatory_rl.agents.simple_grid_dqn import QNetwork

OUTPUT_DIR = Path("runs/dqn_frames")
STATE_DICT = Path("runs/6_dqn/simple_grid_dqn.pt")
HIDDEN_DIM = 192
GRID_SIZE = 6
NUM_OBJECTS = 2
FORCE_FIXED_START = False
AGENT_SPAWN = (1, 1)
OBJECT_UNDER_AGENT = True
# Keep evaluation environment identical to the training run.
SUCCESS_REWARD = 10.0
DISTANCE_REWARD_SCALE = 0.7
# Low epsilon encourages recovery if the greedy policy lands on an out-of-distribution
# state; masking handles obviously invalid moves.
RANDOM_ACTION_PROB = 0.05


def make_env(seed=44):
    env = SimpleGridEnv(
        grid_size=GRID_SIZE,
        num_objects=NUM_OBJECTS,
        success_reward=SUCCESS_REWARD,
        correct_pick_bonus=1.0,
        distance_reward=True,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
    )
    env.reset(seed=seed)
    return env


def load_policy(env):
    input_dim = env.observation_space.shape[0]
    policy = QNetwork(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    state_dict = torch.load(STATE_DICT, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def render_frame(env, target_object_override=None, target_receptacle_override=None):
    grid = env.grid_size
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_facecolor("white")
    ax.invert_yaxis()

    target_obj = target_object_override if target_object_override is not None else env.target_object
    target_rec = target_receptacle_override if target_receptacle_override is not None else env.target_receptacle

    for name, coord in env.receptacles.items():
        color = "lightgreen" if name == target_rec else "lightgray"
        rect = plt.Rectangle((coord[0]-0.5, coord[1]-0.5), 1, 1, color=color, alpha=0.35)
        ax.add_patch(rect)
        ax.text(coord[0], coord[1], name, ha="center", va="center", fontsize=6, color="darkgreen")

    for obj, coord in env.state.objects.items():
        color = "red" if obj == target_obj else "orange"
        ax.scatter(coord[0], coord[1], s=300, c=color, marker="o", edgecolors="black")
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


def reset_env(env: SimpleGridEnv, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
    if FORCE_FIXED_START:
        options = {"agent_pos": AGENT_SPAWN}
        if OBJECT_UNDER_AGENT:
            options["object_under_agent"] = True
        return env.reset(seed=seed, options=options)
    return env.reset(seed=seed)


def valid_action_mask(env: SimpleGridEnv) -> np.ndarray:
    mask = np.ones(env.action_space.n, dtype=bool)
    ax, ay = env.state.agent
    grid_max = env.grid_size - 1
    if ax <= 0:
        mask[SimpleGridEnv.MOVE_LEFT] = False
    if ax >= grid_max:
        mask[SimpleGridEnv.MOVE_RIGHT] = False
    if ay <= 0:
        mask[SimpleGridEnv.MOVE_UP] = False
    if ay >= grid_max:
        mask[SimpleGridEnv.MOVE_DOWN] = False

    carrying = env.state.carrying
    if carrying is None:
        mask[SimpleGridEnv.PLACE] = False
        can_pick = any(
            coord == env.state.agent and obj in env.active_objects
            for obj, coord in env.state.objects.items()
        )
        mask[SimpleGridEnv.PICK] = can_pick
    else:
        mask[SimpleGridEnv.PICK] = False

    return mask


env = make_env()
policy = load_policy(env)
obs, info = reset_env(env)
frames = []
max_steps = GRID_SIZE * GRID_SIZE * 5

rng = np.random.default_rng(0)

for _ in range(max_steps):
    frames.append(render_frame(env))
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
    print(f"Action taken: {action}")
    print(q_values, q_values.argmax())
    # Save target values before step (since they get resampled after success)
    old_target_object = env.target_object
    old_target_receptacle = env.target_receptacle
    obs, reward, success, horizon, info = env.step(action)
    if success:
        print("SUCCESS! Rendering final frame with original task colors")
        frames.append(render_frame(env, target_object_override=old_target_object, target_receptacle_override=old_target_receptacle))
        break
    if horizon:
        obs, info = reset_env(env)

print(f"Number of frames: {len(frames)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for idx, frame in enumerate(frames):
    frame_path = OUTPUT_DIR / f"frame_{idx:04d}.png"
    frame.save(frame_path)
print(f"Saved {len(frames)} frames to {OUTPUT_DIR}")
