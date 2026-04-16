import json
from pathlib import Path

import numpy as np
import torch

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork, VectorEnv
from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv

ROOT = Path('/Users/raghav/raghav/anticipatory_rl')
CONFIG = ROOT / 'anticipatory_rl/configs/config_5x5_3r4o.yaml'
CHECKPOINTS = {
    'myopic': ROOT / 'runs/5_myopic_image_dqn_tpr1/simple_grid_image_dqn.pt',
    'anticipatory': ROOT / 'runs/5_anticipatory_image_dqn_tpr200/simple_grid_image_dqn.pt',
}

NUM_ENVS = 16
TASKS_PER_SETTING = 2000
ENV_RESET_TASKS = 200
SEED = 0

torch.set_num_threads(1)

def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_env():
    return SimpleGridImageEnv(
        grid_size=5,
        max_task_steps=200,
        success_reward=12.0,
        num_objects=4,
        distance_reward=True,
        distance_reward_scale=1.0,
        clear_receptacle_shaping_scale=3.0,
        clear_task_prob=0.5,
        ensure_receptacle_coverage=True,
        config_path=CONFIG,
    )


def evaluate(label, ckpt_path, epsilon):
    print(f'START {label} eps={epsilon}', flush=True)
    device = torch.device('cpu')
    env = VectorEnv(make_env, NUM_ENVS)
    state, infos = env.reset(seed=SEED)
    infos = list(infos)

    q_net = ConvQNetwork(state.shape[1:], hidden_dim=256, num_actions=env.action_space.n).to(device)
    q_net.load_state_dict(load_state_dict(ckpt_path, device))
    q_net.eval()

    rng = np.random.default_rng(SEED)
    pick_action = SimpleGridImageEnv.PICK
    place_action = SimpleGridImageEnv.PLACE
    fallback_actions = np.array([
        SimpleGridImageEnv.MOVE_UP,
        SimpleGridImageEnv.MOVE_DOWN,
        SimpleGridImageEnv.MOVE_LEFT,
        SimpleGridImageEnv.MOVE_RIGHT,
    ], dtype=np.int64)

    successes = 0
    attempted = 0
    auto_attempted = 0
    auto_successes = 0
    active_attempted = 0
    active_successes = 0
    primitive_steps = 0

    tasks_since_reset = np.zeros(NUM_ENVS, dtype=np.int64)
    task_steps = np.zeros(NUM_ENVS, dtype=np.int64)
    task_returns = np.zeros(NUM_ENVS, dtype=np.float64)
    total_task_steps = 0
    total_task_return = 0.0
    current_auto = np.array([bool(info.get('next_auto_satisfied', False)) for info in infos], dtype=bool)

    while attempted < TASKS_PER_SETTING:
        invalid_pick_mask = np.array([not bool(info.get('can_pick', False)) for info in infos], dtype=bool)
        invalid_place_mask = np.array([not bool(info.get('can_place', False)) for info in infos], dtype=bool)

        with torch.no_grad():
            inp = torch.tensor(state, dtype=torch.float32, device=device)
            q_vals = q_net(inp)
            if invalid_pick_mask.any():
                q_vals[torch.from_numpy(invalid_pick_mask).to(device=device), pick_action] = float('-inf')
            if invalid_place_mask.any():
                q_vals[torch.from_numpy(invalid_place_mask).to(device=device), place_action] = float('-inf')
            greedy_actions = torch.argmax(q_vals, dim=1).cpu().numpy()

        if epsilon > 0:
            random_mask = rng.random(NUM_ENVS) < epsilon
            random_actions = rng.integers(0, env.action_space.n, size=NUM_ENVS, dtype=np.int64)
            actions = np.where(random_mask, random_actions, greedy_actions).astype(np.int64)
            invalid_random_pick = invalid_pick_mask & (actions == pick_action)
            if invalid_random_pick.any():
                actions[invalid_random_pick] = rng.choice(fallback_actions, size=int(invalid_random_pick.sum()))
            invalid_random_place = invalid_place_mask & (actions == place_action)
            if invalid_random_place.any():
                actions[invalid_random_place] = rng.choice(fallback_actions, size=int(invalid_random_place.sum()))
        else:
            actions = greedy_actions.astype(np.int64)

        next_state, reward, success, truncated, next_infos = env.step(actions)
        next_infos = list(next_infos)
        primitive_steps += NUM_ENVS
        task_steps += 1
        task_returns += reward

        ended = success | truncated
        for idx in np.where(ended)[0]:
            attempted += 1
            total_task_steps += int(task_steps[idx])
            total_task_return += float(task_returns[idx])
            if success[idx]:
                successes += 1
                tasks_since_reset[idx] += 1
            if current_auto[idx]:
                auto_attempted += 1
                if success[idx]:
                    auto_successes += 1
            else:
                active_attempted += 1
                if success[idx]:
                    active_successes += 1

            task_steps[idx] = 0
            task_returns[idx] = 0.0

            if truncated[idx] or tasks_since_reset[idx] >= ENV_RESET_TASKS:
                new_obs, new_info = env.reset_env(idx)
                next_state[idx] = new_obs
                next_infos[idx] = new_info
                tasks_since_reset[idx] = 0
            current_auto[idx] = bool(next_infos[idx].get('next_auto_satisfied', False))

            if attempted >= TASKS_PER_SETTING:
                break

        state = next_state
        infos = next_infos

    result = {
        'label': label,
        'epsilon': epsilon,
        'num_tasks': attempted,
        'success_rate': successes / attempted,
        'avg_task_steps': total_task_steps / attempted,
        'avg_task_return': total_task_return / attempted,
        'auto_rate': auto_attempted / attempted,
        'auto_success_rate': (auto_successes / auto_attempted) if auto_attempted else None,
        'active_success_rate': (active_successes / active_attempted) if active_attempted else None,
        'active_attempted': active_attempted,
        'auto_attempted': auto_attempted,
        'primitive_steps': primitive_steps,
    }
    print(json.dumps(result), flush=True)
    return result

results = []
for label, ckpt in CHECKPOINTS.items():
    for epsilon in (0.05, 0.0):
        results.append(evaluate(label, ckpt, epsilon))

print('FINAL_RESULTS', flush=True)
print(json.dumps(results, indent=2), flush=True)
