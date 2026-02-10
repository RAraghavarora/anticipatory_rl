"""Task-conditioned Q-learning agent for the MiniWorld rearrangement task.

This script encodes both the environment state and a symbolic task vector, then
trains a DQN-style agent that learns a single Q(s, τ, a) over the augmented
state-task space. Use the CLI below to launch training with default settings.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.miniworld_env import (
    MiniWorldGridRearrange,
    OBJECTS,
    RECEPTACLES,
)


# --------------------------------------------------------------------------- #
# Encoding utilities                                                          #
# --------------------------------------------------------------------------- #
GRID_SIZE = 6
GRID_CELLS = GRID_SIZE * GRID_SIZE
OBJECT_NAMES = [spec.name for spec in OBJECTS]
RECEPTACLE_NAMES = [spec.name for spec in RECEPTACLES]
OBJECT_INDEX = {name: idx for idx, name in enumerate(OBJECT_NAMES)}
RECEPTACLE_INDEX = {name: idx for idx, name in enumerate(RECEPTACLE_NAMES)}
LOCATION_DIM = GRID_CELLS + len(RECEPTACLE_NAMES) + 1  # +1 for gripper slot
NUM_OBJECTS = len(OBJECT_NAMES)
NUM_RECEPTACLES = len(RECEPTACLE_NAMES)
TASK_TYPE_INDEX = {"bring_single": 0, "bring_pair": 1, "clear_receptacle": 2}


def _tile_to_index(tile: Tuple[int, int]) -> int:
    x, y = tile
    return y * GRID_SIZE + x


def encode_state(state: Dict[str, object]) -> np.ndarray:
    """Encode the environment state into a flat feature vector."""
    features: List[float] = []

    agent_x, agent_y = state["agent"]["grid_pos"]
    features.append(agent_x / (GRID_SIZE - 1))
    features.append(agent_y / (GRID_SIZE - 1))

    holding = state["agent"]["carrying"]
    holding_vec = np.zeros(NUM_OBJECTS + 1, dtype=np.float32)
    if holding is None:
        holding_vec[-1] = 1.0
    else:
        holding_vec[OBJECT_INDEX[holding]] = 1.0
    features.extend(holding_vec.tolist())

    for obj_name in OBJECT_NAMES:
        obj_data = state["objects"][obj_name]
        vec = np.zeros(LOCATION_DIM, dtype=np.float32)
        if state["agent"]["carrying"] == obj_name:
            vec[-1] = 1.0  # gripper slot
        elif obj_data["tile"] is not None:
            vec[_tile_to_index(tuple(obj_data["tile"]))] = 1.0
        elif obj_data["region"] is not None:
            offset = GRID_CELLS + RECEPTACLE_INDEX[obj_data["region"]]
            vec[offset] = 1.0
        else:
            raise ValueError(f"Object {obj_name} has no location info.")
        features.extend(vec.tolist())

    return np.asarray(features, dtype=np.float32)


def encode_task(task: Dict[str, object]) -> np.ndarray:
    """Encode a symbolic task into a fixed-length vector."""
    vec = np.zeros(
        3 + 2 * NUM_OBJECTS + 3 * NUM_RECEPTACLES, dtype=np.float32
    )
    task_type = task["task_type"]
    vec[TASK_TYPE_INDEX[task_type]] = 1.0

    objects = task["payload"].get("objects", [])
    for idx, obj_name in enumerate(objects[:2]):
        base = 3 + idx * NUM_OBJECTS
        vec[base + OBJECT_INDEX[obj_name]] = 1.0

    sources = task["payload"].get("sources")
    if sources is None and task_type == "clear_receptacle":
        sources = [task["payload"]["source"]]
    if sources:
        for idx, receptacle in enumerate(sources[:2]):
            base = 3 + 2 * NUM_OBJECTS + idx * NUM_RECEPTACLES
            vec[base + RECEPTACLE_INDEX[receptacle]] = 1.0

    target = task["payload"]["target"]
    target_offset = 3 + 2 * NUM_OBJECTS + 2 * NUM_RECEPTACLES
    vec[target_offset + RECEPTACLE_INDEX[target]] = 1.0
    return vec


def build_augmented_state(state: Dict[str, object], task: Dict[str, object]) -> np.ndarray:
    return np.concatenate([encode_state(state), encode_task(task)])


def task_goal_satisfied(state: Dict[str, object], task: Dict[str, object]) -> bool:
    """Check whether the current environment state satisfies task τ."""
    target = task["payload"]["target"]
    objects = task["payload"].get("objects", [])
    task_type = task["task_type"]

    if task_type == "clear_receptacle":
        objects = task["payload"]["objects"]

    for obj in objects:
        region = state["objects"][obj]["region"]
        if region != target:
            return False
    return True


# --------------------------------------------------------------------------- #
# Q-network and replay buffer                                                 #
# --------------------------------------------------------------------------- #
class TaskConditionedQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 6)  # six discrete actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.memory, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.asarray([t.done for t in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def load_tasks(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text())


def epsilon_by_frame(frame_idx: int, eps_start: float, eps_final: float, eps_decay: int) -> float:
    return eps_final + (eps_start - eps_final) * np.exp(-1.0 * frame_idx / eps_decay)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_task(tasks: Sequence[Dict[str, object]], rng: random.Random) -> Tuple[int, Dict[str, object]]:
    idx = rng.randrange(len(tasks))
    return idx, tasks[idx]


def train(args: argparse.Namespace) -> None:
    device = _select_device()
    env = MiniWorldGridRearrange(render_mode=None)
    env.reset(seed=args.seed)
    tasks = load_tasks(args.tasks_file)
    task_rng = random.Random(args.task_seed if args.task_seed is not None else args.seed)
    current_task_idx, task = _sample_task(tasks, task_rng)

    state = env.get_state()
    aug_state = build_augmented_state(state, task)

    state_dim = aug_state.shape[0]
    q_net = TaskConditionedQNetwork(state_dim, args.hidden_dim).to(device)
    target_net = TaskConditionedQNetwork(state_dim, args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)
    log_file = args.log_path.open("w") if args.log_path else None

    global_step = 0
    episode_return = 0.0
    steps_since_task = 0

    completed_tasks = 0
    successful_tasks = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[int] = deque(maxlen=100)
    recent_lengths: Deque[int] = deque(maxlen=100)

    progress = tqdm(total=args.train_steps, desc="Training steps")
    while global_step < args.train_steps:
        epsilon = epsilon_by_frame(global_step, args.eps_start, args.eps_final, args.eps_decay)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(aug_state, device=device).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        obs, _, _, trunc, info = env.step(action)
        next_state = info["state"]

        reward = -1.0
        done = task_goal_satisfied(next_state, task)
        if done:
            reward = args.goal_reward

        next_aug_state = build_augmented_state(next_state, task)

        replay.push(
            Transition(aug_state, action, reward, next_aug_state, done)
        )

        aug_state = next_aug_state
        state = next_state
        episode_return += reward
        steps_since_task += 1
        global_step += 1

        progress.update(1)
        if done or steps_since_task >= args.max_task_steps or trunc:
            completed_tasks += 1
            success = bool(done)
            successful_tasks += int(success)
            recent_returns.append(episode_return)
            recent_successes.append(1 if success else 0)
            recent_lengths.append(steps_since_task)

            if log_file:
                log_file.write(
                    json.dumps(
                        {
                            "task_index": current_task_idx,
                            "task_type": task["task_type"],
                            "success": success,
                            "steps": steps_since_task,
                            "reward": episode_return,
                            "global_step": global_step,
                        }
                    )
                    + "\n"
                )
                log_file.flush()

            success_rate = (
                float(np.mean(recent_successes)) if recent_successes else 0.0
            )
            avg_return = (
                float(np.mean(recent_returns)) if recent_returns else 0.0
            )
            avg_length = (
                float(np.mean(recent_lengths)) if recent_lengths else 0.0
            )
            progress.set_postfix(
                tasks=completed_tasks,
                success=f"{success_rate:.2f}",
                ret=f"{avg_return:.1f}",
                length=f"{avg_length:.0f}",
            )

            env.reset()
            current_task_idx, task = _sample_task(tasks, task_rng)
            state = env.get_state()
            aug_state = build_augmented_state(state, task)
            steps_since_task = 0
            episode_return = 0.0

        if len(replay) >= args.batch_size:
            batch = replay.sample(args.batch_size)
            states = torch.tensor(batch.state, dtype=torch.float32, device=device)
            actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target = rewards + args.gamma * (1.0 - dones) * next_q

            loss = F.smooth_l1_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

        if global_step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    progress.close()
    if log_file:
        log_file.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a task-conditioned DQN agent.")
    parser.add_argument("--tasks-file", type=Path, default=Path("runs") / "tasks_200.json")
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-final", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=20_000)
    parser.add_argument("--goal-reward", type=float, default=10.0)
    parser.add_argument("--max-task-steps", type=int, default=1500)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=None, help="Seed for task sampling order (defaults to --seed).")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional JSONL log for per-task costs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
