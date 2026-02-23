"""Simple DQN trainer for the 3x3 single-object gridworld."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.simple_grid_env import SimpleGridEnv, OBJECT_NAMES


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if max_prio == 0.0:
            max_prio = 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.atleast_1d(priorities)
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(np.abs(prio) + self.eps)


def train(args: argparse.Namespace) -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curriculum_enabled = args.num_objects > 1 and args.curriculum_single_steps > 0
    initial_objects = 1 if curriculum_enabled else args.num_objects
    env = SimpleGridEnv(
        success_reward=args.success_reward,
        num_objects=initial_objects,
        grid_size=4,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
    )
    curriculum_switch = min(args.curriculum_single_steps, args.total_steps) if curriculum_enabled else None
    curriculum_applied = not curriculum_enabled
    replay = PrioritizedReplayBuffer(args.replay_size)

    state, _ = env.reset(seed=args.seed)
    input_dim = len(state)
    q_net = QNetwork(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = QNetwork(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)

    global_step = 0
    returns: Deque[float] = deque(maxlen=50)
    task_lengths: Deque[int] = deque(maxlen=50)
    tasks_completed = 0
    total_tasks = 0
    step_rewards: List[float] = []
    episode_returns: List[float] = []
    task_length_history: List[int] = []
    tasks_since_reset = 0

    progress = tqdm(total=args.total_steps, desc="SimpleGrid DQN")
    state, _ = env.reset(seed=args.seed)
    task_return = 0.0
    task_steps = 0

    def current_epsilon(step: int) -> float:
        if args.epsilon is not None:
            return args.epsilon
        frac = min(1.0, step / max(1, args.epsilon_decay))
        return args.epsilon_start + frac * (args.epsilon_final - args.epsilon_start)

    while global_step < args.total_steps:
        eps = current_epsilon(global_step)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                inp = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(q_net(inp), dim=1).item())

        obs, reward, success, horizon, info = env.step(action)
        done = success or horizon
        next_state = obs
        replay.push(Transition(state, action, reward, next_state, done))
        step_rewards.append(reward)

        state = next_state
        task_return += reward
        task_steps += 1
        global_step += 1
        progress.update(1)

        if success or horizon:
            tasks_since_reset += 1
            returns.append(task_return)
            task_lengths.append(task_steps)
            episode_returns.append(task_return)
            task_length_history.append(task_steps)
            total_tasks += 1
            if success:
                tasks_completed += 1
            task_return = 0.0
            task_steps = 0
            avg_ret = f"{np.mean(returns):.1f}" if returns else "n/a"
            avg_len = float(np.mean(task_lengths)) if task_lengths else 0.0
            success_rate = tasks_completed / total_tasks if total_tasks > 0 else 0.0
            progress.set_postfix(
                ret=avg_ret,
                steps=f"{avg_len:.1f}" if avg_len else "n/a",
                success=f"{success_rate:.2f}",
                eps=f"{eps:.2f}",
                tasks=tasks_completed,
            )
            if tasks_since_reset >= args.tasks_per_reset:
                state, _ = env.reset()
                tasks_since_reset = 0
            # no epsilon schedule; fixed exploration rate

        if not curriculum_applied and global_step >= curriculum_switch:
            env.set_active_objects(args.num_objects)
            state, _ = env.reset()
            task_return = 0.0
            task_steps = 0
            curriculum_applied = True
            tasks_since_reset = 0

        if len(replay) >= args.batch_size:
            beta = min(
                1.0,
                args.per_beta_start
                + (1.0 - args.per_beta_start) * (global_step / max(1, args.total_steps)),
            )
            batch, indices, weights = replay.sample(args.batch_size, beta=beta)
            states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.tensor(
                np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device
            )
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target = rewards + args.gamma * (1.0 - dones) * next_q

            td_errors = target - q_values
            loss = (weights_t * td_errors.pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            replay.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())

        if global_step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    progress.close()
    torch.save(q_net.state_dict(), args.output)
    print(f"Saved DQN weights to {args.output}")
    _plot_metrics(step_rewards, episode_returns, task_length_history, args.output)


def _plot_metrics(
    step_rewards: List[float],
    episode_returns: List[float],
    task_lengths: List[int],
    weight_path: Path,
) -> None:
    if not step_rewards:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_metrics.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(step_rewards, linewidth=0.5)
    axes[0].set_title("Per-step reward")
    axes[0].set_ylabel("Reward")
    axes[1].plot(episode_returns, linewidth=0.8)
    axes[1].set_title("Per-task return")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved training curves to {plot_path}")
    if task_lengths:
        _plot_avg_task_steps(task_lengths, weight_path)


def _plot_avg_task_steps(task_lengths: List[int], weight_path: Path, window: int = 100) -> None:
    if not task_lengths:
        return
    avg_steps = []
    for idx in range(1, len(task_lengths) + 1):
        start = max(0, idx - window)
        avg_steps.append(float(np.mean(task_lengths[start:idx])))
    plot_path = weight_path.with_name(weight_path.stem + "_task_lengths.png")
    plt.figure(figsize=(10, 4))
    plt.plot(avg_steps, linewidth=0.8)
    plt.title(f"Rolling Avg Task Length (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved avg task length plot to {plot_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on the NxN SimpleGrid pick and place task.")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0, help="Scale for distance-based shaping reward.")
    parser.add_argument("--num-objects", type=int, default=2, help="Number of objects spawned in the grid.")
    parser.add_argument(
        "--curriculum-single-steps",
        type=int,
        default=0,
        help="Train with a single object for this many steps before activating all objects (requires num_objects>1).",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Prioritized replay exponent alpha.")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="Initial importance-sampling exponent beta.")
    parser.add_argument("--per-eps", type=float, default=1e-5, help="Stability epsilon added to TD errors.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="(Optional) Fixed epsilon for epsilon-greedy exploration. Overrides the schedule.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon when using the decay schedule (ignored if --epsilon is set).",
    )
    parser.add_argument(
        "--epsilon-final",
        type=float,
        default=0.05,
        help="Final epsilon once decay is finished (ignored if --epsilon is set).",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=50_000,
        help="Number of global steps to decay epsilon from start to final (ignored if --epsilon is set).",
    )
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / "simple_grid_dqn.pt")
    parser.add_argument("--success-reward", type=float, default=10.0, help="Reward bonus when task is completed.")
    parser.add_argument(
        "--tasks-per-reset",
        type=int,
        default=1,
        help="Number of completed tasks before forcing an environment reset (1 = reset every task).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
