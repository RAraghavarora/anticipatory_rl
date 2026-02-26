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
        grid_size=6,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
    )
    grid_dir = Path("runs") / f"{env.grid_size}_dqn"
    grid_dir.mkdir(parents=True, exist_ok=True)
    args.output = grid_dir / args.output.name
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
    greedy_value_history: List[float] = []
    td_error_history: List[float] = []
    target_value_history: List[float] = []
    episode_returns: List[float] = []
    task_length_history: List[int] = []
    tasks_since_reset = 0
    rolling_success_history: List[float] = []
    rolling_return_history: List[float] = []
    rolling_history_x: List[int] = []
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[int] = deque(maxlen=100)

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
        with torch.no_grad():
            inp = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_vals = q_net(inp)
            greedy_action = int(torch.argmax(q_vals, dim=1).item())
            greedy_value = float(torch.max(q_vals).item())
        greedy_value_history.append(greedy_value)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = greedy_action

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
            recent_returns.append(task_return)
            recent_successes.append(1 if success else 0)
            task_return = 0.0
            task_steps = 0
            avg_ret = f"{np.mean(returns):.1f}" if returns else "n/a"
            avg_len = float(np.mean(task_lengths)) if task_lengths else 0.0
            success_rate = tasks_completed / total_tasks if total_tasks > 0 else 0.0
            rolling_success_history.append(
                float(np.mean(recent_successes)) if recent_successes else 0.0
            )
            rolling_return_history.append(
                float(np.mean(recent_returns)) if recent_returns else 0.0
            )
            rolling_history_x.append(total_tasks)
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
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()
            replay.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())
            td_error_history.append(td_errors.abs().mean().detach().cpu().item())
            target_value_history.append(target.abs().mean().detach().cpu().item())

        if global_step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    progress.close()
    torch.save(q_net.state_dict(), args.output)
    print(f"Saved DQN weights to {args.output}")
    _plot_metrics(
        step_rewards,
        episode_returns,
        task_length_history,
        greedy_value_history,
        td_error_history,
        target_value_history,
        args.output,
    )
    _plot_rolling_stats(
        rolling_history_x,
        rolling_success_history,
        rolling_return_history,
        args.output,
    )


def _plot_metrics(
    step_rewards: List[float],
    episode_returns: List[float],
    task_lengths: List[int],
    greedy_values: List[float],
    td_errors: List[float],
    target_values: List[float],
    weight_path: Path,
) -> None:
    if not step_rewards:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_metrics.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    reward_window = 100
    step_ma = _moving_average(step_rewards, reward_window)
    axes[0].plot(step_rewards, linewidth=0.4, alpha=0.15, color="#1f77b4")
    if step_ma is not None:
        ma_x = np.arange(reward_window - 1, reward_window - 1 + len(step_ma))
        axes[0].plot(ma_x, step_ma, linewidth=1.2, color="#1f77b4")
    axes[0].set_title(f"Per-step reward (MA window={reward_window})")
    axes[0].set_ylabel("Reward")
    return_window = 100
    return_ma = _moving_average(episode_returns, return_window)
    axes[1].plot(episode_returns, linewidth=0.6, alpha=0.2, color="#ff7f0e")
    if return_ma is not None:
        ma_x = np.arange(return_window - 1, return_window - 1 + len(return_ma))
        axes[1].plot(ma_x, return_ma, linewidth=1.2, color="#ff7f0e")
    axes[1].set_title(f"Per-task return (MA window={return_window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    if greedy_values:
        axes[2].plot(greedy_values, linewidth=0.5)
        axes[2].set_title("Greedy action value")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Q-value")
    else:
        axes[2].set_visible(False)
    if td_errors:
        axes[3].plot(td_errors, linewidth=0.5, color="#d62728")
        axes[3].set_title("Mean TD error")
        axes[3].set_xlabel("Update step")
        axes[3].set_ylabel("|δ|")
    else:
        axes[3].set_visible(False)
    if target_values:
        axes[4].plot(target_values, linewidth=0.5, color="#9467bd")
        axes[4].set_title("Target value magnitude")
        axes[4].set_xlabel("Update step")
        axes[4].set_ylabel("|Target|")
    else:
        axes[4].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved training curves to {plot_path}")
    if task_lengths:
        _plot_avg_task_steps(task_lengths, weight_path)


def _plot_rolling_stats(
    task_indices: List[int],
    success_history: List[float],
    return_history: List[float],
    weight_path: Path,
) -> None:
    if not task_indices:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_rolling.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(task_indices, success_history, linewidth=0.8, color="#1f77b4")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Rolling success rate (window≈100)")
    axes[1].plot(task_indices, return_history, linewidth=0.8, color="#ff7f0e")
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Completed task")
    axes[1].set_title("Rolling return (window≈100)")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved rolling metrics to {plot_path}")
    plt.close()
    print(f"Saved rolling metrics to {plot_path}")


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


def _moving_average(series: List[float], window: int) -> np.ndarray | None:
    if len(series) < window or window <= 0:
        return None
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(series, kernel, mode="valid")

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
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping value applied to Q-network updates.",
    )
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
