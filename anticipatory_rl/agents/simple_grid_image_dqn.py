"""DQN trainer for the RGB SimpleGrid image environment."""

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
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv


class ConvQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], hidden_dim: int, num_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.encoder(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


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


class VectorEnv:
    """Lock-step wrapper around multiple SimpleGridImageEnv instances."""

    def __init__(self, make_env, num_envs: int) -> None:
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.grid_size = self.envs[0].grid_size
        self.action_space = self.envs[0].action_space

    def reset(self, seed: int | None = None):
        obs_list = []
        infos = []
        for idx, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + idx
            ob, info = env.reset(seed=env_seed)
            obs_list.append(ob)
            infos.append(info)
        return np.stack(obs_list), infos

    def reset_env(self, idx: int, seed: int | None = None):
        env_seed = None if seed is None else seed + idx
        return self.envs[idx].reset(seed=env_seed)

    def step(self, actions: np.ndarray):
        results = [
            env.step(int(action))
            for env, action in zip(self.envs, actions)
        ]
        obs, rewards, success, horizon, infos = zip(*results)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(success, dtype=bool),
            np.asarray(horizon, dtype=bool),
            infos,
        )

    def set_active_objects(self, count: int) -> None:
        for env in self.envs:
            env.set_active_objects(count)


def train(args: argparse.Namespace) -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env():
        return SimpleGridImageEnv(
            grid_size=args.grid_size,
            success_reward=args.success_reward,
            num_objects=args.num_objects,
            distance_reward=True,
            distance_reward_scale=args.distance_reward_scale,
            clear_task_prob=args.clear_task_prob,
        )

    env = VectorEnv(make_env, max(1, args.num_envs))
    grid_dir = Path("runs") / f"{env.grid_size}_image_dqn"
    grid_dir.mkdir(parents=True, exist_ok=True)
    args.output = grid_dir / args.output.name

    replay = PrioritizedReplayBuffer(
        args.replay_size,
        alpha=args.per_alpha,
        eps=args.per_eps,
    )
    goal_buffer: Deque[Transition] = deque(
        maxlen=args.goal_buffer_size if args.goal_buffer_size > 0 else None
    )

    state, _ = env.reset(seed=args.seed)
    obs_shape = state.shape[1:]
    action_dim = env.action_space.n
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=action_dim).to(device)
    target_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=action_dim).to(device)
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
    num_envs = env.num_envs
    episode_transitions: List[List[Transition]] = [[] for _ in range(num_envs)]
    tasks_since_reset = np.zeros(num_envs, dtype=np.int64)
    rolling_success_history: List[float] = []
    rolling_return_history: List[float] = []
    rolling_history_x: List[int] = []
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[int] = deque(maxlen=100)
    epsilon_restart_until = -1
    last_restart_step = -10**9
    goal_fraction = max(0.0, min(args.goal_buffer_fraction, 1.0))

    progress = tqdm(total=args.total_steps, desc="SimpleGrid Image DQN")
    task_return = np.zeros(num_envs, dtype=np.float32)
    task_steps = np.zeros(num_envs, dtype=np.int64)

    def current_epsilon(step: int) -> float:
        if args.epsilon is not None:
            return args.epsilon
        frac = min(1.0, step / max(1, args.epsilon_decay))
        return args.epsilon_start + frac * (args.epsilon_final - args.epsilon_start)

    while global_step < args.total_steps:
        base_eps = current_epsilon(global_step)
        if global_step < epsilon_restart_until:
            eps = max(base_eps, args.epsilon_restart_value)
        else:
            eps = base_eps
        with torch.no_grad():
            inp = torch.tensor(state, dtype=torch.float32, device=device)
            q_vals = q_net(inp)
            greedy_actions = torch.argmax(q_vals, dim=1).cpu().numpy()
            greedy_value = float(torch.max(q_vals).item())
        greedy_value_history.append(greedy_value)

        random_mask = np.random.rand(num_envs) < eps
        random_actions = np.asarray(
            [env.action_space.sample() for _ in range(num_envs)]
        )
        actions = np.where(random_mask, random_actions, greedy_actions).astype(np.int64)

        next_state, reward, success, horizon, info = env.step(actions)
        done = success | horizon
        for idx in range(num_envs):
            transition = Transition(
                state[idx],
                int(actions[idx]),
                float(reward[idx]),
                next_state[idx],
                bool(done[idx]),
            )
            replay.push(transition)
            episode_transitions[idx].append(transition)
            step_rewards.append(float(reward[idx]))

        state = next_state
        task_return += reward
        task_steps += 1
        global_step += num_envs
        progress.update(num_envs)

        for idx in range(num_envs):
            if done[idx]:
                tasks_since_reset[idx] += 1
                episode_return = float(task_return[idx])
                returns.append(episode_return)
                task_lengths.append(int(task_steps[idx]))
                episode_returns.append(episode_return)
                task_length_history.append(int(task_steps[idx]))
                total_tasks += 1
                if success[idx]:
                    tasks_completed += 1
                recent_returns.append(episode_return)
                recent_successes.append(1 if success[idx] else 0)
                window_success = float(np.mean(recent_successes)) if recent_successes else 0.0
                rolling_success_history.append(window_success)
                rolling_return_history.append(
                    float(np.mean(recent_returns)) if recent_returns else 0.0
                )
                rolling_history_x.append(total_tasks)
                if success[idx] and args.goal_buffer_size > 0 and episode_transitions[idx]:
                    for trans in episode_transitions[idx]:
                        goal_buffer.append(trans)
                episode_transitions[idx].clear()
                task_return[idx] = 0.0
                task_steps[idx] = 0

                if tasks_since_reset[idx] >= args.tasks_per_reset:
                    new_obs, _ = env.reset_env(idx)
                    state[idx] = new_obs
                    tasks_since_reset[idx] = 0
                    episode_transitions[idx].clear()

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
        window_success = float(np.mean(recent_successes)) if recent_successes else 0.0
        if (
            window_success < args.epsilon_restart_threshold
            and global_step >= last_restart_step + args.epsilon_restart_cooldown
        ):
            epsilon_restart_until = global_step + args.epsilon_restart_duration
            last_restart_step = global_step

        if len(replay) >= args.batch_size:
            beta = min(
                1.0,
                args.per_beta_start
                + (1.0 - args.per_beta_start) * (global_step / max(1, args.total_steps)),
            )
            goal_batch_size = 0
            if args.goal_buffer_size > 0 and goal_fraction > 0.0 and goal_buffer:
                desired = int(args.batch_size * goal_fraction)
                if desired > 0:
                    goal_batch_size = min(desired, len(goal_buffer), args.batch_size)
            per_batch_size = args.batch_size - goal_batch_size

            per_batch: List[Transition] = []
            per_indices = np.array([], dtype=np.int64)
            per_weights = np.array([], dtype=np.float32)
            if per_batch_size > 0:
                per_batch, per_indices, per_weights = replay.sample(per_batch_size, beta=beta)
            goal_samples: List[Transition] = []
            if goal_batch_size > 0:
                goal_samples = random.sample(list(goal_buffer), goal_batch_size)

            batch = per_batch + goal_samples
            if not batch:
                continue

            weights_list: List[np.ndarray] = []
            if per_batch_size > 0:
                weights_list.append(per_weights)
            if goal_batch_size > 0:
                weights_list.append(np.ones(goal_batch_size, dtype=np.float32))
            weights_arr = (
                np.concatenate(weights_list)
                if weights_list
                else np.ones(len(batch), dtype=np.float32)
            )

            states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.tensor(
                np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device
            )
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            weights_t = torch.tensor(weights_arr, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_actions = torch.argmax(q_net(next_states), dim=1, keepdim=True)
                next_q = target_net(next_states).gather(1, next_actions)
                target = rewards + args.gamma * (1.0 - dones) * next_q

            td_errors = target - q_values
            loss = (weights_t * td_errors.pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()
            if per_batch_size > 0:
                replay.update_priorities(
                    per_indices,
                    td_errors[:per_batch_size].detach().cpu().numpy().squeeze(),
                )
            td_error_history.append(td_errors.abs().mean().detach().cpu().item())
            target_value_history.append(target.abs().mean().detach().cpu().item())

        if args.tau < 1.0:
            with torch.no_grad():
                tau = args.tau
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        elif global_step % args.target_update == 0:
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
    parser = argparse.ArgumentParser(description="Train DQN on the RGB SimpleGrid environment.")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--num-objects", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel env instances to sample each step.")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-eps", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=80_000)
    parser.add_argument("--epsilon-restart-value", type=float, default=0.4)
    parser.add_argument("--epsilon-restart-threshold", type=float, default=0.6)
    parser.add_argument("--epsilon-restart-duration", type=int, default=15_000)
    parser.add_argument("--epsilon-restart-cooldown", type=int, default=40_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / "simple_grid_image_dqn.pt")
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--tasks-per-reset", type=int, default=1)
    parser.add_argument("--goal-buffer-size", type=int, default=5_000)
    parser.add_argument("--goal-buffer-fraction", type=float, default=0.25)
    parser.add_argument(
        "--clear-task-prob",
        type=float,
        default=0.0,
        help="Probability of sampling a clear-receptacle task instead of a move task.",
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
