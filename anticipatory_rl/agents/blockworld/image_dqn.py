"""Double DQN trainer for the paper1 blockworld image environment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import resource
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from anticipatory_rl.envs.blockworld.env import (
    Paper1BlockworldImageEnv,
)


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
        return self.head(self.encoder(x))

class ReplayBuffer:
    """Fixed-size array-backed replay to avoid Python-object memory blowups."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]) -> None:
        self.capacity = max(1, int(capacity))
        self.obs = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=bool)
        self.task_boundaries = np.empty(self.capacity, dtype=bool)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_boundary: bool,
    ) -> None:
        idx = self.pos
        self.obs[idx] = state
        self.next_obs[idx] = next_state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.task_boundaries[idx] = bool(task_boundary)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
        )


class VectorEnv:
    """Lock-step wrapper around multiple Paper1BlockworldImageEnv instances."""

    def __init__(self, make_env, num_envs: int) -> None:
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.action_space = self.envs[0].action_space

    def reset(self, seed: int | None = None):
        obs_list = []
        infos = []
        for idx, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + idx
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
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
            list(infos),
        )


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label is not None:
        return args.run_label
    return "myopic_blockworld" if args.tasks_per_episode <= 1 else "anticipatory_blockworld"


def _encode_obs_storage(obs: np.ndarray) -> np.ndarray:
    clipped = np.clip(obs, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8, copy=False)


def _decode_obs_batch(obs_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(obs_batch, dtype=np.uint8)
    return torch.tensor(arr, dtype=torch.float32, device=device).div_(255.0)


def _epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    frac = min(1.0, step / max(1, decay))
    return start + frac * (final - start)


def _paper_step_cost(
    env: Paper1BlockworldImageEnv,
    pre_info: Dict[str, Any],
    post_info: Dict[str, Any],
    action: int,
    *,
    auto_before: bool,
) -> float:
    if auto_before:
        return 0.0
    if action in {
        Paper1BlockworldImageEnv.MOVE_UP,
        Paper1BlockworldImageEnv.MOVE_DOWN,
        Paper1BlockworldImageEnv.MOVE_LEFT,
        Paper1BlockworldImageEnv.MOVE_RIGHT,
    }:
        moved = tuple(pre_info["robot"]) != tuple(post_info["robot"])
        if not moved:
            return 0.0
        return float(env.config.move_cost)
    if action == Paper1BlockworldImageEnv.PICK:
        picked = pre_info.get("holding") is None and post_info.get("holding") is not None
        return float(env.config.pick_cost if picked else 0.0)
    if action == Paper1BlockworldImageEnv.PLACE:
        placed = pre_info.get("holding") is not None and post_info.get("holding") is None
        return float(env.config.place_cost if placed else 0.0)
    return 0.0


def _is_invalid_action(
    pre_info: Dict[str, Any],
    post_info: Dict[str, Any],
    action: int,
) -> bool:
    if action in {
        Paper1BlockworldImageEnv.MOVE_UP,
        Paper1BlockworldImageEnv.MOVE_DOWN,
        Paper1BlockworldImageEnv.MOVE_LEFT,
        Paper1BlockworldImageEnv.MOVE_RIGHT,
    }:
        return tuple(pre_info["robot"]) == tuple(post_info["robot"])
    if action == Paper1BlockworldImageEnv.PICK:
        picked = pre_info.get("holding") is None and post_info.get("holding") is not None
        return not picked
    if action == Paper1BlockworldImageEnv.PLACE:
        placed = pre_info.get("holding") is not None and post_info.get("holding") is None
        return not placed
    return False


def _moving_average(series: List[float], window: int) -> np.ndarray | None:
    if window <= 0 or len(series) < window:
        return None
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.asarray(series, dtype=np.float32), kernel, mode="valid")


def _plot_task_curves(
    task_returns: List[float],
    task_costs: List[float],
    task_steps: List[int],
    weight_path: Path,
) -> None:
    if not task_returns:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_metrics.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    series = [
        (task_returns, "Per-task return", "Return", "#2563eb"),
        (task_costs, "Per-task paper cost", "Cost", "#ea580c"),
        ([float(v) for v in task_steps], "Per-task steps", "Steps", "#059669"),
    ]
    for ax, (values, title, ylabel, color) in zip(axes, series):
        ax.plot(values, linewidth=0.6, alpha=0.25, color=color)
        ma = _moving_average([float(v) for v in values], 25)
        if ma is not None:
            x = np.arange(24, 24 + len(ma))
            ax.plot(x, ma, linewidth=1.6, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Completed task")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_cost_by_position(task_records: List[Dict[str, Any]], sequence_len: int, weight_path: Path) -> None:
    if not task_records or sequence_len <= 0:
        return
    grouped: Dict[int, List[float]] = {idx: [] for idx in range(sequence_len)}
    for row in task_records:
        pos = int(row.get("episode_position", -1))
        if 0 <= pos < sequence_len:
            grouped[pos].append(float(row["paper_cost"]))
    means = [float(np.mean(grouped[idx])) if grouped[idx] else 0.0 for idx in range(sequence_len)]
    plot_path = weight_path.with_name(weight_path.stem + "_cost_by_position.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(sequence_len), means, marker="o", linewidth=1.5, color="#7c3aed")
    ax.set_title("Average paper cost by task index")
    ax.set_xlabel("Task index in sequence")
    ax.set_ylabel("Paper cost")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _write_json(data: Dict[str, Any] | List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def _rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1024 * 1024 * 1024:
        return float(rss / (1024.0 ** 3))
    return float(rss / 1024.0 / 1024.0)


def train(args: argparse.Namespace) -> Path:
    device = _select_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    def make_env() -> Paper1BlockworldImageEnv:
        return Paper1BlockworldImageEnv(
            task_library_size=args.task_library_size,
            max_task_steps=args.max_task_steps,
            success_reward=args.success_reward,
            step_penalty=args.step_penalty,
            invalid_action_penalty=args.invalid_action_penalty,
            correct_pick_bonus=args.correct_pick_bonus,
            render_tile_px=args.render_tile_px,
            render_margin_px=args.render_margin_px,
            procedural_layout=args.procedural_layout,
        )

    env = VectorEnv(make_env, max(1, args.num_envs))
    run_label = _resolve_run_label(args)
    run_dir = Path("runs") / f"{run_label}_paper1_blockworld_image_dqn_tpe{args.tasks_per_episode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output.name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")

    tb_writer = None
    tb_log_dir_used: Path | None = None
    if args.tb_log_interval > 0 or args.tb_task_interval > 0:
        tb_base_dir = args.tb_log_dir or (run_dir / "tb")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = tb_base_dir / timestamp
        suffix = 1
        while tb_log_dir.exists():
            tb_log_dir = tb_base_dir / f"{timestamp}_{suffix:02d}"
            suffix += 1
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_log_dir_used = tb_log_dir
        print(f"[train] TensorBoard log dir -> {tb_log_dir.resolve()}")
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    obs, infos = env.reset(seed=args.seed)
    infos = list(infos)
    obs_shape = obs.shape[1:]
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    target_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(max(1, args.replay_size), obs_shape)

    num_envs = env.num_envs
    env_reset_tasks = args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_episode
    is_myopic = args.tasks_per_episode == 1
    global_step = 0
    steps_since_reset = np.zeros(num_envs, dtype=np.int64)
    tasks_since_reset = np.zeros(num_envs, dtype=np.int64)
    env_tasks_since_reset = np.zeros(num_envs, dtype=np.int64)
    task_return = np.zeros(num_envs, dtype=np.float32)
    task_steps = np.zeros(num_envs, dtype=np.int64)
    task_cost = np.zeros(num_envs, dtype=np.float32)
    current_task_auto = np.array([bool(info.get("next_auto_satisfied", False)) for info in infos], dtype=bool)

    total_tasks = 0
    tasks_completed = 0
    total_invalid_actions = 0
    total_horizon_tasks = 0
    total_action_steps = 0
    episode_index = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_success: Deque[int] = deque(maxlen=100)
    recent_costs: Deque[float] = deque(maxlen=100)
    recent_auto: Deque[int] = deque(maxlen=100)
    recent_task_steps: Deque[int] = deque(maxlen=args.tb_task_window)
    recent_step_rewards: Deque[float] = deque(maxlen=args.tb_step_window)
    recent_step_costs: Deque[float] = deque(maxlen=args.tb_step_window)
    recent_invalid: Deque[float] = deque(maxlen=args.tb_step_window)
    recent_horizon: Deque[float] = deque(maxlen=args.tb_step_window)
    recent_random: Deque[float] = deque(maxlen=args.tb_step_window)
    loss_history: List[float] = []
    task_records: List[Dict[str, Any]] = []
    return_history: List[float] = []
    cost_history: List[float] = []
    task_length_history: List[int] = []

    progress = tqdm(total=args.total_steps, desc="Paper1 Blockworld DQN")
    while global_step < args.total_steps:
        epsilon = _epsilon_by_step(global_step, args.epsilon_start, args.epsilon_final, args.epsilon_decay)
        with torch.no_grad():
            inp = torch.tensor(obs, dtype=torch.float32, device=device)
            q_values = q_net(inp)
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        random_mask = np.random.rand(num_envs) < epsilon
        random_actions = np.asarray([env.action_space.sample() for _ in range(num_envs)], dtype=np.int64)
        actions = np.where(random_mask, random_actions, greedy_actions).astype(np.int64)

        pre_infos = infos
        next_obs, rewards, success, horizon, next_infos = env.step(actions)
        task_done = success | horizon

        invalid_flags = np.zeros(num_envs, dtype=np.float32)
        step_paper_costs = np.zeros(num_envs, dtype=np.float32)

        episode_done_flags = np.zeros(num_envs, dtype=bool)
        env_reset_flags = np.zeros(num_envs, dtype=bool)
        bootstrap_done_flags = np.zeros(num_envs, dtype=bool)

        for idx in range(num_envs):
            task_return[idx] += float(rewards[idx])
            task_steps[idx] += 1
            step_cost = float(
                _paper_step_cost(
                    env.envs[idx],
                    pre_infos[idx],
                    next_infos[idx],
                    int(actions[idx]),
                    auto_before=bool(current_task_auto[idx]),
                )
            )
            step_paper_costs[idx] = step_cost
            task_cost[idx] += step_cost
            steps_since_reset[idx] += 1
            invalid_flags[idx] = 1.0 if _is_invalid_action(pre_infos[idx], next_infos[idx], int(actions[idx])) else 0.0
            if bool(task_done[idx]):
                tasks_since_reset[idx] += 1
                env_tasks_since_reset[idx] += 1
                if args.tasks_per_episode > 0 and tasks_since_reset[idx] >= args.tasks_per_episode:
                    episode_done_flags[idx] = True
                if env_reset_tasks is not None and env_reset_tasks > 0 and env_tasks_since_reset[idx] >= env_reset_tasks:
                    env_reset_flags[idx] = True
                    episode_done_flags[idx] = True
            if args.episode_step_limit > 0 and steps_since_reset[idx] >= args.episode_step_limit:
                episode_done_flags[idx] = True
            if env_reset_flags[idx]:
                bootstrap_done_flags[idx] = True
            elif is_myopic and bool(task_done[idx]):
                bootstrap_done_flags[idx] = True
            elif episode_done_flags[idx] and bool(success[idx]):
                bootstrap_done_flags[idx] = True

        recent_step_rewards.append(float(np.mean(rewards)))
        recent_step_costs.append(float(np.mean(step_paper_costs)))
        recent_invalid.append(float(np.mean(invalid_flags)))
        recent_horizon.append(float(np.mean(horizon)))
        recent_random.append(float(np.mean(random_mask)))
        total_invalid_actions += int(np.sum(invalid_flags))
        total_action_steps += int(num_envs)

        for idx in range(num_envs):
            replay.add(
                state=_encode_obs_storage(obs[idx]),
                action=int(actions[idx]),
                reward=float(rewards[idx]),
                next_state=_encode_obs_storage(next_obs[idx]),
                done=bool(bootstrap_done_flags[idx]),
                task_boundary=bool(task_done[idx]),
            )

        if len(replay) >= args.batch_size:
            batch_states, batch_actions_np, batch_rewards_np, batch_next_states, batch_dones_np = replay.sample(args.batch_size)
            states = _decode_obs_batch(batch_states, device)
            batch_actions = torch.tensor(batch_actions_np, dtype=torch.int64, device=device).unsqueeze(1)
            batch_rewards = torch.tensor(batch_rewards_np, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = _decode_obs_batch(batch_next_states, device)
            dones = torch.tensor(batch_dones_np, dtype=torch.float32, device=device).unsqueeze(1)

            q_selected = q_net(states).gather(1, batch_actions)
            with torch.no_grad():
                next_online = q_net(next_states)
                next_actions = torch.argmax(next_online, dim=1, keepdim=True)
                next_target = target_net(next_states).gather(1, next_actions)
                targets = batch_rewards + args.gamma * (1.0 - dones) * next_target

            loss = nn.functional.smooth_l1_loss(q_selected, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_history.append(float(loss.item()))

        if args.tau < 1.0:
            with torch.no_grad():
                tau = float(args.tau)
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        elif (global_step + num_envs) % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        for idx in range(num_envs):
            if bool(task_done[idx]):
                total_tasks += 1
                was_success = bool(success[idx])
                if was_success:
                    tasks_completed += 1
                if bool(horizon[idx]):
                    total_horizon_tasks += 1
                pos = int(tasks_since_reset[idx] - 1)
                snapshot = pre_infos[idx]
                if tb_writer is not None and args.tb_task_interval > 0 and total_tasks % args.tb_task_interval == 0:
                    tb_writer.add_scalar("task/return", float(task_return[idx]), total_tasks)
                    tb_writer.add_scalar("task/steps", float(task_steps[idx]), total_tasks)
                    tb_writer.add_scalar("task/paper_cost", float(task_cost[idx]), total_tasks)
                    tb_writer.add_scalar("task/success", 1.0 if was_success else 0.0, total_tasks)
                    tb_writer.add_scalar("task/horizon", 1.0 if horizon[idx] else 0.0, total_tasks)
                    tb_writer.add_scalar("task/auto_satisfied", 1.0 if current_task_auto[idx] else 0.0, total_tasks)
                task_records.append(
                    {
                        "task_number": total_tasks,
                        "task_assignments": [list(item) for item in snapshot.get("task_assignments", ())],
                        "task_size": int(snapshot.get("task_size", 0)),
                        "success": was_success,
                        "horizon": bool(horizon[idx]),
                        "steps": int(task_steps[idx]),
                        "return": float(task_return[idx]),
                        "paper_cost": float(task_cost[idx]),
                        "auto_satisfied": bool(current_task_auto[idx]),
                        "episode_position": pos,
                    }
                )
                return_history.append(float(task_return[idx]))
                cost_history.append(float(task_cost[idx]))
                task_length_history.append(int(task_steps[idx]))
                recent_task_steps.append(int(task_steps[idx]))
                recent_returns.append(float(task_return[idx]))
                recent_success.append(1 if was_success else 0)
                recent_costs.append(float(task_cost[idx]))
                recent_auto.append(1 if current_task_auto[idx] else 0)
                task_return[idx] = 0.0
                task_steps[idx] = 0
                task_cost[idx] = 0.0
                current_task_auto[idx] = bool(next_infos[idx].get("next_auto_satisfied", False))

        for idx in range(num_envs):
            if env_reset_flags[idx]:
                episode_index += 1
                reset_seed = args.seed + 100_003 * episode_index
                new_obs, new_info = env.reset_env(idx, seed=reset_seed)
                next_obs[idx] = new_obs
                next_infos[idx] = new_info
                env_tasks_since_reset[idx] = 0
                current_task_auto[idx] = bool(new_info.get("next_auto_satisfied", False))
            if episode_done_flags[idx]:
                steps_since_reset[idx] = 0
                tasks_since_reset[idx] = 0
                if not env_reset_flags[idx]:
                    current_task_auto[idx] = bool(next_infos[idx].get("next_auto_satisfied", False))

        obs = next_obs
        infos = next_infos
        global_step += num_envs
        progress.update(num_envs)
        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        avg_success = float(np.mean(recent_success)) if recent_success else 0.0
        avg_cost = float(np.mean(recent_costs)) if recent_costs else 0.0
        avg_auto = float(np.mean(recent_auto)) if recent_auto else 0.0
        avg_loss = float(np.mean(loss_history[-100:])) if loss_history else 0.0
        progress.set_postfix(
            ret=f"{avg_return:.1f}" if recent_returns else "n/a",
            success=f"{avg_success:.2f}",
            cost=f"{avg_cost:.1f}" if recent_costs else "n/a",
            auto=f"{avg_auto:.2f}",
            eps=f"{epsilon:.2f}",
            loss=f"{avg_loss:.3f}" if loss_history else "n/a",
            tasks=tasks_completed,
        )
        if tb_writer is not None and args.tb_log_interval > 0 and global_step % args.tb_log_interval == 0:
            q_mean = float(q_values.mean().item()) if "q_values" in locals() else 0.0
            q_min = float(q_values.min().item()) if "q_values" in locals() else 0.0
            q_max = float(q_values.max().item()) if "q_values" in locals() else 0.0
            q_std = float(q_values.std().item()) if "q_values" in locals() else 0.0
            tb_writer.add_scalar("train/epsilon", float(epsilon), global_step)
            tb_writer.add_scalar("train/loss", float(avg_loss), global_step)
            tb_writer.add_scalar("train/replay_size", float(len(replay)), global_step)
            tb_writer.add_scalar("train/reward_step", float(np.mean(recent_step_rewards)) if recent_step_rewards else 0.0, global_step)
            tb_writer.add_scalar("train/paper_cost_step", float(np.mean(recent_step_costs)) if recent_step_costs else 0.0, global_step)
            tb_writer.add_scalar("train/invalid_action_rate", float(np.mean(recent_invalid)) if recent_invalid else 0.0, global_step)
            tb_writer.add_scalar("train/horizon_rate", float(np.mean(recent_horizon)) if recent_horizon else 0.0, global_step)
            tb_writer.add_scalar("train/random_action_rate", float(np.mean(recent_random)) if recent_random else 0.0, global_step)
            tb_writer.add_scalar("train/success_rate", float(avg_success), global_step)
            tb_writer.add_scalar("train/auto_rate", float(avg_auto), global_step)
            tb_writer.add_scalar("train/avg_task_return", float(avg_return), global_step)
            tb_writer.add_scalar("train/avg_task_cost", float(avg_cost), global_step)
            tb_writer.add_scalar("train/avg_task_steps", float(np.mean(recent_task_steps)) if recent_task_steps else 0.0, global_step)
            tb_writer.add_scalar("train/q_mean", q_mean, global_step)
            tb_writer.add_scalar("train/q_min", q_min, global_step)
            tb_writer.add_scalar("train/q_max", q_max, global_step)
            tb_writer.add_scalar("train/q_std", q_std, global_step)
            tb_writer.add_scalar("train/tasks_completed", float(tasks_completed), global_step)
        if args.log_rss_every > 0 and global_step % args.log_rss_every == 0:
            print(
                f"[mem] step={global_step} rss_gb={_rss_gb():.2f} "
                f"replay={len(replay)} num_envs={num_envs}",
                flush=True,
            )

    progress.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    torch.save(q_net.state_dict(), output_path)
    print(f"Saved DQN weights to {output_path}")

    sequence_len = max(1, int(args.tasks_per_episode))
    pos_costs: Dict[int, List[float]] = {idx: [] for idx in range(sequence_len)}
    pos_auto: Dict[int, List[int]] = {idx: [] for idx in range(sequence_len)}
    full_sequence_costs: List[float] = []
    current_sequence_costs: List[float] = []
    for row in task_records:
        pos = int(row.get("episode_position", -1))
        if 0 <= pos < sequence_len:
            pos_costs[pos].append(float(row["paper_cost"]))
            pos_auto[pos].append(1 if row["auto_satisfied"] else 0)
        current_sequence_costs.append(float(row["paper_cost"]))
        if len(current_sequence_costs) == sequence_len:
            full_sequence_costs.append(float(sum(current_sequence_costs)))
            current_sequence_costs = []

    summary = {
        "run_label": run_label,
        "checkpoint": str(output_path),
        "total_steps": int(args.total_steps),
        "tasks_attempted": int(total_tasks),
        "tasks_completed": int(tasks_completed),
        "success_rate": float(tasks_completed / max(1, total_tasks)),
        "invalid_action_rate": float(total_invalid_actions / max(1, total_action_steps)),
        "horizon_rate": float(total_horizon_tasks / max(1, total_tasks)),
        "avg_task_return": float(np.mean(return_history)) if return_history else 0.0,
        "avg_task_steps": float(np.mean(task_length_history)) if task_length_history else 0.0,
        "avg_task_cost": float(np.mean(cost_history)) if cost_history else 0.0,
        "avg_total_sequence_cost": float(np.mean(full_sequence_costs)) if full_sequence_costs else 0.0,
        "reward_per_step": float(np.mean([row["return"] / max(1, row["steps"]) for row in task_records])) if task_records else 0.0,
        "auto_rate": float(np.mean([1.0 if row["auto_satisfied"] else 0.0 for row in task_records])) if task_records else 0.0,
        "cost_by_task_index": {
            str(idx): float(np.mean(values)) if values else 0.0
            for idx, values in pos_costs.items()
        },
        "auto_rate_by_task_index": {
            str(idx): float(np.mean(values)) if values else 0.0
            for idx, values in pos_auto.items()
        },
        "seed": int(args.seed),
        "tasks_per_episode": int(args.tasks_per_episode),
        "tasks_per_reset": int(args.tasks_per_episode),
        "env_reset_tasks": None if env_reset_tasks is None else int(env_reset_tasks),
        "num_envs": int(args.num_envs),
        "tb_log_dir": None if tb_log_dir_used is None else str(tb_log_dir_used),
    }
    _write_json(summary, run_dir / "train_summary.json")
    _write_json(task_records, run_dir / "task_records.json")
    _write_json(vars(args), run_dir / "train_args.json")
    _plot_task_curves(return_history, cost_history, task_length_history, output_path)
    _plot_cost_by_position(task_records, sequence_len, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Double DQN on the paper1 blockworld image environment.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=500_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--task-library-size", type=int, default=20)
    parser.add_argument("--max-task-steps", type=int, default=128)
    parser.add_argument("--success-reward", type=float, default=100.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--correct-pick-bonus", type=float, default=1.0)
    parser.add_argument("--render-tile-px", type=int, default=24)
    parser.add_argument("--render-margin-px", type=int, default=None)
    parser.add_argument(
        "--procedural-layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use procedurally sampled region layouts on reset.",
    )
    parser.add_argument(
        "--tasks-per-episode",
        type=int,
        default=None,
        help="Number of tasks before marking an episode done (alias: --tasks-per-reset).",
    )
    parser.add_argument("--tasks-per-reset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--env-reset-tasks", type=int, default=None)
    parser.add_argument("--episode-step-limit", type=int, default=0)
    parser.add_argument("--tb-log-dir", type=Path, default=None)
    parser.add_argument("--tb-log-interval", type=int, default=1000)
    parser.add_argument("--tb-task-interval", type=int, default=1)
    parser.add_argument("--tb-step-window", type=int, default=1000)
    parser.add_argument("--tb-task-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("paper1_blockworld_image_dqn.pt"))
    parser.add_argument("--log-rss-every", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tasks_per_episode is None and args.tasks_per_reset is None:
        args.tasks_per_episode = 10
    elif args.tasks_per_episode is None:
        args.tasks_per_episode = args.tasks_per_reset
    elif args.tasks_per_reset is not None and args.tasks_per_reset != args.tasks_per_episode:
        raise ValueError(
            "--tasks-per-episode and --tasks-per-reset must match when both are provided."
        )
    train(args)


if __name__ == "__main__":
    main()
