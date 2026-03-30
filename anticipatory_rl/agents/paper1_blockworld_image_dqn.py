"""Double DQN trainer for the paper1 blockworld image environment."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
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

from anticipatory_rl.envs.paper1_blockworld_image_env import (
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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    task_boundary: bool = False


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
    return "myopic_blockworld" if args.tasks_per_reset <= 1 else "anticipatory_blockworld"


def _encode_obs_storage(obs: np.ndarray) -> np.ndarray:
    clipped = np.clip(obs, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8, copy=False)


def _decode_obs_batch(obs_batch: List[np.ndarray], device: torch.device) -> torch.Tensor:
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
        return float(
            env.config.move_cost
            if tuple(pre_info["robot"]) != tuple(post_info["robot"])
            else env.invalid_action_penalty
        )
    if action == Paper1BlockworldImageEnv.PICK:
        picked = pre_info.get("holding") is None and post_info.get("holding") is not None
        return float(env.config.pick_cost if picked else env.invalid_action_penalty)
    if action == Paper1BlockworldImageEnv.PLACE:
        placed = pre_info.get("holding") is not None and post_info.get("holding") is None
        return float(env.config.place_cost if placed else env.invalid_action_penalty)
    return 0.0


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
    run_dir = Path("runs") / f"{run_label}_paper1_blockworld_image_dqn_tpr{args.tasks_per_reset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output.name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")

    obs, infos = env.reset(seed=args.seed)
    infos = list(infos)
    obs_shape = obs.shape[1:]
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    target_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay: Deque[Transition] = deque(maxlen=max(1, args.replay_size))

    num_envs = env.num_envs
    env_reset_tasks = args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_reset
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
    episode_index = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_success: Deque[int] = deque(maxlen=100)
    recent_costs: Deque[float] = deque(maxlen=100)
    recent_auto: Deque[int] = deque(maxlen=100)
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

        episode_done_flags = np.zeros(num_envs, dtype=bool)
        env_reset_flags = np.zeros(num_envs, dtype=bool)

        for idx in range(num_envs):
            task_return[idx] += float(rewards[idx])
            task_steps[idx] += 1
            task_cost[idx] += float(
                _paper_step_cost(
                    env.envs[idx],
                    pre_infos[idx],
                    next_infos[idx],
                    int(actions[idx]),
                    auto_before=bool(current_task_auto[idx]),
                )
            )
            steps_since_reset[idx] += 1
            if bool(task_done[idx]):
                tasks_since_reset[idx] += 1
                env_tasks_since_reset[idx] += 1
                if args.tasks_per_reset > 0 and tasks_since_reset[idx] >= args.tasks_per_reset:
                    episode_done_flags[idx] = True
                if env_reset_tasks is not None and env_reset_tasks > 0 and env_tasks_since_reset[idx] >= env_reset_tasks:
                    env_reset_flags[idx] = True
                    episode_done_flags[idx] = True
            if args.episode_step_limit > 0 and steps_since_reset[idx] >= args.episode_step_limit:
                episode_done_flags[idx] = True

        for idx in range(num_envs):
            replay.append(
                Transition(
                    state=_encode_obs_storage(obs[idx]),
                    action=int(actions[idx]),
                    reward=float(rewards[idx]),
                    next_state=_encode_obs_storage(next_obs[idx]),
                    done=bool(episode_done_flags[idx]),
                    task_boundary=bool(task_done[idx]),
                )
            )

        if len(replay) >= args.batch_size:
            batch = random.sample(list(replay), args.batch_size)
            states = _decode_obs_batch([t.state for t in batch], device)
            batch_actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
            batch_rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            next_states = _decode_obs_batch([t.next_state for t in batch], device)
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)

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
                pos = int(tasks_since_reset[idx] - 1)
                snapshot = pre_infos[idx]
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

    progress.close()
    torch.save(q_net.state_dict(), output_path)
    print(f"Saved DQN weights to {output_path}")

    sequence_len = max(1, int(args.tasks_per_reset))
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
        "tasks_per_reset": int(args.tasks_per_reset),
        "env_reset_tasks": None if env_reset_tasks is None else int(env_reset_tasks),
        "num_envs": int(args.num_envs),
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--task-library-size", type=int, default=24)
    parser.add_argument("--max-task-steps", type=int, default=64)
    parser.add_argument("--success-reward", type=float, default=12.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=5.0)
    parser.add_argument("--correct-pick-bonus", type=float, default=1.0)
    parser.add_argument("--render-tile-px", type=int, default=24)
    parser.add_argument("--render-margin-px", type=int, default=None)
    parser.add_argument(
        "--procedural-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use procedurally sampled region layouts on reset.",
    )
    parser.add_argument("--tasks-per-reset", type=int, default=10)
    parser.add_argument("--env-reset-tasks", type=int, default=None)
    parser.add_argument("--episode-step-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("paper1_blockworld_image_dqn.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
