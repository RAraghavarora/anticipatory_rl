"""DQN trainer for the symbolic restaurant domain."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.restaurant_symbolic_env import RestaurantSymbolicEnv

from anticipatory_rl.agents.utils import (
    select_device,
    epsilon_by_step,
    resolve_run_label,
)

def _load_layout_corpus(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and isinstance(payload.get("layouts"), list):
        return [x for x in payload["layouts"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError(f"Unsupported layout corpus format in {path}")


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_mask: np.ndarray
    task_boundary: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.memory: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)


class RestaurantQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _masked_argmax(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = q_values.clone()
    invalid = valid_mask <= 0.0
    all_invalid = invalid.all(dim=1, keepdim=True)
    masked[invalid] = float("-inf")
    greedy = torch.argmax(masked, dim=1, keepdim=True)
    if all_invalid.any():
        fallback = torch.argmax(q_values, dim=1, keepdim=True)
        greedy = torch.where(all_invalid, fallback, greedy)
    return greedy


def _select_action(
    q_net: RestaurantQNetwork,
    state: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    valid_indices = np.flatnonzero(valid_mask > 0.0)
    if valid_indices.size == 0:
        valid_indices = np.arange(valid_mask.shape[0], dtype=np.int64)
    if random.random() < epsilon:
        return int(random.choice(valid_indices.tolist()))
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t).squeeze(0)
        invalid = torch.tensor(valid_mask <= 0.0, dtype=torch.bool, device=device)
        if not bool(invalid.all().item()):
            q_values = q_values.clone()
            q_values[invalid] = float("-inf")
        return int(torch.argmax(q_values).item())


def _optimize(
    q_net: RestaurantQNetwork,
    target_net: RestaurantQNetwork,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if len(replay) < args.batch_size:
        return None
    batch = replay.sample(args.batch_size)
    states = torch.tensor(
        np.stack([t.state for t in batch]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(
        np.stack([t.next_state for t in batch]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_valid = torch.tensor(
        np.stack([t.next_valid_mask for t in batch]),
        dtype=torch.float32,
        device=device,
    )

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_online = q_net(next_states)
        next_actions = _masked_argmax(next_online, next_valid)
        next_target = target_net(next_states).gather(1, next_actions)
        targets = rewards + args.gamma * (1.0 - dones) * next_target

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace) -> Path:
    device = select_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = RestaurantSymbolicEnv(
        config_path=args.config_path,
        max_steps_per_task=args.max_steps_per_task,
        success_reward=args.success_reward,
        invalid_action_penalty=args.invalid_action_penalty,
        travel_cost_scale=args.travel_cost_scale,
        pick_cost=args.pick_cost,
        place_cost=args.place_cost,
        wash_cost=args.wash_cost,
        fill_cost=args.fill_cost,
        brew_cost=args.brew_cost,
        fruit_cost=args.fruit_cost,
        rng_seed=args.seed,
    )

    run_label = resolve_run_label(args)
    run_dir = Path("runs") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output_name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")

    layout_pool = _load_layout_corpus(args.layout_corpus)
    if args.layout_id:
        layout_pool = [x for x in layout_pool if str(x.get("layout_id")) == args.layout_id]
        if not layout_pool:
            raise ValueError(f"layout-id '{args.layout_id}' not found in {args.layout_corpus}")
    layout_rng = np.random.default_rng(args.seed + 17_171)

    def _reset_env(reset_seed: int, reset_index: int):
        options: Dict[str, Any] = {}
        if layout_pool:
            if args.sample_layout_per_reset:
                layout = layout_pool[int(layout_rng.integers(0, len(layout_pool)))]
            else:
                layout = layout_pool[reset_index % len(layout_pool)]
            options["layout"] = layout
            if args.task_library_per_layout and isinstance(layout.get("task_library"), list):
                options["task_library"] = layout.get("task_library")
        if options:
            return env.reset(seed=reset_seed, options=options)
        return env.reset(seed=reset_seed)

    obs, info = _reset_env(args.seed, 0)
    obs_dim = int(np.asarray(obs).shape[0])
    action_dim = int(env.action_space.n)
    q_net = RestaurantQNetwork(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = RestaurantQNetwork(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    task_return = 0.0
    task_paper2_cost = 0.0
    task_steps = 0
    total_tasks = 0
    tasks_completed = 0
    current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
    steps_since_reset = 0
    tasks_since_episode = 0
    tasks_since_world_reset = 0
    episode_index = 0

    recent_returns: Deque[float] = deque(maxlen=100)
    recent_success: Deque[int] = deque(maxlen=100)
    recent_auto: Deque[int] = deque(maxlen=100)
    loss_history: List[float] = []
    step_reward_history: List[float] = []
    task_records: List[Dict[str, float | int | bool | str | None]] = []

    progress = tqdm(range(args.total_steps), desc="Restaurant DQN", unit="step")
    for global_step in progress:
        epsilon = epsilon_by_step(
            global_step,
            args.epsilon_start,
            args.epsilon_final,
            args.epsilon_decay,
        )
        current_task_snapshot = dict(info.get("task", {}))
        current_task_auto_snapshot = bool(current_task_auto_satisfied)
        current_layout_snapshot = info.get("layout_id")
        valid_mask = np.asarray(info.get("valid_action_mask"), dtype=np.float32)
        action = _select_action(q_net, obs, valid_mask, epsilon, device)
        next_obs, reward, success, truncated, next_info = env.step(action)
        task_return += float(reward)
        task_paper2_cost += float(next_info.get("paper2_cost_step", 0.0))
        task_steps += 1
        step_reward_history.append(float(reward))

        steps_since_reset += 1
        episode_done_flag = False
        env_reset_flag = False
        bootstrap_done = False

        if success:
            tasks_since_episode += 1
            if args.tasks_per_episode > 0 and tasks_since_episode >= args.tasks_per_episode:
                episode_done_flag = True
                bootstrap_done = True
                tasks_since_episode = 0

        if truncated:
            next_obs, next_info = env.advance_task_after_timeout()
        if args.episode_step_limit > 0 and steps_since_reset >= args.episode_step_limit:
            env_reset_flag = True

        transition = Transition(
            state=np.array(obs, dtype=np.float32, copy=True),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_obs, dtype=np.float32, copy=True),
            done=bool(bootstrap_done),
            next_valid_mask=np.array(next_info.get("valid_action_mask"), dtype=np.float32, copy=True),
            task_boundary=bool(success),
        )
        replay.push(transition)

        loss_value = _optimize(q_net, target_net, replay, optimizer, args, device)
        if loss_value is not None:
            loss_history.append(loss_value)

        if args.tau < 1.0:
            with torch.no_grad():
                tau = float(args.tau)
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        elif (global_step + 1) % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        obs = next_obs
        info = next_info

        if success or truncated:
            total_tasks += 1
            if success:
                tasks_completed += 1
            tasks_since_world_reset += 1
            if args.task_sequence_length > 0 and tasks_since_world_reset >= args.task_sequence_length:
                env_reset_flag = True
                tasks_since_world_reset = 0
            recent_returns.append(task_return)
            recent_success.append(1 if success else 0)
            recent_auto.append(1 if current_task_auto_satisfied else 0)
            task_info = next_info.get("task", {})
            task_records.append(
                {
                    "task_number": total_tasks,
                    "success": bool(success),
                    "truncated": bool(truncated),
                    "steps": int(task_steps),
                    "return": float(task_return),
                    "auto_satisfied": current_task_auto_snapshot,
                    "paper2_cost": float(task_paper2_cost),
                    "task_type": current_task_snapshot.get("task_type"),
                    "target_location": current_task_snapshot.get("target_location"),
                    "target_kind": current_task_snapshot.get("target_kind"),
                    "layout_id": current_layout_snapshot,
                    "task_type_after": task_info.get("task_type"),
                    "target_location_after": task_info.get("target_location"),
                    "target_kind_after": task_info.get("target_kind"),
                }
            )
            task_return = 0.0
            task_paper2_cost = 0.0
            task_steps = 0
            current_task_auto_satisfied = bool(next_info.get("next_auto_satisfied", False))

        if env_reset_flag:
            episode_index += 1
            reset_seed = args.seed + 100_003 * episode_index
            obs, info = _reset_env(reset_seed, episode_index)
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
            tasks_since_world_reset = 0
        if episode_done_flag or env_reset_flag:
            steps_since_reset = 0

        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        success_rate = float(np.mean(recent_success)) if recent_success else 0.0
        auto_rate = float(np.mean(recent_auto)) if recent_auto else 0.0
        avg_loss = float(np.mean(loss_history[-100:])) if loss_history else 0.0
        progress.set_postfix(
            ret=f"{avg_return:.1f}" if recent_returns else "n/a",
            success=f"{success_rate:.2f}",
            auto=f"{auto_rate:.2f}",
            eps=f"{epsilon:.2f}",
            loss=f"{avg_loss:.3f}" if loss_history else "n/a",
            tasks=tasks_completed,
        )

    torch.save(q_net.state_dict(), output_path)
    print(f"Saved DQN weights to {output_path}")

    summary = {
        "run_label": run_label,
        "checkpoint": str(output_path),
        "total_steps": int(args.total_steps),
        "tasks_completed": int(tasks_completed),
        "tasks_attempted": int(total_tasks),
        "success_rate": float(tasks_completed / max(1, total_tasks)),
        "avg_task_return": float(np.mean([r["return"] for r in task_records])) if task_records else 0.0,
        "avg_task_steps": float(np.mean([r["steps"] for r in task_records])) if task_records else 0.0,
        "auto_rate": float(np.mean([1.0 if r["auto_satisfied"] else 0.0 for r in task_records])) if task_records else 0.0,
        "reward_per_step": float(np.mean(step_reward_history)) if step_reward_history else 0.0,
        "paper2_cost_total": float(np.sum([r.get("paper2_cost", 0.0) for r in task_records])) if task_records else 0.0,
        "avg_task_paper2_cost": float(np.mean([r.get("paper2_cost", 0.0) for r in task_records])) if task_records else 0.0,
        "mean_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        "tasks_per_episode": int(args.tasks_per_episode),
        "task_sequence_length": int(args.task_sequence_length),
        "layout_corpus": None if args.layout_corpus is None else str(args.layout_corpus),
        "layout_mode": "sample_per_reset" if args.sample_layout_per_reset else "round_robin",
        "seed": int(args.seed),
    }
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with (run_dir / "task_records.json").open("w", encoding="utf-8") as fh:
        json.dump(task_records, fh, indent=2)
    with (run_dir / "train_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on the symbolic restaurant environment.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tasks-per-episode", type=int, default=200)
    parser.add_argument("--task-sequence-length", type=int, default=200, help="Physical reset interval in task attempts.")
    parser.add_argument(
        "--episode-step-limit",
        type=int,
        default=0,
        help="Maximum primitive steps allowed between resets; <=0 disables.",
    )
    parser.add_argument("--max-steps-per-task", type=int, default=100)
    parser.add_argument("--success-reward", type=float, default=15.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--travel-cost-scale", type=float, default=25.0)
    parser.add_argument("--pick-cost", type=float, default=25.0)
    parser.add_argument("--place-cost", type=float, default=25.0)
    parser.add_argument("--wash-cost", type=float, default=25.0)
    parser.add_argument("--fill-cost", type=float, default=25.0)
    parser.add_argument("--brew-cost", type=float, default=25.0)
    parser.add_argument("--fruit-cost", type=float, default=25.0)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"),
    )
    parser.add_argument("--layout-corpus", type=Path, default=None, help="Optional JSON layout corpus with per-layout schemas.")
    parser.add_argument("--layout-id", type=str, default="", help="Optional fixed layout_id from layout corpus.")
    parser.add_argument("--sample-layout-per-reset", action="store_true", help="Sample a random layout each env reset.")
    parser.add_argument(
        "--task-library-per-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-layout task_library when present.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="restaurant_dqn.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
