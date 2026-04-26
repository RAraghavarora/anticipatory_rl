"""DQN trainer for the symbolic restaurant domain."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.restaurant.env import ACTION_HEADS, ACTION_TYPES, RestaurantSymbolicEnv


OBJECT1_ACTION_MASK = torch.tensor(
    [1.0 if "object1" in ACTION_HEADS[name] else 0.0 for name in ACTION_TYPES],
    dtype=torch.float32,
)
LOCATION_ACTION_MASK = torch.tensor(
    [1.0 if "location" in ACTION_HEADS[name] else 0.0 for name in ACTION_TYPES],
    dtype=torch.float32,
)
OBJECT2_ACTION_MASK = torch.tensor(
    [1.0 if "object2" in ACTION_HEADS[name] else 0.0 for name in ACTION_TYPES],
    dtype=torch.float32,
)


class AimLogger:
    def __init__(self, args: argparse.Namespace, run_label: str) -> None:
        self._run = None
        try:
            from aim import Run  # type: ignore
        except ImportError:
            print("[train] Aim logging disabled: install `aim` to enable experiment tracking.")
            return

        self._run = Run(experiment="restaurant_rl_factored")
        self._run["run_label"] = run_label
        self._run["hparams"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        self._run["action_space"] = {"action_types": list(ACTION_TYPES), "factored": True}
        print("[train] Aim logging enabled. Launch UI with `aim up`.")

    def set_metadata(self, key: str, value: object) -> None:
        if self._run is not None:
            self._run[key] = value

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self._run is None:
            return
        self._run.track(value, name=name, step=step, context=dict(context or {}))

    def close(self) -> None:
        if self._run is None:
            return
        close = getattr(self._run, "close", None)
        if callable(close):
            close()


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    return final + (start - final) * np.exp(-float(step) / float(decay))


def _resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label is not None:
        return args.run_label
    return "myopic_restaurant" if args.tasks_per_episode <= 1 else "anticipatory_restaurant"


def _extract_masks(info: Mapping[str, np.ndarray | List[float] | Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {
        "valid_action_type_mask": np.asarray(info.get("valid_action_type_mask"), dtype=np.float32),
        "valid_object1_mask": np.asarray(info.get("valid_object1_mask"), dtype=np.float32),
        "valid_location_mask": np.asarray(info.get("valid_location_mask"), dtype=np.float32),
        "valid_object2_mask": np.asarray(info.get("valid_object2_mask"), dtype=np.float32),
    }


def _masked_choice(values: torch.Tensor, mask: torch.Tensor) -> int:
    valid = torch.nonzero(mask > 0.0, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return int(torch.argmax(values).item())
    masked = values.clone()
    masked[mask <= 0.0] = float("-inf")
    return int(torch.argmax(masked).item())


def _random_valid_index(mask: np.ndarray) -> int:
    indices = np.flatnonzero(mask > 0.0)
    if indices.size == 0:
        return int(mask.shape[0] - 1)
    return int(random.choice(indices.tolist()))


@dataclass
class Transition:
    state: np.ndarray
    action_type: int
    object1: int
    location: int
    object2: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_type_mask: np.ndarray
    next_object1_mask: np.ndarray
    next_location_mask: np.ndarray
    next_object2_mask: np.ndarray
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
    def __init__(
        self,
        input_dim: int,
        action_type_dim: int,
        object_dim: int,
        location_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_type_head = nn.Linear(hidden_dim, action_type_dim)
        self.object1_head = nn.Linear(hidden_dim, object_dim)
        self.location_head = nn.Linear(hidden_dim, location_dim)
        self.object2_head = nn.Linear(hidden_dim, object_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        return {
            "action_type": self.action_type_head(encoded),
            "object1": self.object1_head(encoded),
            "location": self.location_head(encoded),
            "object2": self.object2_head(encoded),
        }


def _compose_q_values(
    heads: Dict[str, torch.Tensor],
    action_types: torch.Tensor,
    object1: torch.Tensor,
    location: torch.Tensor,
    object2: torch.Tensor,
) -> torch.Tensor:
    q = heads["action_type"].gather(1, action_types)
    q = q + heads["object1"].gather(1, object1) * OBJECT1_ACTION_MASK.to(action_types.device)[action_types.squeeze(1)].unsqueeze(1)
    q = q + heads["location"].gather(1, location) * LOCATION_ACTION_MASK.to(action_types.device)[action_types.squeeze(1)].unsqueeze(1)
    q = q + heads["object2"].gather(1, object2) * OBJECT2_ACTION_MASK.to(action_types.device)[action_types.squeeze(1)].unsqueeze(1)
    return q


def _select_action(
    q_net: RestaurantQNetwork,
    state: np.ndarray,
    masks: Dict[str, np.ndarray],
    epsilon: float,
    device: torch.device,
) -> Dict[str, int]:
    if random.random() < epsilon:
        action_type = _random_valid_index(masks["valid_action_type_mask"])
        object1 = _random_valid_index(masks["valid_object1_mask"][action_type])
        location = _random_valid_index(masks["valid_location_mask"][action_type])
        object2 = _random_valid_index(masks["valid_object2_mask"][action_type, object1])
        return {
            "action_type": int(action_type),
            "object1": int(object1),
            "location": int(location),
            "object2": int(object2),
        }
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        heads = q_net(state_t)
        action_type_mask = torch.tensor(masks["valid_action_type_mask"], dtype=torch.float32, device=device)
        action_type = _masked_choice(heads["action_type"].squeeze(0), action_type_mask)
        object1_mask = torch.tensor(masks["valid_object1_mask"][action_type], dtype=torch.float32, device=device)
        object1 = _masked_choice(heads["object1"].squeeze(0), object1_mask)
        location_mask = torch.tensor(masks["valid_location_mask"][action_type], dtype=torch.float32, device=device)
        location = _masked_choice(heads["location"].squeeze(0), location_mask)
        object2_mask = torch.tensor(masks["valid_object2_mask"][action_type, object1], dtype=torch.float32, device=device)
        object2 = _masked_choice(heads["object2"].squeeze(0), object2_mask)
        return {
            "action_type": int(action_type),
            "object1": int(object1),
            "location": int(location),
            "object2": int(object2),
        }


def _select_greedy_actions_batch(
    heads: Dict[str, torch.Tensor],
    action_type_masks: torch.Tensor,
    object1_masks: torch.Tensor,
    location_masks: torch.Tensor,
    object2_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = heads["action_type"].shape[0]
    chosen_action_type: List[int] = []
    chosen_object1: List[int] = []
    chosen_location: List[int] = []
    chosen_object2: List[int] = []
    for idx in range(batch_size):
        action_type = _masked_choice(heads["action_type"][idx], action_type_masks[idx])
        object1 = _masked_choice(heads["object1"][idx], object1_masks[idx, action_type])
        location = _masked_choice(heads["location"][idx], location_masks[idx, action_type])
        object2 = _masked_choice(heads["object2"][idx], object2_masks[idx, action_type, object1])
        chosen_action_type.append(action_type)
        chosen_object1.append(object1)
        chosen_location.append(location)
        chosen_object2.append(object2)
    device = heads["action_type"].device
    return (
        torch.tensor(chosen_action_type, dtype=torch.int64, device=device).unsqueeze(1),
        torch.tensor(chosen_object1, dtype=torch.int64, device=device).unsqueeze(1),
        torch.tensor(chosen_location, dtype=torch.int64, device=device).unsqueeze(1),
        torch.tensor(chosen_object2, dtype=torch.int64, device=device).unsqueeze(1),
    )


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
    states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    action_type = torch.tensor([t.action_type for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    object1 = torch.tensor([t.object1 for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    location = torch.tensor([t.location for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    object2 = torch.tensor([t.object2 for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_action_type_masks = torch.tensor(np.stack([t.next_action_type_mask for t in batch]), dtype=torch.float32, device=device)
    next_object1_masks = torch.tensor(np.stack([t.next_object1_mask for t in batch]), dtype=torch.float32, device=device)
    next_location_masks = torch.tensor(np.stack([t.next_location_mask for t in batch]), dtype=torch.float32, device=device)
    next_object2_masks = torch.tensor(np.stack([t.next_object2_mask for t in batch]), dtype=torch.float32, device=device)

    heads = q_net(states)
    q_values = _compose_q_values(heads, action_type, object1, location, object2)
    with torch.no_grad():
        next_online = q_net(next_states)
        next_action_type, next_object1, next_location, next_object2 = _select_greedy_actions_batch(
            next_online,
            next_action_type_masks,
            next_object1_masks,
            next_location_masks,
            next_object2_masks,
        )
        next_target = target_net(next_states)
        next_q = _compose_q_values(next_target, next_action_type, next_object1, next_location, next_object2)
        targets = rewards + args.gamma * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace) -> Path:
    device = _select_device()
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

    run_label = _resolve_run_label(args)
    run_dir = Path("runs") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output_name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")
    aim_logger = AimLogger(args, run_label)
    aim_logger.set_metadata("run_dir", str(run_dir.resolve()))
    aim_logger.set_metadata("config_path", str(Path(args.config_path).resolve()))

    obs, info = env.reset(seed=args.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    object_dim = int(env.action_space["object1"].n)
    location_dim = int(env.action_space["location"].n)
    action_type_dim = int(env.action_space["action_type"].n)
    aim_logger.set_metadata(
        "model",
        {
            "observation_dim": obs_dim,
            "action_type_dim": action_type_dim,
            "object_dim": object_dim,
            "location_dim": location_dim,
            "hidden_dim": args.hidden_dim,
        },
    )
    q_net = RestaurantQNetwork(obs_dim, action_type_dim, object_dim, location_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = RestaurantQNetwork(obs_dim, action_type_dim, object_dim, location_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    env_reset_tasks = args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_episode
    if args.tasks_per_episode > 1 and env_reset_tasks != args.tasks_per_episode:
        raise ValueError("For anticipatory runs, env-reset-tasks must equal tasks-per-episode.")

    task_return = 0.0
    task_steps = 0
    total_tasks = 0
    tasks_completed = 0
    current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
    steps_since_reset = 0
    tasks_since_reset = 0
    env_tasks_since_reset = 0
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
        masks = _extract_masks(info)
        action = _select_action(q_net, obs, masks, epsilon, device)
        next_obs, reward, success, truncated, next_info = env.step(action)
        task_return += float(reward)
        task_steps += 1
        step_reward_history.append(float(reward))

        steps_since_reset += 1
        episode_done_flag = False
        env_reset_flag = False
        trunc_reset_flag = False
        bootstrap_done = False

        if success:
            tasks_since_reset += 1
            env_tasks_since_reset += 1
            if args.tasks_per_episode > 0 and tasks_since_reset >= args.tasks_per_episode:
                episode_done_flag = True
                bootstrap_done = True
                tasks_since_reset = 0
            if env_reset_tasks is not None and env_reset_tasks > 0 and env_tasks_since_reset >= env_reset_tasks:
                env_reset_flag = True
                episode_done_flag = True
                bootstrap_done = True
                env_tasks_since_reset = 0

        if truncated:
            trunc_reset_flag = True
            tasks_since_reset = 0
            env_tasks_since_reset = 0
        if args.episode_step_limit > 0 and steps_since_reset >= args.episode_step_limit:
            trunc_reset_flag = True
            tasks_since_reset = 0
            env_tasks_since_reset = 0

        next_masks = _extract_masks(next_info)
        replay.push(
            Transition(
                state=np.array(obs, dtype=np.float32, copy=True),
                action_type=int(action["action_type"]),
                object1=int(action["object1"]),
                location=int(action["location"]),
                object2=int(action["object2"]),
                reward=float(reward),
                next_state=np.array(next_obs, dtype=np.float32, copy=True),
                done=bool(bootstrap_done),
                next_action_type_mask=np.array(next_masks["valid_action_type_mask"], dtype=np.float32, copy=True),
                next_object1_mask=np.array(next_masks["valid_object1_mask"], dtype=np.float32, copy=True),
                next_location_mask=np.array(next_masks["valid_location_mask"], dtype=np.float32, copy=True),
                next_object2_mask=np.array(next_masks["valid_object2_mask"], dtype=np.float32, copy=True),
                task_boundary=bool(success),
            )
        )

        loss_value = _optimize(q_net, target_net, replay, optimizer, args, device)
        if loss_value is not None:
            loss_history.append(loss_value)
            aim_logger.track(loss_value, name="loss", step=global_step, context={"subset": "train"})

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
                    "task_type": current_task_snapshot.get("task_type"),
                    "target_location": current_task_snapshot.get("target_location"),
                    "target_kind": current_task_snapshot.get("target_kind"),
                    "task_type_after": task_info.get("task_type"),
                    "target_location_after": task_info.get("target_location"),
                    "target_kind_after": task_info.get("target_kind"),
                }
            )
            task_return = 0.0
            task_steps = 0
            current_task_auto_satisfied = bool(next_info.get("next_auto_satisfied", False))
            aim_logger.track(
                float(task_records[-1]["return"]),
                name="task_return",
                step=total_tasks,
                context={"task_type": current_task_snapshot.get("task_type", "unknown")},
            )
            aim_logger.track(
                float(task_records[-1]["steps"]),
                name="task_steps",
                step=total_tasks,
                context={"task_type": current_task_snapshot.get("task_type", "unknown")},
            )
            aim_logger.track(
                1.0 if success else 0.0,
                name="task_success",
                step=total_tasks,
                context={"task_type": current_task_snapshot.get("task_type", "unknown")},
            )

        if env_reset_flag or trunc_reset_flag:
            episode_index += 1
            reset_seed = args.seed + 100_003 * episode_index
            obs, info = env.reset(seed=reset_seed)
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
            env_tasks_since_reset = 0
        if episode_done_flag or trunc_reset_flag:
            steps_since_reset = 0

        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        success_rate = float(np.mean(recent_success)) if recent_success else 0.0
        auto_rate = float(np.mean(recent_auto)) if recent_auto else 0.0
        avg_loss = float(np.mean(loss_history[-100:])) if loss_history else 0.0
        aim_logger.track(epsilon, name="epsilon", step=global_step)
        aim_logger.track(success_rate, name="success_rate_rolling", step=global_step, context={"window": 100})
        aim_logger.track(auto_rate, name="auto_rate_rolling", step=global_step, context={"window": 100})
        if recent_returns:
            aim_logger.track(avg_return, name="avg_task_return_rolling", step=global_step, context={"window": 100})
        if loss_history:
            aim_logger.track(avg_loss, name="avg_loss_rolling", step=global_step, context={"window": 100})
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
        "mean_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        "tasks_per_episode": int(args.tasks_per_episode),
        "env_reset_tasks": None if env_reset_tasks is None else int(env_reset_tasks),
        "seed": int(args.seed),
    }
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with (run_dir / "task_records.json").open("w", encoding="utf-8") as fh:
        json.dump(task_records, fh, indent=2)
    with (run_dir / "train_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    aim_logger.set_metadata("summary", summary)
    aim_logger.set_metadata("checkpoint_path", str(output_path))
    aim_logger.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on the symbolic restaurant environment.")
    parser.add_argument("--total-steps", type=int, default=500_000)
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
    parser.add_argument("--tasks-per-episode", type=int, default=1)
    parser.add_argument("--env-reset-tasks", type=int, default=200, help="Physical env reset interval in tasks.")
    parser.add_argument("--episode-step-limit", type=int, default=3000, help="Maximum primitive steps allowed between resets; <=0 disables.")
    parser.add_argument("--max-steps-per-task", type=int, default=64)
    parser.add_argument("--success-reward", type=float, default=15.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--travel-cost-scale", type=float, default=1.0)
    parser.add_argument("--pick-cost", type=float, default=1.0)
    parser.add_argument("--place-cost", type=float, default=1.0)
    parser.add_argument("--wash-cost", type=float, default=2.0)
    parser.add_argument("--fill-cost", type=float, default=1.0)
    parser.add_argument("--brew-cost", type=float, default=2.0)
    parser.add_argument("--fruit-cost", type=float, default=2.0)
    parser.add_argument("--config-path", type=Path, default=Path("anticipatory_rl/configs/restaurant/restaurant_symbolic.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="restaurant_dqn.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
