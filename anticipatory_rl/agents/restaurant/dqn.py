"""Flat-action DQN trainer for the symbolic restaurant domain."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES, RestaurantSymbolicEnv


SUPPORTED_ACTION_TYPES: tuple[str, ...] = ("move", "pick", "place")


class AimLogger:
    """Aim logger with required dependency semantics."""

    def __init__(self, args: argparse.Namespace, run_label: str) -> None:
        try:
            from aim import Run  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Aim logging is required but `aim` is not installed. "
                "Install aim in the active environment before training."
            ) from exc

        self._run = Run(experiment="restaurant_rl_flat")
        self._run["run_label"] = run_label
        self._run["hparams"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        self._run["action_space"] = {"action_types": list(SUPPORTED_ACTION_TYPES), "flat_grounded": True}
        print("[train] Aim logging enabled. Launch UI with `aim up`.")

    def set_metadata(self, key: str, value: object) -> None:
        self._run[key] = value

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self._run.track(value, name=name, step=step, context=dict(context or {}))

    def close(self) -> None:
        close = getattr(self._run, "close", None)
        if callable(close):
            close()

    def track_text(self, text: str, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        try:
            from aim import Text  # type: ignore

            self._run.track(Text(text), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(text, name=name, step=step, context=dict(context or {}))

    def track_image(self, image_path: Path, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        try:
            from aim import Image  # type: ignore

            self._run.track(Image(str(image_path)), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(str(image_path), name=name, step=step, context=dict(context or {}))


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        # Some launchers/scheduler environments can leave torch with an invalid
        # current CUDA ordinal. Force-bind to the first visible GPU.
        try:
            torch.cuda.set_device(0)
        except Exception as exc:  # pragma: no cover - depends on runtime CUDA state
            raise RuntimeError(
                "CUDA is available but failed to select cuda:0. "
                "Check CUDA_VISIBLE_DEVICES and launcher-provided GPU rank env vars."
            ) from exc
        return torch.device("cuda:0")
    return torch.device("cpu")


def epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    return final + (start - final) * np.exp(-float(step) / float(decay))


def _resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label is not None:
        return args.run_label
    return "myopic_restaurant_flat" if args.boundary_mode == "myopic" else "anticipatory_restaurant_flat"


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


def _masked_argmax(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    neg_inf = torch.finfo(values.dtype).min
    masked_values = values.masked_fill(mask <= 0.0, neg_inf)
    indices = torch.argmax(masked_values, dim=1)
    best = masked_values.gather(1, indices.unsqueeze(1)).squeeze(1)
    valid = (mask > 0.0).any(dim=1)
    return indices, best.masked_fill(~valid, neg_inf)


@dataclass(frozen=True)
class FlatAction:
    action_type: int
    object1: int
    location: int
    object2: int

    def to_env_action(self) -> Dict[str, int]:
        return {
            "action_type": int(self.action_type),
            "object1": int(self.object1),
            "location": int(self.location),
            "object2": int(self.object2),
        }


class FlatActionCatalog:
    """Deterministic mapping between action_id and structured env action."""

    def __init__(self, env: RestaurantSymbolicEnv) -> None:
        move_idx = env.action_type_index["move"]
        pick_idx = env.action_type_index["pick"]
        place_idx = env.action_type_index["place"]

        actions: List[FlatAction] = []
        for location in range(env.num_locations):
            actions.append(
                FlatAction(
                    action_type=move_idx,
                    object1=env.none_object_index,
                    location=location,
                    object2=env.none_object_index,
                )
            )
        for obj_idx in range(env.num_objects):
            actions.append(
                FlatAction(
                    action_type=pick_idx,
                    object1=obj_idx,
                    location=env.none_location_index,
                    object2=env.none_object_index,
                )
            )
        for location in range(env.num_locations):
            actions.append(
                FlatAction(
                    action_type=place_idx,
                    object1=env.none_object_index,
                    location=location,
                    object2=env.none_object_index,
                )
            )

        if not actions:
            raise RuntimeError("FlatActionCatalog must contain at least one action.")

        self.actions = actions
        self.num_actions = len(actions)
        self.action_type_idx = np.asarray([a.action_type for a in actions], dtype=np.int64)
        self.object1_idx = np.asarray([a.object1 for a in actions], dtype=np.int64)
        self.location_idx = np.asarray([a.location for a in actions], dtype=np.int64)
        self.object2_idx = np.asarray([a.object2 for a in actions], dtype=np.int64)

    def to_action(self, action_id: int) -> Dict[str, int]:
        return self.actions[int(action_id)].to_env_action()

    def to_string(self, action_id: int, env: RestaurantSymbolicEnv) -> str:
        action = self.actions[int(action_id)]
        action_name = ACTION_TYPES[action.action_type]
        object1_name = "none" if action.object1 >= env.num_objects else env.object_names[action.object1]
        location_name = "none" if action.location >= env.num_locations else env.locations[action.location]
        object2_name = "none" if action.object2 >= env.num_objects else env.object_names[action.object2]
        return f"{action_name}(object1={object1_name}, location={location_name}, object2={object2_name})"

    def project_mask(self, masks: Mapping[str, np.ndarray]) -> np.ndarray:
        action_type_mask = np.asarray(masks["valid_action_type_mask"], dtype=np.float32)
        object1_mask = np.asarray(masks["valid_object1_mask"], dtype=np.float32)
        location_mask = np.asarray(masks["valid_location_mask"], dtype=np.float32)
        object2_mask = np.asarray(masks["valid_object2_mask"], dtype=np.float32)

        flat_valid = action_type_mask[self.action_type_idx] > 0.0
        flat_valid &= object1_mask[self.action_type_idx, self.object1_idx] > 0.0
        flat_valid &= location_mask[self.action_type_idx, self.location_idx] > 0.0
        flat_valid &= object2_mask[self.action_type_idx, self.object1_idx, self.object2_idx] > 0.0
        return flat_valid.astype(np.float32)


@dataclass
class Transition:
    state: np.ndarray
    action_id: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    task_boundary: bool = False


@dataclass
class OptimizeStats:
    loss: float
    q_selected_mean: float
    q_selected_abs_max: float
    target_mean: float
    target_abs_max: float
    td_abs_mean: float
    grad_norm: float


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
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encode(states))


def _choose_oracle_pick_place_action(env: RestaurantSymbolicEnv, task: Mapping[str, object]) -> Dict[str, int]:
    assert task.get("task_type") == "pick_place"
    object_name = str(task["object_name"])
    target_location = str(task["target_location"])
    obj_idx = env.object_name_index[object_name]
    loc_idx = env.location_index[target_location]
    obj = env.state.objects[object_name]
    if env.state.holding == object_name:
        if env.state.agent_location != target_location:
            return {
                "action_type": env.action_type_index["move"],
                "object1": env.none_object_index,
                "location": loc_idx,
                "object2": env.none_object_index,
            }
        return {
            "action_type": env.action_type_index["place"],
            "object1": env.none_object_index,
            "location": loc_idx,
            "object2": env.none_object_index,
        }
    if env.state.holding is not None and env.state.holding != object_name:
        held_target = env.state.agent_location
        return {
            "action_type": env.action_type_index["place"],
            "object1": env.none_object_index,
            "location": env.location_index[held_target],
            "object2": env.none_object_index,
        }
    if obj.location != env.state.agent_location:
        assert obj.location is not None
        return {
            "action_type": env.action_type_index["move"],
            "object1": env.none_object_index,
            "location": env.location_index[obj.location],
            "object2": env.none_object_index,
        }
    return {
        "action_type": env.action_type_index["pick"],
        "object1": obj_idx,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }


def _classify_pick_place_failure(
    env: RestaurantSymbolicEnv,
    task: Mapping[str, object],
    actions: List[Mapping[str, int]],
) -> str:
    object_name = str(task["object_name"])
    target_location = str(task["target_location"])
    picked = False
    placed_at_target = False
    touched_object = False
    for action in actions:
        action_type = ACTION_TYPES[int(action["action_type"])]
        obj_idx = int(action["object1"])
        loc_idx = int(action["location"])
        object1_name = None if obj_idx >= env.num_objects else env.object_names[obj_idx]
        location_name = None if loc_idx >= env.num_locations else env.locations[loc_idx]
        if action_type == "pick" and object1_name == object_name:
            touched_object = True
            picked = True
        if action_type == "place" and location_name == target_location and picked:
            placed_at_target = True
    if not actions:
        return "no_actions"
    if not touched_object:
        return "wrong_object_or_move"
    if touched_object and not picked:
        return "failed_pick"
    if picked and not placed_at_target:
        return "picked_but_failed_place"
    return "timeout_or_mask_issue"


def _select_action_id(
    q_net: RestaurantQNetwork,
    state: np.ndarray,
    flat_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    valid = np.flatnonzero(flat_mask > 0.0)
    if random.random() < epsilon:
        if valid.size > 0:
            return int(random.choice(valid.tolist()))
        return int(random.randrange(int(flat_mask.shape[0])))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t).squeeze(0)
        mask_t = torch.tensor(flat_mask, dtype=torch.float32, device=device)
        return _masked_choice(q_values, mask_t)


def _optimize(
    q_net: RestaurantQNetwork,
    target_net: RestaurantQNetwork,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> OptimizeStats | None:
    if len(replay) < args.batch_size:
        return None

    batch = replay.sample(args.batch_size)
    states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([t.action_id for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=device)
    next_masks = torch.tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.float32, device=device)

    q_selected = q_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        online_next = q_net(next_states)
        target_next_all = target_net(next_states)
        next_indices, _ = _masked_argmax(online_next, next_masks)
        next_q = target_next_all.gather(1, next_indices.unsqueeze(1)).squeeze(1)
        has_valid = (next_masks > 0.0).any(dim=1)
        next_q = next_q.masked_fill(~has_valid, 0.0)
        target = rewards + (1.0 - dones) * args.gamma * next_q

    loss = nn.functional.mse_loss(q_selected, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm_t = nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()

    td = target.detach() - q_selected.detach()
    return OptimizeStats(
        loss=float(loss.item()),
        q_selected_mean=float(q_selected.detach().mean().item()),
        q_selected_abs_max=float(q_selected.detach().abs().max().item()),
        target_mean=float(target.detach().mean().item()),
        target_abs_max=float(target.detach().abs().max().item()),
        td_abs_mean=float(td.abs().mean().item()),
        grad_norm=float(grad_norm_t.item() if hasattr(grad_norm_t, "item") else grad_norm_t),
    )


def _plot_post_train_trajectories(trajectory_records: List[Dict[str, object]], output_path: Path) -> None:
    if not trajectory_records:
        return
    xs = list(range(1, len(trajectory_records) + 1))
    steps = [int(record["steps"]) for record in trajectory_records]
    success = [1.0 if bool(record["success"]) else 0.0 for record in trajectory_records]
    oracle_steps = [int(record["oracle_steps"]) for record in trajectory_records]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(xs, steps, label="policy_steps", color="#1f3b73")
    axes[0].plot(xs, oracle_steps, label="oracle_steps", color="#c2410c", alpha=0.8)
    axes[0].set_ylabel("Steps")
    axes[0].set_title("Post-train rollout trajectories")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].step(xs, success, where="mid", color="#047857")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Success")
    axes[1].set_xlabel("Trajectory index")
    axes[1].grid(alpha=0.3)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)


def _run_post_train_inference(
    q_net: RestaurantQNetwork,
    catalog: FlatActionCatalog,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    aim_logger: AimLogger,
) -> Dict[str, object]:
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
        rng_seed=args.seed + 1_000_000,
    )
    trajectory_dir = run_dir / "post_train_infer"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    q_net.eval()

    trajectory_records: List[Dict[str, object]] = []
    failure_breakdown: Dict[str, int] = {}
    action_type_counts = {name: 0 for name in SUPPORTED_ACTION_TYPES}
    question_counters = {
        "wrong_object_choice": 0,
        "failed_to_move_to_object": 0,
        "failed_after_pick": 0,
        "place_selection_wrong": 0,
        "mask_or_timeout_issue": 0,
    }

    for traj_idx in range(args.post_train_eval_tasks):
        obs, info = env.reset(seed=args.seed + 50_000 + traj_idx)
        task = dict(info.get("task", {}))
        actions: List[Dict[str, int]] = []
        readable_actions: List[str] = []
        total_reward = 0.0
        success = False
        truncated = False
        for _ in range(args.post_train_eval_max_steps):
            masks = _extract_masks(info)
            flat_mask = catalog.project_mask(masks)
            action_id = _select_action_id(q_net, obs, flat_mask, epsilon=0.0, device=device)
            action = catalog.to_action(action_id)
            actions.append(dict(action))
            readable_actions.append(catalog.to_string(action_id, env))
            action_type_counts[ACTION_TYPES[int(action["action_type"])]] += 1
            obs, reward, success, truncated, info = env.step(action)
            total_reward += float(reward)
            if success or truncated:
                break

        oracle_actions: List[str] = []
        oracle_env = RestaurantSymbolicEnv(
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
            rng_seed=args.seed + 1_000_000,
        )
        oracle_env.reset(seed=args.seed + 50_000 + traj_idx)
        oracle_success = False
        oracle_steps = 0
        for _ in range(args.post_train_eval_max_steps):
            oracle_action = _choose_oracle_pick_place_action(oracle_env, task)
            action_type_name = ACTION_TYPES[int(oracle_action["action_type"])]
            oracle_actions.append(
                f"{action_type_name}(object1={oracle_action['object1']}, location={oracle_action['location']}, object2={oracle_action['object2']})"
            )
            _, _, oracle_success, oracle_truncated, _ = oracle_env.step(oracle_action)
            oracle_steps += 1
            if oracle_success or oracle_truncated:
                break

        failure_reason = "success" if success else _classify_pick_place_failure(env, task, actions)
        failure_breakdown[failure_reason] = failure_breakdown.get(failure_reason, 0) + 1
        if not success:
            if failure_reason == "wrong_object_or_move":
                question_counters["wrong_object_choice"] += 1
                question_counters["failed_to_move_to_object"] += 1
            elif failure_reason == "picked_but_failed_place":
                question_counters["failed_after_pick"] += 1
                question_counters["place_selection_wrong"] += 1
            else:
                question_counters["mask_or_timeout_issue"] += 1

        trajectory_records.append(
            {
                "trajectory_index": traj_idx,
                "task": task,
                "success": bool(success),
                "truncated": bool(truncated),
                "steps": len(actions),
                "return": total_reward,
                "failure_reason": failure_reason,
                "actions": readable_actions,
                "oracle_success": bool(oracle_success),
                "oracle_steps": oracle_steps,
                "oracle_actions": oracle_actions,
            }
        )

    summary = {
        "num_trajectories": len(trajectory_records),
        "success_rate": float(np.mean([1.0 if record["success"] else 0.0 for record in trajectory_records])) if trajectory_records else 0.0,
        "avg_steps": float(np.mean([record["steps"] for record in trajectory_records])) if trajectory_records else 0.0,
        "avg_return": float(np.mean([record["return"] for record in trajectory_records])) if trajectory_records else 0.0,
        "oracle_success_rate": float(np.mean([1.0 if record["oracle_success"] else 0.0 for record in trajectory_records])) if trajectory_records else 0.0,
        "oracle_avg_steps": float(np.mean([record["oracle_steps"] for record in trajectory_records])) if trajectory_records else 0.0,
        "failure_breakdown": failure_breakdown,
        "action_type_counts": action_type_counts,
        "debug_questions": question_counters,
    }
    summary_path = trajectory_dir / "trajectory_summary.json"
    trajectories_path = trajectory_dir / "trajectories.json"
    plot_path = trajectory_dir / "trajectory_plot.png"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with trajectories_path.open("w", encoding="utf-8") as fh:
        json.dump(trajectory_records, fh, indent=2)
    _plot_post_train_trajectories(trajectory_records[: args.post_train_plot_trajectories], plot_path)
    aim_logger.set_metadata("post_train_infer", summary)
    aim_logger.track_text(json.dumps(summary, indent=2), name="post_train_summary", step=args.total_steps)
    for record in trajectory_records[: args.post_train_log_trajectories]:
        aim_logger.track_text(
            json.dumps(record, indent=2),
            name="trajectory_trace",
            step=int(record["trajectory_index"]),
            context={"task_type": str(record["task"].get("task_type", "unknown"))},
        )
    if plot_path.exists():
        aim_logger.track_image(plot_path, name="post_train_trajectory_plot", step=args.total_steps)
    return summary


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
    aim_logger.set_metadata("boundary_mode", args.boundary_mode)

    obs, info = env.reset(seed=args.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    catalog = FlatActionCatalog(env)
    aim_logger.set_metadata(
        "model",
        {
            "observation_dim": obs_dim,
            "num_actions": catalog.num_actions,
            "hidden_dim": args.hidden_dim,
            "boundary_mode": args.boundary_mode,
        },
    )
    q_net = RestaurantQNetwork(obs_dim, catalog.num_actions, hidden_dim=args.hidden_dim).to(device)
    target_net = RestaurantQNetwork(obs_dim, catalog.num_actions, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    env_reset_tasks = args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_episode
    task_return = 0.0
    task_steps = 0
    total_tasks = 0
    tasks_completed = 0
    env_tasks_since_reset = 0
    episode_index = 0
    steps_since_reset = 0
    current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
    current_task_actions: List[Dict[str, int]] = []
    current_task_action_strings: List[str] = []

    recent_returns: Deque[float] = deque(maxlen=100)
    recent_success: Deque[int] = deque(maxlen=100)
    recent_auto: Deque[int] = deque(maxlen=100)
    loss_history: List[float] = []
    step_reward_history: List[float] = []
    task_records: List[Dict[str, float | int | bool | str | None]] = []
    optimize_stats_history: List[OptimizeStats] = []
    action_type_counts = {name: 0 for name in SUPPORTED_ACTION_TYPES}
    question_counters = {
        "wrong_object_choice": 0,
        "failed_to_move_to_object": 0,
        "failed_after_pick": 0,
        "place_selection_wrong": 0,
        "mask_or_timeout_issue": 0,
    }

    progress = tqdm(range(args.total_steps), desc="Restaurant DQN Flat", unit="step")
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
        flat_mask = catalog.project_mask(masks)
        action_id = _select_action_id(q_net, obs, flat_mask, epsilon, device)
        action = catalog.to_action(action_id)
        current_task_actions.append(dict(action))
        current_task_action_strings.append(catalog.to_string(action_id, env))
        action_type_name = ACTION_TYPES[int(action["action_type"])]
        if action_type_name in action_type_counts:
            action_type_counts[action_type_name] += 1

        next_obs, reward, success, truncated, next_info = env.step(action)
        task_return += float(reward)
        task_steps += 1
        step_reward_history.append(float(reward))
        steps_since_reset += 1

        next_masks = _extract_masks(next_info)
        next_flat_mask = catalog.project_mask(next_masks)
        success_boundary_terminal = bool(success and args.boundary_mode == "myopic")
        transition_done = bool(truncated or success_boundary_terminal)
        replay.push(
            Transition(
                state=np.array(obs, dtype=np.float32, copy=True),
                action_id=int(action_id),
                reward=float(reward),
                next_state=np.array(next_obs, dtype=np.float32, copy=True),
                done=transition_done,
                next_action_mask=np.array(next_flat_mask, dtype=np.float32, copy=True),
                task_boundary=bool(success),
            )
        )

        optimize_stats = _optimize(q_net, target_net, replay, optimizer, args, device)
        if optimize_stats is not None:
            optimize_stats_history.append(optimize_stats)
            loss_history.append(optimize_stats.loss)
            aim_logger.track(optimize_stats.loss, name="loss", step=global_step, context={"subset": "train"})
            aim_logger.track(optimize_stats.q_selected_mean, name="q_selected_mean", step=global_step)
            aim_logger.track(optimize_stats.q_selected_abs_max, name="q_selected_abs_max", step=global_step)
            aim_logger.track(optimize_stats.target_mean, name="target_mean", step=global_step)
            aim_logger.track(optimize_stats.target_abs_max, name="target_abs_max", step=global_step)
            aim_logger.track(optimize_stats.td_abs_mean, name="td_abs_mean", step=global_step)
            aim_logger.track(optimize_stats.grad_norm, name="grad_norm", step=global_step)

        if args.tau < 1.0:
            with torch.no_grad():
                tau = float(args.tau)
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        elif (global_step + 1) % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        obs = next_obs
        info = next_info

        env_reset_flag = False
        trunc_reset_flag = False
        if success or truncated:
            total_tasks += 1
            if success:
                tasks_completed += 1
            recent_returns.append(task_return)
            recent_success.append(1 if success else 0)
            recent_auto.append(1 if current_task_auto_snapshot else 0)

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
                    "actions": list(current_task_action_strings),
                }
            )

            if not success:
                failure_reason = "timeout_or_mask_issue"
                if current_task_snapshot.get("task_type") == "pick_place":
                    failure_reason = _classify_pick_place_failure(env, current_task_snapshot, current_task_actions)
                    if failure_reason == "wrong_object_or_move":
                        question_counters["wrong_object_choice"] += 1
                        question_counters["failed_to_move_to_object"] += 1
                    elif failure_reason == "picked_but_failed_place":
                        question_counters["failed_after_pick"] += 1
                        question_counters["place_selection_wrong"] += 1
                    else:
                        question_counters["mask_or_timeout_issue"] += 1
                aim_logger.track_text(
                    json.dumps(
                        {
                            "task": current_task_snapshot,
                            "steps": int(task_records[-1]["steps"]),
                            "return": float(task_records[-1]["return"]),
                            "failure_reason": failure_reason,
                            "actions": current_task_action_strings,
                        },
                        indent=2,
                    ),
                    name="failed_task_trace",
                    step=total_tasks,
                    context={"task_type": current_task_snapshot.get("task_type", "unknown")},
                )

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

            task_return = 0.0
            task_steps = 0
            current_task_auto_satisfied = bool(next_info.get("next_auto_satisfied", False))
            current_task_actions = []
            current_task_action_strings = []

            if success:
                env_tasks_since_reset += 1
                if env_reset_tasks is not None and env_reset_tasks > 0 and env_tasks_since_reset >= env_reset_tasks:
                    env_reset_flag = True
            if truncated:
                env_tasks_since_reset = 0
                trunc_reset_flag = True

        if args.episode_step_limit > 0 and steps_since_reset >= args.episode_step_limit:
            trunc_reset_flag = True
            env_tasks_since_reset = 0

        if env_reset_flag or trunc_reset_flag:
            episode_index += 1
            reset_seed = args.seed + 100_003 * episode_index
            obs, info = env.reset(seed=reset_seed)
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
            if env_reset_flag:
                env_tasks_since_reset = 0
            current_task_actions = []
            current_task_action_strings = []
            steps_since_reset = 0

        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        success_rate = float(np.mean(recent_success)) if recent_success else 0.0
        auto_rate = float(np.mean(recent_auto)) if recent_auto else 0.0
        non_auto_success_rate = 0.0
        if recent_success:
            denom = max(1e-8, 1.0 - auto_rate)
            non_auto_success_rate = max(0.0, min(1.0, (success_rate - auto_rate) / denom))
        avg_loss = float(np.mean(loss_history[-100:])) if loss_history else 0.0
        aim_logger.track(epsilon, name="epsilon", step=global_step)
        aim_logger.track(success_rate, name="success_rate_rolling", step=global_step, context={"window": 100})
        aim_logger.track(auto_rate, name="auto_rate_rolling", step=global_step, context={"window": 100})
        aim_logger.track(non_auto_success_rate, name="non_auto_success_rate_rolling", step=global_step, context={"window": 100})
        aim_logger.track(len(replay) / max(1, replay.capacity), name="replay_fill_fraction", step=global_step)
        for action_name, count in action_type_counts.items():
            aim_logger.track(count / max(1, global_step + 1), name="action_type_fraction", step=global_step, context={"action_type": action_name})
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
        "boundary_mode": args.boundary_mode,
        "total_steps": int(args.total_steps),
        "tasks_completed": int(tasks_completed),
        "tasks_attempted": int(total_tasks),
        "success_rate": float(tasks_completed / max(1, total_tasks)),
        "non_auto_success_rate": float(
            max(
                0.0,
                (tasks_completed - sum(1 for r in task_records if bool(r["auto_satisfied"] and r["success"])))
                / max(1, sum(1 for r in task_records if not bool(r["auto_satisfied"]))),
            )
        )
        if task_records
        else 0.0,
        "avg_task_return": float(np.mean([r["return"] for r in task_records])) if task_records else 0.0,
        "avg_task_steps": float(np.mean([r["steps"] for r in task_records])) if task_records else 0.0,
        "auto_rate": float(np.mean([1.0 if r["auto_satisfied"] else 0.0 for r in task_records])) if task_records else 0.0,
        "reward_per_step": float(np.mean(step_reward_history)) if step_reward_history else 0.0,
        "mean_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        "mean_q_selected": float(np.mean([s.q_selected_mean for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "max_abs_q_selected": float(np.max([s.q_selected_abs_max for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "mean_target_q": float(np.mean([s.target_mean for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "max_abs_target_q": float(np.max([s.target_abs_max for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "mean_td_abs": float(np.mean([s.td_abs_mean for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "mean_grad_norm": float(np.mean([s.grad_norm for s in optimize_stats_history])) if optimize_stats_history else 0.0,
        "replay_fill_fraction_final": float(len(replay) / max(1, replay.capacity)),
        "action_type_counts": action_type_counts,
        "debug_questions": question_counters,
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

    post_train_summary = _run_post_train_inference(q_net, catalog, args, device, run_dir, aim_logger)
    summary["post_train_inference"] = post_train_summary
    aim_logger.set_metadata("summary", summary)
    aim_logger.set_metadata("checkpoint_path", str(output_path))
    aim_logger.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train flat-action DQN on the symbolic restaurant environment.")
    parser.add_argument("--boundary-mode", choices=("myopic", "anticipatory"), default="myopic")
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
    parser.add_argument("--tasks-per-episode", type=int, default=1, help="Legacy arg retained for compatibility.")
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
    parser.add_argument("--post-train-eval-tasks", type=int, default=25)
    parser.add_argument("--post-train-eval-max-steps", type=int, default=64)
    parser.add_argument("--post-train-log-trajectories", type=int, default=10)
    parser.add_argument("--post-train-plot-trajectories", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
