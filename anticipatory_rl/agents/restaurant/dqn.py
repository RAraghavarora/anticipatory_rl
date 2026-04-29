"""DQN trainer for the symbolic restaurant domain."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.restaurant.env import ACTION_HEADS, ACTION_TYPES, RestaurantSymbolicEnv


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

    def track_text(self, text: str, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is None:
            return
        try:
            from aim import Text  # type: ignore
            self._run.track(Text(text), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(text, name=name, step=step, context=dict(context or {}))

    def track_image(self, image_path: Path, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is None:
            return
        try:
            from aim import Image  # type: ignore
            self._run.track(Image(str(image_path)), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(str(image_path), name=name, step=step, context=dict(context or {}))


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


def _masked_argmax(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    neg_inf = torch.finfo(values.dtype).min
    masked_values = values.masked_fill(mask <= 0.0, neg_inf)
    indices = torch.argmax(masked_values, dim=1)
    best = masked_values.gather(1, indices.unsqueeze(1)).squeeze(1)
    valid = (mask > 0.0).any(dim=1)
    return indices, best.masked_fill(~valid, neg_inf)


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
    action_type_mask: np.ndarray
    object1_mask: np.ndarray
    location_mask: np.ndarray
    object2_mask: np.ndarray
    next_state: np.ndarray
    done: bool
    next_action_type_mask: np.ndarray
    next_object1_mask: np.ndarray
    next_location_mask: np.ndarray
    next_object2_mask: np.ndarray
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
    def __init__(
        self,
        input_dim: int,
        action_type_dim: int,
        object_dim: int,
        location_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_type_dim = int(action_type_dim)
        self.object_dim = int(object_dim)
        self.location_dim = int(location_dim)
        self.prefix_embed_dim = max(16, hidden_dim // 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.action_type_adv_head = nn.Linear(hidden_dim, action_type_dim)

        self.action_type_embed = nn.Embedding(action_type_dim, self.prefix_embed_dim)
        self.object_embed = nn.Embedding(object_dim, self.prefix_embed_dim)
        self.location_embed = nn.Embedding(location_dim, self.prefix_embed_dim)

        self.object1_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim),
        )
        self.location_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, location_dim),
        )
        self.object2_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 * self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def action_type_scores(self, encoded: torch.Tensor, action_type_mask: torch.Tensor) -> torch.Tensor:
        value = self.value_head(encoded)
        advantages = self.action_type_adv_head(encoded)
        centered = advantages - _masked_mean(advantages, action_type_mask)
        return value + centered

    def object1_scores(
        self,
        encoded: torch.Tensor,
        action_types: torch.Tensor,
        object1_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = torch.cat([encoded, self.action_type_embed(action_types.squeeze(1))], dim=1)
        advantages = self.object1_adv_head(prefix)
        return advantages - _masked_mean(advantages, object1_mask)

    def location_scores(
        self,
        encoded: torch.Tensor,
        action_types: torch.Tensor,
        object1: torch.Tensor,
        location_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = torch.cat(
            [
                encoded,
                self.action_type_embed(action_types.squeeze(1)),
                self.object_embed(object1.squeeze(1)),
            ],
            dim=1,
        )
        advantages = self.location_adv_head(prefix)
        return advantages - _masked_mean(advantages, location_mask)

    def object2_scores(
        self,
        encoded: torch.Tensor,
        action_types: torch.Tensor,
        object1: torch.Tensor,
        location: torch.Tensor,
        object2_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = torch.cat(
            [
                encoded,
                self.action_type_embed(action_types.squeeze(1)),
                self.object_embed(object1.squeeze(1)),
                self.location_embed(location.squeeze(1)),
            ],
            dim=1,
        )
        advantages = self.object2_adv_head(prefix)
        return advantages - _masked_mean(advantages, object2_mask)

    def forward(
        self,
        states: torch.Tensor,
        *,
        action_types: torch.Tensor | None = None,
        object1: torch.Tensor | None = None,
        location: torch.Tensor | None = None,
        object2: torch.Tensor | None = None,
        action_type_masks: torch.Tensor | None = None,
        object1_masks: torch.Tensor | None = None,
        location_masks: torch.Tensor | None = None,
        object2_masks: torch.Tensor | None = None,
        decode_greedy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(states)
        if decode_greedy:
            if action_type_masks is None or object1_masks is None or location_masks is None or object2_masks is None:
                raise ValueError("decode_greedy requires all action masks.")
            return _select_greedy_actions_batch(
                self,
                encoded,
                action_type_masks,
                object1_masks,
                location_masks,
                object2_masks,
            )

        if action_types is None:
            return encoded
        if object1 is None or location is None or object2 is None:
            raise ValueError("compose_q mode requires object1, location, and object2 tensors.")
        if action_type_masks is None or object1_masks is None or location_masks is None or object2_masks is None:
            raise ValueError("compose_q mode requires all action masks.")
        return _compose_q_values(
            self,
            encoded,
            action_types,
            object1,
            location,
            object2,
            action_type_masks,
            object1_masks,
            location_masks,
            object2_masks,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_sum = (values * mask).sum(dim=1, keepdim=True)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked_sum / denom


def _compose_q_values(
    q_net: RestaurantQNetwork,
    encoded: torch.Tensor,
    action_types: torch.Tensor,
    object1: torch.Tensor,
    location: torch.Tensor,
    object2: torch.Tensor,
    action_type_masks: torch.Tensor,
    object1_masks: torch.Tensor,
    location_masks: torch.Tensor,
    object2_masks: torch.Tensor,
) -> torch.Tensor:
    action_type_scores = q_net.action_type_scores(encoded, action_type_masks)
    q = action_type_scores.gather(1, action_types)

    chosen_object1_masks = object1_masks[torch.arange(encoded.shape[0], device=encoded.device), action_types.squeeze(1)]
    object1_scores = q_net.object1_scores(encoded, action_types, chosen_object1_masks)
    q = q + object1_scores.gather(1, object1)

    chosen_location_masks = location_masks[torch.arange(encoded.shape[0], device=encoded.device), action_types.squeeze(1)]
    location_scores = q_net.location_scores(encoded, action_types, object1, chosen_location_masks)
    q = q + location_scores.gather(1, location)

    chosen_object2_masks = object2_masks[
        torch.arange(encoded.shape[0], device=encoded.device),
        action_types.squeeze(1),
        object1.squeeze(1),
    ]
    object2_scores = q_net.object2_scores(encoded, action_types, object1, location, chosen_object2_masks)
    q = q + object2_scores.gather(1, object2)
    return q


def _action_to_string(env: RestaurantSymbolicEnv, action: Mapping[str, int]) -> str:
    action_type = ACTION_TYPES[int(action["action_type"])]
    object1 = int(action["object1"])
    location = int(action["location"])
    object2 = int(action["object2"])
    object1_name = "none" if object1 >= env.num_objects else env.object_names[object1]
    location_name = "none" if location >= env.num_locations else env.locations[location]
    object2_name = "none" if object2 >= env.num_objects else env.object_names[object2]
    return f"{action_type}(object1={object1_name}, location={location_name}, object2={object2_name})"


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


def _classify_pick_place_failure(env: RestaurantSymbolicEnv, task: Mapping[str, object], actions: List[Mapping[str, int]]) -> str:
    object_name = str(task["object_name"])
    target_location = str(task["target_location"])
    picked = False
    placed_at_target = False
    touched_object = False
    reached_object = False
    for action in actions:
        action_type = ACTION_TYPES[int(action["action_type"])]
        obj_idx = int(action["object1"])
        loc_idx = int(action["location"])
        object1_name = None if obj_idx >= env.num_objects else env.object_names[obj_idx]
        location_name = None if loc_idx >= env.num_locations else env.locations[loc_idx]
        if action_type == "move" and location_name == env.state.objects[object_name].location:
            reached_object = True
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
        encoded = q_net.encode(state_t)
        action_type_mask = torch.tensor(masks["valid_action_type_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        object1_masks = torch.tensor(masks["valid_object1_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        location_masks = torch.tensor(masks["valid_location_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        object2_masks = torch.tensor(masks["valid_object2_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        action_type, object1, location, object2 = _select_greedy_actions_batch(
            q_net,
            encoded,
            action_type_mask,
            object1_masks,
            location_masks,
            object2_masks,
        )
        return {
            "action_type": int(action_type.item()),
            "object1": int(object1.item()),
            "location": int(location.item()),
            "object2": int(object2.item()),
        }


def _select_greedy_actions_batch(
    q_net: RestaurantQNetwork,
    encoded: torch.Tensor,
    action_type_masks: torch.Tensor,
    object1_masks: torch.Tensor,
    location_masks: torch.Tensor,
    object2_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = encoded.device
    batch_size, hidden_dim = encoded.shape
    action_type_dim = action_type_masks.shape[1]
    object_dim = object1_masks.shape[2]
    location_dim = location_masks.shape[2]

    if batch_size == 0:
        empty = torch.empty((0, 1), dtype=torch.int64, device=device)
        return empty, empty, empty, empty

    action_type_scores = q_net.action_type_scores(encoded, action_type_masks)
    none_object = object_dim - 1
    none_location = location_dim - 1
    none_object_tensor = torch.full((batch_size, 1), none_object, dtype=torch.int64, device=device)
    none_location_tensor = torch.full((batch_size, 1), none_location, dtype=torch.int64, device=device)

    best_total = torch.full((batch_size,), torch.finfo(action_type_scores.dtype).min, dtype=action_type_scores.dtype, device=device)
    best_action_type = torch.full((batch_size,), none_object, dtype=torch.int64, device=device)
    best_object1 = torch.full((batch_size,), none_object, dtype=torch.int64, device=device)
    best_location = torch.full((batch_size,), none_location, dtype=torch.int64, device=device)
    best_object2 = torch.full((batch_size,), none_object, dtype=torch.int64, device=device)

    action_type_ids = torch.arange(action_type_dim, dtype=torch.int64, device=device)
    object_ids = torch.arange(object_dim, dtype=torch.int64, device=device)
    location_ids = torch.arange(location_dim, dtype=torch.int64, device=device)

    for action_type_t, action_name in enumerate(ACTION_TYPES):
        type_valid = action_type_masks[:, action_type_t] > 0.0
        if not torch.any(type_valid):
            continue
        action_type_tensor = torch.full((batch_size, 1), action_type_t, dtype=torch.int64, device=device)
        action_type_component = action_type_scores[:, action_type_t]
        heads = ACTION_HEADS[action_name]

        if heads == ("object1",):
            object1_mask = object1_masks[:, action_type_t, :]
            object1_scores = q_net.object1_scores(encoded, action_type_tensor, object1_mask)
            candidate_object1, object1_component = _masked_argmax(object1_scores, object1_mask)

            location_mask = location_masks[:, action_type_t, :]
            location_scores = q_net.location_scores(encoded, action_type_tensor, candidate_object1.unsqueeze(1), location_mask)
            location_component = location_scores[:, none_location]

            object2_mask = object2_masks[torch.arange(batch_size, device=device), action_type_t, candidate_object1, :]
            object2_scores = q_net.object2_scores(
                encoded,
                action_type_tensor,
                candidate_object1.unsqueeze(1),
                none_location_tensor,
                object2_mask,
            )
            object2_component = object2_scores[:, none_object]

            total = action_type_component + object1_component + location_component + object2_component
            should_update = type_valid & (total > best_total)
            best_total = torch.where(should_update, total, best_total)
            best_action_type = torch.where(should_update, torch.full_like(best_action_type, action_type_t), best_action_type)
            best_object1 = torch.where(should_update, candidate_object1, best_object1)
            best_location = torch.where(should_update, torch.full_like(best_location, none_location), best_location)
            best_object2 = torch.where(should_update, torch.full_like(best_object2, none_object), best_object2)
            continue

        if heads == ("location",):
            object1_mask = object1_masks[:, action_type_t, :]
            object1_scores = q_net.object1_scores(encoded, action_type_tensor, object1_mask)
            object1_component = object1_scores[:, none_object]

            location_mask = location_masks[:, action_type_t, :]
            location_scores = q_net.location_scores(encoded, action_type_tensor, none_object_tensor, location_mask)
            candidate_location, location_component = _masked_argmax(location_scores, location_mask)

            object2_mask = object2_masks[:, action_type_t, none_object, :]
            object2_scores = q_net.object2_scores(
                encoded,
                action_type_tensor,
                none_object_tensor,
                candidate_location.unsqueeze(1),
                object2_mask,
            )
            object2_component = object2_scores[:, none_object]

            total = action_type_component + object1_component + location_component + object2_component
            should_update = type_valid & (total > best_total)
            best_total = torch.where(should_update, total, best_total)
            best_action_type = torch.where(should_update, torch.full_like(best_action_type, action_type_t), best_action_type)
            best_object1 = torch.where(should_update, torch.full_like(best_object1, none_object), best_object1)
            best_location = torch.where(should_update, candidate_location, best_location)
            best_object2 = torch.where(should_update, torch.full_like(best_object2, none_object), best_object2)
            continue

        if heads == ("object1", "object2"):
            object1_mask = object1_masks[:, action_type_t, :]
            object1_scores = q_net.object1_scores(encoded, action_type_tensor, object1_mask)

            object_ids_b = object_ids.view(1, object_dim).expand(batch_size, object_dim)
            location_mask = location_masks[:, action_type_t, :].unsqueeze(1).expand(batch_size, object_dim, location_dim).reshape(-1, location_dim)
            location_scores = q_net.location_scores(
                encoded.unsqueeze(1).expand(batch_size, object_dim, hidden_dim).reshape(-1, hidden_dim),
                action_type_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object_ids_b.reshape(-1, 1),
                location_mask,
            ).reshape(batch_size, object_dim, location_dim)
            location_component = location_scores[:, :, none_location]

            object2_scores = q_net.object2_scores(
                encoded.unsqueeze(1).expand(batch_size, object_dim, hidden_dim).reshape(-1, hidden_dim),
                action_type_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object_ids_b.reshape(-1, 1),
                none_location_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object2_masks[:, action_type_t, :, :].reshape(-1, object_dim),
            ).reshape(batch_size, object_dim, object_dim)

            total_scores = (
                action_type_component.unsqueeze(1).unsqueeze(2)
                + object1_scores.unsqueeze(2)
                + location_component.unsqueeze(2)
                + object2_scores
            )
            valid_combo = (
                type_valid.unsqueeze(1).unsqueeze(2)
                & (object1_mask > 0.0).unsqueeze(2)
                & (object2_masks[:, action_type_t, :, :] > 0.0)
            )
            neg_inf = torch.finfo(total_scores.dtype).min
            total_scores = total_scores.masked_fill(~valid_combo, neg_inf)
            flat_scores = total_scores.reshape(batch_size, -1)
            best_pair_flat = torch.argmax(flat_scores, dim=1)
            pair_valid = valid_combo.reshape(batch_size, -1).any(dim=1)
            candidate_object1 = torch.div(best_pair_flat, object_dim, rounding_mode="floor")
            candidate_object2 = torch.remainder(best_pair_flat, object_dim)
            total = flat_scores.gather(1, best_pair_flat.unsqueeze(1)).squeeze(1)
            should_update = pair_valid & (total > best_total)
            best_total = torch.where(should_update, total, best_total)
            best_action_type = torch.where(should_update, torch.full_like(best_action_type, action_type_t), best_action_type)
            best_object1 = torch.where(should_update, candidate_object1, best_object1)
            best_location = torch.where(should_update, torch.full_like(best_location, none_location), best_location)
            best_object2 = torch.where(should_update, candidate_object2, best_object2)
            continue

        raise ValueError(f"Unsupported action signature for decoder: {heads}")

    has_valid = best_total > torch.finfo(best_total.dtype).min / 2
    fallback_indices = torch.nonzero(~has_valid, as_tuple=False).squeeze(-1)
    if fallback_indices.numel() > 0:
        for idx in fallback_indices.tolist():
            action_type = _masked_choice(action_type_scores[idx], action_type_masks[idx])
            object1 = _random_valid_index(object1_masks[idx, action_type].detach().cpu().numpy())
            location = _random_valid_index(location_masks[idx, action_type].detach().cpu().numpy())
            object2 = _random_valid_index(object2_masks[idx, action_type, object1].detach().cpu().numpy())
            best_action_type[idx] = int(action_type)
            best_object1[idx] = int(object1)
            best_location[idx] = int(location)
            best_object2[idx] = int(object2)

    return (
        best_action_type.unsqueeze(1),
        best_object1.unsqueeze(1),
        best_location.unsqueeze(1),
        best_object2.unsqueeze(1),
    )


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
    action_type = torch.tensor([t.action_type for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    object1 = torch.tensor([t.object1 for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    location = torch.tensor([t.location for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    object2 = torch.tensor([t.object2 for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    action_type_masks = torch.tensor(np.stack([t.action_type_mask for t in batch]), dtype=torch.float32, device=device)
    object1_masks = torch.tensor(np.stack([t.object1_mask for t in batch]), dtype=torch.float32, device=device)
    location_masks = torch.tensor(np.stack([t.location_mask for t in batch]), dtype=torch.float32, device=device)
    object2_masks = torch.tensor(np.stack([t.object2_mask for t in batch]), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_action_type_masks = torch.tensor(np.stack([t.next_action_type_mask for t in batch]), dtype=torch.float32, device=device)
    next_object1_masks = torch.tensor(np.stack([t.next_object1_mask for t in batch]), dtype=torch.float32, device=device)
    next_location_masks = torch.tensor(np.stack([t.next_location_mask for t in batch]), dtype=torch.float32, device=device)
    next_object2_masks = torch.tensor(np.stack([t.next_object2_mask for t in batch]), dtype=torch.float32, device=device)

    q_values = q_net(
        states,
        action_types=action_type,
        object1=object1,
        location=location,
        object2=object2,
        action_type_masks=action_type_masks,
        object1_masks=object1_masks,
        location_masks=location_masks,
        object2_masks=object2_masks,
    )
    with torch.no_grad():
        next_action_type, next_object1, next_location, next_object2 = q_net(
            next_states,
            action_type_masks=next_action_type_masks,
            object1_masks=next_object1_masks,
            location_masks=next_location_masks,
            object2_masks=next_object2_masks,
            decode_greedy=True,
        )
        next_q = target_net(
            next_states,
            action_types=next_action_type,
            object1=next_object1,
            location=next_location,
            object2=next_object2,
            action_type_masks=next_action_type_masks,
            object1_masks=next_object1_masks,
            location_masks=next_location_masks,
            object2_masks=next_object2_masks,
        )
        targets = rewards + args.gamma * (1.0 - dones) * next_q

    td = q_values - targets
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    grad_norm_t = nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()
    return OptimizeStats(
        loss=float(loss.item()),
        q_selected_mean=float(q_values.mean().item()),
        q_selected_abs_max=float(q_values.abs().max().item()),
        target_mean=float(targets.mean().item()),
        target_abs_max=float(targets.abs().max().item()),
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
    action_type_counts = {name: 0 for name in ACTION_TYPES}
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
            action = _select_action(q_net, obs, masks, epsilon=0.0, device=device)
            actions.append(dict(action))
            readable_actions.append(_action_to_string(env, action))
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
        oracle_obs, oracle_info = oracle_env.reset(seed=args.seed + 50_000 + traj_idx)
        del oracle_obs, oracle_info
        oracle_success = False
        oracle_steps = 0
        for _ in range(args.post_train_eval_max_steps):
            oracle_action = _choose_oracle_pick_place_action(oracle_env, task)
            oracle_actions.append(_action_to_string(oracle_env, oracle_action))
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
    current_task_actions: List[Dict[str, int]] = []
    current_task_action_strings: List[str] = []
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
    optimize_stats_history: List[OptimizeStats] = []
    action_type_counts = {name: 0 for name in ACTION_TYPES}
    question_counters = {
        "wrong_object_choice": 0,
        "failed_to_move_to_object": 0,
        "failed_after_pick": 0,
        "place_selection_wrong": 0,
        "mask_or_timeout_issue": 0,
    }

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
        current_task_actions.append(dict(action))
        current_task_action_strings.append(_action_to_string(env, action))
        action_type_counts[ACTION_TYPES[int(action["action_type"])]] += 1
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
        transition_done = bool(bootstrap_done or truncated)
        replay.push(
            Transition(
                state=np.array(obs, dtype=np.float32, copy=True),
                action_type=int(action["action_type"]),
                object1=int(action["object1"]),
                location=int(action["location"]),
                object2=int(action["object2"]),
                reward=float(reward),
                action_type_mask=np.array(masks["valid_action_type_mask"], dtype=np.float32, copy=True),
                object1_mask=np.array(masks["valid_object1_mask"], dtype=np.float32, copy=True),
                location_mask=np.array(masks["valid_location_mask"], dtype=np.float32, copy=True),
                object2_mask=np.array(masks["valid_object2_mask"], dtype=np.float32, copy=True),
                next_state=np.array(next_obs, dtype=np.float32, copy=True),
                done=transition_done,
                next_action_type_mask=np.array(next_masks["valid_action_type_mask"], dtype=np.float32, copy=True),
                next_object1_mask=np.array(next_masks["valid_object1_mask"], dtype=np.float32, copy=True),
                next_location_mask=np.array(next_masks["valid_location_mask"], dtype=np.float32, copy=True),
                next_object2_mask=np.array(next_masks["valid_object2_mask"], dtype=np.float32, copy=True),
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
                    "actions": list(current_task_action_strings),
                }
            )
            task_return = 0.0
            task_steps = 0
            current_task_auto_satisfied = bool(next_info.get("next_auto_satisfied", False))
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
            current_task_actions = []
            current_task_action_strings = []

        if env_reset_flag or trunc_reset_flag:
            episode_index += 1
            reset_seed = args.seed + 100_003 * episode_index
            obs, info = env.reset(seed=reset_seed)
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
            env_tasks_since_reset = 0
            current_task_actions = []
            current_task_action_strings = []
        if episode_done_flag or trunc_reset_flag:
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
        ) if task_records else 0.0,
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
    post_train_summary = _run_post_train_inference(q_net, args, device, run_dir, aim_logger)
    summary["post_train_inference"] = post_train_summary
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
