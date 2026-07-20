"""DQN trainer for the symbolic restaurant domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, PrioritizedReplayBuffer, ReplayBuffer
from tqdm import tqdm

from anticipatory_rl.envs.restaurant.env import ACTION_HEADS, ACTION_TYPES, RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import RestaurantPlannerState, solve_restaurant_task_with_fd


from anticipatory_rl.logging import AimLogger, CSVLogger, LoggerPair
from anticipatory_rl.utils import extract_masks, masked_choice, random_valid_index, select_device


def epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    return final + (start - final) * np.exp(-float(step) / float(decay))


def _resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label is not None:
        return args.run_label
    return "myopic_restaurant" if args.tasks_per_episode <= 1 else "anticipatory_restaurant"


class TaskTransition(NamedTuple):
    episode_done_flag: bool
    env_reset_flag: bool
    trunc_reset_flag: bool
    bootstrap_done: bool
    tasks_since_reset: int
    env_tasks_since_reset: int


def _decide_task_transition(
    success: bool,
    truncated: bool,
    tasks_since_reset: int,
    env_tasks_since_reset: int,
    steps_since_reset: int,
    tasks_per_episode: int,
    env_reset_tasks: int | None,
    episode_step_limit: int,
) -> TaskTransition:
    episode_done_flag = False
    env_reset_flag = False
    trunc_reset_flag = False
    bootstrap_done = False
    new_tasks = tasks_since_reset
    new_env_tasks = env_tasks_since_reset

    if success or truncated:
        new_tasks += 1
        new_env_tasks += 1
        if tasks_per_episode > 0 and new_tasks >= tasks_per_episode:
            episode_done_flag = True
            bootstrap_done = True
            new_tasks = 0
        if env_reset_tasks is not None and env_reset_tasks > 0 and new_env_tasks >= env_reset_tasks:
            env_reset_flag = True
            episode_done_flag = True
            bootstrap_done = True
            new_env_tasks = 0
    if episode_step_limit > 0 and steps_since_reset >= episode_step_limit:
        trunc_reset_flag = True
        bootstrap_done = True
        new_tasks = 0
        new_env_tasks = 0
    return TaskTransition(episode_done_flag, env_reset_flag, trunc_reset_flag, bootstrap_done,
                          new_tasks, new_env_tasks)


@dataclass
class OptimizeStats:
    loss: float
    q_selected_mean: float
    q_selected_abs_max: float
    target_mean: float
    target_abs_max: float
    td_abs_mean: float
    grad_norm: float
    # Diagnostics (populated only when args.diagnostics is set; NaN otherwise).
    td_abs_mean_pos_reward: float = float("nan")
    td_abs_mean_nonpos_reward: float = float("nan")
    value_vs_meanq_gap: float = float("nan")


class RestaurantQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_type_dim: int,
        object_dim: int,
        location_dim: int,
        hidden_dim: int = 256,
        center_advantages: bool = True,
    ) -> None:
        super().__init__()
        self.center_advantages = bool(center_advantages)
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
        self.value_head = nn.Linear(hidden_dim, 1) # V(s)
        self.action_type_adv_head = nn.Linear(hidden_dim, action_type_dim) # A_t(s,t)

        self.action_type_embed = nn.Embedding(action_type_dim, self.prefix_embed_dim) 
        self.object_embed = nn.Embedding(object_dim, self.prefix_embed_dim)
        self.location_embed = nn.Embedding(location_dim, self.prefix_embed_dim)

        self.object1_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim),
        ) # A_x(s,t,x)
        self.location_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, location_dim),
        ) # A_y(s,t,x,y)
        self.object2_adv_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 * self.prefix_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim),
        ) # A_z(s,t,x,y,z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def action_type_scores(self, encoded: torch.Tensor, action_type_mask: torch.Tensor) -> torch.Tensor:
        value = self.value_head(encoded) #V(s)
        advantages = self.action_type_adv_head(encoded) # A_t(s,t)
        if self.center_advantages:
            advantages = advantages - _masked_mean(advantages, action_type_mask)
        return value + advantages # V(s) + A_t(s,t) [- mean_{t'} A_t(s,t')]

    def object1_scores(
        self,
        encoded: torch.Tensor,
        action_types: torch.Tensor,
        object1_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = torch.cat([encoded, self.action_type_embed(action_types.squeeze(1))], dim=1) # [s; e_t(t)]
        advantages = self.object1_adv_head(prefix) # A_x(s,t,x)
        if not self.center_advantages:
            return advantages
        return advantages - _masked_mean(advantages, object1_mask) # A_x(s,t,x) - mean_{x'} A_x(s,t,x')

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
        ) # [s; e_t(t); e_x(x)]
        advantages = self.location_adv_head(prefix) # A_y(s,t,x,y)
        if not self.center_advantages:
            return advantages
        return advantages - _masked_mean(advantages, location_mask) # A_y(s,t,x,y) - mean_{y'} A_y(s,t,x,y')

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
        ) # [s; e_t(t); e_x(x); e_y(y)]
        advantages = self.object2_adv_head(prefix) # A_z(s,t,x,y,z)
        if not self.center_advantages:
            return advantages
        return advantages - _masked_mean(advantages, object2_mask) # A_z(s,t,x,y,z) - mean_{z'} A_z(s,t,x,y,z')

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

    head_signatures = [ACTION_HEADS[ACTION_TYPES[int(t.item())]] for t in action_types.squeeze(1)]
    no_arg_mask = torch.tensor([len(heads) == 0 for heads in head_signatures], dtype=torch.bool, device=encoded.device)
    if bool(no_arg_mask.all().item()):
        return q

    chosen_object1_masks = object1_masks[torch.arange(encoded.shape[0], device=encoded.device), action_types.squeeze(1)]
    object1_scores = q_net.object1_scores(encoded, action_types, chosen_object1_masks)
    object1_component = object1_scores.gather(1, object1)
    q = q + torch.where(no_arg_mask.unsqueeze(1), torch.zeros_like(object1_component), object1_component)

    chosen_location_masks = location_masks[torch.arange(encoded.shape[0], device=encoded.device), action_types.squeeze(1)]
    location_scores = q_net.location_scores(encoded, action_types, object1, chosen_location_masks)
    location_component = location_scores.gather(1, location)
    q = q + torch.where(no_arg_mask.unsqueeze(1), torch.zeros_like(location_component), location_component)

    chosen_object2_masks = object2_masks[
        torch.arange(encoded.shape[0], device=encoded.device),
        action_types.squeeze(1),
        object1.squeeze(1),
    ]
    object2_scores = q_net.object2_scores(encoded, action_types, object1, location, chosen_object2_masks)
    object2_component = object2_scores.gather(1, object2)
    q = q + torch.where(no_arg_mask.unsqueeze(1), torch.zeros_like(object2_component), object2_component)
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


def _classify_pick_place_failure(env: RestaurantSymbolicEnv, task: Mapping[str, object], actions: List[Mapping[str, int]]) -> str:
    if task.get("task_type") != "pick_place":
        return "non_pick_place_task"
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


def _store_transition(
    replay: ReplayBuffer,
    obs: np.ndarray,
    action: Mapping[str, int],
    reward: float,
    masks: Dict[str, np.ndarray],
    next_obs: np.ndarray,
    next_masks: Dict[str, np.ndarray],
    transition_done: bool,
    success: bool,
) -> None:
    """Append one transition to the replay buffer (shared by training loop and oracle seeding)."""
    replay.add(
        TensorDict({
            "state": torch.tensor(np.asarray(obs, dtype=np.float32)),
            "action_type": torch.tensor(int(action["action_type"])),
            "object1": torch.tensor(int(action["object1"])),
            "location": torch.tensor(int(action["location"])),
            "object2": torch.tensor(int(action["object2"])),
            "reward": torch.tensor(float(reward)),
            "action_type_mask": torch.tensor(np.asarray(masks["valid_action_type_mask"], dtype=np.float32)),
            "object1_mask": torch.tensor(np.asarray(masks["valid_object1_mask"], dtype=np.float32)),
            "location_mask": torch.tensor(np.asarray(masks["valid_location_mask"], dtype=np.float32)),
            "object2_mask": torch.tensor(np.asarray(masks["valid_object2_mask"], dtype=np.float32)),
            "next_state": torch.tensor(np.asarray(next_obs, dtype=np.float32)),
            "done": torch.tensor(float(transition_done)),
            "next_action_type_mask": torch.tensor(np.asarray(next_masks["valid_action_type_mask"], dtype=np.float32)),
            "next_object1_mask": torch.tensor(np.asarray(next_masks["valid_object1_mask"], dtype=np.float32)),
            "next_location_mask": torch.tensor(np.asarray(next_masks["valid_location_mask"], dtype=np.float32)),
            "next_object2_mask": torch.tensor(np.asarray(next_masks["valid_object2_mask"], dtype=np.float32)),
            "task_boundary": torch.tensor(float(success)),
        }, batch_size=torch.Size([]))
    )


def _auto_complete_replay_action_and_masks(
    env: RestaurantSymbolicEnv,
    masks: Dict[str, np.ndarray],
) -> tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Build a replay-only internal event action for an auto-satisfied task."""
    auto_idx = env.action_type_index["auto_complete"]
    action = {
        "action_type": auto_idx,
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }
    replay_masks = {key: np.asarray(value, dtype=np.float32).copy() for key, value in masks.items()}
    replay_masks["valid_action_type_mask"][:] = 0.0
    replay_masks["valid_action_type_mask"][auto_idx] = 1.0
    replay_masks["valid_object1_mask"][:] = 0.0
    replay_masks["valid_object1_mask"][auto_idx, env.none_object_index] = 1.0
    replay_masks["valid_location_mask"][:] = 0.0
    replay_masks["valid_location_mask"][auto_idx, env.none_location_index] = 1.0
    replay_masks["valid_object2_mask"][:] = 0.0
    replay_masks["valid_object2_mask"][auto_idx, env.none_object_index, env.none_object_index] = 1.0
    if "valid_action_mask" in replay_masks:
        replay_masks["valid_action_mask"] = replay_masks["valid_action_type_mask"].copy()
    return action, replay_masks


def _planner_action_to_env_action(env: RestaurantSymbolicEnv, action: Tuple[str, List[str]]) -> Dict[str, int]:
    """Convert a planner (name, args) action to an env factored action dict (int indices)."""
    name, args = action
    none_obj = env.none_object_index
    none_loc = env.none_location_index
    if name == "move":
        return {"action_type": env.action_type_index["move"], "object1": none_obj, "location": env.location_index[args[-1]], "object2": none_obj}
    if name == "pick":
        return {"action_type": env.action_type_index["pick"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "place":
        return {"action_type": env.action_type_index["place"], "object1": none_obj, "location": env.location_index[args[-1]], "object2": none_obj}
    if name == "wash":
        return {"action_type": env.action_type_index["wash"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "fill":
        return {"action_type": env.action_type_index["fill"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "drain":
        return {"action_type": env.action_type_index["drain"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "make-coffee":
        return {"action_type": env.action_type_index["make_coffee"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "pour":
        return {"action_type": env.action_type_index["pour"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": none_obj}
    if name == "refill_water":
        obj2 = env.object_name_index[args[2]] if len(args) > 2 else none_obj
        return {"action_type": env.action_type_index["refill_water"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": obj2}
    if name == "make-fruit-bowl":
        return {"action_type": env.action_type_index["make_fruit_bowl"], "object1": env.object_name_index[args[0]], "location": none_loc, "object2": env.object_name_index[args[1]]}
    raise ValueError(f"Unknown planner action: {name}")


def _persistent_oracle_rollout(
    env: RestaurantSymbolicEnv,
    n_outcomes: int,
    max_steps: int,
    seed_base: int,
    planner_path: Path,
    domain_path: Path,
    alias: str = "seq-sat-lama-2011",
    timeout_s: float = 10.0,
    transition_store: ReplayBuffer | None = None,
    env_reset_tasks: int = 200,
) -> Dict[str, object]:
    """Persistent-world myopic oracle roll-out.

    Resets the env once at the start and then every ``env_reset_tasks`` task
    outcomes. Auto-successes and planner failures advance the world without
    storing transitions. Successful plans are executed step-by-step with
    myopic terminal flags (``done=True`` on the task-boundary step).

    Returns
    -------
    dict with keys ``stored`` (int), ``outcomes`` (int), and
    ``successful_plan_rewards`` (list of lists of per-step rewards).
    """
    stored = 0
    outcomes = 0
    successful_plan_rewards: List[List[float]] = []
    world_index = 0

    obs, info = env.reset(seed=seed_base + 100_003 * world_index)
    world_index += 1

    while outcomes < n_outcomes:
        if env_reset_tasks > 0 and outcomes > 0 and outcomes % env_reset_tasks == 0:
            obs, info = env.reset(seed=seed_base + 100_003 * world_index)
            world_index += 1

        if env._pending_auto_success:
            # Settle the auto-success and refresh obs/info for the next task.
            obs, _, _, _, info = env.step(env.action_space.sample())
            outcomes += 1
            continue

        state = RestaurantPlannerState.from_env(env)
        result = solve_restaurant_task_with_fd(
            env, state, env.task,
            planner_path=planner_path, domain_path=domain_path, alias=alias, timeout_s=timeout_s,
        )
        if not result.success or len(result.plan_actions) > max_steps:
            env._resample_task()
            env._task_steps = 0
            # Refresh obs/info for the next task without storing a transition.
            obs, info = env._obs(), env._info(success=False)
            outcomes += 1
            continue

        plan_rewards: List[float] = []
        for plan_action in result.plan_actions:
            masks = extract_masks(info)
            action = _planner_action_to_env_action(env, plan_action)
            parsed = env._normalize_action(action)
            if not env._is_action_valid(parsed):
                raise RuntimeError(f"FD plan action invalid in env: {plan_action} → {action}")
            next_obs, reward, success, truncated, next_info = env.step(action)
            next_masks = extract_masks(next_info)
            transition_done = bool(success or truncated)
            if transition_store is not None:
                _store_transition(
                    transition_store, obs, action, reward, masks,
                    next_obs, next_masks, transition_done, success,
                )
                stored += 1
            plan_rewards.append(float(reward))
            obs, info = next_obs, next_info
            if success or truncated:
                break

        if plan_rewards:
            successful_plan_rewards.append(plan_rewards)
        outcomes += 1

    return {"stored": stored, "outcomes": outcomes, "successful_plan_rewards": successful_plan_rewards}


def _seed_replay_with_oracle(
    replay: ReplayBuffer,
    env: RestaurantSymbolicEnv,
    *,
    n_outcomes: int,
    max_steps: int,
    seed_base: int,
    planner_path: Path,
    domain_path: Path,
    alias: str = "seq-sat-lama-2011",
    timeout_s: float = 10.0,
    env_reset_tasks: int = 200,
) -> int:
    """Thin wrapper around ``_persistent_oracle_rollout`` for replay seeding.

    Returns the number of transitions stored.
    """
    result = _persistent_oracle_rollout(
        env, n_outcomes, max_steps, seed_base,
        planner_path, domain_path, alias=alias, timeout_s=timeout_s,
        transition_store=replay, env_reset_tasks=env_reset_tasks,
    )
    return int(result["stored"])


def _select_action(
    q_net: RestaurantQNetwork,
    state: np.ndarray,
    masks: Dict[str, np.ndarray],
    epsilon: float,
    device: torch.device,
) -> Dict[str, int]:
    if random.random() < epsilon:
        action_type = random_valid_index(masks["valid_action_type_mask"])
        object1 = random_valid_index(masks["valid_object1_mask"][action_type])
        location = random_valid_index(masks["valid_location_mask"][action_type])
        object2 = random_valid_index(masks["valid_object2_mask"][action_type, object1])
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

        if heads == ():
            total = action_type_component
            should_update = type_valid & (total > best_total)
            best_total = torch.where(should_update, total, best_total)
            best_action_type = torch.where(should_update, torch.full_like(best_action_type, action_type_t), best_action_type)
            best_object1 = torch.where(should_update, torch.full_like(best_object1, none_object), best_object1)
            best_location = torch.where(should_update, torch.full_like(best_location, none_location), best_location)
            best_object2 = torch.where(should_update, torch.full_like(best_object2, none_object), best_object2)
            continue

        if heads == ("object1",):
            # Full within-type joint argmax over object1.
            # The sentinel-dependent components (location at none_location and
            # object2 at none_object) vary with the choice of object1 due to
            # conditional embeddings.  Computing the full sum for every valid
            # object1 eliminates the approximation gap.
            object1_mask = object1_masks[:, action_type_t, :]
            object1_scores = q_net.object1_scores(encoded, action_type_tensor, object1_mask)

            # location(none_location) for every candidate object1.
            object_ids_b = object_ids.view(1, object_dim).expand(batch_size, object_dim)
            location_mask = location_masks[:, action_type_t, :].unsqueeze(1).expand(batch_size, object_dim, location_dim).reshape(-1, location_dim)
            location_scores = q_net.location_scores(
                encoded.unsqueeze(1).expand(batch_size, object_dim, hidden_dim).reshape(-1, hidden_dim),
                action_type_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object_ids_b.reshape(-1, 1),
                location_mask,
            ).reshape(batch_size, object_dim, location_dim)
            location_component = location_scores[:, :, none_location]

            # object2(none_object) for every candidate object1.
            object2_mask = object2_masks[:, action_type_t, :, :].reshape(batch_size * object_dim, object_dim)
            object2_scores = q_net.object2_scores(
                encoded.unsqueeze(1).expand(batch_size, object_dim, hidden_dim).reshape(-1, hidden_dim),
                action_type_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object_ids_b.reshape(-1, 1),
                none_location_tensor.unsqueeze(1).expand(batch_size, object_dim, 1).reshape(-1, 1),
                object2_mask,
            ).reshape(batch_size, object_dim, object_dim)
            object2_component = object2_scores[:, :, none_object]

            total_scores = (
                action_type_component.unsqueeze(1)
                + object1_scores
                + location_component
                + object2_component
            )
            neg_inf = torch.finfo(total_scores.dtype).min
            valid_o1 = object1_mask > 0.0
            total_scores = total_scores.masked_fill(~valid_o1, neg_inf)
            candidate_object1 = torch.argmax(total_scores, dim=1)
            total = total_scores.gather(1, candidate_object1.unsqueeze(1)).squeeze(1)
            should_update = type_valid & (total > best_total)
            best_total = torch.where(should_update, total, best_total)
            best_action_type = torch.where(should_update, torch.full_like(best_action_type, action_type_t), best_action_type)
            best_object1 = torch.where(should_update, candidate_object1, best_object1)
            best_location = torch.where(should_update, torch.full_like(best_location, none_location), best_location)
            best_object2 = torch.where(should_update, torch.full_like(best_object2, none_object), best_object2)
            continue

        if heads == ("location",):
            # Full within-type joint argmax over location.
            # The sentinel-dependent component (object2 at none_object) varies
            # with the choice of location due to conditional embeddings.
            object1_mask = object1_masks[:, action_type_t, :]
            object1_scores = q_net.object1_scores(encoded, action_type_tensor, object1_mask)
            object1_component = object1_scores[:, none_object]

            # location scores for every candidate location.
            location_mask = location_masks[:, action_type_t, :]
            location_scores = q_net.location_scores(encoded, action_type_tensor, none_object_tensor, location_mask)

            # object2(none_object) for every candidate location.
            location_ids_b = location_ids.view(1, location_dim).expand(batch_size, location_dim)
            object2_mask = object2_masks[:, action_type_t, none_object, :].unsqueeze(1).expand(batch_size, location_dim, object_dim).reshape(-1, object_dim)
            object2_scores = q_net.object2_scores(
                encoded.unsqueeze(1).expand(batch_size, location_dim, hidden_dim).reshape(-1, hidden_dim),
                action_type_tensor.unsqueeze(1).expand(batch_size, location_dim, 1).reshape(-1, 1),
                none_object_tensor.unsqueeze(1).expand(batch_size, location_dim, 1).reshape(-1, 1),
                location_ids_b.reshape(-1, 1),
                object2_mask,
            ).reshape(batch_size, location_dim, object_dim)
            object2_component = object2_scores[:, :, none_object]

            total_scores = (
                action_type_component.unsqueeze(1)
                + object1_component.unsqueeze(1)
                + location_scores
                + object2_component
            )
            neg_inf = torch.finfo(total_scores.dtype).min
            valid_loc = location_mask > 0.0
            total_scores = total_scores.masked_fill(~valid_loc, neg_inf)
            candidate_location = torch.argmax(total_scores, dim=1)
            total = total_scores.gather(1, candidate_location.unsqueeze(1)).squeeze(1)
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
            action_type = masked_choice(action_type_scores[idx], action_type_masks[idx])
            object1 = random_valid_index(object1_masks[idx, action_type].detach().cpu().numpy())
            location = random_valid_index(location_masks[idx, action_type].detach().cpu().numpy())
            object2 = random_valid_index(object2_masks[idx, action_type, object1].detach().cpu().numpy())
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
    demo_replay: ReplayBuffer | None = None,
    demo_fraction: float = 0.0,
) -> OptimizeStats | None:
    if len(replay) < args.batch_size:
        return None
    per_alpha = float(getattr(args, "per_alpha", 0.0) or 0.0)
    per_eps = float(getattr(args, "per_eps", 1e-6) or 1e-6)
    per_info, main_indices, n_demo = None, None, 0
    # Protected-demo sampling: reserve a fixed fraction of every minibatch for
    # never-evicting oracle demonstrations so positive-reward transitions cannot
    # wash out of the agent's gradient signal.
    if demo_replay is not None and demo_fraction > 0.0 and len(demo_replay) > 0:
        n_demo = min(int(round(args.batch_size * demo_fraction)), len(demo_replay))
        n_main = args.batch_size - n_demo
        if n_main <= 0:
            n_main = args.batch_size
            n_demo = 0
        if per_alpha > 0.0:
            main_data, per_info = replay.sample(n_main, return_info=True)
            main_indices = per_info["index"]
        else:
            main_data = replay.sample(n_main)
        if n_demo > 0:
            demo_batch = demo_replay.sample(n_demo)
            batch = torch.cat([main_data, demo_batch], dim=0)
        else:
            batch = main_data
    else:
        if per_alpha > 0.0:
            data, per_info = replay.sample(args.batch_size, return_info=True)
            main_indices = per_info["index"]
            batch = data
        else:
            batch = replay.sample(args.batch_size)
    td = batch.to(device)

    q_values = q_net(
        td["state"],
        action_types=td["action_type"].unsqueeze(1),
        object1=td["object1"].unsqueeze(1),
        location=td["location"].unsqueeze(1),
        object2=td["object2"].unsqueeze(1),
        action_type_masks=td["action_type_mask"],
        object1_masks=td["object1_mask"],
        location_masks=td["location_mask"],
        object2_masks=td["object2_mask"],
    )
    with torch.no_grad():
        next_action_type, next_object1, next_location, next_object2 = q_net(
            td["next_state"],
            action_type_masks=td["next_action_type_mask"],
            object1_masks=td["next_object1_mask"],
            location_masks=td["next_location_mask"],
            object2_masks=td["next_object2_mask"],
            decode_greedy=True,
        )
        next_q = target_net(
            td["next_state"],
            action_types=next_action_type,
            object1=next_object1,
            location=next_location,
            object2=next_object2,
            action_type_masks=td["next_action_type_mask"],
            object1_masks=td["next_object1_mask"],
            location_masks=td["next_location_mask"],
            object2_masks=td["next_object2_mask"],
        )
        targets = td["reward"].unsqueeze(1) + args.gamma * (1.0 - td["done"].unsqueeze(1)) * next_q

    td_error = q_values - targets
    loss = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
    if per_info is not None:
        is_weights = per_info["priority_weight"].to(device)  # (n_main,)
        if n_demo > 0:
            demo_ones = torch.ones(n_demo, dtype=is_weights.dtype, device=device)
            is_weights = torch.cat([is_weights, demo_ones], dim=0)
        is_weights = is_weights.unsqueeze(1)  # (batch_size, 1)
        loss = (loss * is_weights).mean()
    else:
        loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm_t = nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()

    td_pos = td_neg = v_gap = float("nan")
    if getattr(args, "diagnostics", False):
        with torch.no_grad():
            rewards = td["reward"].unsqueeze(1)
            td_abs = td_error.abs()
            pos_mask = rewards > 0.0
            if pos_mask.any():
                td_pos = float(td_abs[pos_mask].mean().item())
            if (~pos_mask).any():
                td_neg = float(td_abs[~pos_mask].mean().item())
            # V(s) vs mean over valid action-type scores: how much absolute level
            # the dueling value head absorbs relative to the action-type advantages.
            encoded = q_net.encode(td["state"])
            value = q_net.value_head(encoded)
            type_scores = q_net.action_type_scores(encoded, td["action_type_mask"])
            mean_type_q = _masked_mean(type_scores, td["action_type_mask"])
            v_gap = float((value - mean_type_q).mean().item())

    if main_indices is not None:
        n_main = main_indices.shape[0]
        main_td_abs = td_error.abs()[:n_main].squeeze(1)
        per_clip = float(getattr(args, "per_clip", 10.0) or 10.0)
        priorities = main_td_abs.clamp(max=per_clip) + per_eps
        replay.update_priority(main_indices, priorities)

    return OptimizeStats(
        loss=float(loss.item()),
        q_selected_mean=float(q_values.mean().item()),
        q_selected_abs_max=float(q_values.abs().max().item()),
        target_mean=float(targets.mean().item()),
        target_abs_max=float(targets.abs().max().item()),
        td_abs_mean=float(td_error.abs().mean().item()),
        grad_norm=float(grad_norm_t.item() if hasattr(grad_norm_t, "item") else grad_norm_t),
        td_abs_mean_pos_reward=td_pos,
        td_abs_mean_nonpos_reward=td_neg,
        value_vs_meanq_gap=v_gap,
    )


def _greedy_success_rate(
    q_net: RestaurantQNetwork,
    env: RestaurantSymbolicEnv,
    device: torch.device,
    *,
    n_tasks: int,
    max_steps: int,
    seed_base: int,
) -> float:
    """Greedy (epsilon=0) success rate over n_tasks fresh tasks. Diagnostics only."""
    was_training = q_net.training
    q_net.eval()
    successes = 0
    for i in range(n_tasks):
        obs, info = env.reset(seed=seed_base + i)
        for _ in range(max_steps):
            masks = extract_masks(info)
            action = _select_action(q_net, obs, masks, epsilon=0.0, device=device)
            obs, reward, success, truncated, info = env.step(action)
            if success:
                successes += 1
                break
            if truncated:
                break
    if was_training:
        q_net.train()
    return successes / max(1, n_tasks)


def _q_by_action_type(
    q_net: RestaurantQNetwork,
    env: RestaurantSymbolicEnv,
    device: torch.device,
    *,
    seed: int,
) -> Dict[str, float]:
    """Best composed-Q per valid action_type at one fixed probe state.

    Watches whether Q(move) ever overtakes Q(pick) as replay fills, which is the
    crux of the dueling-decomposition hypothesis.
    """
    obs, info = env.reset(seed=seed)
    masks = extract_masks(info)
    was_training = q_net.training
    q_net.eval()
    out: Dict[str, float] = {}
    with torch.no_grad():
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = q_net.encode(state_t)
        atm = torch.tensor(masks["valid_action_type_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        o1m = torch.tensor(masks["valid_object1_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        lm = torch.tensor(masks["valid_location_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        o2m = torch.tensor(masks["valid_object2_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        type_scores = q_net.action_type_scores(encoded, atm)
        for t_idx, name in enumerate(ACTION_TYPES):
            if atm[0, t_idx] <= 0.0:
                continue
            # Reuse the per-type greedy decode by masking to this single type.
            single_type_mask = torch.zeros_like(atm)
            single_type_mask[0, t_idx] = 1.0
            at, o1, loc, o2 = _select_greedy_actions_batch(q_net, encoded, single_type_mask, o1m, lm, o2m)
            q = _compose_q_values(q_net, encoded, at, o1, loc, o2, atm, o1m, lm, o2m)
            out[name] = float(q.item())
    if was_training:
        q_net.train()
    return out


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
        rng_seed=args.seed,
    )

    run_label = _resolve_run_label(args)
    run_dir = Path("runs") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output_name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")
    aim_logger = AimLogger(args, run_label)
    csv_logger = CSVLogger(args, run_label, run_dir)
    logger = LoggerPair(aim_logger, csv_logger)
    logger.set_metadata("run_dir", str(run_dir.resolve()))
    logger.set_metadata("config_path", str(Path(args.config_path).resolve()))

    obs, info = env.reset(seed=args.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    object_dim = int(env.action_space["object1"].n)
    location_dim = int(env.action_space["location"].n)
    action_type_dim = int(env.action_space["action_type"].n)
    logger.set_metadata(
        "model",
        {
            "observation_dim": obs_dim,
            "action_type_dim": action_type_dim,
            "object_dim": object_dim,
            "location_dim": location_dim,
            "hidden_dim": args.hidden_dim,
        },
    )
    center_advantages = not getattr(args, "no_dueling_centering", False)
    q_net = RestaurantQNetwork(obs_dim, action_type_dim, object_dim, location_dim, hidden_dim=args.hidden_dim, center_advantages=center_advantages).to(device)
    target_net = RestaurantQNetwork(obs_dim, action_type_dim, object_dim, location_dim, hidden_dim=args.hidden_dim, center_advantages=center_advantages).to(device)
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        print(f"[train] Resuming q_net weights from {resume_path}")
        q_net.load_state_dict(torch.load(resume_path, map_location=device))
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    per_alpha = float(getattr(args, "per_alpha", 0.0) or 0.0)
    per_beta = float(getattr(args, "per_beta", 0.4) or 0.4)
    per_eps = float(getattr(args, "per_eps", 1e-6) or 1e-6)
    if per_alpha > 0.0:
        replay: ReplayBuffer = PrioritizedReplayBuffer(
            alpha=float(per_alpha),
            beta=float(per_beta),
            eps=float(per_eps),
            storage=LazyTensorStorage(max_size=args.replay_size),
        )
        print(f"[train] PrioritizedReplayBuffer (α={per_alpha}, β={per_beta}, ε={per_eps})")
    else:
        replay = ReplayBuffer(storage=LazyTensorStorage(max_size=args.replay_size))

    env_reset_tasks = args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_episode
    if args.episode_step_limit is None:
        args.episode_step_limit = int(env_reset_tasks * args.max_steps_per_task * 1.5)
    if args.episode_step_limit > 0 and env_reset_tasks > 0:
        assert args.episode_step_limit > env_reset_tasks * args.max_steps_per_task, (
            "episode_step_limit must exceed env_reset_tasks * max_steps_per_task "
            "or paired agents desync on world-reset cadence."
        )
    # if args.tasks_per_episode > 1 and env_reset_tasks != args.tasks_per_episode:
    #     raise ValueError("For anticipatory runs, env-reset-tasks must equal tasks-per-episode.")

    diagnostics = bool(getattr(args, "diagnostics", False))
    diag_interval = max(1, int(getattr(args, "diagnostics_interval", 1000)))
    positive_reward_transitions = 0
    diag_env: RestaurantSymbolicEnv | None = None
    if diagnostics:
        diag_env = RestaurantSymbolicEnv(
            config_path=args.config_path,
            max_steps_per_task=args.max_steps_per_task,
            success_reward=args.success_reward,
            invalid_action_penalty=args.invalid_action_penalty,
            rng_seed=args.seed + 7_000_000,
        )

    demo_path = getattr(args, "demo_transitions", None)
    offline_transitions = None
    offline_count = 0
    if demo_path is not None:
        data = torch.load(demo_path, weights_only=False)
        offline_metadata = data["metadata"]
        offline_transitions = data["transitions"]
        offline_count = len(offline_transitions)
        for field in ("object_names", "locations", "object_kinds", "contents", "task_types"):
            stored = offline_metadata.get(field)
            if stored is None:
                raise ValueError(f"Demo metadata missing field '{field}'. File may be from an older script version.")
            if list(stored) != list(getattr(env, field)):
                raise ValueError(f"Demo {field} mismatch. Demo: {list(stored)}, Env: {list(getattr(env, field))}")
        for field in ("success_reward", "invalid_action_penalty", "travel_cost_scale",
                      "pick_cost", "place_cost", "wash_cost", "fill_cost", "brew_cost",
                      "fruit_cost", "spread_cost", "pour_cost", "refill_cost", "drain_cost"):
            stored = offline_metadata.get(field)
            if stored is None:
                raise ValueError(f"Demo metadata missing cost field '{field}'. File may be from an older script version.")
            if float(stored) != float(getattr(env, field)):
                raise ValueError(f"Demo cost '{field}' mismatch ({stored} vs {getattr(env, field)}).")
        for field in ("max_steps_per_task",):
            stored = offline_metadata.get(field)
            if stored is None:
                raise ValueError(f"Demo metadata missing field '{field}'. File may be from an older script version.")
            if int(stored) != int(getattr(args, field)):
                raise ValueError(f"Demo {field} mismatch ({stored} vs {getattr(args, field)}).")
        stored_hash = offline_metadata.get("config_hash")
        current_hash = hashlib.sha256(args.config_path.read_bytes()).hexdigest()
        if stored_hash is not None and stored_hash != current_hash:
            raise ValueError(f"Demo config_hash mismatch — config file may have changed since demo generation. "
                             f"Demo: {stored_hash[:12]}... Current: {current_hash[:12]}...")
        stored_obs_dim = offline_metadata.get("obs_dim")
        if stored_obs_dim is None:
            raise ValueError("Demo metadata missing field 'obs_dim'. File may be from an older script version.")
        if int(stored_obs_dim) != int(obs_dim):
            raise ValueError(f"Demo obs_dim {stored_obs_dim} != env obs_dim {obs_dim}. Config drift?")
        credit = offline_metadata.get("credit_horizon", "unknown")
        if args.tasks_per_episode > 1 and credit == "myopic":
            print("[train] WARNING: loading myopic demos (done=True per task) into anticipatory training "
                  "(tasks_per_episode > 1). Bootstrap mismatch — demos teach terminal targets.")
        print(f"[train] Validated {offline_count} demo transitions from {demo_path}.")

    seed_oracle = int(getattr(args, "seed_replay_oracle", 0) or 0)
    demo_fraction = float(getattr(args, "protect_demo_fraction", 0.0) or 0.0)
    demo_replay: ReplayBuffer | None = None
    if seed_oracle > 0:
        oracle_env = RestaurantSymbolicEnv(
            config_path=args.config_path,
            max_steps_per_task=args.max_steps_per_task,
            success_reward=args.success_reward,
            invalid_action_penalty=args.invalid_action_penalty,
            rng_seed=args.seed + 3_000_000,
        )
        # When protecting demos, route them into a separate never-evicting buffer
        # that _optimize blends into every minibatch. Otherwise behave as before
        # (demos mixed into the main buffer, free to age out).
        if demo_fraction > 0.0:
            total_cap = max(seed_oracle * args.max_steps_per_task + offline_count, args.batch_size)
            demo_replay = ReplayBuffer(storage=LazyTensorStorage(max_size=total_cap))
            target_buffer = demo_replay
        else:
            target_buffer = replay
        stored = _seed_replay_with_oracle(
            target_buffer, oracle_env, n_outcomes=seed_oracle, max_steps=args.max_steps_per_task,
            seed_base=args.seed + 3_000_000,
            planner_path=args.planner_path, domain_path=args.domain_path,
            env_reset_tasks=args.env_reset_tasks,
        )
        dest = "protected demo buffer" if demo_fraction > 0.0 else "main replay"
        print(f"[train] Seeded {dest} with {stored} oracle transitions from {seed_oracle} outcomes (demo_fraction={demo_fraction}).")
        logger.set_metadata("oracle_seed_transitions", int(stored))
        logger.set_metadata("protect_demo_fraction", demo_fraction)

    if offline_transitions is not None:
        if demo_fraction > 0.0:
            if demo_replay is None:
                demo_replay = ReplayBuffer(storage=LazyTensorStorage(max_size=max(offline_count, args.batch_size)))
            target = demo_replay
        else:
            target = replay
        if offline_count > 0:
            target.extend(torch.stack(offline_transitions))
        dest = "protected demo buffer" if demo_fraction > 0.0 else "main replay"
        print(f"[train] Loaded {offline_count} demo transitions from {demo_path} into {dest}.")

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
    replay_auto_complete_count = 0
    # Diagnostic counters for Option 3 self-loop detection.
    # Any auto-success where world is unchanged is terminal for bootstrapping (no task_equality check).
    # This prevents multi-task loops (A→B→A) from generating degenerate infinite Q-values.
    self_loop_terminal_count = 0           # both conditions match: terminal-for-bootstrap event fired
    self_loop_world_only_count = 0         # auto_success AND world_unchanged (independent of task equality)
    self_loop_task_only_count = 0           # auto_success AND task==pre-task (independent of world check)
    question_counters = {
        "wrong_object_choice": 0,
        "failed_to_move_to_object": 0,
        "failed_after_pick": 0,
        "place_selection_wrong": 0,
        "mask_or_timeout_issue": 0,
    }

    consecutive_auto_successes = 0

    progress = tqdm(range(args.total_steps), desc="Restaurant DQN", unit="step")
    for global_step in progress:
        epsilon = epsilon_by_step(
            global_step,
            args.epsilon_start,
            args.epsilon_final,
            args.epsilon_decay,
        )
        current_task_snapshot = dict(info.get("task", {}))  # captured BEFORE env.step() below
        current_world_state_key = env._action_mask_state_key()  # snapshot of s (world only, excludes task)
        current_task_auto_snapshot = bool(current_task_auto_satisfied)
        masks = extract_masks(info)
        action = _select_action(q_net, obs, masks, epsilon, device)
        current_task_actions.append(dict(action))
        current_task_action_strings.append(_action_to_string(env, action))
        action_type_counts[ACTION_TYPES[int(action["action_type"])]] += 1
        next_obs, reward, success, truncated, next_info = env.step(action)
        task_return += float(reward)
        task_steps += 1
        step_reward_history.append(float(reward))

        steps_since_reset += 1
        t = _decide_task_transition(
            success, truncated, tasks_since_reset, env_tasks_since_reset, steps_since_reset,
            args.tasks_per_episode, env_reset_tasks, args.episode_step_limit,
        )
        episode_done_flag = t.episode_done_flag
        env_reset_flag = t.env_reset_flag
        trunc_reset_flag = t.trunc_reset_flag
        bootstrap_done = t.bootstrap_done
        tasks_since_reset = t.tasks_since_reset
        env_tasks_since_reset = t.env_tasks_since_reset

        next_masks = extract_masks(next_info)
        store_action = dict(action)
        store_masks = masks
        if bool(next_info.get("auto_success", False)):
            store_action, store_masks = _auto_complete_replay_action_and_masks(env, masks)
            replay_auto_complete_count += 1
        store_next_masks = next_masks
        if bool(next_info.get("next_auto_satisfied", False)):
            _, store_next_masks = _auto_complete_replay_action_and_masks(env, next_masks)
        # Option 3: treat an auto-success that leaves the world unchanged AS A
        # self-transition (s,τ)->(s,τ) as terminal for bootstrapping. auto_success=True
        # implies world unchanged per env.step() contract, but we check the world-
        # state key defensively. Same-task equality on the augmented state is what
        # creates the degenerate fixed point V=r/(1-γ); making it terminal kills the
        # self-bootstrap without touching cross-task (s,τ)->(s,τ') bootstraps.
        world_unchanged = env._action_mask_state_key() == current_world_state_key
        auto_success_flag = bool(next_info.get("auto_success", False))
        task_equality = next_info.get("task") == current_task_snapshot
        
        # Option 3 (fixed): Treat ANY auto-success that leaves the world unchanged 
        # as terminal for bootstrapping, regardless of task_equality. This prevents
        # multi-task loops (A -> B -> A) from generating degenerate infinite values.
        auto_success_flag = bool(next_info.get("auto_success", False))
        if auto_success_flag and world_unchanged:
            consecutive_auto_successes += 1
        else:
            consecutive_auto_successes = 0

        self_loop_auto_success = bool(
            (auto_success_flag and world_unchanged and task_equality) or 
            (consecutive_auto_successes > 1)
        )
        
        if auto_success_flag and world_unchanged:
            self_loop_world_only_count += 1
        if auto_success_flag and task_equality:
            self_loop_task_only_count += 1
        if self_loop_auto_success:
            self_loop_terminal_count += 1
        transition_done = bool(bootstrap_done or self_loop_auto_success)
        _store_transition(replay, obs, store_action, reward, store_masks, next_obs, store_next_masks, transition_done, success)
        if reward > 0.0:
            positive_reward_transitions += 1

        optimize_stats = _optimize(q_net, target_net, replay, optimizer, args, device, demo_replay=demo_replay, demo_fraction=demo_fraction)
        if per_alpha > 0.0:
            frac = min(1.0, float(global_step) / max(1, args.total_steps))
            current_beta = per_beta + (1.0 - per_beta) * frac
            replay._sampler._beta = float(current_beta)

        if optimize_stats is not None:
            optimize_stats_history.append(optimize_stats)
            loss_history.append(optimize_stats.loss)
            logger.track(optimize_stats.loss, name="loss", step=global_step, context={"subset": "train"})
            logger.track(optimize_stats.q_selected_mean, name="q_selected_mean", step=global_step)
            logger.track(optimize_stats.q_selected_abs_max, name="q_selected_abs_max", step=global_step)
            logger.track(optimize_stats.target_mean, name="target_mean", step=global_step)
            logger.track(optimize_stats.target_abs_max, name="target_abs_max", step=global_step)
            logger.track(optimize_stats.td_abs_mean, name="td_abs_mean", step=global_step)
            logger.track(optimize_stats.grad_norm, name="grad_norm", step=global_step)
            if diagnostics:
                if not np.isnan(optimize_stats.td_abs_mean_pos_reward):
                    logger.track(optimize_stats.td_abs_mean_pos_reward, name="td_abs_mean_pos_reward", step=global_step)
                if not np.isnan(optimize_stats.td_abs_mean_nonpos_reward):
                    logger.track(optimize_stats.td_abs_mean_nonpos_reward, name="td_abs_mean_nonpos_reward", step=global_step)
                if not np.isnan(optimize_stats.value_vs_meanq_gap):
                    logger.track(optimize_stats.value_vs_meanq_gap, name="value_vs_meanq_gap", step=global_step)

        if diagnostics and diag_env is not None and (global_step + 1) % diag_interval == 0:
            logger.track(
                positive_reward_transitions / float(global_step + 1),
                name="replay_positive_reward_fraction",
                step=global_step,
            )
            greedy_success = _greedy_success_rate(
                q_net, diag_env, device,
                n_tasks=20, max_steps=args.max_steps_per_task, seed_base=args.seed + 8_000_000,
            )
            logger.track(greedy_success, name="greedy_success_rolling", step=global_step)
            q_by_type = _q_by_action_type(q_net, diag_env, device, seed=args.seed + 9_000_000)
            for name, q_val in q_by_type.items():
                logger.track(q_val, name="q_by_action_type", step=global_step, context={"action_type": name})

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
                logger.track_text(
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
            logger.track(
                float(task_records[-1]["return"]),
                name="task_return",
                step=total_tasks,
                context={"task_type": current_task_snapshot.get("task_type", "unknown")},
            )
            logger.track(
                float(task_records[-1]["steps"]),
                name="task_steps",
                step=total_tasks,
                context={"task_type": current_task_snapshot.get("task_type", "unknown")},
            )
            logger.track(
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
        logger.track(epsilon, name="epsilon", step=global_step)
        logger.track(success_rate, name="success_rate_rolling", step=global_step, context={"window": 100})
        logger.track(auto_rate, name="auto_rate_rolling", step=global_step, context={"window": 100})
        logger.track(non_auto_success_rate, name="non_auto_success_rate_rolling", step=global_step, context={"window": 100})
        logger.track(len(replay) / max(1, args.replay_size), name="replay_fill_fraction", step=global_step)
        for action_name, count in action_type_counts.items():
            logger.track(count / max(1, global_step + 1), name="action_type_fraction", step=global_step, context={"action_type": action_name})
        if recent_returns:
            logger.track(avg_return, name="avg_task_return_rolling", step=global_step, context={"window": 100})
        if loss_history:
            logger.track(avg_loss, name="avg_loss_rolling", step=global_step, context={"window": 100})
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
        "replay_fill_fraction_final": float(len(replay) / max(1, args.replay_size)),
        "replay_auto_complete_count": int(replay_auto_complete_count),
        "self_loop_terminal_count": int(self_loop_terminal_count),
        "self_loop_world_only_count": int(self_loop_world_only_count),
        "self_loop_task_only_count": int(self_loop_task_only_count),
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
    logger.set_metadata("summary", summary)
    logger.set_metadata("checkpoint_path", str(output_path))
    logger.close()
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
    parser.add_argument("--episode-step-limit", type=int, default=None, help="Maximum primitive steps allowed between resets; <=0 disables. None -> derived from env_reset_tasks * max_steps_per_task.")
    parser.add_argument("--max-steps-per-task", type=int, default=64)
    parser.add_argument("--success-reward", type=float, default=15.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="restaurant_dqn.pt")
    parser.add_argument("--resume-from", type=Path, default=None, help="Load q_net weights from this checkpoint before training. Target net is synced from q_net; replay/optimizer/epsilon schedules start fresh.")
    parser.add_argument("--diagnostics", action="store_true", help="Enable root-cause diagnostic logging (off by default).")
    parser.add_argument("--diagnostics-interval", type=int, default=1000, help="Steps between periodic greedy-eval / Q-probe diagnostics.")
    parser.add_argument("--no-dueling-centering", action="store_true", help="Ablation C: disable masked-mean advantage centering in all heads.")
    parser.add_argument("--seed-replay-oracle", type=int, default=0, help="Ablation B: seed replay with myopic (FD) oracle demos from N tasks before training.")
    parser.add_argument("--protect-demo-fraction", type=float, default=0.0, help="If >0, store oracle demos in a separate never-evicting buffer and draw this fraction of every minibatch from it (prevents demo washout).")
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"), help="Path to Fast Downward planner (for oracle demo seeding).")
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"), help="Path to PDDL domain file (for oracle demo seeding).")
    parser.add_argument("--demo-transitions", type=Path, default=None, help="Load pre-generated demo transitions (.pt) into replay at training start.")
    parser.add_argument("--per-alpha", type=float, default=0.0, help="PER α: 0 = uniform, >0 enables PrioritizedReplayBuffer with this α.")
    parser.add_argument("--per-beta", type=float, default=0.4, help="PER β initial value (annealed linearly to 1.0 over total_steps).")
    parser.add_argument("--per-eps", type=float, default=1e-6, help="PER ε added to |TD| before raising to α.")
    parser.add_argument("--per-clip", type=float, default=10.0, help="Clip |TD| at this value before computing PER priority (breaks feedback loop).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
