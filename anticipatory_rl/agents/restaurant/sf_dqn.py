"""Successor Features DQN trainer for the symbolic restaurant domain with structured actions."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, ACTION_TYPES
from anticipatory_rl.agents.restaurant.utils import (
    select_device,
    epsilon_by_step,
    resolve_run_label,
)


@dataclass
class Transition:
    state: np.ndarray
    action: Dict[str, int]  # Structured action dict - not flat int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_action_type_mask: np.ndarray
    next_valid_object1_mask: np.ndarray  # 2D: [action_type_idx, object1_idx]
    next_valid_location_mask: np.ndarray  # 2D: [action_type_idx, location_idx]
    next_valid_object2_mask: np.ndarray  # 3D: [action_type_idx, object1_idx, object2_idx]
    task: Dict[str, str | None]
    next_task: Dict[str, str | None]
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


class SuccessorFeatureNetwork(nn.Module):
    """Successor Feature network with structured action embeddings."""

    def __init__(
        self,
        obs_dim: int,
        action_type_count: int,
        object_count: int,
        location_count: int,
        object_names: Tuple[str, ...],
        locations: Tuple[str, ...],
        action_types: Tuple[str, ...],
        object_kinds: Tuple[str, ...],
        task_types: Tuple[str, ...],
        sf_dim: int = 64,
        hidden_dim: int = 256
    ) -> None:
        super().__init__()

        # Shared state encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Embeddings for different action components
        self.action_type_embed = nn.Embedding(action_type_count, 64)
        self.object1_embed = nn.Embedding(object_count + 1, 64)  # +1 for None token
        self.location_embed = nn.Embedding(location_count + 1, 64)  # +1 for None token
        self.object2_embed = nn.Embedding(object_count + 1, 64)  # +1 for None token

        # Conditional branch SF heads - each produces sf_dim dimensional vector
        # psi_t(s, t) - SF feature for action type (conditions on state + action_type)
        self.psi_t_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, sf_dim),
        )

        # psi_x(s, t, x) - SF feature for first argument
        self.psi_x_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim//2),  # state + action_type_emb + object1_emb
            nn.ReLU(),
            nn.Linear(hidden_dim//2, sf_dim),
        )

        # psi_y(s, t, x, y) - SF feature for second argument (location)
        self.psi_y_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64 + 64, hidden_dim//2),  # state+type+obj1+loc
            nn.ReLU(),
            nn.Linear(hidden_dim//2, sf_dim),
        )

        # psi_z(s, t, x, y, z) - SF feature for third argument
        self.psi_z_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64 + 64 + 64, hidden_dim//2),  # state+type+obj1+loc+obj2
            nn.ReLU(),
            nn.Linear(hidden_dim//2, sf_dim),
        )

        # Task embedding head
        self.task_head = nn.Sequential(
            nn.Linear(len(task_types) + len(locations) + len(object_kinds) + len(object_names), 128),  # task one-hot dim
            nn.ReLU(),
            nn.Linear(128, sf_dim),
            nn.LayerNorm(sf_dim),  # Normalize to maintain scale stability
        )

        self.sf_dim = sf_dim
        self.action_type_count = action_type_count
        self.object_count = object_count
        self.location_count = location_count

    def _enumerate_valid_actions(
        self,
        valid_action_type_mask: np.ndarray,
        valid_object1_mask: np.ndarray,
        valid_location_mask: np.ndarray,
        valid_object2_mask: np.ndarray,
        action_dims: Tuple[int, int, int, int],
    ) -> List[Dict[str, int]]:
        """Enumerate all valid action combinations."""
        # Convert to torch tensor for faster operation
        with torch.no_grad():
            action_type_mask = torch.from_numpy(valid_action_type_mask).float()
            object1_mask = torch.from_numpy(valid_object1_mask).float()  # [action_type_idx, object1_idx]
            location_mask = torch.from_numpy(valid_location_mask).float()  # [action_type_idx, location_idx]
            object2_mask = torch.from_numpy(valid_object2_mask).float()  # [action_type_idx, object1_idx, object2_idx]

            # Get valid action types
            valid_action_types = torch.nonzero(action_type_mask > 0.9).squeeze(-1).tolist()
            if isinstance(valid_action_types, int):
                valid_action_types = [valid_action_types]

            valid_combinations = []
            for action_type_idx in valid_action_types:
                # Get valid objects for this action type
                obj_mask = object1_mask[action_type_idx]
                valid_objects = torch.nonzero(obj_mask > 0.9).squeeze(-1).tolist()
                if len(valid_objects) == 1 and isinstance(valid_objects[0], int):  # Handle scalar single element
                    valid_objects = [valid_objects[0]]

                # Get valid locations for this action type
                loc_mask = location_mask[action_type_idx]
                valid_locations = torch.nonzero(loc_mask > 0.9).squeeze(-1).tolist()
                if len(valid_locations) == 1 and isinstance(valid_locations[0], int):
                    valid_locations = [valid_locations[0]]

                # Iterate through valid combinations and check if object2 is valid
                for obj1_idx in valid_objects:
                    for loc_idx in valid_locations:
                        # Check valid object2 for this type+object1 combination
                        obj2_mask_slice = object2_mask[action_type_idx, obj1_idx]
                        valid_object2s = torch.nonzero(obj2_mask_slice > 0.9).squeeze(-1).tolist()
                        if len(valid_object2s) == 1 and isinstance(valid_object2s[0], int):
                            valid_object2s = [valid_object2s[0]]

                        for obj2_idx in valid_object2s:
                            action_combo = {
                                "action_type": int(action_type_idx),
                                "object1": int(obj1_idx),
                                "location": int(loc_idx),
                                "object2": int(obj2_idx)
                            }
                            valid_combinations.append(action_combo)

        return valid_combinations

    def compute_successor_features_batch(
        self,
        states: torch.Tensor,  # [batch, state_dim]
        actions: List[Dict[str, torch.Tensor]], # List of batch action dicts
    ) -> torch.Tensor:  # [batch, sf_dim]
        """Compute SF for a batch of state-action pairs."""
        batch_size = states.shape[0]

        # Encode states
        h = self.encoder(states)  # [batch, hidden_dim]

        # Prepare action embeddings for the batch (squeeze dim=1 since each component is a scalar)
        action_type_indices = torch.cat([a["action_type"] for a in actions])  # [batch]
        object1_indices = torch.cat([a["object1"] for a in actions])  # [batch]
        location_indices = torch.cat([a["location"] for a in actions])  # [batch]
        object2_indices = torch.cat([a["object2"] for a in actions])  # [batch]

        action_type_embs = self.action_type_embed(action_type_indices)  # [batch, 64]
        object1_embs = self.object1_embed(object1_indices)  # [batch, 64]
        location_embs = self.location_embed(location_indices)  # [batch, 64]
        object2_embs = self.object2_embed(object2_indices)  # [batch, 64]

        # Compute action-specific successor features
        psi_t = self.psi_t_head(torch.cat([h, action_type_embs], dim=-1))  # [batch, sf_dim]
        psi_x = self.psi_x_head(torch.cat([h, action_type_embs, object1_embs], dim=-1))  # [batch, sf_dim]
        psi_y = self.psi_y_head(torch.cat([h, action_type_embs, object1_embs, location_embs], dim=-1))  # [batch, sf_dim]
        psi_z = self.psi_z_head(torch.cat([h, action_type_embs, object1_embs, location_embs, object2_embs], dim=-1))  # [batch, sf_dim]

        # Combine them into full SF (element-wise sum)
        sf = psi_t + psi_x + psi_y + psi_z  # [batch, sf_dim]

        return sf

    def forward(
        self,
        state: torch.Tensor | None = None,
        action: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute successor features ψ(s,a)."""

        if state is not None and action is not None:
            # Compute successor features ψ(s, a)
            h = self.encoder(state)
            action_type_emb = self.action_type_embed(action["action_type"])
            object1_emb = self.object1_embed(action["object1"])
            location_emb = self.location_embed(action["location"])
            object2_emb = self.object2_embed(action["object2"])

            psi_t = self.psi_t_head(torch.cat([h, action_type_emb], dim=-1))
            psi_x = self.psi_x_head(torch.cat([h, action_type_emb, object1_emb], dim=-1))
            psi_y = self.psi_y_head(torch.cat([h, action_type_emb, object1_emb, location_emb], dim=-1))
            psi_z = self.psi_z_head(torch.cat([h, action_type_emb, object1_emb, location_emb, object2_emb], dim=-1))

            return psi_t + psi_x + psi_y + psi_z
        else:
            raise ValueError("Must provide state and action for SF computation")

    def compute_task_weight(self, task_vector: torch.Tensor) -> torch.Tensor:
        """Compute task weight vector w(τ)."""
        return self.task_head(task_vector)  # [batch, sf_dim]


def _task_to_vector(
    task: Dict[str, str | None],
    task_types: Tuple[str, ...],
    locations: Tuple[str, ...],
    object_kinds: Tuple[str, ...],
    object_names: Tuple[str, ...],
) -> np.ndarray:
    """Convert task dict to one-hot vector."""
    task_type_idx = task_types.index(task.get("task_type", "")) if task.get("task_type") in task_types else -1
    target_location_idx = locations.index(task.get("target_location")) if task.get("target_location") in locations else -1
    target_kind_idx = object_kinds.index(task.get("target_kind")) if task.get("target_kind") in object_kinds else -1
    object_name_idx = object_names.index(task.get("object_name")) if task.get("object_name") in object_names else -1

    vec = np.zeros(len(task_types) + len(locations) + len(object_kinds) + len(object_names), dtype=np.float32)
    if task_type_idx >= 0:
        vec[task_type_idx] = 1.0

    if target_location_idx >= 0:
        vec[len(task_types) + target_location_idx] = 1.0

    if target_kind_idx >= 0:
        vec[len(task_types) + len(locations) + target_kind_idx] = 1.0

    if object_name_idx >= 0:
        vec[len(task_types) + len(locations) + len(object_kinds) + object_name_idx] = 1.0

    return vec


class AimLogger:
    def __init__(self, args: argparse.Namespace, run_label: str) -> None:
        self._run = None
        try:
            from aim import Run  # type: ignore
        except ImportError:
            print("[train] Aim logging disabled: install `aim` to enable experiment tracking.")
            return

        self._run = Run(experiment="restaurant_sf")
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
        name: str,
        *,
        step: int | None = None,
        context: Mapping[str, str] | None = None,
    ) -> None:
        if self._run is not None:
            self._run.track(value, name, step=step, context=context)

    def close(self) -> None:
        if self._run is not None:
            self._run.close()


def _select_action(
    sf_net: SuccessorFeatureNetwork,
    state: np.ndarray,
    valid_action_type_mask: np.ndarray,
    valid_object1_mask: np.ndarray,
    valid_location_mask: np.ndarray,
    valid_object2_mask: np.ndarray,
    task: Dict[str, str | None],
    epsilon: float,
    device: torch.device,
    action_dims: Tuple[int, int, int, int],
    task_types: Tuple[str, ...],
    locations: Tuple[str, ...],
    object_names: Tuple[str, ...],
    object_kinds: Tuple[str, ...],
) -> Dict[str, int]:  # Return structured action dict
    """Select action using successor features and task weights."""

    if random.random() < epsilon:
        # Random action from valid ones
        valid_actions = sf_net._enumerate_valid_actions(
            valid_action_type_mask,
            valid_object1_mask,
            valid_location_mask,
            valid_object2_mask,
            action_dims,
        )
        if valid_actions:
            return random.choice(valid_actions)
        # Fallback if no valid actions
        return {
            "action_type": 0,
            "object1": 0,
            "location": 0,
            "object2": 0,
        }

    with torch.no_grad():
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, state_dim]

        # Convert task to weight vector
        task_vector = _task_to_vector(task, task_types, locations, object_kinds, object_names)
        task_weight = torch.tensor(task_vector, dtype=torch.float32, device=device).unsqueeze(0)  # [1, task_len]
        w_tau = sf_net.compute_task_weight(task_weight)  # [1, sf_dim]

        # Enumerate all valid actions to score them
        valid_actions = sf_net._enumerate_valid_actions(
            valid_action_type_mask,
            valid_object1_mask,
            valid_location_mask,
            valid_object2_mask,
            action_dims,
        )

        if not valid_actions:
            # Fallback if no valid actions
            return {
                "action_type": 0,
                "object1": 0,
                "location": 0,
                "object2": 0,
            }

        # Batch all valid actions through compute_successor_features_batch
        state_batch = state_tensor.repeat(len(valid_actions), 1)  # [num_valid, state_dim]
        action_list = [
            {
                "action_type": torch.tensor([a["action_type"]], dtype=torch.long, device=device),
                "object1": torch.tensor([a["object1"]], dtype=torch.long, device=device),
                "location": torch.tensor([a["location"]], dtype=torch.long, device=device),
                "object2": torch.tensor([a["object2"]], dtype=torch.long, device=device),
            }
            for a in valid_actions
        ]
        psi_batch = sf_net.compute_successor_features_batch(state_batch, action_list)  # [num_valid, sf_dim]
        q_batch = (psi_batch * w_tau).sum(dim=1)  # [num_valid]
        best_idx = int(torch.argmax(q_batch).item())
        return valid_actions[best_idx]


def _optimize(
    sf_net: SuccessorFeatureNetwork,
    target_sf_net: SuccessorFeatureNetwork,
    task_net: SuccessorFeatureNetwork,  # Separate net for task weights (share params generally, but this is clean for updates)
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    action_types: Tuple[str, ...],
    task_types: Tuple[str, ...],
    locations: Tuple[str, ...],
    object_names: Tuple[str, ...],
    object_kinds: Tuple[str, ...],
) -> float | None:
    if len(replay) < args.batch_size:
        return None

    batch = replay.sample(args.batch_size)

    # Convert batch data to tensors
    states = torch.tensor(
        np.stack([t.state for t in batch]),
        dtype=torch.float32,
        device=device,
    )  # [batch, state_dim]

    # Convert actions to tensors
    batch_actions = []
    for t in batch:
        action_dict = {
            "action_type": torch.tensor([t.action["action_type"]], dtype=torch.long, device=device),
            "object1": torch.tensor([t.action["object1"]], dtype=torch.long, device=device),
            "location": torch.tensor([t.action["location"]], dtype=torch.long, device=device),
            "object2": torch.tensor([t.action["object2"]], dtype=torch.long, device=device),
        }
        batch_actions.append(action_dict)

    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)  # [batch]
    dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=device)  # [batch]

    # Convert next state masks
    next_valid_action_types = np.stack([t.next_valid_action_type_mask for t in batch]) # [batch, action_type_count]
    next_valid_obj1 = np.stack([t.next_valid_object1_mask for t in batch])    # [batch, action_type_count, object_count+1]
    next_valid_loc = np.stack([t.next_valid_location_mask for t in batch])    # [batch, action_type_count, loc_count+1]
    next_valid_obj2 = np.stack([t.next_valid_object2_mask for t in batch])    # [batch, action_type_count, obj_count+1, obj_count+1]

    # Convert next tasks to tensor and compute task weights (online for argmax, target for eval)
    next_tasks = []
    for t in batch:
        task_vec = _task_to_vector(t.next_task, task_types, locations, object_kinds, object_names)
        next_tasks.append(task_vec)
    next_task_vectors = torch.tensor(np.stack(next_tasks), dtype=torch.float32, device=device)  # [batch, task_vector_dim]
    next_task_weights_online = sf_net.compute_task_weight(next_task_vectors)  # [batch, sf_dim] — for argmax only
    next_task_weights_tgt = target_sf_net.compute_task_weight(next_task_vectors)  # [batch, sf_dim] — for evaluation

    # Compute current SF predictions: psi(s, a) for taken actions
    current_psi_sa = sf_net.compute_successor_features_batch(states, batch_actions)  # [batch, sf_dim]

    # Compute Q values: psi(s,a)^T * w(τ)
    current_task_vectors = torch.tensor(
        np.stack([_task_to_vector(t.task, task_types, locations, object_kinds, object_names) for t in batch]),
        dtype=torch.float32, device=device
    )  # [batch, task_vector_dim]
    current_task_weights = sf_net.compute_task_weight(current_task_vectors)  # Use online net for current tasks
    current_q_values = (current_psi_sa * current_task_weights).sum(dim=1)  # [batch]

    # Compute targets: r + γ * max_a psi(s', a)^T * w(τ')
    next_states = torch.tensor(
        np.stack([t.next_state for t in batch]),
        dtype=torch.float32,
        device=device,
    )

    # For each next state, enumerate valid actions and compute max Q'
    with torch.no_grad():
        target_q_values = []
        for i in range(len(batch)):
            state = next_states[i:i+1]  # [1, state_dim]
            w_tgt = next_task_weights_tgt[i:i+1]  # [1, sf_dim] — for evaluation
            w_online = next_task_weights_online[i:i+1]  # [1, sf_dim] — for argmax

            # Get valid masks for this specific transition
            valid_action_types = next_valid_action_types[i]
            valid_obj1 = next_valid_obj1[i]
            valid_loc = next_valid_loc[i]
            valid_obj2 = next_valid_obj2[i]

            # Find all valid actions
            valid_combs = sf_net._enumerate_valid_actions(
                valid_action_types,
                valid_obj1,
                valid_loc,
                valid_obj2,
                (len(action_types), len(object_names)+1, len(locations)+1, len(object_names)+1),
            )

            if not valid_combs:
                target_q_values.append(0.0)
                continue

            # Batch ψ_online(s', a) for all valid actions, then argmax
            state_batch = state.repeat(len(valid_combs), 1)  # [num_valid, state_dim]
            action_list = [
                {
                    "action_type": torch.tensor([c["action_type"]], dtype=torch.long, device=device),
                    "object1": torch.tensor([c["object1"]], dtype=torch.long, device=device),
                    "location": torch.tensor([c["location"]], dtype=torch.long, device=device),
                    "object2": torch.tensor([c["object2"]], dtype=torch.long, device=device),
                }
                for c in valid_combs
            ]
            # a* = argmax ψ_online(s', a)^T w_online(τ')  (full online, eq. 108)
            online_sf_batch = sf_net.compute_successor_features_batch(state_batch, action_list)  # [num_valid, sf_dim]
            online_qs = (online_sf_batch * w_online).sum(dim=1)  # [num_valid]
            best_idx = torch.argmax(online_qs).item()
            best_comb = valid_combs[best_idx]

            # Evaluate with target network: y = ψ_tgt(s', a*)^T w_tgt(τ')
            target_sf = target_sf_net.compute_successor_features_batch(state, [{
                "action_type": torch.tensor([best_comb["action_type"]], dtype=torch.long, device=device),
                "object1": torch.tensor([best_comb["object1"]], dtype=torch.long, device=device),
                "location": torch.tensor([best_comb["location"]], dtype=torch.long, device=device),
                "object2": torch.tensor([best_comb["object2"]], dtype=torch.long, device=device),
            }])[0]
            target_q = torch.dot(target_sf, w_tgt[0]).item()
            target_q_values.append(target_q)

        target_q_values = torch.tensor(target_q_values, dtype=torch.float32, device=device)

        # Standard TD target: r + γ(1-done) * max_Q'. The done flag controls
        # cross-task bootstrapping: myopic → done=True at every task boundary;
        # anticipatory → done=False (γ continues into next task).
        gamma_term = args.gamma * (1.0 - dones) * target_q_values
        targets = rewards + gamma_term

    # Compute loss and backpropagate
    loss = nn.functional.mse_loss(current_q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(sf_net.parameters(), args.max_grad_norm)
    optimizer.step()

    return float(loss.item())


def _classify_pick_place_failure(
    env: RestaurantSymbolicEnv,
    start_task: Mapping[str, str | None] | None,
    end_task: Mapping[str, str | None] | None,
    end_holding: str | None,
    task_steps: int,
    max_steps_per_task: int,
    success: bool,
) -> str:
    """Classify the reason for task failure in pick-place tasks."""
    if success:
        return "success"
    if task_steps >= max_steps_per_task:
        return "timeout"

    if start_task is None:
        return "error-unexpected"

    expected_target = start_task.get("object_name")
    target_location = start_task.get("target_location")

    if expected_target is None or target_location is None:
        return "error-invalid-task"

    if end_task is not None and end_task.get("task_type") != "pick_place":
        # Advanced to different task type (could be success)
        return "success-next-type"

    held_obj = getattr(env.state, 'holding', end_holding)

    # Check if we picked the object first
    if held_obj != expected_target:
        # Check all objects to see if target is still available
        target_available = any(
            obj.name == expected_target and obj.location == target_location
            for obj in env.state.objects.values()
        )
        if target_available:
            return "picked-wrong-object"
        return "object-not-at-dest-location"

    # We're holding the right object but haven't put it in target location
    current_loc = getattr(env.state, 'agent_location', "unknown")
    if current_loc == target_location:
        # We're at right place but didn't place the object
        return "reachable-but-not-placed"
    else:
        # We either don't have the right object, or object is at destination, or timeout
        return "held-incorrect-object"


def _run_post_train_inference(
    args: argparse.Namespace,
    output_path: Path,
    env: RestaurantSymbolicEnv,
    eval_task_generator: Mapping[str, Mapping[str, object]],
) -> None:
    """Run post-training greedy inference and save trajectory information."""
    device = torch.device("cpu")
    action_types = tuple(ACTION_TYPES)
    task_types = tuple(env.task_types)
    locations = tuple(env.locations)
    object_names = tuple(env.object_names)
    object_kinds = tuple(env.object_kinds)

    # Reload the trained weights
    obs_dim = int(np.asarray(env.observation_space.shape[0]))
    model = SuccessorFeatureNetwork(
        obs_dim=obs_dim,
        action_type_count=len(action_types),
        object_count=len(object_names),
        location_count=len(locations),
        object_names=object_names,
        locations=locations,
        action_types=action_types,
        object_kinds=object_kinds,
        task_types=task_types,
        sf_dim=args.sf_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))

    # Run inference trajectories
    trajectories = []
    max_steps = 10_000  # Max total inference episodes to prevent infinite loop

    eval_tasks = [eval_task_generator["pick_place"][0] for _ in range(100)]  # Run 100 eval episodes

    for i, task in enumerate(eval_tasks):
        obs, info = env.reset()
        env.set_task(task["task_type"], target_location=task["target_location"], object_name=task["object_name"])
        obs = env._obs()
        info = env._info(success=False)
        start_task = info.get("task")

        trajectory = {
            "episode_id": i,
            "task": dict(task),
            "states": [],
            "actions": [],
            "observations": [],
            "rewards": [],
            "steps": 0,
            "final_state": {},
            "success": False,
        }
        state = obs
        steps = 0
        total_reward = 0.0
        success = False
        truncated = False

        action_dims = (
            len(action_types), len(object_names) + 1, len(locations) + 1, len(object_names) + 1
        )

        while steps < args.max_steps_per_task and not success and not truncated:
            # Extract all valid masks from info (top-level keys, not nested)
            action_type_mask = info.get("valid_action_type_mask")
            object1_mask = info.get("valid_object1_mask")
            location_mask = info.get("valid_location_mask")
            object2_mask = info.get("valid_object2_mask")

            if action_type_mask is None or object1_mask is None or location_mask is None or object2_mask is None:
                valid_action_dict = env._compute_action_masks()
                action_type_mask = valid_action_dict["valid_action_type_mask"]
                object1_mask = valid_action_dict["valid_object1_mask"]
                location_mask = valid_action_dict["valid_location_mask"]
                object2_mask = valid_action_dict["valid_object2_mask"]

            action = _select_action(
                model, state, action_type_mask, object1_mask, location_mask, object2_mask,
                info.get("task", {}), epsilon=0.0,  # Greedy inference
                device=device,
                action_dims=action_dims,
                task_types=task_types,
                locations=locations,
                object_names=object_names,
                object_kinds=object_kinds,
            )

            trajectory["actions"].append(dict(action))
            trajectory["observations"].append(obs.tolist())
            trajectory["states"].append({
                "agent_location": env.state.agent_location,
                "holding": env.state.holding,
                "objects": {
                    name: {"kind": obj.kind, "location": obj.location, "dirty": obj.dirty,
                           "filled_with": obj.filled_with, "contained_in": obj.contained_in}
                    for name, obj in env.state.objects.items()
                },
                "bread_spread": env.state.bread_spread,
            })

            # Take action
            obs, reward, success, truncated, info = env.step(action)
            state = obs

            trajectory["rewards"].append(reward)
            total_reward += reward
            steps += 1

            if success:
                trajectory["success"] = True
                break

        trajectory["steps"] = steps
        trajectory["total_reward"] = total_reward
        trajectory["final_state"] = {
            "agent_location": env.state.agent_location,
            "holding": env.state.holding,
            "objects": {
                name: {"kind": obj.kind, "location": obj.location, "dirty": obj.dirty,
                       "filled_with": obj.filled_with, "contained_in": obj.contained_in}
                for name, obj in env.state.objects.items()
            },
            "bread_spread": env.state.bread_spread,
        }
        trajectory["final_info"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
        trajectory["failure_reason"] = "success" if success else _classify_pick_place_failure(
            env,
            start_task,
            info.get("task"),  # post-step task (same as start_task if success, resampled if truncated)
            info.get("holding"),
            steps,
            args.max_steps_per_task,
            success,
        )

        trajectories.append(trajectory)

        if i >= len(eval_tasks) - 1:  # Completed all eval tasks
            break

    # Save trajectory information
    infer_dir = output_path.parent / "post_train_infer"
    infer_dir.mkdir(exist_ok=True)

    # Compute aggregate stats
    successes = len([t for t in trajectories if t["success"]])
    avg_steps = np.mean([t["steps"] for t in trajectories])
    avg_reward = np.mean([t["total_reward"] for t in trajectories])
    failure_breakdown = {}
    for traj in trajectories:
        reason = traj["failure_reason"]
        failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1

    summary = {
        "inference_run_label": f"{output_path.stem}_post_eval",
        "episodes_run": len(trajectories),
        "eval_episodes": len(eval_tasks),
        "successful_episodes": successes,
        "success_rate": 0.0 if not trajectories else successes / len(trajectories),
        "average_steps": float(avg_steps),
        "average_reward": float(avg_reward),
        "failure_analysis": failure_breakdown,
    }

    with (infer_dir / "trajectory_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (infer_dir / "trajectories.json").open("w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"[inference] Success rate: {successes}/{len(trajectories)} ({summary['success_rate']:.3f})")
    print(f"[inference] Avg steps: {avg_steps:.1f}, Avg reward: {avg_reward:.1f}")

    # Create a basic visualization of success rates
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        success_counts = [successes, len(trajectories) - successes]
        labels = ['Success', 'Failure']
        colors = ['#28a745', '#dc3545']
        ax.bar(labels, success_counts, color=colors)
        ax.set_ylabel('Episode Count')
        ax.set_title(f'Training Result Summary\n({successes}/{len(trajectories)} Successful)')

        for i, v in enumerate(success_counts):
            ax.text(i, v + 0.05*max(success_counts), str(v), ha='center', va='bottom')

        plt.tight_layout()
        plot_path = infer_dir / "trajectory_plot.png"
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[inference] Plot saved to {plot_path}")
    except Exception as e:
        print(f"[inference] Could not create plot: {e}")


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

    # Get environment dimensions
    action_types = tuple(ACTION_TYPES)
    task_types = tuple(env.task_types)
    locations = tuple(env.locations)
    object_names = tuple(env.object_names)
    object_kinds = tuple(env.object_kinds)

    run_label = resolve_run_label(args)
    run_dir = Path("runs") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / args.output_name
    print(f"[train] Run artifacts -> {run_dir.resolve()} ({run_label})")

    obs, info = env.reset()
    obs_dim = int(np.asarray(obs).shape[0])

    # Create SF networks
    sf_net = SuccessorFeatureNetwork(
        obs_dim=obs_dim,
        action_type_count=len(action_types),
        object_count=len(object_names),
        location_count=len(locations),
        object_names=object_names,
        locations=locations,
        action_types=action_types,
        object_kinds=object_kinds,
        task_types=task_types,
        sf_dim=args.sf_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    target_sf_net = SuccessorFeatureNetwork(
        obs_dim=obs_dim,
        action_type_count=len(action_types),
        object_count=len(object_names),
        location_count=len(locations),
        object_names=object_names,
        locations=locations,
        action_types=action_types,
        object_kinds=object_kinds,
        task_types=task_types,
        sf_dim=args.sf_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    target_sf_net.load_state_dict(sf_net.state_dict())
    target_sf_net.eval()

    optimizer = optim.Adam(sf_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    aim_logger = AimLogger(args, run_label)

    task_return = 0.0
    task_paper2_cost = 0.0
    task_steps = 0
    total_tasks = 0
    tasks_completed = 0
    current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
    steps_since_reset = 0
    tasks_since_world_reset = 0
    episode_index = 0

    recent_returns: Deque[float] = deque(maxlen=100)
    recent_success: Deque[int] = deque(maxlen=100)
    recent_auto: Deque[int] = deque(maxlen=100)
    loss_history: List[float] = []
    step_reward_history: List[float] = []
    task_records: List[Dict[str, float | int | bool | str | None]] = []


    progress = tqdm(range(args.total_steps), desc="Restaurant SF-DQN", unit="step")
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

        # Extract structured action masks (top-level keys in info, not nested)
        valid_action_type_mask = info.get("valid_action_type_mask")
        valid_object1_mask = info.get("valid_object1_mask")
        valid_location_mask = info.get("valid_location_mask")
        valid_object2_mask = info.get("valid_object2_mask")

        # Fallback: recompute if masks not in info
        if valid_action_type_mask is None:
            computed_masks = env._compute_action_masks()
            valid_action_type_mask = computed_masks["valid_action_type_mask"]
            valid_object1_mask = computed_masks["valid_object1_mask"]
            valid_location_mask = computed_masks["valid_location_mask"]
            valid_object2_mask = computed_masks["valid_object2_mask"]

        action_dims = (
            len(action_types), len(object_names) + 1, len(locations) + 1, len(object_names) + 1
        )

        action = _select_action(
            sf_net, obs, valid_action_type_mask, valid_object1_mask, valid_location_mask, valid_object2_mask,
            info.get("task", {}), epsilon=epsilon,
            device=device,
            action_dims=action_dims,
            task_types=task_types,
            locations=locations,
            object_names=object_names,
            object_kinds=object_kinds,
        )

        next_obs, reward, success, truncated, next_info = env.step(action)
        task_return += float(reward)
        task_paper2_cost += float(next_info.get("paper2_cost_step", 0.0))
        task_steps += 1
        step_reward_history.append(float(reward))

        steps_since_reset += 1
        env_reset_flag = False

        if args.episode_step_limit > 0 and steps_since_reset >= args.episode_step_limit:
            env_reset_flag = True

        # Track task_sequence_length boundary BEFORE computing done, so
        # env_reset_flag is already set and the transition correctly reflects
        # whether γ-bootstrapping should occur across this step.
        if success or truncated:
            tasks_since_world_reset += 1
            if args.task_sequence_length > 0 and tasks_since_world_reset >= args.task_sequence_length:
                env_reset_flag = True
                tasks_since_world_reset = 0

        # done=True stops γ-bootstrapping. Myopic: every task boundary is terminal.
        # Anticipatory: terminal only at physical resets (task_sequence_length).
        done = args.myopic and (success or truncated)
        if env_reset_flag:
            done = True

        # Create and store transition with task info (masks are top-level keys in info)
        next_valid_action_type_mask = next_info.get("valid_action_type_mask", np.zeros(len(action_types), dtype=np.float32))
        next_valid_object1_mask = next_info.get("valid_object1_mask", np.zeros((len(action_types), len(object_names)+1), dtype=np.float32))
        next_valid_location_mask = next_info.get("valid_location_mask", np.zeros((len(action_types), len(locations)+1), dtype=np.float32))
        next_valid_object2_mask = next_info.get("valid_object2_mask", np.zeros((len(action_types), len(object_names)+1, len(object_names)+1), dtype=np.float32))

        transition = Transition(
            state=np.array(obs, dtype=np.float32, copy=True),
            action=action,
            reward=float(reward),
            next_state=np.array(next_obs, dtype=np.float32, copy=True),
            done=done,
            next_valid_action_type_mask=next_valid_action_type_mask.copy(),
            next_valid_object1_mask=next_valid_object1_mask.copy(),
            next_valid_location_mask=next_valid_location_mask.copy(),
            next_valid_object2_mask=next_valid_object2_mask.copy(),
            task=current_task_snapshot,
            next_task=dict(next_info.get("task", {})),
            task_boundary=bool(success),
        )
        replay.push(transition)

        loss_value = _optimize(
            sf_net, target_sf_net, sf_net,  # task_net can be same as sf_net for simplicity in bi-linear version
            replay, optimizer, args, device,
            action_types, task_types, locations, object_names, object_kinds
        )
        if loss_value is not None:
            loss_history.append(loss_value)

        # Update target network with Polyak averaging or hard update
        if args.tau < 1.0:
            with torch.no_grad():
                tau = float(args.tau)
                for target_param, param in zip(target_sf_net.parameters(), sf_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        elif (global_step + 1) % args.target_update == 0:
            target_sf_net.load_state_dict(sf_net.state_dict())

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
            task_records.append({
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
                "object_name": current_task_snapshot.get("object_name"),
                "layout_id": current_layout_snapshot,
                "task_type_after": task_info.get("task_type"),
                "target_location_after": task_info.get("target_location"),
                "target_kind_after": task_info.get("target_kind"),
                "object_name_after": task_info.get("object_name"),
            })
            task_return = 0.0
            task_paper2_cost = 0.0
            task_steps = 0
            current_task_auto_satisfied = bool(next_info.get("next_auto_satisfied", False))

        if env_reset_flag:
            episode_index += 1
            reset_seed = args.seed + 100_003 * episode_index
            obs, info = env.reset(seed=reset_seed)
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
            tasks_since_world_reset = 0
            steps_since_reset = 0

        # Logging
        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        success_rate = float(np.mean(recent_success)) if recent_success else 0.0
        auto_rate = float(np.mean(recent_auto)) if recent_auto else 0.0
        avg_loss = float(np.mean(loss_history[-100:])) if loss_history else 0.0

        # Log metrics
        logs = {
            "train/return": avg_return,
            "train/success_rate": success_rate,
            "train/auto_satisfaction_rate": auto_rate,
            "train/epsilon": epsilon,
            "train/loss": avg_loss,
            "train/tasks_completed": tasks_completed,
            "train/total_steps": global_step,
        }

        if device.type == "cuda":
            logs.update({
                "train/grad_norm": float(torch.norm(
                    torch.stack([torch.norm(p.grad) for p in sf_net.parameters() if p.grad is not None])
                )) if any(p.grad is not None for p in sf_net.parameters()) else 0.0,
            })

        for metric_name, value in logs.items():
            aim_logger.track(value, metric_name, step=global_step, context={"split": "train"})

        progress.set_postfix(
            ret=f"{avg_return:.1f}" if recent_returns else "n/a",
            success=f"{success_rate:.2f}",
            auto=f"{auto_rate:.2f}",
            loss=f"{avg_loss:.3f}" if loss_history else "n/a",
            tasks=tasks_completed,
        )

    # Save trained model
    torch.save(sf_net.state_dict(), output_path)
    print(f"Saved SF-DQN weights to {output_path}")

    # Calculate and save summaries
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
        "task_sequence_length": int(args.task_sequence_length),
        "seed": int(args.seed),
        "sf_dim": int(args.sf_dim),
        "model_type": "Successor Features DQN (bilinear ψ^T w)",
    }

    with (run_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (run_dir / "task_records.json").open("w") as f:
        json.dump(task_records, f, indent=2)

    with (run_dir / "train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Run post-train inference only for pick_place tasks for now
    try:
        task_lib = [{"task_type": "pick_place", "target_location": "table_left", "object_name": "mug_red"}]
        eval_tasks = {"pick_place": task_lib}
        _run_post_train_inference(args, output_path, env, eval_tasks)
    except Exception as e:
        print(f"[train] Skipping post-train inference due to error: {e}")

    aim_logger.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Successor Features DQN on the symbolic restaurant environment.")
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--replay-size", type=int, default=50_000, help="Experience replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size for training")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size for networks")
    parser.add_argument("--sf-dim", type=int, default=64, help="Successor feature dimension (size of w vector)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Final epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=int, default=100_000, help="Steps to decay epsilon")
    parser.add_argument("--target-update", type=int, default=1_000, help="Steps between target network updates")
    parser.add_argument("--tau", type=float, default=1.0, help="Polyak averaging factor (tau=1.0 means hard update)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--task-sequence-length", type=int, default=200, help="Physical reset interval in task attempts.")
    parser.add_argument("--myopic", action="store_true", help="Myopic mode: don't bootstrap at task boundaries")
    parser.add_argument(
        "--episode-step-limit",
        type=int, default=0,
        help="Maximum primitive steps between resets; <=0 disables."
    )
    parser.add_argument("--max-steps-per-task", type=int, default=24, help="Max steps per individual task")
    parser.add_argument("--success-reward", type=float, default=15.0, help="Reward for completing a task")
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0, help="Penalty for invalid actions")
    parser.add_argument("--travel-cost-scale", type=float, default=25.0, help="Travel cost scale factor")
    parser.add_argument("--pick-cost", type=float, default=25.0, help="Cost of pick actions")
    parser.add_argument("--place-cost", type=float, default=25.0, help="Cost of place actions")
    parser.add_argument("--wash-cost", type=float, default=25.0, help="Cost of wash actions")
    parser.add_argument("--fill-cost", type=float, default=25.0, help="Cost of fill actions")
    parser.add_argument("--brew-cost", type=float, default=25.0, help="Cost of brew actions")
    parser.add_argument("--fruit-cost", type=float, default=25.0, help="Cost of fruit action")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("anticipatory_rl/configs/restaurant/restaurant_symbolic.yaml"),
        help="Path to the configuration file"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--run-label", type=str, default=None, help="Run label for logging")
    parser.add_argument("--output-name", type=str, default="restaurant_sf_dqn.pt", help="Output model filename")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()