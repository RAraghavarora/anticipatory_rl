"""Task-conditioned Q-learning agent for the MiniWorld rearrangement task.

This script encodes both the environment state and a symbolic task vector, then
trains a DQN-style agent that learns a single Q(s, τ, a) over the augmented
state-task space. Use the CLI below to launch training with default settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

MPL_CACHE_DIR = (Path("runs") / ".matplotlib").resolve()
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
Path(MPL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

XDG_CACHE_DIR = (Path("runs") / ".cache").resolve()
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
Path(XDG_CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from anticipatory_rl.envs.miniworld_env import (
    MiniWorldGridRearrange,
    OBJECT_SPECS,
    RECEPTACLES,
    OBJECT_NAMES,
    SixAction,
)
from anticipatory_rl.tasks.build_problem_from_task import (
    CONFIG as PROBLEM_CONFIG,
    _parse_template,
    build_problem_text_for_task,
)
from anticipatory_rl.tasks.generator import OBJECT_SOURCE_DIST
from anticipatory_rl.tasks.planner_utils import run_planner


# --------------------------------------------------------------------------- #
# Encoding utilities                                                          #
# --------------------------------------------------------------------------- #
GRID_SIZE = 6
GRID_CELLS = GRID_SIZE * GRID_SIZE
OBJECT_NAMES = [spec.name for spec in OBJECT_SPECS]
RECEPTACLE_NAMES = [spec.name for spec in RECEPTACLES]
OBJECT_INDEX = {name: idx for idx, name in enumerate(OBJECT_NAMES)}
RECEPTACLE_INDEX = {name: idx for idx, name in enumerate(RECEPTACLE_NAMES)}
LOCATION_DIM = GRID_CELLS + len(RECEPTACLE_NAMES) + 1  # +1 for gripper slot
NUM_OBJECTS = len(OBJECT_NAMES)
NUM_RECEPTACLES = len(RECEPTACLE_NAMES)
TASK_TYPE_INDEX = {"bring_single": 0, "bring_pair": 1, "clear_receptacle": 2}
ACTION_NAME_TO_ID = {
    "move_up": SixAction.move_up,
    "move_down": SixAction.move_down,
    "move_left": SixAction.move_left,
    "move_right": SixAction.move_right,
    "pick": SixAction.pick,
    "place": SixAction.place,
}
OBJECT_HOME_REGION = {spec.name: spec.home_region for spec in OBJECT_SPECS}


def _tile_to_index(tile: Tuple[int, int]) -> int:
    x, y = tile
    return y * GRID_SIZE + x


def encode_state(state: Dict[str, object]) -> np.ndarray:
    """Encode the environment state into a flat feature vector."""
    features: List[float] = []

    agent_x, agent_y = state["agent"]["grid_pos"]
    features.append(agent_x / (GRID_SIZE - 1))
    features.append(agent_y / (GRID_SIZE - 1))

    holding = state["agent"]["carrying"]
    holding_vec = np.zeros(NUM_OBJECTS + 1, dtype=np.float32)
    if holding is None:
        holding_vec[-1] = 1.0
    else:
        holding_vec[OBJECT_INDEX[holding]] = 1.0
    features.extend(holding_vec.tolist())

    for obj_name in OBJECT_NAMES:
        obj_data = state["objects"][obj_name]
        vec = np.zeros(LOCATION_DIM, dtype=np.float32)
        if state["agent"]["carrying"] == obj_name:
            vec[-1] = 1.0  # gripper slot
        elif obj_data["tile"] is not None:
            vec[_tile_to_index(tuple(obj_data["tile"]))] = 1.0
        elif obj_data["region"] is not None:
            offset = GRID_CELLS + RECEPTACLE_INDEX[obj_data["region"]]
            vec[offset] = 1.0
        else:
            raise ValueError(f"Object {obj_name} has no location info.")
        features.extend(vec.tolist())

    return np.asarray(features, dtype=np.float32)


def encode_task(task: Dict[str, object]) -> np.ndarray:
    """Encode a symbolic task into a fixed-length vector."""
    vec = np.zeros(
        3 + 2 * NUM_OBJECTS + 3 * NUM_RECEPTACLES, dtype=np.float32
    )
    task_type = task["task_type"]
    vec[TASK_TYPE_INDEX[task_type]] = 1.0

    payload = task["payload"]
    objects = payload.get("objects", [])
    for idx, obj_name in enumerate(objects[:2]):
        base = 3 + idx * NUM_OBJECTS
        vec[base + OBJECT_INDEX[obj_name]] = 1.0

    sources_field = payload.get("sources")
    if sources_field:
        sources = list(sources_field)
    elif task_type == "clear_receptacle":
        source = payload.get("source")
        sources = [source] if source else []
    else:
        sources = []
    if sources:
        for idx, receptacle in enumerate(sources[:2]):
            base = 3 + 2 * NUM_OBJECTS + idx * NUM_RECEPTACLES
            vec[base + RECEPTACLE_INDEX[receptacle]] = 1.0

    target = payload.get("target")
    if target:
        target_offset = 3 + 2 * NUM_OBJECTS + 2 * NUM_RECEPTACLES
        vec[target_offset + RECEPTACLE_INDEX[target]] = 1.0
    return vec


def build_augmented_state(state: Dict[str, object], task: Dict[str, object]) -> np.ndarray:
    return np.concatenate([encode_state(state), encode_task(task)])


def task_goal_satisfied(state: Dict[str, object], task: Dict[str, object]) -> bool:
    """Check whether the current environment state satisfies task τ."""
    task_type = task["task_type"]
    payload = task["payload"]

    if task_type in {"bring_single", "bring_pair"}:
        target = payload["target"]
        for obj in payload["objects"]:
            region = state["objects"][obj]["region"]
            if region != target:
                return False
        return True

    if task_type == "clear_receptacle":
        source = payload["source"]
        return all(info["region"] != source for info in state["objects"].values())

    raise ValueError(f"Unknown task type '{task_type}'")


def _save_task_metrics(
    entries: List[Dict[str, object]],
    csv_path: Path,
    plot_path: Path,
) -> None:
    if not entries:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("task_number,task_index,task_type,reward,steps,success,global_step\n")
        for entry in entries:
            fh.write(
                f"{entry['task_number']},{entry['task_index']},{entry['task_type']},"
                f"{entry['reward']},{entry['steps']},{int(entry['success'])},"
                f"{entry['global_step']}\n"
            )

    rewards = [entry["reward"] for entry in entries]
    task_numbers = [entry["task_number"] for entry in entries]
    plt.figure(figsize=(9, 4))
    plt.plot(task_numbers, rewards, label="Task return", color="#1f77b4")
    if len(rewards) >= 5:
        window = max(1, len(rewards) // 20)
        window = max(window, 5)
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="same")
        plt.plot(task_numbers, smoothed, label=f"Moving avg ({window})", color="#ff7f0e")
    plt.axhline(0.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
    plt.xlabel("Completed task")
    plt.ylabel("Return")
    plt.title("Per-task returns during training")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=160)
    plt.close()


class PhiModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PhiPotential:
    def __init__(self, weights_path: Path, device: torch.device):
        if not weights_path.exists():
            raise FileNotFoundError(
                f"φ_θ weights not found at {weights_path}. Train via anticipatory_rl.agents.train_phi."
            )
        checkpoint = torch.load(weights_path, map_location=device)
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint.get("hidden_dim", 32))
        self.device = device
        self.model = PhiModel(input_dim, hidden_dim).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def value(self, features: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return float(self.model(tensor).item())


# --------------------------------------------------------------------------- #
# Q-network and replay buffer                                                 #
# --------------------------------------------------------------------------- #
class TaskConditionedQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 6)  # six discrete actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def load_tasks(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text())


def epsilon_by_frame(frame_idx: int, eps_start: float, eps_final: float, eps_decay: int) -> float:
    return eps_final + (eps_start - eps_final) * np.exp(-1.0 * frame_idx / eps_decay)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_task(tasks: Sequence[Dict[str, object]], rng: random.Random) -> Tuple[int, Dict[str, object]]:
    idx = rng.randrange(len(tasks))
    return idx, tasks[idx]


def _prepare_task_instance(task: Dict[str, object], state: Dict[str, object]) -> Dict[str, object] | None:
    """
    Create a task instance whose payload contains any dynamically-computed metadata
    (e.g., which objects currently sit on the source receptacle for clear tasks).
    """
    inst = copy.deepcopy(task)
    payload = inst.setdefault("payload", {})
    if inst["task_type"] == "clear_receptacle":
        source = payload.get("source")
        if not source:
            raise ValueError("clear_receptacle task missing 'source'")
        if not any(obj_state["region"] == source for obj_state in state["objects"].values()):
            return None
    return inst


def _sample_task_instance(
    tasks: Sequence[Dict[str, object]],
    rng: random.Random,
    state: Dict[str, object],
    max_attempts: int = 128,
) -> Tuple[int, Dict[str, object]]:
    for _ in range(max_attempts):
        idx, base = _sample_task(tasks, rng)
        inst = _prepare_task_instance(base, state)
        if inst is not None:
            return idx, inst
    raise RuntimeError(
        "Unable to sample a compatible task instance for the current state; "
        "consider regenerating tasks or adjusting object placements."
    )


def _init_region_memory(state: Dict[str, object]) -> Dict[str, str]:
    memory: Dict[str, str] = {}
    for obj in OBJECT_NAMES:
        region = state["objects"][obj]["region"]
        if region is None:
            region = OBJECT_HOME_REGION[obj]
        memory[obj] = region
    return memory


def _update_region_memory(memory: Dict[str, str], state: Dict[str, object]) -> None:
    for obj in OBJECT_NAMES:
        region = state["objects"][obj]["region"]
        if region is not None:
            memory[obj] = region


def _phi_features(state: Dict[str, object], memory: Dict[str, str]) -> np.ndarray:
    vec = np.zeros(NUM_OBJECTS * NUM_RECEPTACLES, dtype=np.float32)
    for obj_idx, obj in enumerate(OBJECT_NAMES):
        region = state["objects"][obj]["region"]
        if region is None:
            region = memory.get(obj, OBJECT_HOME_REGION[obj])
        rec_idx = RECEPTACLE_INDEX.get(region)
        if rec_idx is None:
            continue
        vec[obj_idx * NUM_RECEPTACLES + rec_idx] = 1.0
    return vec


def _sample_object_placements(rng: random.Random) -> Dict[str, str]:
    placements: Dict[str, str] = {}
    for obj in OBJECT_NAMES:
        dist = OBJECT_SOURCE_DIST.get(obj)
        if dist:
            options = list(dist.keys())
            weights = [dist[name] for name in options]
            placements[obj] = rng.choices(options, weights=weights, k=1)[0]
        else:
            placements[obj] = rng.choice(RECEPTACLE_NAMES)
    return placements


def _planner_prefill_replay(
    replay: "ReplayBuffer",
    demo_replay: "ReplayBuffer",
    env_kwargs: Dict[str, object],
    *,
    args: argparse.Namespace,
    tasks: Sequence[Dict[str, object]],
    phi: "PhiPotential",
) -> None:
    if args.planner_prefill <= 0:
        return

    template = _parse_template(args.planner_template)
    surface_dist = PROBLEM_CONFIG.get("surface_distribution", {})
    pref_env = MiniWorldGridRearrange(render_mode=None, **env_kwargs)
    rng = random.Random(args.seed + 4242)
    solved = 0

    while solved < args.planner_prefill:
        placements = _sample_object_placements(rng)
        fake_state = {
            "objects": {obj: {"region": region} for obj, region in placements.items()}
        }

        idx = rng.randrange(len(tasks))
        base_task = tasks[idx]
        inst = _prepare_task_instance(base_task, fake_state)
        if inst is None:
            continue

        task_name = f"planner-prefill-{solved}"
        try:
            problem_text = build_problem_text_for_task(
                inst,
                template,
                task_name,
                surface_dist=surface_dist,
                rng=random.Random(rng.randint(0, 10**9)),
                placements=placements,
            )
        except ValueError:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            problem_path = tmp_path / "problem.pddl"
            problem_path.write_text(problem_text)
            plan_path = run_planner(
                args.planner_binary,
                args.planner_domain,
                problem_path,
                args.planner_search,
                tmp_path,
            )
            plan_text = plan_path.read_text()

        actions = plan_to_action_sequence(plan_text)
        if not actions:
            continue

        pref_env.reset()
        pref_env.apply_object_placements(placements)
        state = pref_env.get_state()
        phi_memory = _init_region_memory(state)
        aug_state = build_augmented_state(state, inst)

        success = False
        for act_name in actions:
            action_enum = ACTION_NAME_TO_ID.get(act_name)
            if action_enum is None:
                raise KeyError(f"Unknown action name '{act_name}' from planner")
            action = int(action_enum)

            phi_state_vec = _phi_features(state, phi_memory)
            phi_value = phi.value(phi_state_vec)

            obs, _, _, trunc, info = pref_env.step(action)
            next_state = info["state"]
            _update_region_memory(phi_memory, next_state)
            phi_next_vec = _phi_features(next_state, phi_memory)
            phi_next_value = phi.value(phi_next_vec)

            done = task_goal_satisfied(next_state, inst)
            base_reward = args.goal_reward if done else args.step_penalty
            # reward = base_reward + args.gamma * phi_next_value - phi_value
            reward = base_reward # Ignoring phi for now!

            next_aug_state = build_augmented_state(next_state, inst)
            transition = Transition(aug_state, action, reward, next_aug_state, trunc)
            replay.push(transition)
            demo_replay.push(transition)

            state = next_state
            aug_state = next_aug_state

            if done:
                success = True
                break

        if success:
            solved += 1

    pref_env.close()


def parse_plan(plan_text: str) -> List[List[str]]:
    commands = []
    for raw in plan_text.splitlines():
        line = raw.strip().lower()
        if not line or line.startswith(";"):
            continue
        tokens = line.strip("()").split()
        commands.append(tokens)
    return commands


def grid_from_loc(loc: str) -> Tuple[int, int]:
    _, suffix = loc.split("_")
    return (int(suffix[0]), int(suffix[1]))


def plan_to_action_sequence(plan_text: str) -> List[str]:
    commands = parse_plan(plan_text)
    moves = {
        (1, 0): "move_right",
        (-1, 0): "move_left",
        (0, 1): "move_down",
        (0, -1): "move_up",
    }
    sequence: List[str] = []
    current = (0, 0)
    for tokens in commands:
        name = tokens[0]
        if name == "move":
            target = grid_from_loc(tokens[3])
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            action = moves.get((dx, dy))
            if action is None:
                raise ValueError(f"Unsupported move from {current} to {target}")
            sequence.append(action)
            current = target
        elif name.startswith("pick"):
            sequence.append("pick")
        elif name.startswith("place") or name.startswith("stack") or name.startswith("drop"):
            sequence.append("place")
        else:
            raise ValueError(f"Unknown plan token '{name}'")
    return sequence


def train(args: argparse.Namespace) -> None:
    device = _select_device()
    env_kwargs = {"max_episode_steps": args.env_max_steps}
    env = MiniWorldGridRearrange(render_mode=None, **env_kwargs)
    env.reset(seed=args.seed)

    args.planner_domain = args.planner_domain.resolve()
    args.planner_template = args.planner_template.resolve()
    args.planner_binary = args.planner_binary.resolve()

    if args.train_steps is None:
        args.train_steps = args.tasks_per_run * args.max_task_steps

    tasks = load_tasks(args.tasks_file)
    task_rng = random.Random(args.task_seed if args.task_seed is not None else args.seed)
    phi = PhiPotential(args.phi_weights, device)
    state = env.get_state()
    phi_region_memory = _init_region_memory(state)
    current_task_idx, task = _sample_task_instance(tasks, task_rng, state)
    aug_state = build_augmented_state(state, task)

    state_dim = aug_state.shape[0]
    q_net = TaskConditionedQNetwork(state_dim, args.hidden_dim).to(device)
    target_net = TaskConditionedQNetwork(state_dim, args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)
    demo_replay = ReplayBuffer(args.replay_size)
    log_file = args.log_path.open("w") if args.log_path else None

    _planner_prefill_replay(
        replay,
        demo_replay,
        env_kwargs,
        args=args,
        tasks=tasks,
        phi=phi,
    )

    global_step = 0
    episode_return = 0.0
    steps_since_task = 0

    completed_tasks = 0
    successful_tasks = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[int] = deque(maxlen=100)
    recent_lengths: Deque[int] = deque(maxlen=100)
    task_metrics: List[Dict[str, object]] = []

    progress = tqdm(total=args.train_steps, desc="Training steps")
    while global_step < args.train_steps and completed_tasks < args.tasks_per_run:
        epsilon = epsilon_by_frame(global_step, args.eps_start, args.eps_final, args.eps_decay)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(aug_state, device=device).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        phi_state_vec = _phi_features(state, phi_region_memory)
        phi_value = phi.value(phi_state_vec)

        obs, _, _, trunc, info = env.step(action)
        next_state = info["state"]
        _update_region_memory(phi_region_memory, next_state)
        phi_next_vec = _phi_features(next_state, phi_region_memory)
        phi_next_value = phi.value(phi_next_vec)

        base_reward = args.step_penalty
        done = task_goal_satisfied(next_state, task)
        if done:
            base_reward = args.goal_reward

        # reward = base_reward + args.gamma * phi_next_value - phi_value
        reward = base_reward # Ignoring phi for now!
        next_aug_state = build_augmented_state(next_state, task)
        transition_done = trunc  # only true environment truncations stop bootstrapping

        replay.push(
            Transition(aug_state, action, reward, next_aug_state, transition_done)
        )

        episode_return += reward
        steps_since_task += 1
        global_step += 1

        progress.update(1)
        state = next_state
        boundary = done or steps_since_task >= args.max_task_steps or trunc
        if not boundary:
            aug_state = next_aug_state
        else:
            completed_tasks += 1
            success = bool(done)
            successful_tasks += int(success)
            recent_returns.append(episode_return)
            recent_successes.append(1 if success else 0)
            recent_lengths.append(steps_since_task)
            task_metrics.append(
                {
                    "task_number": completed_tasks,
                    "task_index": current_task_idx,
                    "task_type": task["task_type"],
                    "reward": episode_return,
                    "steps": steps_since_task,
                    "success": success,
                    "global_step": global_step,
                }
            )

            if log_file:
                log_file.write(
                    json.dumps(
                        {
                            "task_index": current_task_idx,
                            "task_type": task["task_type"],
                            "success": success,
                            "steps": steps_since_task,
                            "reward": episode_return,
                            "global_step": global_step,
                        }
                    )
                    + "\n"
                )
                log_file.flush()

            success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            progress.set_postfix(
                tasks=completed_tasks,
                success=f"{success_rate:.2f}",
                ret=f"{avg_return:.1f}",
                length=f"{avg_length:.0f}",
            )

            if trunc:
                env.reset()
                state = env.get_state()
                phi_region_memory = _init_region_memory(state)
            current_task_idx, task = _sample_task_instance(tasks, task_rng, state)
            aug_state = build_augmented_state(state, task)
            steps_since_task = 0
            episode_return = 0.0

        if len(replay) >= args.batch_size:
            demo_quota = args.batch_size // 2
            demo_count = min(demo_quota, len(demo_replay))
            demo_batch = demo_replay.sample(demo_count) if demo_count > 0 else []
            live_batch = replay.sample(args.batch_size - demo_count)
            combined = demo_batch + live_batch
            random.shuffle(combined)
            states = torch.tensor(
                np.stack([t.state for t in combined]), dtype=torch.float32, device=device
            )
            actions = (
                torch.tensor([t.action for t in combined], dtype=torch.int64, device=device)
                .unsqueeze(1)
            )
            rewards = (
                torch.tensor([t.reward for t in combined], dtype=torch.float32, device=device)
                .unsqueeze(1)
            )
            next_states = torch.tensor(
                np.stack([t.next_state for t in combined]), dtype=torch.float32, device=device
            )
            dones = (
                torch.tensor([t.done for t in combined], dtype=torch.float32, device=device)
                .unsqueeze(1)
            )

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target = rewards + args.gamma * (1.0 - dones) * next_q

            loss = F.smooth_l1_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

        if global_step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    progress.close()
    if log_file:
        log_file.close()
    _save_task_metrics(task_metrics, args.reward_log, args.reward_plot)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a task-conditioned DQN agent.")
    parser.add_argument("--tasks-file", type=Path, default=Path("runs") / "tasks_1000.json")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Total primitive steps (defaults to tasks_per_run * max_task_steps).",
    )
    parser.add_argument(
        "--tasks-per-run",
        type=int,
        default=250,
        help="Number of distinct tasks sampled per training run.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-final", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=20_000)
    parser.add_argument("--goal-reward", type=float, default=10.0)
    parser.add_argument(
        "--max-task-steps",
        type=int,
        default=100,
        help="Per-task interaction budget (horizon).",
    )
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=None, help="Seed for task sampling order (defaults to --seed).")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional JSONL log for per-task costs.")
    parser.add_argument(
        "--reward-log",
        type=Path,
        default=Path("runs") / "task_rewards.csv",
        help="CSV file capturing per-task returns.",
    )
    parser.add_argument(
        "--reward-plot",
        type=Path,
        default=Path("runs") / "task_rewards.png",
        help="PNG plot path for per-task reward curve.",
    )
    parser.add_argument(
        "--phi-weights",
        type=Path,
        default=Path("runs") / "phi_model.pt",
        help="Path to φ_θ weights produced by train_phi.",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=-1.0,
        help="Base reward applied each step before shaping.",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=1_000_000,
        help="Max primitive steps before forcing an environment reset.",
    )
    parser.add_argument(
        "--planner-prefill",
        type=int,
        default=0,
        help="Number of planner-solved tasks to insert into the replay buffer before training.",
    )
    parser.add_argument(
        "--planner-domain",
        type=Path,
        default=Path("pddl") / "gridworld_domain.pddl",
        help="Domain PDDL used for planner-prefill trajectories.",
    )
    parser.add_argument(
        "--planner-template",
        type=Path,
        default=Path("pddl") / "gridworld_problem.pddl",
        help="Problem template used to instantiate planner tasks.",
    )
    parser.add_argument(
        "--planner-binary",
        type=Path,
        default=Path("downward") / "fast-downward.py",
        help="Path to Fast Downward entrypoint.",
    )
    parser.add_argument(
        "--planner-search",
        type=str,
        default="astar(lmcut())",
        help="Search configuration passed to Fast Downward.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
