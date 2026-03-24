"""DQN trainer for the RGB SimpleGrid image environment."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from anticipatory_rl.envs.simple_grid_image_env import OBJECT_NAMES, SimpleGridImageEnv


# --- Matplotlib / TensorBoard visuals -----------------------------------------
_PRETTY_MPL_RC: Dict[str, Any] = {
    "figure.facecolor": "#f4f4f5",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d4d4d8",
    "axes.labelcolor": "#3f3f46",
    "axes.titlecolor": "#18181b",
    "text.color": "#3f3f46",
    "xtick.color": "#71717a",
    "ytick.color": "#71717a",
    "grid.color": "#e4e4e7",
    "grid.linestyle": "-",
    "grid.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "600",
    "lines.linewidth": 2.0,
    "lines.antialiased": True,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#e4e4e7",
    "legend.facecolor": "#ffffff",
    "legend.fontsize": 9,
}

_TB_COLORS = {
    "epsilon": "#7c3aed",
    "return": "#059669",
    "success": "#2563eb",
    "replay": "#64748b",
    "move": "#db2777",
    "clear": "#ea580c",
    "clear_prob": "#0d9488",
    "fill": "#a1a1aa",
}


def _apply_pretty_mpl_defaults() -> None:
    plt.rcParams.update(_PRETTY_MPL_RC)


def _tb_smooth(y: List[float], window: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return arr.copy()
    w = max(3, min(window | 1, n))  # odd, at least 3
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return arr.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(arr, kernel, mode="same")


def _build_tensorboard_dashboard_figure(
    steps: List[int],
    epsilon: List[float],
    recent_return: List[float],
    recent_success: List[float],
    replay_size: List[float],
    clear_task_prob: List[float],
    move_sr: List[float],
    clear_sr: List[float],
    title_suffix: str = "",
    smooth_window: int = 15,
) -> plt.Figure:
    """Multi-panel figure for SummaryWriter.add_figure (Scalars tab stays raw)."""
    with plt.rc_context(_PRETTY_MPL_RC):
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
        st = np.asarray(steps, dtype=np.float64)

        def _style_axis(ax: Any) -> None:
            ax.tick_params(axis="both", which="major", length=4, width=0.8)
            ax.set_xlabel("Environment step", fontsize=9, color="#71717a")

        # (0,0) Exploration ε
        ax = axes[0, 0]
        ye = np.asarray(epsilon, dtype=np.float64)
        ax.fill_between(st, ye, alpha=0.12, color=_TB_COLORS["epsilon"], linewidth=0)
        ax.plot(st, ye, color=_TB_COLORS["epsilon"], linewidth=2.0)
        ax.set_ylabel("ε", fontsize=10)
        ax.set_title("Exploration (ε)", loc="left", pad=8)
        if len(ye):
            ax.set_ylim(0.0, max(1.05, float(np.nanmax(ye)) * 1.08))
        else:
            ax.set_ylim(0.0, 1.0)
        _style_axis(ax)

        # (0,1) Recent return (window ≈100 tasks) + smooth
        ax = axes[0, 1]
        yr = np.asarray(recent_return, dtype=np.float64)
        ax.plot(st, yr, color=_TB_COLORS["return"], linewidth=1.0, alpha=0.35, label="raw")
        if len(yr) >= 3:
            ax.plot(st, _tb_smooth(list(yr), smooth_window), color=_TB_COLORS["return"], linewidth=2.2, label="smoothed")
        else:
            ax.plot(st, yr, color=_TB_COLORS["return"], linewidth=2.2)
        ax.set_ylabel("Mean return", fontsize=10)
        ax.set_title("Recent per-task return (~100 tasks)", loc="left", pad=8)
        ax.legend(loc="upper right", fontsize=8)
        _style_axis(ax)

        # (0,2) Success rate
        ax = axes[0, 2]
        ys = np.asarray(recent_success, dtype=np.float64)
        ax.fill_between(st, ys, alpha=0.15, color=_TB_COLORS["success"], linewidth=0)
        if len(ys) >= 3:
            ax.plot(st, _tb_smooth(list(ys), smooth_window), color=_TB_COLORS["success"], linewidth=2.2)
        else:
            ax.plot(st, ys, color=_TB_COLORS["success"], linewidth=2.2)
        ax.axhline(0.5, color="#d4d4d8", linestyle="--", linewidth=1, zorder=0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("Success rate", fontsize=10)
        ax.set_title("Rolling task success (~100)", loc="left", pad=8)
        _style_axis(ax)

        # (1,0) Replay buffer
        ax = axes[1, 0]
        yrep = np.asarray(replay_size, dtype=np.float64)
        ax.fill_between(st, yrep, alpha=0.2, color=_TB_COLORS["replay"], linewidth=0)
        ax.plot(st, yrep, color=_TB_COLORS["replay"], linewidth=2.0)
        ax.set_ylabel("Transitions", fontsize=10)
        ax.set_title("Replay buffer size", loc="left", pad=8)
        _style_axis(ax)

        # (1,1) Move vs clear success
        ax = axes[1, 1]
        ym = np.asarray(move_sr, dtype=np.float64)
        yc = np.asarray(clear_sr, dtype=np.float64)
        if np.isfinite(ym).any():
            ax.plot(st, ym, color=_TB_COLORS["move"], linewidth=2.0, label="Move tasks", alpha=0.95)
        if np.isfinite(yc).any():
            ax.plot(st, yc, color=_TB_COLORS["clear"], linewidth=2.0, label="Clear tasks", alpha=0.95)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("Success rate", fontsize=10)
        ax.set_title("Success by task type (~100)", loc="left", pad=8)
        ax.legend(loc="lower right", fontsize=8)
        _style_axis(ax)

        # (1,2) Clear-task sampling probability
        ax = axes[1, 2]
        yp = np.asarray(clear_task_prob, dtype=np.float64)
        ax.fill_between(st, yp, alpha=0.15, color=_TB_COLORS["clear_prob"], linewidth=0)
        ax.plot(st, yp, color=_TB_COLORS["clear_prob"], linewidth=2.0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("P(clear)", fontsize=10)
        ax.set_title("Clear-task probability (env)", loc="left", pad=8)
        _style_axis(ax)

        fig.suptitle(f"SimpleGrid image DQN — training dashboard{title_suffix}", fontsize=14, y=1.01)
        fig.patch.set_facecolor(_PRETTY_MPL_RC["figure.facecolor"])
    return fig


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
    can_pick_next: bool = False
    can_place_next: bool = False
    task_boundary: bool = False


def _encode_obs_storage(obs: np.ndarray) -> np.ndarray:
    clipped = np.clip(obs, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8, copy=False)


def _decode_obs_batch(obs_batch: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.asarray(obs_batch, dtype=np.uint8)
    return torch.tensor(arr, dtype=torch.float32, device=device).div_(255.0)


@dataclass(frozen=True)
class RuntimeResources:
    allocated_cpus: int
    allocated_gpus: int
    visible_cuda_devices: Tuple[str, ...]
    torch_threads: int
    torch_interop_threads: int
    num_envs: int


def _parse_int_env(var_name: str) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    for token in (stripped, stripped.split("(")[0], stripped.split(",")[0]):
        try:
            parsed = int(token)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


def _parse_visible_cuda_devices() -> Tuple[str, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return ()
    stripped = value.strip()
    if not stripped or stripped in {"-1", "NoDevFiles"}:
        return ()
    return tuple(device.strip() for device in stripped.split(",") if device.strip())


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_runtime_resources(args: argparse.Namespace, device: torch.device) -> RuntimeResources:
    allocated_cpus = (
        _parse_int_env("SLURM_CPUS_PER_TASK")
        or _parse_int_env("SLURM_CPUS_ON_NODE")
        or os.cpu_count()
        or 1
    )
    visible_cuda_devices = _parse_visible_cuda_devices()
    allocated_gpus = (
        len(visible_cuda_devices)
        or _parse_int_env("SLURM_GPUS_PER_NODE")
        or _parse_int_env("SLURM_GPUS")
        or (1 if device.type == "cuda" else 0)
    )

    if args.num_envs is not None and args.num_envs > 0:
        num_envs = args.num_envs
    else:
        num_envs = max(1, allocated_cpus)

    if args.torch_threads is not None and args.torch_threads > 0:
        torch_threads = args.torch_threads
    else:
        if device.type == "cuda":
            torch_threads = max(1, allocated_cpus // max(1, num_envs))
        else:
            torch_threads = max(1, allocated_cpus // max(1, min(num_envs, allocated_cpus)))

    if args.torch_interop_threads is not None and args.torch_interop_threads > 0:
        torch_interop_threads = args.torch_interop_threads
    else:
        torch_interop_threads = 1 if device.type == "cuda" else min(4, max(1, torch_threads))

    torch_threads = min(torch_threads, allocated_cpus)
    torch_interop_threads = min(torch_interop_threads, allocated_cpus)
    return RuntimeResources(
        allocated_cpus=allocated_cpus,
        allocated_gpus=max(0, allocated_gpus),
        visible_cuda_devices=visible_cuda_devices,
        torch_threads=max(1, torch_threads),
        torch_interop_threads=max(1, torch_interop_threads),
        num_envs=max(1, num_envs),
    )


def _apply_runtime_resources(resources: RuntimeResources) -> None:
    torch.set_num_threads(resources.torch_threads)
    torch.set_num_interop_threads(resources.torch_interop_threads)


def _log_runtime_resources(resources: RuntimeResources, device: torch.device) -> None:
    visible_cuda = ",".join(resources.visible_cuda_devices) if resources.visible_cuda_devices else "auto"
    print(
        "Runtime resources: "
        f"device={device}, "
        f"allocated_cpus={resources.allocated_cpus}, "
        f"allocated_gpus={resources.allocated_gpus}, "
        f"cuda_visible_devices={visible_cuda}, "
        f"num_envs={resources.num_envs}, "
        f"torch_threads={resources.torch_threads}, "
        f"torch_interop_threads={resources.torch_interop_threads}"
    )


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


def _resolve_run_label(args: argparse.Namespace) -> str:
    """Label run directories as myopic vs anticipatory (overridable via --run-label)."""
    if args.run_label is not None:
        return args.run_label
    return "myopic" if args.tasks_per_reset <= 1 else "anticipatory"


def train(args: argparse.Namespace, device: torch.device) -> None:
    def make_env():
        return SimpleGridImageEnv(
            grid_size=args.grid_size,
            success_reward=args.success_reward,
            num_objects=args.num_objects,
            distance_reward=True,
            distance_reward_scale=args.distance_reward_scale,
            clear_receptacle_shaping_scale=args.clear_receptacle_shaping_scale,
            clear_task_prob=args.clear_task_prob,
            config_path=args.config_path,
        )

    env = VectorEnv(make_env, max(1, args.num_envs))
    run_label = _resolve_run_label(args)
    grid_dir = Path("runs") / f"{env.grid_size}_{run_label}_image_dqn_tpr{args.tasks_per_reset}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Run artifacts → {grid_dir.resolve()} ({run_label})")
    args.output = grid_dir / args.output.name
    tb_dir = args.tb_log_dir or (grid_dir / "tb" / args.output.stem)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    _apply_pretty_mpl_defaults()
    env_reset_tasks = (
        args.env_reset_tasks if args.env_reset_tasks is not None else args.tasks_per_reset
    )

    replay = PrioritizedReplayBuffer(
        args.replay_size,
        alpha=args.per_alpha,
        eps=args.per_eps,
    )
    goal_buffer: Deque[Transition] = deque(
        maxlen=args.goal_buffer_size if args.goal_buffer_size > 0 else None
    )

    state, infos = env.reset(seed=args.seed)
    infos = list(infos)
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
    pick_action = SimpleGridImageEnv.PICK
    place_action = SimpleGridImageEnv.PLACE
    fallback_actions = np.array(
        [
            SimpleGridImageEnv.MOVE_UP,
            SimpleGridImageEnv.MOVE_DOWN,
            SimpleGridImageEnv.MOVE_LEFT,
            SimpleGridImageEnv.MOVE_RIGHT,
        ],
        dtype=np.int64,
    )
    episode_transitions: List[List[Transition]] = [[] for _ in range(num_envs)]
    tasks_since_reset = np.zeros(num_envs, dtype=np.int64)
    env_tasks_since_reset = np.zeros(num_envs, dtype=np.int64)
    steps_since_reset = np.zeros(num_envs, dtype=np.int64)

    # Anticipation tracking (per-position auto-success rates)
    _episode_len = args.tasks_per_reset if args.tasks_per_reset > 0 else 0
    current_task_auto_satisfied = np.array(
        [info.get("next_auto_satisfied", False) for info in infos], dtype=bool
    )
    auto_counts_by_pos = np.zeros(max(1, _episode_len), dtype=np.int64)
    total_counts_by_pos = np.zeros(max(1, _episode_len), dtype=np.int64)
    recent_auto_by_pos: Dict[int, Deque[int]] = {
        p: deque(maxlen=200) for p in range(_episode_len)
    }
    antic_delta_history: List[float] = []
    antic_history_x: List[int] = []
    _ANTIC_LOG_EVERY = 500

    rolling_success_history: List[float] = []
    rolling_return_history: List[float] = []
    rolling_history_x: List[int] = []
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[int] = deque(maxlen=100)
    epsilon_restart_until = -1
    last_restart_step = -10**9
    goal_fraction = max(0.0, min(args.goal_buffer_fraction, 1.0))
    log_interval = max(1, args.log_interval)
    if args.tb_dashboard_interval is not None and args.tb_dashboard_interval > 0:
        tb_dash_interval = max(1, args.tb_dashboard_interval)
    elif args.tb_dashboard_interval is not None and args.tb_dashboard_interval <= 0:
        tb_dash_interval = log_interval
    else:
        tb_dash_interval = max(2500, log_interval * 4)

    tb_dash_steps: List[int] = []
    tb_dash_eps: List[float] = []
    tb_dash_return: List[float] = []
    tb_dash_success: List[float] = []
    tb_dash_replay: List[float] = []
    tb_dash_clear_prob: List[float] = []
    tb_dash_move_sr: List[float] = []
    tb_dash_clear_sr: List[float] = []

    # --- Diagnostic accumulators ---
    action_counts = np.zeros(6, dtype=np.int64)
    invalid_pick_attempts = 0
    invalid_place_attempts = 0

    move_tasks_total = 0
    move_tasks_success = 0
    clear_tasks_total = 0
    clear_tasks_success = 0
    rolling_move_success: Deque[int] = deque(maxlen=100)
    rolling_clear_success: Deque[int] = deque(maxlen=100)
    rolling_move_success_history: List[float] = []
    rolling_clear_success_history: List[float] = []
    rolling_task_type_x: List[int] = []

    fail_horizon = 0
    fail_episode_step_limit = 0

    correct_picks = 0
    wrong_picks = 0
    place_on_target = 0
    place_off_target = 0

    success_reward_total = 0.0
    non_success_reward_total = 0.0
    success_step_count = 0
    non_success_step_count = 0

    steps_carrying_nothing = 0
    steps_carrying_target = 0
    steps_carrying_wrong = 0

    initial_obj_dists: List[float] = []
    task_start_obj_dist: List[Optional[float]] = [None] * num_envs

    post_task_dist_to_next_obj: List[int] = []

    invalid_argmax_frac_history: List[float] = []
    q_per_action_history: List[List[float]] = []
    boundary_td_mean_history: List[float] = []
    normal_td_mean_history: List[float] = []
    q_diag_update_count = 0

    ACTION_NAMES = ["up", "down", "left", "right", "pick", "place"]

    def _obj_dist_from_info(info_dict: Dict[str, Any]) -> Optional[float]:
        target_obj = info_dict.get("target_object")
        agent_pos = info_dict.get("agent")
        objects = info_dict.get("objects", {})
        if target_obj and agent_pos and target_obj in objects:
            op = objects[target_obj]
            return float(abs(agent_pos[0] - op[0]) + abs(agent_pos[1] - op[1]))
        return None

    for idx in range(num_envs):
        if idx < len(infos):
            task_start_obj_dist[idx] = _obj_dist_from_info(infos[idx])

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
        invalid_pick_mask = np.zeros(num_envs, dtype=bool)
        invalid_place_mask = np.zeros(num_envs, dtype=bool)
        for idx in range(num_envs):
            if idx < len(infos):
                can_pick = bool(infos[idx].get("can_pick", False))
                can_place = bool(infos[idx].get("can_place", False))
            else:
                can_pick = False
                can_place = False
            invalid_pick_mask[idx] = not can_pick
            invalid_place_mask[idx] = not can_place

        with torch.no_grad():
            inp = torch.tensor(state, dtype=torch.float32, device=device)
            q_vals = q_net(inp)
            masked_q_vals = q_vals.clone()
            if invalid_pick_mask.any():
                mask_tensor = torch.from_numpy(invalid_pick_mask).to(device=device)
                masked_q_vals[mask_tensor, pick_action] = float("-inf")
            if invalid_place_mask.any():
                mask_tensor = torch.from_numpy(invalid_place_mask).to(device=device)
                masked_q_vals[mask_tensor, place_action] = float("-inf")
            greedy_actions = torch.argmax(masked_q_vals, dim=1).cpu().numpy()
            greedy_value = float(torch.max(masked_q_vals).item())
        greedy_value_history.append(greedy_value)

        random_mask = np.random.rand(num_envs) < eps
        random_actions = np.asarray(
            [env.action_space.sample() for _ in range(num_envs)]
        )
        actions = np.where(random_mask, random_actions, greedy_actions).astype(np.int64)
        invalid_random_pick = invalid_pick_mask & (actions == pick_action)
        if np.any(invalid_random_pick):
            actions[invalid_random_pick] = np.random.choice(
                fallback_actions, size=int(invalid_random_pick.sum())
            )
        invalid_random = invalid_place_mask & (actions == place_action)
        if np.any(invalid_random):
            actions[invalid_random] = np.random.choice(
                fallback_actions, size=int(invalid_random.sum())
            )

        for idx in range(num_envs):
            action_counts[actions[idx]] += 1
            info_i = infos[idx] if idx < len(infos) else {}
            carrying = info_i.get("carrying")
            target_obj = info_i.get("target_object")
            if carrying is None:
                steps_carrying_nothing += 1
            elif carrying == target_obj:
                steps_carrying_target += 1
            else:
                steps_carrying_wrong += 1

        next_state, reward, success, horizon, info = env.step(actions)
        next_infos = list(info)
        task_done = success | horizon

        for idx in range(num_envs):
            act = int(actions[idx])
            pre_info = infos[idx] if idx < len(infos) else {}
            post_info = next_infos[idx] if idx < len(next_infos) else {}
            pre_carrying = pre_info.get("carrying")
            post_carrying = post_info.get("carrying")
            target_obj = pre_info.get("target_object")

            if act == pick_action:
                if pre_carrying is None and post_carrying is not None:
                    if post_carrying == target_obj:
                        correct_picks += 1
                    else:
                        wrong_picks += 1
                else:
                    invalid_pick_attempts += 1
            elif act == place_action:
                if pre_carrying is not None and post_carrying is None:
                    if success[idx]:
                        place_on_target += 1
                    else:
                        place_off_target += 1
                else:
                    invalid_place_attempts += 1

            if success[idx]:
                success_reward_total += float(reward[idx])
                success_step_count += 1
            else:
                non_success_reward_total += float(reward[idx])
                non_success_step_count += 1

        episode_done_flags = np.zeros(num_envs, dtype=bool)
        env_reset_flags = np.zeros(num_envs, dtype=bool)
        for idx in range(num_envs):
            steps_since_reset[idx] += 1
            finished_task = bool(task_done[idx])
            if finished_task:
                tasks_since_reset[idx] += 1
                env_tasks_since_reset[idx] += 1
                if args.tasks_per_reset > 0 and tasks_since_reset[idx] >= args.tasks_per_reset:
                    episode_done_flags[idx] = True
                    tasks_since_reset[idx] = 0
                if (
                    env_reset_tasks is not None
                    and env_reset_tasks > 0
                    and env_tasks_since_reset[idx] >= env_reset_tasks
                ):
                    env_reset_flags[idx] = True
                    env_tasks_since_reset[idx] = 0
                    episode_done_flags[idx] = True
            if args.episode_step_limit > 0 and steps_since_reset[idx] >= args.episode_step_limit:
                episode_done_flags[idx] = True

        for idx in range(num_envs):
            post_info = next_infos[idx] if idx < len(next_infos) else {}
            transition = Transition(
                _encode_obs_storage(state[idx]),
                int(actions[idx]),
                float(reward[idx]),
                _encode_obs_storage(next_state[idx]),
                bool(episode_done_flags[idx]),
                can_pick_next=bool(post_info.get("can_pick", False)),
                can_place_next=bool(post_info.get("can_place", False)),
                task_boundary=bool(task_done[idx]),
            )
            replay.push(transition)
            episode_transitions[idx].append(transition)
            step_rewards.append(float(reward[idx]))

        state = next_state
        task_return += reward
        task_steps += 1
        global_step += num_envs
        progress.update(num_envs)
        if global_step % log_interval == 0:
            writer.add_scalar("train/epsilon", eps, global_step)
            clear_p = float(env.envs[0].clear_task_prob)
            writer.add_scalar("env/clear_task_prob", clear_p, global_step)
            writer.add_scalar("train/replay_size", len(replay), global_step)
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            writer.add_scalar("train/recent_return", avg_return, global_step)
            window_success = float(np.mean(recent_successes)) if recent_successes else 0.0
            writer.add_scalar("train/recent_success", window_success, global_step)
            if rolling_move_success:
                writer.add_scalar(
                    "tasks/move_success_rate",
                    float(np.mean(rolling_move_success)),
                    global_step,
                )
            if rolling_clear_success:
                writer.add_scalar(
                    "tasks/clear_success_rate",
                    float(np.mean(rolling_clear_success)),
                    global_step,
                )

            tb_dash_steps.append(global_step)
            tb_dash_eps.append(float(eps))
            tb_dash_return.append(avg_return)
            tb_dash_success.append(window_success)
            tb_dash_replay.append(float(len(replay)))
            tb_dash_clear_prob.append(clear_p)
            tb_dash_move_sr.append(
                float(np.mean(rolling_move_success)) if rolling_move_success else float("nan")
            )
            tb_dash_clear_sr.append(
                float(np.mean(rolling_clear_success)) if rolling_clear_success else float("nan")
            )

            if global_step % tb_dash_interval == 0 and tb_dash_steps:
                dash = _build_tensorboard_dashboard_figure(
                    tb_dash_steps,
                    tb_dash_eps,
                    tb_dash_return,
                    tb_dash_success,
                    tb_dash_replay,
                    tb_dash_clear_prob,
                    tb_dash_move_sr,
                    tb_dash_clear_sr,
                    title_suffix=f" · {run_label}",
                )
                writer.add_figure("visuals/training_dashboard", dash, global_step)
                plt.close(dash)

        for idx in range(num_envs):
            if task_done[idx]:
                episode_return = float(task_return[idx])
                returns.append(episode_return)
                task_lengths.append(int(task_steps[idx]))
                episode_returns.append(episode_return)
                task_length_history.append(int(task_steps[idx]))
                total_tasks += 1

                pre_info = infos[idx] if idx < len(infos) else {}
                post_info = next_infos[idx] if idx < len(next_infos) else {}
                task_type = pre_info.get("task_type", "move")
                was_success = bool(success[idx])

                if task_type == "move":
                    move_tasks_total += 1
                    if was_success:
                        move_tasks_success += 1
                    rolling_move_success.append(1 if was_success else 0)
                else:
                    clear_tasks_total += 1
                    if was_success:
                        clear_tasks_success += 1
                    rolling_clear_success.append(1 if was_success else 0)

                rolling_task_type_x.append(total_tasks)
                rolling_move_success_history.append(
                    float(np.mean(rolling_move_success)) if rolling_move_success else 0.0
                )
                rolling_clear_success_history.append(
                    float(np.mean(rolling_clear_success)) if rolling_clear_success else 0.0
                )

                if not was_success:
                    if bool(horizon[idx]):
                        fail_horizon += 1
                    elif bool(episode_done_flags[idx]):
                        fail_episode_step_limit += 1

                if task_start_obj_dist[idx] is not None:
                    initial_obj_dists.append(task_start_obj_dist[idx])

                new_dist = _obj_dist_from_info(post_info)
                if new_dist is not None and post_info.get("task_type") == "move":
                    post_task_dist_to_next_obj.append(int(new_dist))
                task_start_obj_dist[idx] = new_dist

                if was_success:
                    tasks_completed += 1
                recent_returns.append(episode_return)
                recent_successes.append(1 if was_success else 0)
                window_success = float(np.mean(recent_successes)) if recent_successes else 0.0
                rolling_success_history.append(window_success)
                rolling_return_history.append(
                    float(np.mean(recent_returns)) if recent_returns else 0.0
                )
                rolling_history_x.append(total_tasks)

                # Anticipation: record this task's auto-satisfied status by position
                if _episode_len > 0:
                    ep_pos = int(tasks_since_reset[idx]) - 1
                    if 0 <= ep_pos < _episode_len:
                        auto_sat = bool(current_task_auto_satisfied[idx])
                        auto_counts_by_pos[ep_pos] += int(auto_sat)
                        total_counts_by_pos[ep_pos] += 1
                        recent_auto_by_pos[ep_pos].append(int(auto_sat))
                    # The env already resampled the next task; capture its flag
                    current_task_auto_satisfied[idx] = bool(
                        next_infos[idx].get("next_auto_satisfied", False)
                    )
                    # Periodically log the rolling anticipation delta
                    if (
                        total_tasks % _ANTIC_LOG_EVERY == 0
                        and recent_auto_by_pos.get(0)
                    ):
                        baseline = float(np.mean(list(recent_auto_by_pos[0])))
                        later_means = [
                            float(np.mean(list(recent_auto_by_pos[p])))
                            for p in range(1, _episode_len)
                            if recent_auto_by_pos[p]
                        ]
                        if later_means:
                            antic_delta_history.append(
                                float(np.mean(later_means)) - baseline
                            )
                            antic_history_x.append(total_tasks)

                if was_success and args.goal_buffer_size > 0 and episode_transitions[idx]:
                    for trans in episode_transitions[idx]:
                        goal_buffer.append(trans)
                episode_transitions[idx].clear()
                task_return[idx] = 0.0
                task_steps[idx] = 0

        for idx in range(num_envs):
            if env_reset_flags[idx]:
                new_obs, new_info = env.reset_env(idx)
                state[idx] = new_obs
                next_infos[idx] = new_info
                task_start_obj_dist[idx] = _obj_dist_from_info(new_info)
                env_tasks_since_reset[idx] = 0
            if episode_done_flags[idx]:
                tasks_since_reset[idx] = 0
                steps_since_reset[idx] = 0
                task_return[idx] = 0.0
                task_steps[idx] = 0
                episode_transitions[idx].clear()
                if not env_reset_flags[idx]:
                    task_start_obj_dist[idx] = _obj_dist_from_info(
                        next_infos[idx] if idx < len(next_infos) else {}
                    )
                if _episode_len > 0:
                    current_task_auto_satisfied[idx] = bool(
                        next_infos[idx].get("next_auto_satisfied", False)
                    )

        infos = next_infos

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

            states = _decode_obs_batch([t.state for t in batch], device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            next_states = _decode_obs_batch([t.next_state for t in batch], device)
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
            weights_t = torch.tensor(weights_arr, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_all = q_net(next_states)
                batch_can_pick = torch.tensor(
                    [t.can_pick_next for t in batch], dtype=torch.bool, device=device
                )
                batch_can_place = torch.tensor(
                    [t.can_place_next for t in batch], dtype=torch.bool, device=device
                )
                next_q_all[~batch_can_pick, pick_action] = float("-inf")
                next_q_all[~batch_can_place, place_action] = float("-inf")
                next_actions = torch.argmax(next_q_all, dim=1, keepdim=True)
                next_q = target_net(next_states).gather(1, next_actions)
                target = rewards + args.gamma * (1.0 - dones) * next_q

            td_errors = target - q_values
            loss = (weights_t * nn.functional.smooth_l1_loss(q_values, target.detach(), reduction="none")).mean()
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

            can_pick_flags = torch.tensor(
                [t.can_pick_next for t in batch], dtype=torch.bool, device=device
            )
            can_place_flags = torch.tensor(
                [t.can_place_next for t in batch], dtype=torch.bool, device=device
            )
            next_greedy = torch.argmax(q_net(next_states), dim=1)
            pick_invalid = (~can_pick_flags) & (next_greedy == pick_action)
            place_invalid = (~can_place_flags) & (next_greedy == place_action)
            invalid_frac = float((pick_invalid | place_invalid).float().mean().item())
            invalid_argmax_frac_history.append(invalid_frac)

            boundary_flags = torch.tensor(
                [t.task_boundary for t in batch], dtype=torch.bool, device=device
            )
            td_abs = td_errors.abs().squeeze()
            if boundary_flags.any():
                boundary_td_mean_history.append(
                    float(td_abs[boundary_flags].mean().item())
                )
            if (~boundary_flags).any():
                normal_td_mean_history.append(
                    float(td_abs[~boundary_flags].mean().item())
                )

            q_diag_update_count += 1
            if q_diag_update_count % 100 == 0:
                with torch.no_grad():
                    all_q = q_net(states)
                    mean_q = all_q.mean(dim=0).cpu().tolist()
                q_per_action_history.append(mean_q)

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
    _plot_action_diagnostics(
        action_counts,
        invalid_pick_attempts,
        invalid_place_attempts,
        correct_picks,
        wrong_picks,
        place_on_target,
        place_off_target,
        ACTION_NAMES,
        args.output,
    )
    _plot_task_type_stats(
        rolling_move_success_history,
        rolling_clear_success_history,
        rolling_task_type_x,
        fail_horizon,
        fail_episode_step_limit,
        tasks_completed,
        total_tasks,
        args.output,
    )
    _plot_reward_decomposition(
        success_reward_total,
        non_success_reward_total,
        success_step_count,
        non_success_step_count,
        args.output,
    )
    _plot_carry_state(
        steps_carrying_nothing,
        steps_carrying_target,
        steps_carrying_wrong,
        args.output,
    )
    _plot_q_diagnostics(
        q_per_action_history,
        invalid_argmax_frac_history,
        boundary_td_mean_history,
        normal_td_mean_history,
        ACTION_NAMES,
        args.output,
    )
    _plot_distance_progress(initial_obj_dists, args.output)
    _plot_anticipation(post_task_dist_to_next_obj, args.output)
    _plot_auto_rate_by_pos(
        auto_counts_by_pos,
        total_counts_by_pos,
        antic_delta_history,
        antic_history_x,
        _episode_len,
        args.output,
    )

    total_act = int(action_counts.sum())
    act_pcts = {
        name: float(action_counts[i]) / max(1, total_act) * 100
        for i, name in enumerate(ACTION_NAMES)
    }
    total_carry = steps_carrying_nothing + steps_carrying_target + steps_carrying_wrong
    diag: Dict[str, Any] = {
        "action_distribution_pct": act_pcts,
        "invalid_pick_attempts": invalid_pick_attempts,
        "invalid_place_attempts": invalid_place_attempts,
        "correct_picks": correct_picks,
        "wrong_picks": wrong_picks,
        "place_on_target": place_on_target,
        "place_off_target": place_off_target,
        "move_tasks_total": move_tasks_total,
        "move_tasks_success": move_tasks_success,
        "move_success_rate": move_tasks_success / max(1, move_tasks_total),
        "clear_tasks_total": clear_tasks_total,
        "clear_tasks_success": clear_tasks_success,
        "clear_success_rate": clear_tasks_success / max(1, clear_tasks_total),
        "fail_horizon": fail_horizon,
        "fail_episode_step_limit": fail_episode_step_limit,
        "success_reward_total": success_reward_total,
        "non_success_reward_total": non_success_reward_total,
        "success_step_count": success_step_count,
        "non_success_step_count": non_success_step_count,
        "carry_nothing_frac": steps_carrying_nothing / max(1, total_carry),
        "carry_target_frac": steps_carrying_target / max(1, total_carry),
        "carry_wrong_frac": steps_carrying_wrong / max(1, total_carry),
        "invalid_argmax_frac_final": (
            float(np.mean(invalid_argmax_frac_history[-100:]))
            if invalid_argmax_frac_history else None
        ),
        "q_per_action_final": (
            q_per_action_history[-1] if q_per_action_history else None
        ),
        "total_tasks": total_tasks,
        "tasks_completed": tasks_completed,
        "overall_success_rate": tasks_completed / max(1, total_tasks),
        "auto_rate_by_pos": {
            str(p): float(auto_counts_by_pos[p]) / max(1.0, float(total_counts_by_pos[p]))
            for p in range(_episode_len)
        } if _episode_len > 0 else None,
        "baseline_auto_rate": (
            float(auto_counts_by_pos[0]) / max(1.0, float(total_counts_by_pos[0]))
            if _episode_len > 0 else None
        ),
        "anticipation_delta_final": float(antic_delta_history[-1]) if antic_delta_history else None,
    }
    _write_diagnostics_json(diag, args.output)
    writer.close()


def _plot_auto_rate_by_pos(
    auto_counts: np.ndarray,
    total_counts: np.ndarray,
    delta_history: List[float],
    delta_x: List[int],
    episode_len: int,
    output: Path,
) -> None:
    """Plot per-position auto-success rates and rolling anticipation delta."""
    if episode_len == 0:
        return
    stem = output.stem
    out_dir = output.parent
    rates = [
        int(auto_counts[p]) / max(1, int(total_counts[p]))
        for p in range(episode_len)
    ]
    baseline = rates[0] if rates else 0.0
    positions = list(range(episode_len))

    n_panels = 2 if delta_history else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax0 = axes[0]
    ax0.bar(positions, rates, color="steelblue", alpha=0.8, label="Auto-satisfied rate")
    ax0.axhline(baseline, color="crimson", linestyle="--",
                label=f"Baseline (pos 0): {baseline:.3f}")
    if len(rates) > 1:
        later_mean = float(np.mean(rates[1:]))
        delta = later_mean - baseline
        ax0.axhline(later_mean, color="green", linestyle="--",
                    label=f"Later mean: {later_mean:.3f}")
        ax0.set_title(f"Auto-success by Episode Position\n(Δ over baseline: {delta:+.3f})")
    else:
        ax0.set_title("Auto-success by Episode Position")
    ax0.set_xlabel("Episode Position (0 = first task after reset)")
    ax0.set_ylabel("Auto-satisfied Rate")
    ax0.legend(fontsize=8)

    if delta_history:
        ax1 = axes[1]
        ax1.plot(delta_x, delta_history, color="green", linewidth=1.5)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_title("Rolling Anticipation Δ over Training")
        ax1.set_xlabel("Total Tasks")
        ax1.set_ylabel("Δ Auto-success Rate (later – baseline)")

    fig.tight_layout()
    save_path = out_dir / f"{stem}_auto_rate_by_pos.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[diag] Saved auto_rate_by_pos plot → {save_path}")


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
    c_rwd, c_ret, c_q, c_td, c_tgt = (
        _TB_COLORS["success"],
        _TB_COLORS["return"],
        _TB_COLORS["epsilon"],
        "#e11d48",
        _TB_COLORS["clear"],
    )
    with plt.rc_context(_PRETTY_MPL_RC):
        fig, axes = plt.subplots(5, 1, figsize=(10.5, 12.5), constrained_layout=True)
        reward_window = 100
        step_ma = _moving_average(step_rewards, reward_window)
        x_r = np.arange(len(step_rewards))
        axes[0].fill_between(x_r, step_rewards, alpha=0.12, color=c_rwd, linewidth=0)
        axes[0].plot(x_r, step_rewards, linewidth=0.35, alpha=0.25, color=c_rwd)
        if step_ma is not None:
            ma_x = np.arange(reward_window - 1, reward_window - 1 + len(step_ma))
            axes[0].plot(ma_x, step_ma, linewidth=2.0, color=c_rwd)
        axes[0].set_title(f"Per-step reward (MA window={reward_window})")
        axes[0].set_ylabel("Reward")
        return_window = 100
        return_ma = _moving_average(episode_returns, return_window)
        x_e = np.arange(len(episode_returns))
        axes[1].fill_between(x_e, episode_returns, alpha=0.12, color=c_ret, linewidth=0)
        axes[1].plot(x_e, episode_returns, linewidth=0.5, alpha=0.35, color=c_ret)
        if return_ma is not None:
            ma_x = np.arange(return_window - 1, return_window - 1 + len(return_ma))
            axes[1].plot(ma_x, return_ma, linewidth=2.0, color=c_ret)
        axes[1].set_title(f"Per-task return (MA window={return_window})")
        axes[1].set_xlabel("Task index")
        axes[1].set_ylabel("Return")
        if greedy_values:
            axes[2].plot(greedy_values, linewidth=1.0, color=c_q, alpha=0.85)
            axes[2].set_title("Greedy action value (masked max Q)")
            axes[2].set_xlabel("Environment step")
            axes[2].set_ylabel("Q-value")
        else:
            axes[2].set_visible(False)
        if td_errors:
            axes[3].plot(td_errors, linewidth=1.0, color=c_td, alpha=0.9)
            axes[3].set_title("Mean |TD error|")
            axes[3].set_xlabel("Update step")
            axes[3].set_ylabel("|δ|")
        else:
            axes[3].set_visible(False)
        if target_values:
            axes[4].plot(target_values, linewidth=1.0, color=c_tgt, alpha=0.9)
            axes[4].set_title("Target value magnitude")
            axes[4].set_xlabel("Update step")
            axes[4].set_ylabel("|Target|")
        else:
            axes[4].set_visible(False)
        fig.patch.set_facecolor(_PRETTY_MPL_RC["figure.facecolor"])
        fig.savefig(
            plot_path,
            dpi=140,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
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
    with plt.rc_context(_PRETTY_MPL_RC):
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
        axes[0].fill_between(task_indices, success_history, alpha=0.15, color=_TB_COLORS["success"], linewidth=0)
        axes[0].plot(task_indices, success_history, linewidth=2.0, color=_TB_COLORS["success"])
        axes[0].set_ylabel("Success rate")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_title("Rolling success rate (window ≈ 100)")
        axes[1].fill_between(task_indices, return_history, alpha=0.12, color=_TB_COLORS["return"], linewidth=0)
        axes[1].plot(task_indices, return_history, linewidth=2.0, color=_TB_COLORS["return"])
        axes[1].set_ylabel("Return")
        axes[1].set_xlabel("Completed task")
        axes[1].set_title("Rolling return (window ≈ 100)")
        fig.patch.set_facecolor(_PRETTY_MPL_RC["figure.facecolor"])
        fig.savefig(
            plot_path,
            dpi=140,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
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


def _plot_action_diagnostics(
    action_counts: np.ndarray,
    invalid_pick: int,
    invalid_place: int,
    correct_picks: int,
    wrong_picks: int,
    place_on_target: int,
    place_off_target: int,
    action_names: List[str],
    weight_path: Path,
) -> None:
    plot_path = weight_path.with_name(weight_path.stem + "_action_diag.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    total = action_counts.sum()
    pcts = action_counts / max(1, total) * 100
    axes[0].bar(action_names, pcts, color="#1f77b4")
    axes[0].set_ylabel("% of actions")
    axes[0].set_title("Action distribution")
    for i, v in enumerate(pcts):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    inv_labels = ["invalid pick", "invalid place"]
    inv_vals = [invalid_pick, invalid_place]
    axes[1].bar(inv_labels, inv_vals, color=["#d62728", "#ff7f0e"])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Invalid action attempts")

    pick_labels = ["correct pick", "wrong pick", "place on target", "place off target"]
    pick_vals = [correct_picks, wrong_picks, place_on_target, place_off_target]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
    axes[2].bar(pick_labels, pick_vals, color=colors)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pick / place quality")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved action diagnostics to {plot_path}")


def _plot_task_type_stats(
    rolling_move_history: List[float],
    rolling_clear_history: List[float],
    rolling_x: List[int],
    fail_horizon: int,
    fail_episode_step_limit: int,
    tasks_completed: int,
    total_tasks: int,
    weight_path: Path,
) -> None:
    plot_path = weight_path.with_name(weight_path.stem + "_task_type.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if rolling_x:
        if rolling_move_history:
            axes[0].plot(rolling_x, rolling_move_history, linewidth=0.8, label="move")
        if rolling_clear_history:
            axes[0].plot(rolling_x, rolling_clear_history, linewidth=0.8, label="clear")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel("Rolling success rate")
        axes[0].set_xlabel("Completed task")
        axes[0].set_title("Success rate by task type (window=100)")
        axes[0].legend()
    else:
        axes[0].set_visible(False)

    fail_other = max(0, total_tasks - tasks_completed - fail_horizon - fail_episode_step_limit)
    labels, sizes, colors = [], [], []
    for lbl, val, col in [
        ("success", tasks_completed, "#2ca02c"),
        ("task horizon", fail_horizon, "#ff7f0e"),
        ("episode step limit", fail_episode_step_limit, "#d62728"),
        ("other failure", fail_other, "#9467bd"),
    ]:
        if val > 0:
            labels.append(lbl)
            sizes.append(val)
            colors.append(col)
    if sizes:
        axes[1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Task outcome breakdown")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved task type stats to {plot_path}")


def _plot_reward_decomposition(
    success_reward_total: float,
    non_success_reward_total: float,
    success_step_count: int,
    non_success_step_count: int,
    weight_path: Path,
) -> None:
    plot_path = weight_path.with_name(weight_path.stem + "_reward_decomp.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        f"Success steps\n(n={success_step_count})",
        f"Non-success steps\n(n={non_success_step_count})",
    ]
    vals = [success_reward_total, non_success_reward_total]
    colors = ["#2ca02c", "#d62728"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Reward decomposition")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved reward decomposition to {plot_path}")


def _plot_carry_state(
    nothing: int,
    target: int,
    wrong: int,
    weight_path: Path,
) -> None:
    plot_path = weight_path.with_name(weight_path.stem + "_carry_state.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    labels, sizes, colors = [], [], []
    for lbl, val, col in [
        ("nothing", nothing, "#1f77b4"),
        ("target obj", target, "#2ca02c"),
        ("wrong obj", wrong, "#d62728"),
    ]:
        if val > 0:
            labels.append(lbl)
            sizes.append(val)
            colors.append(col)
    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Carry-state occupancy")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved carry state plot to {plot_path}")


def _plot_q_diagnostics(
    q_per_action_history: List[List[float]],
    invalid_argmax_frac_history: List[float],
    boundary_td_mean_history: List[float],
    normal_td_mean_history: List[float],
    action_names: List[str],
    weight_path: Path,
) -> None:
    plot_path = weight_path.with_name(weight_path.stem + "_q_diag.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    if q_per_action_history:
        arr = np.array(q_per_action_history)
        for i, name in enumerate(action_names):
            axes[0].plot(arr[:, i], linewidth=0.8, label=name)
        axes[0].set_title("Mean Q by action (sampled every 100 updates)")
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("Mean Q")
        axes[0].legend(fontsize=8, ncol=3)
    else:
        axes[0].set_visible(False)

    if invalid_argmax_frac_history:
        ma = _moving_average(invalid_argmax_frac_history, 100)
        axes[1].plot(invalid_argmax_frac_history, linewidth=0.3, alpha=0.2, color="#d62728")
        if ma is not None:
            ma_x = np.arange(99, 99 + len(ma))
            axes[1].plot(ma_x, ma, linewidth=1.2, color="#d62728")
        axes[1].set_title("Fraction of replay argmax on invalid action")
        axes[1].set_xlabel("Update step")
        axes[1].set_ylabel("Fraction")
        axes[1].set_ylim(-0.02, 1.02)
    else:
        axes[1].set_visible(False)

    has_boundary = len(boundary_td_mean_history) > 0
    has_normal = len(normal_td_mean_history) > 0
    if has_boundary or has_normal:
        if has_boundary:
            axes[2].plot(boundary_td_mean_history, linewidth=0.5, label="boundary", color="#ff7f0e")
        if has_normal:
            axes[2].plot(normal_td_mean_history, linewidth=0.5, label="normal", color="#1f77b4")
        axes[2].set_title("Mean |TD error|: boundary vs normal transitions")
        axes[2].set_xlabel("Update step")
        axes[2].set_ylabel("|TD|")
        axes[2].legend()
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved Q diagnostics to {plot_path}")


def _plot_distance_progress(
    initial_obj_dists: List[float],
    weight_path: Path,
    window: int = 100,
) -> None:
    if not initial_obj_dists:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_distance.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ma = _moving_average(initial_obj_dists, window)
    ax.plot(initial_obj_dists, linewidth=0.3, alpha=0.15, color="#1f77b4")
    if ma is not None:
        ma_x = np.arange(window - 1, window - 1 + len(ma))
        ax.plot(ma_x, ma, linewidth=1.2, color="#1f77b4")
    ax.set_title(f"Initial distance to target object (MA window={window})")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Manhattan distance")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved distance progress to {plot_path}")


def _plot_anticipation(
    post_task_dist: List[int],
    weight_path: Path,
    window: int = 100,
) -> None:
    if not post_task_dist:
        return
    plot_path = weight_path.with_name(weight_path.stem + "_anticipation.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    floats = [float(d) for d in post_task_dist]
    ma = _moving_average(floats, window)
    ax.plot(floats, linewidth=0.3, alpha=0.15, color="#ff7f0e")
    if ma is not None:
        ma_x = np.arange(window - 1, window - 1 + len(ma))
        ax.plot(ma_x, ma, linewidth=1.2, color="#ff7f0e")
    ax.set_title(f"Post-task distance to next target object (MA window={window})")
    ax.set_xlabel("Task transition index")
    ax.set_ylabel("Manhattan distance")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved anticipation plot to {plot_path}")


def _write_diagnostics_json(
    diag: Dict[str, Any],
    weight_path: Path,
) -> None:
    json_path = weight_path.with_name(weight_path.stem + "_diagnostics.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, default=str)
    print(f"Saved diagnostics JSON to {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on the RGB SimpleGrid environment.")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--num-objects", type=int, default=len(OBJECT_NAMES))
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel env instances to sample each step (default: auto from launcher CPUs).",
    )
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
        help="Environment steps between TensorBoard scalar logs.",
    )
    parser.add_argument(
        "--tb-dashboard-interval",
        type=int,
        default=None,
        help=(
            "Steps between TensorBoard dashboard figures (IMAGES tab). "
            "Default: max(2500, 4×log-interval). Use 0 or negative to match --log-interval."
        ),
    )
    parser.add_argument(
        "--tb-log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory (defaults to runs/<grid>_<label>_image_dqn_tpr*/tb/<run_name>).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Torch intra-op CPU threads (default: auto from launcher CPUs).",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Torch inter-op CPU threads (default: auto from launcher CPUs).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / "simple_grid_image_dqn.pt")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to task/object/receptacle distribution YAML.",
    )
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument(
        "--run-label",
        type=str,
        choices=("anticipatory", "myopic"),
        default=None,
        help="Subdirectory label under runs/ (default: myopic if tasks-per-reset<=1, else anticipatory).",
    )
    parser.add_argument(
        "--tasks-per-reset",
        type=int,
        default=1,
        help="Number of completed tasks that make up one RL episode (done=True only after this many).",
    )
    parser.add_argument(
        "--env-reset-tasks",
        type=int,
        default=None,
        help="Physical environment reset interval in tasks (default: same as tasks-per-reset).",
    )
    parser.add_argument(
        "--episode-step-limit",
        type=int,
        default=20_000,
        help="Maximum environment steps allowed between resets; overrides tasks-per-reset if exceeded (<=0 disables).",
    )
    parser.add_argument("--goal-buffer-size", type=int, default=5_000)
    parser.add_argument("--goal-buffer-fraction", type=float, default=0.25)
    parser.add_argument(
        "--clear-task-prob",
        type=float,
        default=None,
        help="Override probability of sampling clear tasks (default pulled from configs/config.yaml).",
    )
    parser.add_argument(
        "--clear-receptacle-shaping-scale",
        type=float,
        default=2.0,
        help="Per-object reward for clear_receptacle when objects leave the target surface (default 2.0).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _select_device()
    resources = _resolve_runtime_resources(args, device)
    args.num_envs = resources.num_envs
    args.torch_threads = resources.torch_threads
    args.torch_interop_threads = resources.torch_interop_threads
    _apply_runtime_resources(resources)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _log_runtime_resources(resources, device)
    train(args, device)


if __name__ == "__main__":
    main()
