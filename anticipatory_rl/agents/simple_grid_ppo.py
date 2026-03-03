"""PPO training script that leverages Stable-Baselines3 for the SimpleGrid task."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from anticipatory_rl.envs.simple_grid_image_env import SimpleGridImageEnv



class SimpleGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(conv_out, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.proj(self.conv(observations))


class SimpleGridWrapper(gym.Wrapper):
    """Wrapper to ensure SimpleGridEnv follows Gym API and resets properly."""
    
    def __init__(self, env: SimpleGridImageEnv, tasks_per_episode: int = 1):
        super().__init__(env)
        self._tasks_per_episode = max(1, tasks_per_episode)
        self._tasks_completed = 0
    
    def reset(self, **kwargs):
        self._tasks_completed = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, success, horizon, info = self.env.step(action)
        terminated = False
        truncated = horizon

        if success:
            self._tasks_completed += 1
            if self._tasks_completed >= self._tasks_per_episode:
                terminated = True
                self._tasks_completed = 0

        if truncated:
            self._tasks_completed = 0

        info["success"] = success
        return obs, reward, terminated, truncated, info


def make_env(
    seed: int | None,
    args: argparse.Namespace,
    num_objects: int,
    *,
    success_reward: float | None = None,
    distance_reward_scale: float | None = None,
) -> DummyVecEnv:
    def _init():
        env = SimpleGridImageEnv(
            success_reward=success_reward if success_reward is not None else args.success_reward,
            num_objects=num_objects,
            grid_size=args.grid_size,
            correct_pick_bonus=args.correct_pick_bonus,
            distance_reward=args.distance_reward,
            distance_reward_scale=distance_reward_scale
            if distance_reward_scale is not None
            else args.distance_reward_scale,
        )
        # Wrap with SimpleGridWrapper for proper Gym API and auto-reset
        env = SimpleGridWrapper(env, tasks_per_episode=args.tasks_per_episode)
        env.reset(seed=seed)
        return env

    return _init


class EntropyBoostCallback(BaseCallback):
    def __init__(self, boost_steps: int, boost_coef: float, base_coef: float):
        super().__init__()
        self.boost_steps = boost_steps
        self.boost_coef = boost_coef
        self.base_coef = base_coef
        self._boosting = boost_steps > 0

    def _on_training_start(self) -> None:
        if self._boosting and hasattr(self.model, "ent_coef"):
            self.model.ent_coef = self.boost_coef  # type: ignore[attr-defined]

    def _on_step(self) -> bool:
        if not self._boosting:
            return True
        if self.num_timesteps >= self.boost_steps:
            if hasattr(self.model, "ent_coef"):
                self.model.ent_coef = self.base_coef  # type: ignore[attr-defined]
            self._boosting = False
        return True


class MetricsCallback(BaseCallback):
    def __init__(self, plot_path: Path, window: int = 200):
        super().__init__()
        self.plot_path = plot_path
        self.window = window
        self.step_rewards: List[float] = []
        self.episode_returns: List[float] = []
        self.task_lengths: List[int] = []
        self.success_flags: List[float] = []
        self._task_returns: np.ndarray | None = None
        self._task_lengths_accum: np.ndarray | None = None
        self.kl_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.critic_mean_history: List[float] = []

    def _on_training_start(self) -> None:
        envs = self.training_env.num_envs
        self._task_returns = np.zeros(envs, dtype=np.float32)
        self._task_lengths_accum = np.zeros(envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards: np.ndarray = self.locals["rewards"]
        dones: np.ndarray = self.locals["dones"]
        infos: Sequence[dict] = self.locals.get("infos", [{} for _ in range(len(dones))])
        self.step_rewards.extend(rewards.tolist())
        if self._task_returns is None or self._task_lengths_accum is None:
            envs = len(rewards)
            self._task_returns = np.zeros(envs, dtype=np.float32)
            self._task_lengths_accum = np.zeros(envs, dtype=np.int32)
        self._task_returns += rewards
        self._task_lengths_accum += 1
        for idx, done in enumerate(dones):
            info_success = bool(infos[idx].get("success", False))
            if info_success:
                self.episode_returns.append(float(self._task_returns[idx]))
                self.task_lengths.append(int(self._task_lengths_accum[idx]))
                self.success_flags.append(1.0)
                self._task_returns[idx] = 0.0
                self._task_lengths_accum[idx] = 0
            elif done:
                self.episode_returns.append(float(self._task_returns[idx]))
                self.task_lengths.append(int(self._task_lengths_accum[idx]))
                self.success_flags.append(0.0)
                self._task_returns[idx] = 0.0
                self._task_lengths_accum[idx] = 0
        self._log_rollups()
        return True

    def _on_rollout_end(self) -> None:
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        if approx_kl is not None:
            self.kl_history.append(float(approx_kl))
        value_loss = self.logger.name_to_value.get("train/value_loss")
        if value_loss is not None:
            self.value_loss_history.append(float(value_loss))
        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.values is not None:
            critic_mean = float(np.mean(self.model.rollout_buffer.values))
            self.critic_mean_history.append(critic_mean)

    def _log_rollups(self) -> None:
        window_slice = slice(-self.window, None)
        if self.task_lengths:
            avg_steps = float(np.mean(self.task_lengths[window_slice]))
            self.logger.record("custom/avg_task_steps", avg_steps)
        if self.success_flags:
            success_rate = float(np.mean(self.success_flags[window_slice]))
            self.logger.record("custom/success_rate", success_rate)

    def save_plot(self) -> None:
        if not self.step_rewards:
            return
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(self.step_rewards, linewidth=0.5)
        axes[0].set_title("Per-step reward")
        axes[0].set_ylabel("Reward")
        axes[1].plot(self.episode_returns, linewidth=0.8)
        axes[1].set_title("Per-task return")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Return")
        fig.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_path)
        plt.close(fig)
        self._save_task_length_plot()
        self._save_diagnostics_plot()

    def _save_task_length_plot(self, window: int = 100) -> None:
        if not self.task_lengths:
            return
        rolling = []
        for idx in range(1, len(self.task_lengths) + 1):
            start = max(0, idx - window)
            rolling.append(float(np.mean(self.task_lengths[start:idx])))
        plot_path = self.plot_path.with_name(self.plot_path.stem.replace("_metrics", "_task_lengths") + ".png")
        plt.figure(figsize=(10, 4))
        plt.plot(rolling, linewidth=0.8)
        plt.title(f"Rolling Avg Task Length (window={window})")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved avg task length plot to {plot_path}")

    def _save_diagnostics_plot(self) -> None:
        if not self.episode_returns and not self.kl_history and not self.value_loss_history:
            return
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()

        if self.episode_returns:
            axes[0].plot(self.episode_returns, linewidth=0.8)
            axes[0].set_title("Return per Task")
            axes[0].set_xlabel("Task")
            axes[0].set_ylabel("Return")
        else:
            axes[0].set_visible(False)

        if self.kl_history:
            axes[1].plot(self.kl_history, linewidth=0.8, color="tab:orange")
            axes[1].set_title("Approx KL per Rollout")
            axes[1].set_xlabel("Rollout")
            axes[1].set_ylabel("KL")
        else:
            axes[1].set_visible(False)

        if self.critic_mean_history:
            axes[2].plot(self.critic_mean_history, linewidth=0.8, color="tab:green")
            axes[2].set_title("Critic Value Mean")
            axes[2].set_xlabel("Rollout")
            axes[2].set_ylabel("V(s)")
        else:
            axes[2].set_visible(False)

        if self.value_loss_history:
            axes[3].plot(self.value_loss_history, linewidth=0.8, color="tab:red")
            axes[3].set_title("Value Loss")
            axes[3].set_xlabel("Rollout")
            axes[3].set_ylabel("Loss")
        else:
            axes[3].set_visible(False)

        fig.tight_layout()
        diag_path = self.plot_path.with_name(self.plot_path.stem.replace("_metrics", "_diagnostics") + ".png")
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(diag_path)
        plt.close(fig)
        print(f"Saved diagnostics plot to {diag_path}")


class SuccessReplayCallback(BaseCallback):
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        updates_per_rollout: int,
        bc_coef: float,
        start_step: int,
    ):
        super().__init__()
        self.capacity = capacity
        self.batch_size = batch_size
        self.updates_per_rollout = updates_per_rollout
        self.bc_coef = bc_coef
        self.start_step = start_step
        self.buffer: Deque[Tuple[np.ndarray, int]] = deque(maxlen=capacity)
        self._episode_obs: List[List[np.ndarray]] | None = None
        self._episode_actions: List[List[int]] | None = None

    def _on_training_start(self) -> None:
        envs = self.training_env.num_envs
        self._episode_obs = [[] for _ in range(envs)]
        self._episode_actions = [[] for _ in range(envs)]

    def _on_step(self) -> bool:
        if self._episode_obs is None or self._episode_actions is None:
            return True
        observations: np.ndarray = self.locals.get("observations") or self.locals.get("new_obs")  # type: ignore[assignment]
        actions: np.ndarray = self.locals["actions"]
        dones: np.ndarray = self.locals["dones"]
        infos: Sequence[dict] = self.locals.get("infos", [{} for _ in range(len(dones))])
        for idx in range(len(dones)):
            self._episode_obs[idx].append(np.array(observations[idx], copy=True))
            self._episode_actions[idx].append(int(actions[idx]))
            if dones[idx]:
                if infos[idx].get("success"):
                    for obs, act in zip(self._episode_obs[idx], self._episode_actions[idx]):
                        self.buffer.append((obs, act))
                self._episode_obs[idx].clear()
                self._episode_actions[idx].clear()
        return True

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps < self.start_step:
            return True
        if len(self.buffer) < self.batch_size:
            return True
        device = self.model.policy.device
        for _ in range(self.updates_per_rollout):
            batch = random.sample(self.buffer, k=min(self.batch_size, len(self.buffer)))
            obs_batch = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
            action_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device)
            _, log_prob, _ = self.model.policy.evaluate_actions(obs_batch, action_batch)
            bc_loss = -log_prob.mean() * self.bc_coef
            self.model.policy.optimizer.zero_grad()
            bc_loss.backward()
            clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
            self.model.policy.optimizer.step()
        return True


def _build_vec_env(args: argparse.Namespace, num_objects: int, success_reward: float, distance_scale: float) -> DummyVecEnv:
    return DummyVecEnv(
        [
            make_env(
                (args.seed or 0) + i,
                args,
                num_objects,
                success_reward=success_reward,
                distance_reward_scale=distance_scale,
            )
            for i in range(args.num_envs)
        ]
    )


def _build_model(
    args: argparse.Namespace,
    env: DummyVecEnv,
    *,
    ent_coef: float,
) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=SimpleGridCNN,
        features_extractor_kwargs={"features_dim": args.hidden_dim},
        net_arch=dict(pi=[args.hidden_dim], vf=[args.hidden_dim]),
    )
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return PPO(
        "MlpPolicy",
        env,
        n_steps=args.rollout_steps,
        batch_size=args.mini_batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        ent_coef=ent_coef,
        vf_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.lr,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=device,
    )


def train(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / f"{args.grid_size}_ppo_{args.num_objects}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output if args.output.is_absolute() else run_dir / args.output
    metrics_path = run_dir / f"{output_path.stem}_metrics.png"
    metrics_cb = MetricsCallback(metrics_path)

    curriculum_enabled = args.num_objects > 1 and args.curriculum_single_steps > 0
    stage1_steps = min(args.curriculum_single_steps, args.total_steps) if curriculum_enabled else 0
    remaining_steps = args.total_steps
    policy_state = None

    if stage1_steps > 0:
        stage1_success = args.curriculum_stage1_success_reward or args.success_reward
        stage1_distance = args.curriculum_stage1_distance_reward_scale or args.distance_reward_scale
        env_stage1 = _build_vec_env(args, num_objects=1, success_reward=stage1_success, distance_scale=stage1_distance)
        model_stage1 = _build_model(args, env_stage1, ent_coef=args.curriculum_stage1_entropy_coef or args.entropy_coef)
        model_stage1.learn(
            total_timesteps=stage1_steps,
            callback=CallbackList([metrics_cb]),
            progress_bar=True,
        )
        policy_state = model_stage1.policy.state_dict()
        remaining_steps -= stage1_steps
        del model_stage1

    env_stage2 = _build_vec_env(
        args,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_scale=args.distance_reward_scale,
    )
    model = _build_model(args, env_stage2, ent_coef=args.entropy_coef)
    if policy_state is not None:
        model.policy.load_state_dict(policy_state)

    callbacks: List[BaseCallback] = [metrics_cb]
    if args.curriculum_entropy_steps > 0:
        callbacks.append(
            EntropyBoostCallback(
                boost_steps=args.curriculum_entropy_steps,
                boost_coef=args.curriculum_entropy_coef,
                base_coef=args.entropy_coef,
            )
        )
    if args.success_bc_coef > 0.0:
        callbacks.append(
            SuccessReplayCallback(
                capacity=args.success_buffer_size,
                batch_size=args.success_bc_batch,
                updates_per_rollout=args.success_bc_updates,
                bc_coef=args.success_bc_coef,
                start_step=0,
            )
        )

    callback_list = CallbackList(callbacks)
    model.learn(
        total_timesteps=max(remaining_steps, 0),
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(str(output_path))
    print(f"Saved PPO weights to {output_path}")
    metrics_cb.save_plot()
    print(f"Saved training curves to {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO (SB3) on the SimpleGrid pick-and-place task.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--tasks-per-episode",
        type=int,
        default=1,
        help="Number of tasks grouped into a single episode termination signal.",
    )
    parser.add_argument(
        "--curriculum-single-steps",
        type=int,
        default=0,
        help="Train with one object for this many steps before enabling all objects (requires num_objects>1).",
    )
    parser.add_argument(
        "--curriculum-entropy-steps",
        type=int,
        default=10000,
        help="Number of steps to keep the entropy boost active after curriculum switch.",
    )
    parser.add_argument(
        "--curriculum-entropy-coef",
        type=float,
        default=0.05,
        help="Temporary entropy coefficient applied right after curriculum switch.",
    )
    parser.add_argument(
        "--success-buffer-size",
        type=int,
        default=5000,
        help="Number of transitions to keep from successful tasks for imitation updates.",
    )
    parser.add_argument(
        "--success-bc-batch",
        type=int,
        default=128,
        help="Batch size for imitation updates from the success buffer.",
    )
    parser.add_argument(
        "--success-bc-updates",
        type=int,
        default=4,
        help="Number of imitation gradient steps to run after each rollout.",
    )
    parser.add_argument(
        "--success-bc-coef",
        type=float,
        default=0.5,
        help="Weight applied to the imitation loss from successful transitions (set 0 to disable).",
    )
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--correct-pick-bonus", type=float, default=1.0)
    parser.add_argument(
        "--curriculum-stage1-success-reward",
        type=float,
        default=None,
        help="Optional success reward to use during the single-object curriculum stage.",
    )
    parser.add_argument(
        "--curriculum-stage1-distance-reward-scale",
        type=float,
        default=None,
        help="Optional distance reward scale to use during the single-object curriculum stage.",
    )
    parser.add_argument(
        "--curriculum-stage1-entropy-coef",
        type=float,
        default=None,
        help="Entropy coefficient to use during the single-object curriculum stage (defaults to --entropy-coef).",
    )
    parser.add_argument(
        "--distance-reward",
        action="store_true",
        help="Enable distance-based shaping reward (agent→object before pickup, object→target after).",
    )
    parser.add_argument(
        "--distance-reward-scale",
        type=float,
        default=1.0,
        help="Scale for the distance-based shaping reward.",
    )
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("simple_grid_ppo.zip"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
