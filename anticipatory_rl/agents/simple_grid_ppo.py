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
from torch.nn.utils import clip_grad_norm_
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from anticipatory_rl.envs.simple_grid_env import SimpleGridEnv


class SimpleGridWrapper(gym.Wrapper):
    """Wrapper to ensure SimpleGridEnv follows Gym API and resets properly."""
    
    def __init__(self, env: SimpleGridEnv):
        super().__init__(env)
        self._max_steps_per_task = 100  # Maximum steps before forced reset
        self._current_steps = 0
    
    def reset(self, **kwargs):
        self._current_steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        self._current_steps += 1
        obs, reward, success, horizon, info = self.env.step(action)
        
        # Convert to standard Gym API: (obs, reward, terminated, truncated, info)
        terminated = success  # Task completed successfully
        truncated = horizon or (self._current_steps >= self._max_steps_per_task)  # Max steps reached
        
        # Add success flag to info for metrics
        info["success"] = success
        
        # Auto-reset if episode is done
        if terminated or truncated:
            self._current_steps = 0
        
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
        env = SimpleGridEnv(
            success_reward=success_reward if success_reward is not None else args.success_reward,
            num_objects=num_objects,
            grid_size=3,
            correct_pick_bonus=args.correct_pick_bonus,
            distance_reward=args.distance_reward,
            distance_reward_scale=distance_reward_scale
            if distance_reward_scale is not None
            else args.distance_reward_scale,
        )
        # Wrap with SimpleGridWrapper for proper Gym API and auto-reset
        env = SimpleGridWrapper(env)
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
        self._running_returns: np.ndarray | None = None
        self._running_lengths: np.ndarray | None = None

    def _on_training_start(self) -> None:
        envs = self.training_env.num_envs
        self._running_returns = np.zeros(envs, dtype=np.float32)
        self._running_lengths = np.zeros(envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards: np.ndarray = self.locals["rewards"]
        dones: np.ndarray = self.locals["dones"]
        infos: Sequence[dict] = self.locals.get("infos", [{} for _ in range(len(dones))])
        self.step_rewards.extend(rewards.tolist())
        if self._running_returns is None or self._running_lengths is None:
            envs = len(rewards)
            self._running_returns = np.zeros(envs, dtype=np.float32)
            self._running_lengths = np.zeros(envs, dtype=np.int32)
        self._running_returns += rewards
        self._running_lengths += 1
        for idx, done in enumerate(dones):
            if done:
                self.episode_returns.append(float(self._running_returns[idx]))
                self.task_lengths.append(int(self._running_lengths[idx]))
                success = float(infos[idx].get("success", False))
                self.success_flags.append(success)
                self._running_returns[idx] = 0.0
                self._running_lengths[idx] = 0
        self._log_rollups()
        return True

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
    policy_kwargs = dict(net_arch=[args.hidden_dim, args.hidden_dim])
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
    metrics_path = args.output.with_name(args.output.stem + "_metrics.png")
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
    model.save(str(args.output))
    print(f"Saved PPO weights to {args.output}")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs") / "simple_grid_ppo.zip")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
