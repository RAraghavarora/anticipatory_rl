"""DQN trainer for the Three-Box anticipation toy problem.

Trains two agents back-to-back on the same Three-Box environment:

  1. **Anticipatory** — standard multi-task episodes.  The Bellman backup
     flows from Task 2 back into Task 1, so the agent learns *where* to
     drop the apple during Task 1 based on the Task 2 distribution.

  2. **Myopic** — identical, except the replay-buffer transition at the
     Task 1 boundary has ``done=True``.  This severs the Bellman backup,
     so the agent treats each task independently and cannot anticipate.

After training, both agents are evaluated greedily and a comparison
summary + plots are saved to ``--output-dir`` (default ``runs/three_box``).

Usage
-----
    python -m anticipatory_rl.agents.three_box_dqn          # defaults
    python -m anticipatory_rl.agents.three_box_dqn --total-steps 300000
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from tqdm import tqdm  # noqa: E402

from anticipatory_rl.envs.three_box_env import ThreeBoxEnv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Network ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConvQNetwork(nn.Module):
    """Same CNN encoder used in the full SimpleGrid DQN."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        hidden_dim: int,
        num_actions: int,
    ):
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
            conv_out = self.encoder(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Replay ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple uniform-sampling replay buffer."""

    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(list(self.buf), n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Metrics container ━━━━━━━━━━━━━━━━━━━━


@dataclass
class Metrics:
    episode_returns: List[float] = field(default_factory=list)
    rolling_drop_b: List[float] = field(default_factory=list)
    rolling_drop_x: List[int] = field(default_factory=list)
    task2_step_log: List[int] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _epsilon(step: int, args: argparse.Namespace) -> float:
    frac = min(1.0, step / max(1, args.epsilon_decay))
    return args.epsilon_start + frac * (args.epsilon_final - args.epsilon_start)


def _eval_action(q: torch.Tensor, args: argparse.Namespace) -> int:
    if args.eval_temperature <= 0:
        return int(q.argmax(dim=0).item())
    probs = torch.softmax(q / args.eval_temperature, dim=0)
    return int(torch.multinomial(probs, num_samples=1).item())


def _build_env(args: argparse.Namespace) -> ThreeBoxEnv:
    return ThreeBoxEnv(
        success_reward=args.success_reward,
        step_cost=args.step_cost,
        max_episode_steps=args.max_episode_steps,
        prob_a=args.prob_a,
        render_tile_px=args.render_tile_px,
    )


def _load_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> ConvQNetwork:
    device = _device()
    env = _build_env(args)
    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)
    q_net = ConvQNetwork(obs_shape, args.hidden_dim, n_actions).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    q_net.load_state_dict(state_dict)
    q_net.eval()
    return q_net


def rollout_policy(
    q_net: ConvQNetwork,
    args: argparse.Namespace,
    num_episodes: int = 500,
) -> Tuple[Metrics, Dict[str, object]]:
    """Run evaluation rollouts and collect both traces and summary stats."""

    device = _device()
    q_net.eval()
    env = _build_env(args)

    metrics = Metrics()
    recent_drops: Deque[str] = deque(maxlen=200)
    drop_count = 0
    drop_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    successes = 0

    for ep in range(num_episodes):
        state, info = env.reset(seed=args.seed + 100_000 + ep)
        ep_return = 0.0
        done = False
        while not done:
            with torch.no_grad():
                q = q_net(
                    torch.tensor(state, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                ).squeeze(0)
                action = _eval_action(q, args)
            state, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

            dl = info.get("drop_location")
            if dl is not None:
                drop_counts[dl] = drop_counts.get(dl, 0) + 1
                recent_drops.append(dl)
                drop_count += 1
                pct = sum(1 for d in recent_drops if d == "B") / len(recent_drops)
                metrics.rolling_drop_b.append(pct)
                metrics.rolling_drop_x.append(drop_count)

        metrics.episode_returns.append(ep_return)
        if terminated:
            successes += 1
            t2 = 0 if info.get("task2_auto", False) else info.get("task2_steps", 0)
            metrics.task2_step_log.append(int(t2))

    total_drops = sum(drop_counts.values())
    stats = {
        "mean_return": float(np.mean(metrics.episode_returns)),
        "std_return": float(np.std(metrics.episode_returns)),
        "success_rate": successes / max(1, num_episodes),
        "drop_A_pct": drop_counts["A"] / max(1, total_drops),
        "drop_B_pct": drop_counts["B"] / max(1, total_drops),
        "drop_counts": drop_counts,
        "mean_task2_steps": (
            float(np.mean(metrics.task2_step_log))
            if metrics.task2_step_log
            else float("nan")
        ),
        "num_episodes": num_episodes,
    }
    return metrics, stats


def train_agent(
    args: argparse.Namespace,
    myopic: bool,
    label: str,
) -> Tuple[ConvQNetwork, Metrics]:
    """Train a single DQN agent and return the network + training metrics."""

    device = _device()
    env = _build_env(args)

    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)

    q_net = ConvQNetwork(obs_shape, args.hidden_dim, n_actions).to(device)
    target_net = ConvQNetwork(obs_shape, args.hidden_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)
    metrics = Metrics()

    state, info = env.reset(seed=args.seed)
    ep_return: float = 0.0

    recent_drops: Deque[str] = deque(maxlen=200)
    drop_count = 0

    progress = tqdm(range(args.total_steps), desc=label, leave=True)

    for step in progress:
        # ---- action selection (ε-greedy) ----
        eps = _epsilon(step, args)
        if random.random() < eps:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                q = q_net(
                    torch.tensor(state, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                )
                action = int(q.argmax(dim=1).item())

        # ---- environment step ----
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # For the myopic agent the Bellman backup is severed at the
        # Task 1 → Task 2 boundary: the transition gets done=True
        # even though the environment continues to Task 2.
        bellman_done = done or (myopic and info.get("task1_done", False))

        # For the myopic agent, only store Task 1 transitions.
        # Storing Task 2 transitions would let Task 2 gradients
        # contaminate the shared network weights, leaking information
        # about the Task 2 distribution into Task 1 Q-values.
        in_task2 = myopic and info.get("task_phase") == 2 and not info.get("task1_done", False)
        if not in_task2:
            replay.push(
                Transition(state, action, float(reward), next_state, bellman_done)
            )
        ep_return += reward

        # ---- track drop location ----
        dl = info.get("drop_location")
        if dl is not None:
            recent_drops.append(dl)
            drop_count += 1
            pct = sum(1 for d in recent_drops if d == "B") / len(recent_drops)
            metrics.rolling_drop_b.append(pct)
            metrics.rolling_drop_x.append(drop_count)

        # ---- episode boundary ----
        if done:
            metrics.episode_returns.append(ep_return)
            if terminated:
                t2 = (
                    0
                    if info.get("task2_auto", False)
                    else info.get("task2_steps", 0)
                )
                metrics.task2_step_log.append(int(t2))
            ep_return = 0.0
            state, info = env.reset()
        else:
            state = next_state

        # ---- network update (Double DQN) ----
        if len(replay) >= args.batch_size:
            batch = replay.sample(args.batch_size)
            s  = torch.tensor(np.stack([t.state for t in batch]),      dtype=torch.float32, device=device)  # noqa: E221
            a  = torch.tensor([t.action for t in batch],               dtype=torch.int64,   device=device).unsqueeze(1)  # noqa: E221,E501
            r  = torch.tensor([t.reward for t in batch],               dtype=torch.float32, device=device).unsqueeze(1)  # noqa: E221,E501
            ns = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)  # noqa: E221
            d  = torch.tensor([t.done for t in batch],                 dtype=torch.float32, device=device).unsqueeze(1)  # noqa: E221,E501

            q_vals = q_net(s).gather(1, a)
            with torch.no_grad():
                next_a = q_net(ns).argmax(dim=1, keepdim=True)
                next_q = target_net(ns).gather(1, next_a)
                target = r + args.gamma * (1.0 - d) * next_q

            loss = nn.functional.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()

            metrics.td_errors.append(
                float((target - q_vals).abs().mean().item())
            )

            # soft target update
            if args.tau < 1.0:
                with torch.no_grad():
                    for tp, p in zip(
                        target_net.parameters(), q_net.parameters()
                    ):
                        tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
            elif (step + 1) % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        # ---- progress bar ----
        if step % 500 == 0:
            avg_ret = (
                f"{np.mean(metrics.episode_returns[-50:]):.1f}"
                if metrics.episode_returns
                else "n/a"
            )
            drop_pct = (
                f"{metrics.rolling_drop_b[-1] * 100:.0f}%"
                if metrics.rolling_drop_b
                else "n/a"
            )
            progress.set_postfix(
                ret=avg_ret, drop_B=drop_pct, eps=f"{eps:.2f}"
            )

    progress.close()
    return q_net, metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━


def evaluate(
    q_net: ConvQNetwork,
    args: argparse.Namespace,
    num_episodes: int = 500,
) -> Dict[str, object]:
    """Run the softmax evaluation policy and collect statistics."""

    _, stats = rollout_policy(q_net, args, num_episodes)
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Plotting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _moving_avg(data: List[float], w: int) -> np.ndarray | None:
    if len(data) < w or w <= 0:
        return None
    return np.convolve(data, np.ones(w) / w, mode="valid")


def plot_comparison(
    anti_m: Metrics,
    myopic_m: Metrics,
    anti_eval: Dict,
    myopic_eval: Dict,
    out_dir: Path,
    prob_a: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"anti": "#2196F3", "myopic": "#FF5722"}

    # ---- 1. Episode returns ----
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 50
    for m, lbl, c in [
        (anti_m, "Anticipatory", colors["anti"]),
        (myopic_m, "Myopic", colors["myopic"]),
    ]:
        ax.plot(m.episode_returns, alpha=0.06, color=c, linewidth=0.5)
        ma = _moving_avg(m.episode_returns, w)
        if ma is not None:
            xs = np.arange(w - 1, w - 1 + len(ma))
            ax.plot(xs, ma, label=lbl, color=c, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"Episode Return  (rolling avg, window={w})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "returns.png", dpi=150)
    plt.close(fig)

    # ---- 2. Drop % on B (rolling) ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for m, lbl, c in [
        (anti_m, "Anticipatory", colors["anti"]),
        (myopic_m, "Myopic", colors["myopic"]),
    ]:
        if m.rolling_drop_b:
            ax.plot(
                m.rolling_drop_x,
                [p * 100 for p in m.rolling_drop_b],
                label=lbl,
                color=c,
                linewidth=1.2,
            )
    ax.axhline(y=50, color="grey", ls="--", lw=0.8, label="Random (50%)")
    ax.axhline(y=100, color="green", ls=":", lw=0.8, label="Optimal (100%)")
    ax.set_xlabel("Task 1 completions")
    ax.set_ylabel("Rolling % drops on B (window=200)")
    ax.set_title(
        f"Anticipation Metric: Task 1 drop location  (P(Task2=B)={1 - prob_a})"
    )
    ax.set_ylim(-5, 110)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "drop_location.png", dpi=150)
    plt.close(fig)

    # ---- 3. Evaluation bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["Anticipatory", "Myopic"]
    bar_colors = [colors["anti"], colors["myopic"]]

    axes[0].bar(
        labels,
        [anti_eval["mean_return"], myopic_eval["mean_return"]],
        color=bar_colors,
    )
    axes[0].set_title("Avg Episode Return")
    axes[0].set_ylabel("Return")

    axes[1].bar(
        labels,
        [anti_eval["drop_B_pct"] * 100, myopic_eval["drop_B_pct"] * 100],
        color=bar_colors,
    )
    axes[1].axhline(y=50, color="grey", ls="--", lw=0.8)
    axes[1].set_title("% Drops on B (eval)")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 110)

    axes[2].bar(
        labels,
        [anti_eval["mean_task2_steps"], myopic_eval["mean_task2_steps"]],
        color=bar_colors,
    )
    axes[2].set_title("Avg Task 2 Steps")
    axes[2].set_ylabel("Steps")

    fig.suptitle("Anticipatory vs Myopic — Evaluation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    plt.close(fig)

    # ---- 4. Task 2 steps over training ----
    fig, ax = plt.subplots(figsize=(10, 4))
    w2 = 100
    for m, lbl, c in [
        (anti_m, "Anticipatory", colors["anti"]),
        (myopic_m, "Myopic", colors["myopic"]),
    ]:
        if m.task2_step_log:
            ma = _moving_avg([float(x) for x in m.task2_step_log], w2)
            if ma is not None:
                ax.plot(
                    np.arange(w2 - 1, w2 - 1 + len(ma)),
                    ma,
                    label=lbl,
                    color=c,
                    linewidth=1.2,
                )
    ax.set_xlabel("Successful episode")
    ax.set_ylabel("Task 2 steps")
    ax.set_title(f"Task 2 completion effort  (rolling avg, window={w2})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "task2_steps.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


def print_eval_summary(
    anti: Dict, myopic: Dict, prob_a: float
) -> None:
    print()
    print("=" * 62)
    print("        THREE-BOX ANTICIPATION TEST RESULTS")
    print("=" * 62)
    print(f"  Task 2 distribution:  A = {prob_a * 100:.0f}%,  B = {(1 - prob_a) * 100:.0f}%")
    print()

    for label, stats in [("ANTICIPATORY", anti), ("MYOPIC", myopic)]:
        print(f"  [{label}]")
        print(f"    Avg return:        {stats['mean_return']:>7.2f}  (± {stats['std_return']:.2f})")
        print(f"    Success rate:      {stats['success_rate'] * 100:>6.1f}%")
        print(f"    Drops on A:        {stats['drop_A_pct'] * 100:>6.1f}%")
        print(f"    Drops on B:        {stats['drop_B_pct'] * 100:>6.1f}%")
        print(f"    Avg Task 2 steps:  {stats['mean_task2_steps']:>7.2f}")
        print()

    delta = anti["mean_return"] - myopic["mean_return"]
    print(f"  Return advantage (anticipatory - myopic):  {delta:+.2f}")

    if anti["drop_B_pct"] > 0.70:
        verdict = "ANTICIPATORY — agent learned to prefer B"
    elif anti["drop_B_pct"] > 0.55:
        verdict = "PARTIAL — some anticipation detected"
    else:
        verdict = "NOT LEARNED — increase --total-steps or tune hyperparams"
    print(f"  Verdict:  {verdict}")
    print("=" * 62)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CLI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Three-Box anticipation toy: train & compare DQN agents."
    )
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--replay-size", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=80_000)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--success-reward", type=float, default=10.0)
    p.add_argument("--step-cost", type=float, default=0.05)
    p.add_argument("--prob-a", type=float, default=0.2)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--render-tile-px", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference-only", action="store_true")
    p.add_argument(
        "--anticipatory-checkpoint",
        type=Path,
        default=Path("runs") / "three_box" / "anticipatory.pt",
    )
    p.add_argument(
        "--myopic-checkpoint",
        type=Path,
        default=Path("runs") / "three_box" / "myopic.pt",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "three_box",
    )
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    args = build_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.inference_only:
        print(">>> Loading saved checkpoints for inference")
        anti_net = _load_checkpoint(args.anticipatory_checkpoint, args)
        myopic_net = _load_checkpoint(args.myopic_checkpoint, args)

        print(">>> Running rollout comparison …")
        anti_metrics, anti_eval = rollout_policy(
            anti_net, args, args.eval_episodes
        )
        myopic_metrics, myopic_eval = rollout_policy(
            myopic_net, args, args.eval_episodes
        )

        print_eval_summary(anti_eval, myopic_eval, args.prob_a)
        plot_comparison(
            anti_metrics,
            myopic_metrics,
            anti_eval,
            myopic_eval,
            out,
            args.prob_a,
        )
        return

    # ---- Train anticipatory agent ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(">>> Training ANTICIPATORY agent")
    anti_net, anti_metrics = train_agent(
        args, myopic=False, label="Anticipatory"
    )
    torch.save(anti_net.state_dict(), out / "anticipatory.pt")
    print(f"    Saved weights → {out / 'anticipatory.pt'}")

    # ---- Train myopic agent ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n>>> Training MYOPIC agent")
    myopic_net, myopic_metrics = train_agent(
        args, myopic=True, label="Myopic"
    )
    torch.save(myopic_net.state_dict(), out / "myopic.pt")
    print(f"    Saved weights → {out / 'myopic.pt'}")

    # ---- Evaluate both ----
    print("\n>>> Evaluating both agents …")
    anti_eval = evaluate(anti_net, args, args.eval_episodes)
    myopic_eval = evaluate(myopic_net, args, args.eval_episodes)

    # ---- Report ----
    print_eval_summary(anti_eval, myopic_eval, args.prob_a)
    plot_comparison(
        anti_metrics,
        myopic_metrics,
        anti_eval,
        myopic_eval,
        out,
        args.prob_a,
    )


if __name__ == "__main__":
    main()
