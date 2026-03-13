"""Greedy inference utility for SimpleGrid image DQN checkpoints.

This script mirrors the network/env setup from the training script but runs a
single-environment rollout, saving the pixel channels of every visited state
to disk. Useful for debugging datasets and verifying that trained policies
behave as expected under the latest environment dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from anticipatory_rl.agents.simple_grid_image_dqn import ConvQNetwork
from anticipatory_rl.envs.simple_grid_image_env import (
    OBJECT_NAMES,
    SimpleGridImageEnv,
)


@dataclass
class SampledTask:
    task_type: str  # "move" or "clear"
    object_name: Optional[str]
    receptacle_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy DQN inference and save pixel channels.")
    parser.add_argument("--state-dict", type=Path, required=True, help="Checkpoint containing ConvQNetwork weights.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/image_dqn_pixels"), help="Where to store saved pixel arrays.")
    parser.add_argument("--total-steps", type=int, default=5_000, help="Number of primitive steps to execute.")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-objects", type=int, default=len(OBJECT_NAMES))
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to task/object/receptacle distribution YAML.",
    )
    parser.add_argument("--clear-task-prob", type=float, default=None, help="Override clear-task probability (defaults to config).")
    parser.add_argument("--tasks-per-reset", type=int, default=1_000, help="Force env reset after this many tasks (match training).")
    parser.add_argument(
        "--tasks-per-sequence",
        type=int,
        default=100,
        help="Number of sampled tasks to queue at a time (must be > 0).",
    )
    parser.add_argument(
        "--save-format",
        choices=("png", "npy"),
        default="png",
        help="How to serialize each observation (default: png).",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> SimpleGridImageEnv:
    return SimpleGridImageEnv(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_task_prob=args.clear_task_prob,
        config_path=args.config_path,
    )


def weighted_choice(
    distribution: Mapping[str, float],
    candidates: Sequence[str],
    rng: np.random.Generator,
) -> str:
    if not candidates:
        raise ValueError("Cannot sample from an empty candidate set.")
    weights = np.array([max(distribution.get(name, 0.0), 0.0) for name in candidates], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return str(rng.choice(candidates))
    probs = weights / total
    return str(rng.choice(candidates, p=probs))


def _eligible_receptacles(receptacles: Sequence[str], last_sampled: Optional[str]) -> List[str]:
    if last_sampled is None or len(receptacles) <= 1:
        return list(receptacles)
    filtered = [rec for rec in receptacles if rec != last_sampled]
    return filtered or list(receptacles)


def sample_task_sequence(
    env: SimpleGridImageEnv,
    num_tasks: int,
    rng: np.random.Generator,
) -> List[SampledTask]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be > 0 when sampling task sequences.")
    tasks: List[SampledTask] = []
    last_rec: Optional[str] = None
    active_objects = list(env.active_objects)
    receptacles = list(env.receptacle_names)
    object_distribution = getattr(env, "object_distribution", {})
    surface_distribution = getattr(env, "surface_distribution", {})
    object_source_distribution = getattr(env, "object_source_distribution", {})
    clear_prob = getattr(env, "clear_task_prob", 0.0)
    for _ in range(num_tasks):
        if clear_prob > 0.0 and rng.random() < clear_prob:
            rec_choices = _eligible_receptacles(receptacles, last_rec)
            rec = weighted_choice(surface_distribution, rec_choices, rng)
            tasks.append(SampledTask("clear", None, rec))
            last_rec = rec
            continue
        obj = weighted_choice(object_distribution, active_objects, rng)
        source_dist = object_source_distribution.get(obj, surface_distribution)
        rec_choices = _eligible_receptacles(receptacles, last_rec)
        rec = weighted_choice(source_dist, rec_choices, rng)
        tasks.append(SampledTask("move", obj, rec))
        last_rec = rec
    return tasks


def apply_sampled_task(env: SimpleGridImageEnv, task: SampledTask):
    env.task_type = task.task_type
    env.target_object = task.object_name
    env.target_receptacle = task.receptacle_name
    if hasattr(env, "_last_target_receptacle"):
        env._last_target_receptacle = task.receptacle_name  # noqa: SLF001
    if hasattr(env, "_task_steps"):
        env._task_steps = 0  # noqa: SLF001
    if hasattr(env, "_pending_auto_success") and hasattr(env, "_task_already_satisfied"):
        env._pending_auto_success = env._task_already_satisfied()  # noqa: SLF001
    return env._obs(), env._info()


def _describe_record(record: dict) -> str:
    if record.get("task_type") == "clear":
        return f"CLEAR {record.get('target_receptacle')}"
    obj = record.get("target_object")
    rec = record.get("target_receptacle")
    if obj is None:
        return f"MOVE (?) -> {rec}"
    return f"MOVE {obj} -> {rec}"


def compute_anticipation_metrics(
    task_records: List[dict],
    episode_len: int,
) -> dict:
    """Aggregate per-task records into anticipation evaluation metrics.

    Key quantities:
    - baseline_auto_rate: fraction of position-0 tasks already satisfied at
      assignment (pure luck, no agent contribution).
    - auto_rate_by_pos: same fraction for every episode position 0..episode_len-1.
    - anticipation_delta: mean(auto_rate[pos>0]) - baseline; positive values
      mean the agent leaves the world in a state that satisfies future tasks
      more often than chance.
    - active_success_rate: success rate only for tasks that were NOT
      pre-satisfied (measures real solving ability separately from luck).
    """
    by_pos: Dict[int, List[dict]] = {}
    for rec in task_records:
        pos = rec.get("episode_position")
        if pos is None:
            continue
        by_pos.setdefault(pos, []).append(rec)

    positions = sorted(by_pos.keys())
    auto_rate_by_pos: Dict[int, float] = {}
    success_rate_by_pos: Dict[int, float] = {}
    avg_steps_by_pos: Dict[int, float] = {}
    n_by_pos: Dict[int, int] = {}
    for pos in positions:
        recs = by_pos[pos]
        auto_rate_by_pos[pos] = float(np.mean([r["auto_satisfied"] for r in recs]))
        success_rate_by_pos[pos] = float(np.mean([1.0 if r["success"] else 0.0 for r in recs]))
        avg_steps_by_pos[pos] = float(np.mean([r["steps"] for r in recs]))
        n_by_pos[pos] = len(recs)

    baseline = auto_rate_by_pos.get(0, 0.0)
    delta_by_pos = {pos: auto_rate_by_pos[pos] - baseline for pos in positions}
    later = [pos for pos in positions if pos > 0]
    anticipation_delta = float(np.mean([delta_by_pos[p] for p in later])) if later else 0.0

    auto_recs = [r for r in task_records if r.get("auto_satisfied", False)]
    active_recs = [r for r in task_records if not r.get("auto_satisfied", False)]

    def _safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    active_solved_steps = [r["steps"] for r in active_recs if r["success"]]

    return {
        "episode_len": episode_len,
        "baseline_auto_rate": baseline,
        "mean_auto_rate": _safe_mean(list(auto_rate_by_pos.values())),
        "anticipation_delta": anticipation_delta,
        "auto_rate_by_pos": {str(k): v for k, v in auto_rate_by_pos.items()},
        "success_rate_by_pos": {str(k): v for k, v in success_rate_by_pos.items()},
        "avg_steps_by_pos": {str(k): v for k, v in avg_steps_by_pos.items()},
        "delta_by_pos": {str(k): v for k, v in delta_by_pos.items()},
        "n_by_pos": {str(k): v for k, v in n_by_pos.items()},
        "n_tasks_total": len(task_records),
        "n_auto_tasks": len(auto_recs),
        "n_active_tasks": len(active_recs),
        "auto_success_rate": _safe_mean([1.0 if r["success"] else 0.0 for r in auto_recs]),
        "active_success_rate": _safe_mean([1.0 if r["success"] else 0.0 for r in active_recs]),
        "active_avg_steps_when_solved": _safe_mean(active_solved_steps),
    }


def _plot_anticipation_eval(metrics: dict, output_dir: Path) -> None:
    """Four-panel anticipation evaluation figure saved to output_dir."""
    auto_by_pos = {int(k): v for k, v in metrics["auto_rate_by_pos"].items()}
    delta_by_pos = {int(k): v for k, v in metrics["delta_by_pos"].items()}
    steps_by_pos = {int(k): v for k, v in metrics["avg_steps_by_pos"].items()}
    succ_by_pos = {int(k): v for k, v in metrics["success_rate_by_pos"].items()}
    n_by_pos = {int(k): v for k, v in metrics["n_by_pos"].items()}

    positions = sorted(auto_by_pos.keys())
    if not positions:
        return

    baseline = metrics["baseline_auto_rate"]
    antic_delta = metrics["anticipation_delta"]

    fig, axes = plt.subplots(4, 1, figsize=(max(10, len(positions) * 0.6), 16))

    # ── Panel 1: auto-success rate by position ──────────────────────────────
    ax = axes[0]
    ax.bar(positions, [auto_by_pos[p] for p in positions], alpha=0.75, color="#1f77b4",
           label="Auto-success rate")
    ax.axhline(baseline, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Luck baseline (pos 0) = {baseline:.1%}")
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Auto-success rate")
    ax.set_title("Auto-success rate by episode position")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_xticks(positions)
    for p in positions:
        ax.text(p, auto_by_pos[p] + 0.02, f"{auto_by_pos[p]:.0%}", ha="center", va="bottom",
                fontsize=7)

    # ── Panel 2: anticipation delta (Δ auto-rate over baseline) ────────────
    ax = axes[1]
    later = [p for p in positions if p > 0]
    colors = ["#2ca02c" if delta_by_pos[p] >= 0 else "#d62728" for p in later]
    ax.bar(later, [delta_by_pos[p] for p in later], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Δ auto-rate")
    ax.set_title(f"Anticipation delta (auto-rate − baseline) | mean Δ = {antic_delta:+.1%}")
    ax.set_xticks(later)

    # ── Panel 3: task success rate by position ──────────────────────────────
    ax = axes[2]
    ax.bar(positions, [succ_by_pos[p] for p in positions], alpha=0.75, color="#ff7f0e")
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Success rate")
    ax.set_title("Task success rate by episode position")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(positions)

    # ── Panel 4: avg steps to complete by position ──────────────────────────
    ax = axes[3]
    ax.bar(positions, [steps_by_pos[p] for p in positions], alpha=0.75, color="#9467bd")
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Avg steps")
    ax.set_title("Avg steps to complete by episode position")
    ax.set_xticks(positions)

    fig.tight_layout()
    plot_path = output_dir / "anticipation_eval.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved anticipation eval plot to {plot_path}")


def _print_anticipation_report(metrics: dict) -> None:
    """Print a human-readable anticipation evaluation summary to stdout."""
    sep = "=" * 54
    print()
    print(sep)
    print("         ANTICIPATION EVALUATION REPORT")
    print(sep)
    print(f"  Luck baseline  (pos-0 auto-rate) : {metrics['baseline_auto_rate']:.1%}")
    print(f"  Mean auto-rate (all positions)   : {metrics['mean_auto_rate']:.1%}")
    print(f"  Anticipation Δ (mean, pos > 0)   : {metrics['anticipation_delta']:+.1%}")
    print()
    print(f"  Auto tasks   : n={metrics['n_auto_tasks']:<5d} success={metrics['auto_success_rate']:.1%}")
    print(f"  Active tasks : n={metrics['n_active_tasks']:<5d} success={metrics['active_success_rate']:.1%}"
          f"  avg_steps={metrics['active_avg_steps_when_solved']:.1f}")
    print()
    print("  By episode position:")
    auto_by_pos = {int(k): v for k, v in metrics["auto_rate_by_pos"].items()}
    delta_by_pos = {int(k): v for k, v in metrics["delta_by_pos"].items()}
    steps_by_pos = {int(k): v for k, v in metrics["avg_steps_by_pos"].items()}
    n_by_pos = {int(k): v for k, v in metrics["n_by_pos"].items()}
    for pos in sorted(auto_by_pos.keys()):
        tag = "(baseline)" if pos == 0 else f"Δ={delta_by_pos[pos]:+.1%}"
        print(f"    pos {pos:2d}: auto={auto_by_pos[pos]:.1%}  {tag:<14}"
              f"  avg_steps={steps_by_pos[pos]:5.1f}  n={n_by_pos[pos]}")
    print(sep)
    print()


def save_observation(output_dir: Path, step: int, obs: np.ndarray, save_format: str) -> None:
    if save_format == "png":
        rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"frame_{step:06d}.png")
    else:
        np.save(output_dir / f"frame_{step:06d}.npy", obs)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    obs_shape = obs.shape
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    state_dict = torch.load(args.state_dict, map_location=device)
    q_net.load_state_dict(state_dict)
    q_net.eval()

    if args.tasks_per_sequence <= 0:
        raise ValueError("--tasks-per-sequence must be >= 1.")

    task_rng = np.random.default_rng(args.seed + 1)
    task_buffer: List[SampledTask] = []
    task_cursor = 0

    def dequeue_task() -> SampledTask:
        nonlocal task_buffer, task_cursor
        if not task_buffer or task_cursor >= len(task_buffer):
            task_buffer = sample_task_sequence(env, args.tasks_per_sequence, task_rng)
            task_cursor = 0
        task = task_buffer[task_cursor]
        task_cursor += 1
        return task

    current_task = dequeue_task()
    obs, info = apply_sampled_task(env, current_task)
    # Capture whether the initial task was already satisfied at assignment time
    current_task_auto_satisfied: bool = bool(getattr(env, "_pending_auto_success", False))

    pick_action = SimpleGridImageEnv.PICK
    place_action = SimpleGridImageEnv.PLACE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_steps": 0,
        "tasks_attempted": 0,
        "successes": 0,
        "failures": 0,
    }
    tasks_since_reset = 0
    task_step_counter = 0
    task_return = 0.0
    task_records = []
    task_frame_ranges: List[Tuple[dict, int, int]] = []
    current_task_start = 0
    progress = tqdm(total=args.total_steps, desc="Inference rollout")

    for step in range(args.total_steps):
        save_observation(args.output_dir, step, obs, args.save_format)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            can_pick = info.get("can_pick", True)
            can_place = info.get("can_place", True)
            if not can_pick:
                q_values[0, pick_action] = float("-inf")
            if not can_place:
                q_values[0, place_action] = float("-inf")
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, success, horizon, info = env.step(action)
        progress.update(1)
        stats["total_steps"] += 1
        task_step_counter += 1
        task_return += float(reward)

        if success or horizon:
            stats["tasks_attempted"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            record = {
                "task_number": stats["tasks_attempted"],
                "task_type": current_task.task_type,
                "target_object": current_task.object_name,
                "target_receptacle": current_task.receptacle_name,
                "success": bool(success),
                "steps": task_step_counter,
                "return": task_return,
                # Anticipation tracking fields
                "auto_satisfied": current_task_auto_satisfied,
                # tasks_since_reset is the position within the current episode
                # (0 = first task after reset, 1 = second, ...).  We read it
                # before incrementing so it reflects the completed task.
                "episode_position": tasks_since_reset,
            }
            task_records.append(record)
            task_step_counter = 0
            task_return = 0.0
            tasks_since_reset += 1
            task_frame_ranges.append((record, current_task_start, step))
            current_task_start = step + 1
            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                obs, info = env.reset()
                tasks_since_reset = 0
            current_task = dequeue_task()
            obs, info = apply_sampled_task(env, current_task)
            current_task_auto_satisfied = bool(getattr(env, "_pending_auto_success", False))

    progress.close()
    success_rate = stats["successes"] / max(1, stats["tasks_attempted"])
    stats["success_rate"] = success_rate
    task_steps = [rec["steps"] for rec in task_records]
    stats["avg_task_steps"] = float(np.mean(task_steps)) if task_steps else 0.0
    stats["median_task_steps"] = float(np.median(task_steps)) if task_steps else 0.0

    # ── Anticipation evaluation ─────────────────────────────────────────────
    antic_metrics: Optional[dict] = None
    if args.tasks_per_reset > 0 and task_records:
        antic_metrics = compute_anticipation_metrics(task_records, args.tasks_per_reset)
        _print_anticipation_report(antic_metrics)
        _plot_anticipation_eval(antic_metrics, args.output_dir)
        with (args.output_dir / "anticipation_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(antic_metrics, fh, indent=2)

    report = {
        "stats": stats,
        "anticipation": antic_metrics,
        "tasks": task_records,
    }
    with (args.output_dir / "rollout_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    summary = (
        f"Saved {args.total_steps} frames/pixels to {args.output_dir.resolve()} "
        f"| success rate {success_rate:.1%} ({stats['successes']}/{stats['tasks_attempted']}) "
        f"| avg steps/task {stats['avg_task_steps']:.1f}"
    )
    if task_steps:
        summary += f" | median steps {stats['median_task_steps']:.1f}"
    if antic_metrics is not None:
        summary += (
            f" | antic Δ {antic_metrics['anticipation_delta']:+.1%}"
            f" (baseline {antic_metrics['baseline_auto_rate']:.1%})"
        )
    print(summary)
    if task_step_counter > 0:
        print(
            f"Step budget expired with an unfinished task "
            f"({task_step_counter} steps so far); that attempt is not included in the stats."
        )
        pending_record = {
            "task_number": stats["tasks_attempted"] + 1,
            "task_type": current_task.task_type,
            "target_object": current_task.object_name,
            "target_receptacle": current_task.receptacle_name,
            "success": False,
            "steps": task_step_counter,
            "return": task_return,
        }
        task_frame_ranges.append(
            (pending_record, current_task_start, args.total_steps - 1)
        )
    if task_frame_ranges:
        print("Frame ranges per sampled task:")
        for record, start_idx, end_idx in task_frame_ranges:
            label = _describe_record(record)
            print(
                f"  Task {record['task_number']:>4}: frames {start_idx}–{end_idx} | {label}"
            )
    else:
        print("No completed tasks to report frame ranges for.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
