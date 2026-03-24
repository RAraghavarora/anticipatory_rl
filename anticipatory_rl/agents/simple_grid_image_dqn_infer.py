"""Greedy inference for SimpleGrid image DQN checkpoints.

Supports a single checkpoint or a side-by-side comparison of anticipatory vs myopic
frozen weights on the same task RNG and env reset schedule.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy DQN inference; optionally compare anticipatory vs myopic checkpoints."
    )
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="Single checkpoint (ConvQNetwork weights). Omit if using --anticipatory-weights and --myopic-weights.",
    )
    parser.add_argument(
        "--anticipatory-weights",
        type=Path,
        default=None,
        help="Frozen weights from anticipatory training (used with --myopic-weights for comparison).",
    )
    parser.add_argument(
        "--myopic-weights",
        type=Path,
        default=None,
        help="Frozen weights from myopic training (used with --anticipatory-weights for comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (single run: default <checkpoint_dir>/infer; compare: default runs/compare_image_dqn_infer).",
    )
    parser.add_argument("--total-steps", type=int, default=5_000, help="Primitive env steps per rollout.")
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
    parser.add_argument(
        "--clear-task-prob",
        type=float,
        default=None,
        help="Override clear-task probability (defaults to config).",
    )
    parser.add_argument(
        "--clear-receptacle-shaping-scale",
        type=float,
        default=2.0,
        help="Per-object reward for clear when objects leave target surface.",
    )
    parser.add_argument(
        "--tasks-per-reset",
        type=int,
        default=1_000,
        help="Episode length in tasks (env reset + deterministic reset seed each episode).",
    )
    parser.add_argument(
        "--tasks-per-sequence",
        type=int,
        default=100,
        help="Tasks sampled ahead per buffer refill (must be > 0).",
    )
    parser.add_argument(
        "--save-format",
        choices=("png", "npy"),
        default="png",
        help="How to serialize each observation when saving frames.",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Skip writing per-step frames (still writes JSON/plots).",
    )
    args = parser.parse_args()

    ant = args.anticipatory_weights
    myo = args.myopic_weights
    single = args.state_dict
    if ant is not None or myo is not None:
        if ant is None or myo is None:
            parser.error("Compare mode requires both --anticipatory-weights and --myopic-weights.")
        if single is not None:
            parser.error("Do not pass --state-dict together with compare checkpoints.")
        args.compare_mode = True
    else:
        if single is None:
            parser.error("Pass --state-dict, or pass both --anticipatory-weights and --myopic-weights.")
        args.compare_mode = False
    return args


def make_env(args: argparse.Namespace) -> SimpleGridImageEnv:
    return SimpleGridImageEnv(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_receptacle_shaping_scale=args.clear_receptacle_shaping_scale,
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

    ax = axes[0]
    ax.bar(positions, [auto_by_pos[p] for p in positions], alpha=0.75, color="#2563eb", label="Auto-success rate")
    ax.axhline(
        baseline,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Luck baseline (pos 0) = {baseline:.1%}",
    )
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Auto-success rate")
    ax.set_title("Auto-success rate by episode position")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_xticks(positions)
    for p in positions:
        ax.text(
            p,
            auto_by_pos[p] + 0.02,
            f"{auto_by_pos[p]:.0%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax = axes[1]
    later = [p for p in positions if p > 0]
    colors = ["#16a34a" if delta_by_pos[p] >= 0 else "#dc2626" for p in later]
    ax.bar(later, [delta_by_pos[p] for p in later], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Δ auto-rate")
    ax.set_title(f"Anticipation delta (auto-rate − baseline) | mean Δ = {antic_delta:+.1%}")
    ax.set_xticks(later)

    ax = axes[2]
    ax.bar(positions, [succ_by_pos[p] for p in positions], alpha=0.75, color="#ea580c")
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Success rate")
    ax.set_title("Task success rate by episode position")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(positions)

    ax = axes[3]
    ax.bar(positions, [steps_by_pos[p] for p in positions], alpha=0.75, color="#7c3aed")
    ax.set_xlabel("Episode position")
    ax.set_ylabel("Avg steps")
    ax.set_title("Avg steps to complete by episode position")
    ax.set_xticks(positions)

    fig.tight_layout()
    plot_path = output_dir / "anticipation_eval.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved anticipation eval plot to {plot_path}")


def _print_anticipation_report(metrics: dict, label: str) -> None:
    sep = "=" * 54
    print()
    print(sep)
    print(f"         ANTICIPATION EVAL — {label}")
    print(sep)
    print(f"  Luck baseline  (pos-0 auto-rate) : {metrics['baseline_auto_rate']:.1%}")
    print(f"  Mean auto-rate (all positions)   : {metrics['mean_auto_rate']:.1%}")
    print(f"  Anticipation Δ (mean, pos > 0)   : {metrics['anticipation_delta']:+.1%}")
    print()
    print(f"  Auto tasks   : n={metrics['n_auto_tasks']:<5d} success={metrics['auto_success_rate']:.1%}")
    print(
        f"  Active tasks : n={metrics['n_active_tasks']:<5d} success={metrics['active_success_rate']:.1%}"
        f"  avg_steps={metrics['active_avg_steps_when_solved']:.1f}"
    )
    print()
    print("  By episode position:")
    auto_by_pos = {int(k): v for k, v in metrics["auto_rate_by_pos"].items()}
    delta_by_pos = {int(k): v for k, v in metrics["delta_by_pos"].items()}
    steps_by_pos = {int(k): v for k, v in metrics["avg_steps_by_pos"].items()}
    n_by_pos = {int(k): v for k, v in metrics["n_by_pos"].items()}
    for pos in sorted(auto_by_pos.keys()):
        tag = "(baseline)" if pos == 0 else f"Δ={delta_by_pos[pos]:+.1%}"
        print(
            f"    pos {pos:2d}: auto={auto_by_pos[pos]:.1%}  {tag:<14}"
            f"  avg_steps={steps_by_pos[pos]:5.1f}  n={n_by_pos[pos]}"
        )
    print(sep)
    print()


def save_observation(output_dir: Path, step: int, obs: np.ndarray, save_format: str) -> None:
    if save_format == "png":
        rgb = (np.transpose(obs[:3], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"frame_{step:06d}.png")
    else:
        np.save(output_dir / f"frame_{step:06d}.npy", obs)


def _load_state_dict(path: Path, device: torch.device) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run_single_rollout(
    args: argparse.Namespace,
    state_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    save_frames: bool,
) -> Dict[str, Any]:
    device = _select_device()
    state_path = state_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    obs_shape = obs.shape
    q_net = ConvQNetwork(obs_shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    q_net.load_state_dict(_load_state_dict(state_path, device))
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
    current_task_auto_satisfied: bool = bool(getattr(env, "_pending_auto_success", False))

    pick_action = SimpleGridImageEnv.PICK
    place_action = SimpleGridImageEnv.PLACE

    stats = {
        "total_steps": 0,
        "tasks_attempted": 0,
        "successes": 0,
        "failures": 0,
    }
    tasks_since_reset = 0
    episode_index = 0
    task_step_counter = 0
    task_return = 0.0
    task_records: List[dict] = []
    task_frame_ranges: List[Tuple[dict, int, int]] = []
    current_task_start = 0
    progress = tqdm(total=args.total_steps, desc=f"Inference [{run_label}]")

    for step in range(args.total_steps):
        if save_frames:
            save_observation(output_dir, step, obs, args.save_format)
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
                "auto_satisfied": current_task_auto_satisfied,
                "episode_position": tasks_since_reset,
            }
            task_records.append(record)
            task_step_counter = 0
            task_return = 0.0
            tasks_since_reset += 1
            task_frame_ranges.append((record, current_task_start, step))
            current_task_start = step + 1
            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                episode_index += 1
                reset_seed = args.seed + 100_003 * episode_index
                obs, info = env.reset(seed=reset_seed)
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

    antic_metrics: Optional[dict] = None
    if args.tasks_per_reset > 0 and task_records:
        antic_metrics = compute_anticipation_metrics(task_records, args.tasks_per_reset)
        _print_anticipation_report(antic_metrics, run_label)
        _plot_anticipation_eval(antic_metrics, output_dir)
        with (output_dir / "anticipation_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(antic_metrics, fh, indent=2, default=str)

    report = {
        "run_label": run_label,
        "checkpoint": str(state_path),
        "stats": stats,
        "anticipation": antic_metrics,
        "tasks": task_records,
    }
    with (output_dir / "rollout_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    summary = (
        f"[{run_label}] saved rollout → {output_dir} "
        f"| success {success_rate:.1%} ({stats['successes']}/{stats['tasks_attempted']}) "
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
            f"[{run_label}] Step budget expired with an unfinished task "
            f"({task_step_counter} steps so far); not counted in stats."
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
        task_frame_ranges.append((pending_record, current_task_start, args.total_steps - 1))
    if task_frame_ranges:
        print(f"[{run_label}] Frame ranges per sampled task:")
        for record, start_idx, end_idx in task_frame_ranges:
            label = _describe_record(record)
            print(f"  Task {record['task_number']:>4}: frames {start_idx}–{end_idx} | {label}")
    else:
        print(f"[{run_label}] No completed tasks to report frame ranges for.")

    return report


def _plot_comparison(anticipatory: Dict[str, Any], myopic: Dict[str, Any], out_path: Path) -> None:
    ant_stats = anticipatory["stats"]
    myo_stats = myopic["stats"]
    ant_a = anticipatory.get("anticipation") or {}
    myo_a = myopic.get("anticipation") or {}

    ant_vals = [
        ant_stats["success_rate"],
        ant_stats["avg_task_steps"],
        float(ant_a.get("anticipation_delta", 0.0)),
    ]
    myo_vals = [
        myo_stats["success_rate"],
        myo_stats["avg_task_steps"],
        float(myo_a.get("anticipation_delta", 0.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    titles = ("Success rate", "Avg steps / task", "Anticipation Δ (signed)")
    ylabels = ("Fraction", "Steps", "Δ auto-rate (later − baseline)")
    for i, (ax, title, ylab) in enumerate(zip(axes, titles, ylabels)):
        ax.bar(
            [0, 1],
            [ant_vals[i], myo_vals[i]],
            color=["#2563eb", "#ea580c"],
            alpha=0.9,
            width=0.55,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Anticipatory", "Myopic"], rotation=12)
        ax.set_title(title, fontweight="600")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.35)
        if i == 0:
            ax.set_ylim(0, 1.05)
        if i == 2:
            ax.axhline(0.0, color="#71717a", linewidth=0.9, linestyle="--")
    fig.suptitle("Frozen weights — same task RNG & env reset seeds", fontsize=12, fontweight="600")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    print(f"Saved comparison figure → {out_path}")


def run_compare(args: argparse.Namespace) -> None:
    ant_path = args.anticipatory_weights.expanduser().resolve()
    myo_path = args.myopic_weights.expanduser().resolve()
    base_out = args.output_dir
    if base_out is None:
        base_out = Path("runs") / "compare_image_dqn_infer"
    base_out = base_out.expanduser().resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    save_frames = not args.no_save_frames
    ant_report = run_single_rollout(
        args,
        ant_path,
        base_out / "anticipatory",
        run_label="anticipatory",
        save_frames=save_frames,
    )
    myo_report = run_single_rollout(
        args,
        myo_path,
        base_out / "myopic",
        run_label="myopic",
        save_frames=save_frames,
    )

    ant_s = ant_report["stats"]
    myo_s = myo_report["stats"]
    ant_a = ant_report.get("anticipation") or {}
    myo_a = myo_report.get("anticipation") or {}

    comparison = {
        "seed": args.seed,
        "total_steps": args.total_steps,
        "tasks_per_reset": args.tasks_per_reset,
        "anticipatory_checkpoint": str(ant_path),
        "myopic_checkpoint": str(myo_path),
        "anticipatory": {"stats": ant_s, "anticipation": ant_report.get("anticipation")},
        "myopic": {"stats": myo_s, "anticipation": myo_report.get("anticipation")},
        "delta_anticipatory_minus_myopic": {
            "success_rate": float(ant_s["success_rate"] - myo_s["success_rate"]),
            "avg_task_steps": float(ant_s["avg_task_steps"] - myo_s["avg_task_steps"]),
            "anticipation_delta": float(
                (ant_a.get("anticipation_delta") or 0.0) - (myo_a.get("anticipation_delta") or 0.0)
            ),
            "baseline_auto_rate": float(
                (ant_a.get("baseline_auto_rate") or 0.0) - (myo_a.get("baseline_auto_rate") or 0.0)
            ),
        },
    }
    cmp_path = base_out / "comparison.json"
    with cmp_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    print(f"Wrote comparison JSON → {cmp_path}")

    _plot_comparison(ant_report, myo_report, base_out / "comparison_summary.png")

    d = comparison["delta_anticipatory_minus_myopic"]
    print()
    print("=" * 60)
    print(" SUMMARY (anticipatory − myopic)")
    print("=" * 60)
    print(f"  Δ success rate      : {d['success_rate']:+.4f}")
    print(f"  Δ avg steps / task  : {d['avg_task_steps']:+.2f}  (negative is better if anticipatory)")
    print(f"  Δ anticipation Δ    : {d['anticipation_delta']:+.4f}")
    print(f"  Δ luck baseline     : {d['baseline_auto_rate']:+.4f}")
    print("=" * 60)


def run_single(args: argparse.Namespace) -> None:
    state_path = args.state_dict.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = state_path.parent / "infer"
    run_single_rollout(
        args,
        state_path,
        output_dir,
        run_label="single",
        save_frames=not args.no_save_frames,
    )


def main() -> None:
    args = parse_args()
    if args.compare_mode:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
