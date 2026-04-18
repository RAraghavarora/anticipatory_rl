"""Inference for symbolic restaurant DQN checkpoints.

Supports single-checkpoint evaluation or side-by-side comparison of anticipatory
and myopic policies on matched environment reset seeds and task RNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from anticipatory_rl.agents.restaurant_dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant_symbolic_env import RestaurantSymbolicEnv, TASK_TYPES


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_action_from_q(
    q_values_row: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(q_values_row).item())
    logits = q_values_row / temperature
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0.0:
        return int(torch.argmax(q_values_row).item())
    draw = torch.multinomial(probs.detach().float().cpu(), 1, generator=generator)
    return int(draw.item())


def _action_policy_label(temperature: float) -> str:
    return "argmax (greedy)" if temperature <= 0.0 else f"softmax (tau={float(temperature):g})"


def _load_layout_corpus(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and isinstance(payload.get("layouts"), list):
        return [x for x in payload["layouts"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError(f"Unsupported layout corpus format in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for restaurant DQN checkpoints; compare anticipatory vs myopic if both are provided."
    )
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--anticipatory-weights", type=Path, default=None)
    parser.add_argument("--myopic-weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-tasks", type=int, default=5_000)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument("--tasks-per-reset", type=int, default=200)
    parser.add_argument("--task-sequence-length", type=int, default=40)
    parser.add_argument("--eval-layout-count", type=int, default=0, help="When >0, evaluate this many layout sequences.")
    parser.add_argument("--max-task-steps", type=int, default=24)
    parser.add_argument("--success-reward", type=float, default=15.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=6.0)
    parser.add_argument("--travel-cost-scale", type=float, default=25.0)
    parser.add_argument("--pick-cost", type=float, default=25.0)
    parser.add_argument("--place-cost", type=float, default=25.0)
    parser.add_argument("--wash-cost", type=float, default=25.0)
    parser.add_argument("--fill-cost", type=float, default=25.0)
    parser.add_argument("--brew-cost", type=float, default=25.0)
    parser.add_argument("--fruit-cost", type=float, default=25.0)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"),
    )
    parser.add_argument("--layout-corpus", type=Path, default=None, help="Optional JSON layout corpus.")
    parser.add_argument("--layout-id", type=str, default="", help="Evaluate only one layout_id from corpus.")
    parser.add_argument("--sample-layout-per-reset", action="store_true", help="Sample random layout each sequence reset.")
    parser.add_argument(
        "--task-library-per-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-layout task_library when available.",
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


def make_env(args: argparse.Namespace) -> RestaurantSymbolicEnv:
    return RestaurantSymbolicEnv(
        config_path=args.config_path,
        max_task_steps=args.max_task_steps,
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


def _load_model(
    state_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    *,
    input_dim: int,
    action_dim: int,
) -> RestaurantQNetwork:
    model = RestaurantQNetwork(
        input_dim=int(input_dim),
        action_dim=int(action_dim),
        hidden_dim=args.hidden_dim,
    ).to(device)
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _summarize_by_task_type(task_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for task_type in TASK_TYPES:
        rows = [r for r in task_records if r.get("task_type") == task_type]
        if not rows:
            continue
        succ = [r for r in rows if r["success"]]
        summary[task_type] = {
            "count": float(len(rows)),
            "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in rows])),
            "auto_rate": float(np.mean([1.0 if r["auto_satisfied"] else 0.0 for r in rows])),
            "avg_steps": float(np.mean([r["steps"] for r in rows])),
            "avg_return": float(np.mean([r["return"] for r in rows])),
            "avg_paper2_cost": float(np.mean([r.get("paper2_cost", 0.0) for r in rows])),
            "avg_steps_when_success": float(np.mean([r["steps"] for r in succ])) if succ else 0.0,
        }
    return summary


def _print_report(report: Dict[str, Any], run_label: str) -> None:
    stats = report["stats"]
    print()
    print("=" * 54)
    print(f"         RESTAURANT EVAL -- {run_label}")
    print("=" * 54)
    print(f"  Success rate                : {stats['success_rate']:.1%}")
    print(f"  Overall auto-rate           : {stats['auto_rate']:.1%}")
    print(f"  Avg steps / task            : {stats['avg_task_steps']:.2f}")
    print(f"  Avg task return             : {stats['avg_task_return']:.3f}")
    print(f"  Reward / step               : {stats['reward_per_step']:.3f}")
    print(f"  Avg paper2 cost / task      : {stats['avg_task_paper2_cost']:.3f}")
    print(f"  paper2 cumulative cost      : {stats['paper2_cost_total']:.3f}")
    print(f"  Discounted return           : {stats['discounted_return']:.3f}")
    print(f"  Tasks attempted             : {stats['tasks_attempted']}")
    print(f"  Primitive steps             : {stats['total_steps']}")
    print()
    print("  Per-task-type:")
    for task_type, vals in report["by_task_type"].items():
        print(
            f"    {task_type:<18} n={int(vals['count']):5d} "
            f"success={vals['success_rate']:.1%} auto={vals['auto_rate']:.1%} "
            f"steps={vals['avg_steps']:.2f} ret={vals['avg_return']:.2f} p2={vals['avg_paper2_cost']:.2f}"
        )


def _plot_comparison(
    anticipatory: Dict[str, Any],
    myopic: Dict[str, Any],
    out_path: Path,
    *,
    action_policy_line: str,
) -> None:
    ant_stats = anticipatory["stats"]
    myo_stats = myopic["stats"]
    ant_vals = [
        ant_stats["success_rate"],
        ant_stats["avg_task_steps"],
        ant_stats["reward_per_step"],
        ant_stats["auto_rate"],
    ]
    myo_vals = [
        myo_stats["success_rate"],
        myo_stats["avg_task_steps"],
        myo_stats["reward_per_step"],
        myo_stats["auto_rate"],
    ]
    titles = ("Success rate", "Avg steps / task", "Reward / step", "Auto rate")
    ylabels = ("Fraction", "Steps", "Reward", "Fraction")
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2), constrained_layout=True)
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
        if i in {0, 3}:
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"Restaurant DQN -- {action_policy_line}", fontsize=12, fontweight="600")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    print(f"Saved comparison figure -> {out_path}")


def evaluate(
    state_path: Path,
    *,
    run_label: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    env = make_env(args)
    layout_pool = _load_layout_corpus(args.layout_corpus)
    if args.layout_id:
        layout_pool = [x for x in layout_pool if str(x.get("layout_id")) == args.layout_id]
        if not layout_pool:
            raise ValueError(f"layout-id '{args.layout_id}' not found in {args.layout_corpus}")
    layout_rng = np.random.default_rng(args.seed + 42_001)

    target_tasks = int(args.num_tasks)
    if args.eval_layout_count > 0:
        target_tasks = int(args.eval_layout_count) * max(1, int(args.task_sequence_length))

    sequence_index = 0
    sequence_task_count = 0

    def _reset_env(reset_seed: int, seq_index: int):
        options: Dict[str, Any] = {}
        if layout_pool:
            if args.sample_layout_per_reset:
                layout = layout_pool[int(layout_rng.integers(0, len(layout_pool)))]
            else:
                layout = layout_pool[seq_index % len(layout_pool)]
            options["layout"] = layout
            if args.task_library_per_layout and isinstance(layout.get("task_library"), list):
                options["task_library"] = layout.get("task_library")
        if options:
            return env.reset(seed=reset_seed, options=options)
        return env.reset(seed=reset_seed)

    obs, info = _reset_env(args.seed, sequence_index)
    model = _load_model(
        state_path,
        args,
        device,
        input_dim=int(np.asarray(obs).shape[0]),
        action_dim=int(env.action_space.n),
    )

    total_reward = 0.0
    discounted_return = 0.0
    discount = 1.0
    total_steps = 0
    total_tasks = 0
    successes = 0
    task_return = 0.0
    task_steps = 0
    task_paper2_cost = 0.0
    tasks_since_reset = 0
    episode_index = 0
    current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))
    task_records: List[Dict[str, Any]] = []
    action_gen = torch.Generator(device="cpu")
    action_gen.manual_seed(args.seed)

    max_steps = None if args.total_steps <= 0 else int(args.total_steps)
    progress = tqdm(total=target_tasks, desc=f"Inference [{run_label}]", unit="task")

    while total_tasks < target_tasks:
        if max_steps is not None and total_steps >= max_steps:
            break
        task_snapshot = dict(info.get("task", {}))
        auto_snapshot = bool(current_task_auto_satisfied)
        layout_snapshot = info.get("layout_id")
        valid_mask = np.asarray(info.get("valid_action_mask"), dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_tensor).squeeze(0)
            invalid = torch.tensor(valid_mask <= 0.0, dtype=torch.bool, device=device)
            if not bool(invalid.all().item()):
                q_values = q_values.clone()
                q_values[invalid] = float("-inf")
            action = _sample_action_from_q(
                q_values,
                temperature=float(args.softmax_temperature),
                generator=action_gen,
            )

        obs, reward, success, truncated, info = env.step(action)
        total_steps += 1
        task_steps += 1
        task_return += float(reward)
        task_paper2_cost += float(info.get("paper2_cost_step", 0.0))
        total_reward += float(reward)
        discounted_return += discount * float(reward)
        discount *= float(args.gamma)

        if success or truncated:
            total_tasks += 1
            progress.update(1)
            if success:
                successes += 1
                tasks_since_reset += 1
            task_records.append(
                {
                    "task_number": total_tasks,
                    "task_type": task_snapshot.get("task_type"),
                    "target_location": task_snapshot.get("target_location"),
                    "target_kind": task_snapshot.get("target_kind"),
                    "success": bool(success),
                    "truncated": bool(truncated),
                    "steps": int(task_steps),
                    "return": float(task_return),
                    "paper2_cost": float(task_paper2_cost),
                    "auto_satisfied": auto_snapshot,
                    "layout_id": layout_snapshot,
                }
            )
            task_return = 0.0
            task_paper2_cost = 0.0
            task_steps = 0
            sequence_task_count += 1
            reset_required = bool(truncated)
            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                reset_required = True
            if args.task_sequence_length > 0 and sequence_task_count >= args.task_sequence_length:
                reset_required = True
                sequence_task_count = 0
                sequence_index += 1
            if reset_required:
                episode_index += 1
                obs, info = _reset_env(args.seed + 100_003 * episode_index, sequence_index)
                tasks_since_reset = 0
            current_task_auto_satisfied = bool(info.get("next_auto_satisfied", False))

    progress.close()

    stats = {
        "tasks_attempted": int(total_tasks),
        "successes": int(successes),
        "success_rate": float(successes / max(1, total_tasks)),
        "avg_task_steps": float(np.mean([r["steps"] for r in task_records])) if task_records else 0.0,
        "avg_task_return": float(np.mean([r["return"] for r in task_records])) if task_records else 0.0,
        "reward_per_step": float(total_reward / max(1, total_steps)),
        "paper2_cost_total": float(np.sum([r.get("paper2_cost", 0.0) for r in task_records])) if task_records else 0.0,
        "avg_task_paper2_cost": float(np.mean([r.get("paper2_cost", 0.0) for r in task_records])) if task_records else 0.0,
        "discounted_return": float(discounted_return),
        "auto_rate": float(np.mean([1.0 if r["auto_satisfied"] else 0.0 for r in task_records])) if task_records else 0.0,
        "total_steps": int(total_steps),
        "cumulative_reward": float(total_reward),
    }
    report = {
        "run_label": run_label,
        "checkpoint": str(state_path),
        "stats": stats,
        "by_task_type": _summarize_by_task_type(task_records),
        "tasks": task_records,
        "action_policy": _action_policy_label(float(args.softmax_temperature)),
        "task_sequence_length": int(args.task_sequence_length),
        "eval_layout_count": int(args.eval_layout_count),
        "layout_corpus": None if args.layout_corpus is None else str(args.layout_corpus),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rollout_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    _print_report(report, run_label)
    if total_tasks < target_tasks:
        print(
            f"[{run_label}] Stopped early after {total_steps} primitive steps "
            f"with {total_tasks}/{target_tasks} tasks evaluated."
        )
    return report


def run_compare(args: argparse.Namespace) -> None:
    device = _select_device()
    output_dir = args.output_dir or (Path("runs") / "compare_restaurant_dqn_infer")
    ant_dir = output_dir / "anticipatory"
    myo_dir = output_dir / "myopic"
    anticipatory = evaluate(
        args.anticipatory_weights.expanduser().resolve(),
        run_label="anticipatory",
        args=args,
        device=device,
        output_dir=ant_dir,
    )
    myopic = evaluate(
        args.myopic_weights.expanduser().resolve(),
        run_label="myopic",
        args=args,
        device=device,
        output_dir=myo_dir,
    )
    ant_stats = anticipatory["stats"]
    myo_stats = myopic["stats"]
    comparison = {
        "action_policy": _action_policy_label(float(args.softmax_temperature)),
        "seed": int(args.seed),
        "num_tasks": int(args.num_tasks),
        "tasks_per_reset": int(args.tasks_per_reset),
        "anticipatory": anticipatory,
        "myopic": myopic,
        "delta": {
            "success_rate": float(ant_stats["success_rate"] - myo_stats["success_rate"]),
            "avg_task_steps": float(ant_stats["avg_task_steps"] - myo_stats["avg_task_steps"]),
            "avg_task_return": float(ant_stats["avg_task_return"] - myo_stats["avg_task_return"]),
            "avg_task_paper2_cost": float(ant_stats["avg_task_paper2_cost"] - myo_stats["avg_task_paper2_cost"]),
            "paper2_cost_total": float(ant_stats["paper2_cost_total"] - myo_stats["paper2_cost_total"]),
            "reward_per_step": float(ant_stats["reward_per_step"] - myo_stats["reward_per_step"]),
            "discounted_return": float(ant_stats["discounted_return"] - myo_stats["discounted_return"]),
            "auto_rate": float(ant_stats["auto_rate"] - myo_stats["auto_rate"]),
        },
    }
    comparison["delta"]["delta_success_rate"] = comparison["delta"]["success_rate"]
    comparison["delta"]["delta_avg_steps"] = comparison["delta"]["avg_task_steps"]
    comparison["delta"]["delta_avg_return"] = comparison["delta"]["avg_task_return"]
    comparison["delta"]["delta_reward_per_step"] = comparison["delta"]["reward_per_step"]
    comparison["delta"]["delta_auto_rate"] = comparison["delta"]["auto_rate"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    _plot_comparison(
        anticipatory,
        myopic,
        output_dir / "comparison.png",
        action_policy_line=_action_policy_label(float(args.softmax_temperature)),
    )
    print()
    print("=" * 60)
    print(" SUMMARY (anticipatory - myopic)")
    print(f"  Policy              : {_action_policy_label(float(args.softmax_temperature))}")
    print("=" * 60)
    print(f"  Delta success rate  : {comparison['delta']['success_rate']:+.4f}")
    print(f"  Delta avg steps     : {comparison['delta']['avg_task_steps']:+.4f}")
    print(f"  Delta avg return    : {comparison['delta']['avg_task_return']:+.4f}")
    print(f"  Delta avg paper2    : {comparison['delta']['avg_task_paper2_cost']:+.4f}")
    print(f"  Delta reward/step   : {comparison['delta']['reward_per_step']:+.4f}")
    print(f"  Delta disc return   : {comparison['delta']['discounted_return']:+.4f}")
    print(f"  Delta auto-rate     : {comparison['delta']['auto_rate']:+.4f}")
    print("=" * 60)


def run_single(args: argparse.Namespace) -> None:
    device = _select_device()
    state_path = args.state_dict.expanduser().resolve()
    output_dir = args.output_dir or (state_path.parent / "infer")
    evaluate(
        state_path,
        run_label=state_path.stem,
        args=args,
        device=device,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    if args.compare_mode:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
