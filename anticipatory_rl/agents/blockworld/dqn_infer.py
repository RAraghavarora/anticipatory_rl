"""Inference and comparison for paper1 blockworld image DQN checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from .image_dqn import ConvQNetwork
from anticipatory_rl.envs.blockworld.env import (
    Paper1BlockworldImageEnv,
    Task,
)


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


def _obs_to_rgb_uint8(obs: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(obs[:3], dtype=np.float32), 0.0, 1.0)
    return np.rint(rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8, copy=False)


def _save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames recorded; cannot save GIF.")
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / max(1, fps))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for paper1 blockworld image DQN checkpoints."
    )
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--anticipatory-weights", type=Path, default=None)
    parser.add_argument("--myopic-weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument(
        "--tasks-per-episode",
        type=int,
        default=None,
        help="Number of tasks per sequence/episode (alias: --tasks-per-reset).",
    )
    parser.add_argument("--tasks-per-reset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument("--task-library-size", type=int, default=20)
    parser.add_argument("--max-task-steps", type=int, default=64)
    parser.add_argument("--success-reward", type=float, default=12.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=5.0)
    parser.add_argument("--correct-pick-bonus", type=float, default=1.0)
    parser.add_argument("--tb-log-dir", type=Path, default=None)
    parser.add_argument("--trajectory-log-dir", type=Path, default=None)
    parser.add_argument("--trajectory-log-fps", type=int, default=6)
    parser.add_argument("--render-tile-px", type=int, default=24)
    parser.add_argument("--render-margin-px", type=int, default=None)
    parser.add_argument(
        "--procedural-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
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
            parser.error("Pass --state-dict or both compare checkpoints.")
        args.compare_mode = False
    return args


def make_env(args: argparse.Namespace) -> Paper1BlockworldImageEnv:
    return Paper1BlockworldImageEnv(
        task_library_size=args.task_library_size,
        max_task_steps=args.max_task_steps,
        success_reward=args.success_reward,
        step_penalty=args.step_penalty,
        invalid_action_penalty=args.invalid_action_penalty,
        correct_pick_bonus=args.correct_pick_bonus,
        render_tile_px=args.render_tile_px,
        render_margin_px=args.render_margin_px,
        procedural_layout=args.procedural_layout,
    )


def _load_model(state_path: Path, args: argparse.Namespace, device: torch.device) -> ConvQNetwork:
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    model = ConvQNetwork(obs.shape, hidden_dim=args.hidden_dim, num_actions=env.action_space.n).to(device)
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _apply_task(env: Paper1BlockworldImageEnv, task: Task):
    env.current_task = task
    env._task_steps = 0  # noqa: SLF001
    env._update_pending_auto_success()  # noqa: SLF001
    return env._obs(), env._info()


def _paper_step_cost(
    env: Paper1BlockworldImageEnv,
    pre_info: Dict[str, Any],
    post_info: Dict[str, Any],
    action: int,
    *,
    auto_before: bool,
) -> float:
    if auto_before:
        return 0.0
    if action in {
        Paper1BlockworldImageEnv.MOVE_UP,
        Paper1BlockworldImageEnv.MOVE_DOWN,
        Paper1BlockworldImageEnv.MOVE_LEFT,
        Paper1BlockworldImageEnv.MOVE_RIGHT,
    }:
        moved = tuple(pre_info["robot"]) != tuple(post_info["robot"])
        if not moved:
            return 0.0
        return float(env.config.move_cost)
    if action == Paper1BlockworldImageEnv.PICK:
        picked = pre_info.get("holding") is None and post_info.get("holding") is not None
        return float(env.config.pick_cost if picked else 0.0)
    if action == Paper1BlockworldImageEnv.PLACE:
        placed = pre_info.get("holding") is not None and post_info.get("holding") is None
        return float(env.config.place_cost if placed else 0.0)
    return 0.0


def _summarize_task_index_metrics(task_records: List[Dict[str, Any]], sequence_len: int) -> Dict[str, Dict[str, float]]:
    cost_by_pos: Dict[str, float] = {}
    auto_by_pos: Dict[str, float] = {}
    for idx in range(sequence_len):
        rows = [row for row in task_records if int(row.get("episode_position", -1)) == idx]
        cost_by_pos[str(idx)] = float(np.mean([row["paper_cost"] for row in rows])) if rows else 0.0
        auto_by_pos[str(idx)] = float(np.mean([1.0 if row["auto_satisfied"] else 0.0 for row in rows])) if rows else 0.0
    return {"cost_by_task_index": cost_by_pos, "auto_rate_by_task_index": auto_by_pos}


def _plot_comparison_summary(
    anticipatory: Dict[str, Any],
    myopic: Dict[str, Any],
    out_path: Path,
    *,
    action_policy_line: str,
) -> None:
    ant = anticipatory["stats"]
    myo = myopic["stats"]
    ant_vals = [
        ant["success_rate"],
        ant["avg_task_steps"],
        ant["avg_task_return"],
        ant["avg_task_cost"],
        ant["avg_total_sequence_cost"],
        ant["auto_rate"],
    ]
    myo_vals = [
        myo["success_rate"],
        myo["avg_task_steps"],
        myo["avg_task_return"],
        myo["avg_task_cost"],
        myo["avg_total_sequence_cost"],
        myo["auto_rate"],
    ]
    titles = (
        "Success rate",
        "Avg steps / task",
        "Avg return / task",
        "Avg paper cost",
        "Avg 10-task cost",
        "Auto rate",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        ax.bar([0, 1], [ant_vals[idx], myo_vals[idx]], color=["#2563eb", "#ea580c"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Anticipatory", "Myopic"], rotation=10)
        ax.set_title(titles[idx], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Paper1 Blockworld DQN -- {action_policy_line}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)


def _plot_cost_curve(
    anticipatory: Dict[str, Any],
    myopic: Dict[str, Any],
    out_path: Path,
) -> None:
    ant_curve = anticipatory["stats"]["cost_by_task_index"]
    myo_curve = myopic["stats"]["cost_by_task_index"]
    xs = [int(key) for key in ant_curve.keys()]
    ant_vals = [float(ant_curve[str(x)]) for x in xs]
    myo_vals = [float(myo_curve[str(x)]) for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ant_vals, marker="o", linewidth=1.8, color="#2563eb", label="Anticipatory")
    ax.plot(xs, myo_vals, marker="o", linewidth=1.8, color="#ea580c", label="Myopic")
    ax.set_title("Average paper cost by task index")
    ax.set_xlabel("Task index in 10-task sequence")
    ax.set_ylabel("Paper cost")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)


def evaluate(
    state_path: Path,
    *,
    run_label: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    tb_writer = None
    if args.tb_log_dir is not None:
        tb_log_dir = args.tb_log_dir / run_label
    else:
        tb_log_dir = output_dir / "tb"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    env = make_env(args)
    model = _load_model(state_path, args, device)
    action_gen = torch.Generator(device="cpu")
    action_gen.manual_seed(args.seed)
    task_rng = random.Random(args.seed + 1)

    total_steps = 0
    total_tasks = 0
    successes = 0
    horizon_tasks = 0
    task_return = 0.0
    task_steps = 0
    task_cost = 0.0
    total_reward = 0.0
    discounted_return = 0.0
    discount = 1.0
    task_records: List[Dict[str, Any]] = []
    sequence_costs: List[float] = []
    trajectory_log_dir = args.trajectory_log_dir
    collected_trajectories: Dict[str, Dict[str, Any] | None] = {
        "success": None,
        "failure": None,
    }

    sequence_count = 0
    progress = tqdm(
        total=args.num_sequences * args.tasks_per_episode,
        desc=f"Inference [{run_label}]",
        unit="task",
    )
    while sequence_count < args.num_sequences and total_steps < args.total_steps:
        reset_seed = args.seed + 100_003 * sequence_count
        obs, info = env.reset(seed=reset_seed)
        sequence_tasks = [task_rng.choice(env.task_library) for _ in range(args.tasks_per_episode)]
        obs, info = _apply_task(env, sequence_tasks[0])
        current_task_auto = bool(info.get("next_auto_satisfied", False))
        sequence_running_cost = 0.0

        for task_idx, _task in enumerate(sequence_tasks):
            trajectory_frames: List[np.ndarray] = [_obs_to_rgb_uint8(obs)]
            trajectory_meta: Dict[str, Any] = {
                "sequence_index": int(sequence_count),
                "task_index": int(task_idx),
                "task_assignments": [list(item) for item in env.current_task.assignments],
                "task_size": int(env.current_task.assignments.__len__()),
                "start_auto_satisfied": bool(current_task_auto),
            }
            while total_steps < args.total_steps:
                pre_info = info
                auto_snapshot = bool(current_task_auto)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(obs_tensor).squeeze(0)
                    action = _sample_action_from_q(
                        q_values,
                        temperature=float(args.softmax_temperature),
                        generator=action_gen,
                    )
                obs, reward, success, horizon, info = env.step(action)
                trajectory_frames.append(_obs_to_rgb_uint8(obs))
                total_steps += 1
                task_steps += 1
                task_return += float(reward)
                total_reward += float(reward)
                discounted_return += discount * float(reward)
                discount *= float(args.gamma)
                task_cost += _paper_step_cost(env, pre_info, info, int(action), auto_before=auto_snapshot)

                if success or horizon:
                    total_tasks += 1
                    progress.update(1)
                    if success:
                        successes += 1
                        trajectory_meta["outcome"] = "success"
                    if horizon:
                        horizon_tasks += 1
                        trajectory_meta["outcome"] = "failure"
                    if trajectory_log_dir is not None:
                        if bool(success) and collected_trajectories["success"] is None:
                            collected_trajectories["success"] = {
                                "frames": list(trajectory_frames),
                                "meta": dict(trajectory_meta),
                            }
                        elif bool(horizon) and collected_trajectories["failure"] is None:
                            collected_trajectories["failure"] = {
                                "frames": list(trajectory_frames),
                                "meta": dict(trajectory_meta),
                            }
                    task_records.append(
                        {
                            "task_number": total_tasks,
                            "task_assignments": [list(item) for item in pre_info.get("task_assignments", ())],
                            "task_size": int(pre_info.get("task_size", 0)),
                            "success": bool(success),
                            "horizon": bool(horizon),
                            "steps": int(task_steps),
                            "return": float(task_return),
                            "paper_cost": float(task_cost),
                            "auto_satisfied": auto_snapshot,
                            "episode_position": task_idx,
                        }
                    )
                    sequence_running_cost += float(task_cost)
                    task_return = 0.0
                    task_steps = 0
                    task_cost = 0.0
                    if task_idx + 1 < len(sequence_tasks):
                        obs, info = _apply_task(env, sequence_tasks[task_idx + 1])
                        current_task_auto = bool(info.get("next_auto_satisfied", False))
                    break
            if total_steps >= args.total_steps:
                break
            if trajectory_log_dir is not None and collected_trajectories["success"] is not None and collected_trajectories["failure"] is not None:
                break
        if total_steps >= args.total_steps:
            break
        if trajectory_log_dir is not None and collected_trajectories["success"] is not None and collected_trajectories["failure"] is not None:
            break
        sequence_costs.append(sequence_running_cost)
        sequence_count += 1

    progress.close()
    task_index_metrics = _summarize_task_index_metrics(task_records, args.tasks_per_episode)
    stats = {
        "tasks_attempted": int(total_tasks),
        "successes": int(successes),
        "success_rate": float(successes / max(1, total_tasks)),
        "avg_task_steps": float(np.mean([row["steps"] for row in task_records])) if task_records else 0.0,
        "avg_task_return": float(np.mean([row["return"] for row in task_records])) if task_records else 0.0,
        "avg_task_cost": float(np.mean([row["paper_cost"] for row in task_records])) if task_records else 0.0,
        "avg_total_sequence_cost": float(np.mean(sequence_costs)) if sequence_costs else 0.0,
        "auto_rate": float(np.mean([1.0 if row["auto_satisfied"] else 0.0 for row in task_records])) if task_records else 0.0,
        "horizon_rate": float(horizon_tasks / max(1, total_tasks)),
        "reward_per_step": float(total_reward / max(1, total_steps)),
        "discounted_return": float(discounted_return),
        "total_steps": int(total_steps),
        "num_sequences_completed": int(sequence_count),
        **task_index_metrics,
    }
    if tb_writer is not None:
        step = int(total_tasks)
        tb_writer.add_scalar("eval/success_rate", stats["success_rate"], step)
        tb_writer.add_scalar("eval/avg_task_steps", stats["avg_task_steps"], step)
        tb_writer.add_scalar("eval/avg_task_return", stats["avg_task_return"], step)
        tb_writer.add_scalar("eval/avg_task_cost", stats["avg_task_cost"], step)
        tb_writer.add_scalar("eval/avg_total_sequence_cost", stats["avg_total_sequence_cost"], step)
        tb_writer.add_scalar("eval/auto_rate", stats["auto_rate"], step)
        tb_writer.add_scalar("eval/horizon_rate", stats["horizon_rate"], step)
        tb_writer.add_scalar("eval/reward_per_step", stats["reward_per_step"], step)
        tb_writer.add_scalar("eval/discounted_return", stats["discounted_return"], step)
        for idx, value in stats["cost_by_task_index"].items():
            tb_writer.add_scalar(f"eval/cost_by_task_index/{idx}", value, step)
        for idx, value in stats["auto_rate_by_task_index"].items():
            tb_writer.add_scalar(f"eval/auto_rate_by_task_index/{idx}", value, step)
        tb_writer.flush()
        tb_writer.close()
    report = {
        "run_label": run_label,
        "checkpoint": str(state_path),
        "stats": stats,
        "tasks": task_records,
        "action_policy": _action_policy_label(float(args.softmax_temperature)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rollout_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    if trajectory_log_dir is not None:
        trajectory_log_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, str] = {}
        for kind, payload in collected_trajectories.items():
            if payload is None:
                continue
            frames = payload["frames"]
            meta = payload["meta"]
            gif_path = trajectory_log_dir / f"{kind}.gif"
            _save_gif(frames, gif_path, fps=int(args.trajectory_log_fps))
            with (trajectory_log_dir / f"{kind}.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)
            written[kind] = str(gif_path)
        with (trajectory_log_dir / "index.json").open("w", encoding="utf-8") as fh:
            json.dump({"written": written, "seed": int(args.seed), "checkpoint": str(state_path)}, fh, indent=2, default=str)
    return report


def run_compare(args: argparse.Namespace) -> None:
    device = _select_device()
    output_dir = args.output_dir or (Path("runs") / "compare_blockworld_image_dqn_infer")
    ant_report = evaluate(
        args.anticipatory_weights.expanduser().resolve(),
        run_label="anticipatory",
        args=args,
        device=device,
        output_dir=output_dir / "anticipatory",
    )
    myo_report = evaluate(
        args.myopic_weights.expanduser().resolve(),
        run_label="myopic",
        args=args,
        device=device,
        output_dir=output_dir / "myopic",
    )
    ant = ant_report["stats"]
    myo = myo_report["stats"]
    comparison = {
        "action_policy": _action_policy_label(float(args.softmax_temperature)),
        "seed": int(args.seed),
        "num_sequences": int(args.num_sequences),
        "tasks_per_episode": int(args.tasks_per_episode),
        "tasks_per_reset": int(args.tasks_per_episode),
        "anticipatory": ant_report,
        "myopic": myo_report,
        "delta": {
            "success_rate": float(ant["success_rate"] - myo["success_rate"]),
            "avg_task_steps": float(ant["avg_task_steps"] - myo["avg_task_steps"]),
            "avg_task_return": float(ant["avg_task_return"] - myo["avg_task_return"]),
            "avg_task_cost": float(ant["avg_task_cost"] - myo["avg_task_cost"]),
            "avg_total_sequence_cost": float(ant["avg_total_sequence_cost"] - myo["avg_total_sequence_cost"]),
            "auto_rate": float(ant["auto_rate"] - myo["auto_rate"]),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    _plot_comparison_summary(
        ant_report,
        myo_report,
        output_dir / "comparison_summary.png",
        action_policy_line=_action_policy_label(float(args.softmax_temperature)),
    )
    _plot_cost_curve(ant_report, myo_report, output_dir / "per_task_cost_curve.png")


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
    if args.tasks_per_episode is None and args.tasks_per_reset is None:
        args.tasks_per_episode = 10
    elif args.tasks_per_episode is None:
        args.tasks_per_episode = args.tasks_per_reset
    elif args.tasks_per_reset is not None and args.tasks_per_reset != args.tasks_per_episode:
        raise ValueError(
            "--tasks-per-episode and --tasks-per-reset must match when both are provided."
        )
    if args.compare_mode:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
