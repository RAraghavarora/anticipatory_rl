"""Privileged-state scripted evaluation for SimpleGrid image tasks.

This compares two hand-coded controllers on the same fixed task sequence:

1. myopic: clear tasks are solved with minimum immediate effort
2. anticipatory: clear tasks trade immediate effort against expected next-task value

The goal is diagnostic, not fair policy learning. The controllers read the full
environment state directly to estimate whether the benchmark itself contains a
meaningful anticipatory advantage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from anticipatory_rl.envs.simple_grid_image_env import (
    OBJECT_NAMES,
    SimpleGridImageEnv,
)

Coord = Tuple[int, int]


@dataclass
class SampledTask:
    task_type: str
    object_name: Optional[str]
    receptacle_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scripted myopic vs anticipatory policies on fixed SimpleGrid task sequences."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "simple_grid_scripted_eval")
    parser.add_argument("--num-tasks", type=int, default=1_000)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-objects", type=int, default=len(OBJECT_NAMES))
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--correct-pick-bonus", type=float, default=1.0)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--max-task-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--clear-task-prob", type=float, default=None)
    parser.add_argument(
        "--clear-receptacle-shaping-scale",
        type=float,
        default=2.0,
        help="Per-object reward for clear when objects leave the target surface.",
    )
    parser.add_argument(
        "--ensure-receptacle-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize resets so each receptacle starts with at least one object when feasible.",
    )
    parser.add_argument(
        "--tasks-per-reset",
        type=int,
        default=1_000,
        help="Episode length in tasks before a deterministic reset.",
    )
    parser.add_argument(
        "--tasks-per-sequence",
        type=int,
        default=100,
        help="Tasks sampled ahead per refill of the task buffer.",
    )
    parser.add_argument(
        "--anticipation-weight",
        type=float,
        default=1.0,
        help="Multiplier on the scripted future-value term for the anticipatory clear policy.",
    )
    parser.add_argument(
        "--use-env-task-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the environment's internal task process instead of an external iid task list.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> SimpleGridImageEnv:
    return SimpleGridImageEnv(
        grid_size=args.grid_size,
        max_task_steps=args.max_task_steps,
        num_objects=args.num_objects,
        success_reward=args.success_reward,
        correct_pick_bonus=args.correct_pick_bonus,
        distance_reward=True,
        distance_reward_scale=args.distance_reward_scale,
        clear_receptacle_shaping_scale=args.clear_receptacle_shaping_scale,
        clear_task_prob=args.clear_task_prob,
        ensure_receptacle_coverage=args.ensure_receptacle_coverage,
        config_path=args.config_path,
    )


def weighted_choice(
    distribution: Mapping[str, float],
    candidates: Sequence[str],
    rng: np.random.Generator,
) -> str:
    if not candidates:
        raise ValueError("Cannot sample from an empty candidate set.")
    weights = np.array(
        [max(distribution.get(name, 0.0), 0.0) for name in candidates],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0:
        return str(rng.choice(candidates))
    probs = weights / total
    return str(rng.choice(candidates, p=probs))


def _eligible_receptacles(
    receptacles: Sequence[str],
    last_sampled: Optional[str],
) -> List[str]:
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
        target_dist = object_source_distribution.get(obj, surface_distribution)
        rec_choices = _eligible_receptacles(receptacles, last_rec)
        rec = weighted_choice(target_dist, rec_choices, rng)
        tasks.append(SampledTask("move", obj, rec))
        last_rec = rec
    return tasks


def apply_sampled_task(env: SimpleGridImageEnv, task: SampledTask):
    env.set_task(
        task.task_type,
        task.object_name,
        task.receptacle_name,
        task_source="external",
    )
    env._task_steps = 0  # noqa: SLF001
    return env._obs(), env._info()


def compute_anticipation_metrics(task_records: List[dict], episode_len: int) -> dict:
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
        success_rate_by_pos[pos] = float(
            np.mean([1.0 if r["success"] else 0.0 for r in recs])
        )
        avg_steps_by_pos[pos] = float(np.mean([r["steps"] for r in recs]))
        n_by_pos[pos] = len(recs)

    auto_recs = [r for r in task_records if r.get("auto_satisfied", False)]
    active_recs = [r for r in task_records if not r.get("auto_satisfied", False)]

    def _safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    active_solved_steps = [r["steps"] for r in active_recs if r["success"]]

    return {
        "episode_len": episode_len,
        "mean_auto_rate": _safe_mean(list(auto_rate_by_pos.values())),
        "overall_auto_rate": float(len(auto_recs)) / max(1, len(task_records)),
        "auto_rate_by_pos": {str(k): v for k, v in auto_rate_by_pos.items()},
        "success_rate_by_pos": {str(k): v for k, v in success_rate_by_pos.items()},
        "avg_steps_by_pos": {str(k): v for k, v in avg_steps_by_pos.items()},
        "n_by_pos": {str(k): v for k, v in n_by_pos.items()},
        "n_tasks_total": len(task_records),
        "n_auto_tasks": len(auto_recs),
        "n_active_tasks": len(active_recs),
        "auto_success_rate": _safe_mean(
            [1.0 if r["success"] else 0.0 for r in auto_recs]
        ),
        "active_success_rate": _safe_mean(
            [1.0 if r["success"] else 0.0 for r in active_recs]
        ),
        "active_avg_steps_when_solved": _safe_mean(active_solved_steps),
    }


def _safe_prob(
    distribution: Mapping[str, float],
    candidates: Sequence[str],
    name: str,
) -> float:
    weights = np.array(
        [max(distribution.get(candidate, 0.0), 0.0) for candidate in candidates],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0:
        return 1.0 / max(1, len(candidates)) if name in candidates else 0.0
    return max(distribution.get(name, 0.0), 0.0) / total


def _distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _best_tile_for_receptacle(
    env: SimpleGridImageEnv,
    receptacle_name: str,
    from_coord: Coord,
) -> Coord:
    tiles = list(env.receptacles[receptacle_name])
    return min(tiles, key=lambda tile: (_distance(from_coord, tile), tile[1], tile[0]))


def _objects_on_receptacle(env: SimpleGridImageEnv, receptacle_name: str) -> List[str]:
    tiles = set(env.receptacles[receptacle_name])
    return sorted(
        [
            name
            for name, coord in env.state.objects.items()
            if name in env.active_objects
            and name != env.state.carrying
            and coord in tiles
        ]
    )


def _object_position(env: SimpleGridImageEnv, object_name: str) -> Coord:
    if env.state.carrying == object_name:
        return env.state.agent
    coord = env.state.objects.get(object_name)
    if coord is None:
        raise KeyError(f"Object not found in state: {object_name}")
    return coord


def _next_step_towards(start: Coord, goal: Coord) -> int:
    sx, sy = start
    gx, gy = goal
    if sx < gx:
        return SimpleGridImageEnv.MOVE_RIGHT
    if sx > gx:
        return SimpleGridImageEnv.MOVE_LEFT
    if sy < gy:
        return SimpleGridImageEnv.MOVE_DOWN
    if sy > gy:
        return SimpleGridImageEnv.MOVE_UP
    return SimpleGridImageEnv.MOVE_UP


def _immediate_relocation_cost(
    env: SimpleGridImageEnv,
    object_name: str,
    dest_receptacle: str,
    *,
    carrying: bool,
) -> int:
    dest_tile = _best_tile_for_receptacle(env, dest_receptacle, env.state.agent)
    if carrying:
        return _distance(env.state.agent, dest_tile) + 1
    obj_pos = _object_position(env, object_name)
    return (
        _distance(env.state.agent, obj_pos)
        + 1
        + _distance(obj_pos, dest_tile)
        + 1
    )


def _candidate_destinations(env: SimpleGridImageEnv) -> List[str]:
    return [
        rec
        for rec in env.receptacle_names
        if rec != env.target_receptacle
    ]


def _choose_myopic_object_and_dest(env: SimpleGridImageEnv) -> Tuple[str, str]:
    target_objects = _objects_on_receptacle(env, env.target_receptacle)
    if not target_objects:
        raise RuntimeError("Myopic clear policy asked to move from an already-empty target.")
    candidates: List[Tuple[int, str, str]] = []
    for obj_name in target_objects:
        for dest_rec in _candidate_destinations(env):
            cost = _immediate_relocation_cost(
                env,
                obj_name,
                dest_rec,
                carrying=False,
            )
            candidates.append((cost, obj_name, dest_rec))
    _, obj_name, dest_rec = min(candidates)
    return obj_name, dest_rec


def _choose_lowest_cost_object_for_dest(
    env: SimpleGridImageEnv,
    dest_rec: str,
) -> str:
    target_objects = _objects_on_receptacle(env, env.target_receptacle)
    if not target_objects:
        raise RuntimeError("Clear policy asked to move from an already-empty target.")
    candidates = [
        (
            _immediate_relocation_cost(
                env,
                obj_name,
                dest_rec,
                carrying=False,
            ),
            obj_name,
        )
        for obj_name in target_objects
    ]
    _, obj_name = min(candidates)
    return obj_name


def _choose_myopic_dest_for_carried(env: SimpleGridImageEnv, object_name: str) -> str:
    candidates: List[Tuple[int, str]] = []
    for dest_rec in _candidate_destinations(env):
        cost = _immediate_relocation_cost(env, object_name, dest_rec, carrying=True)
        candidates.append((cost, dest_rec))
    _, dest_rec = min(candidates)
    return dest_rec


def _anticipated_future_score(
    env: SimpleGridImageEnv,
    object_name: str,
    dest_receptacle: str,
    *,
    anticipation_weight: float,
) -> float:
    p_clear = float(np.clip(env.clear_task_prob, 0.0, 1.0))
    next_recs = _eligible_receptacles(env.receptacle_names, env.target_receptacle)
    p_next_clear = p_clear * _safe_prob(
        env.surface_distribution,
        next_recs,
        dest_receptacle,
    )
    target_distribution = env.object_source_distribution.get(
        object_name,
        env.surface_distribution,
    )
    p_next_move_auto = (1.0 - p_clear) * _safe_prob(
        env.object_distribution,
        env.active_objects,
        object_name,
    ) * _safe_prob(target_distribution, next_recs, dest_receptacle)

    dest_was_empty = len(_objects_on_receptacle(env, dest_receptacle)) == 0
    clear_block_penalty = env.success_reward if dest_was_empty else 0.0
    move_bonus = env.success_reward
    return anticipation_weight * (
        move_bonus * p_next_move_auto - clear_block_penalty * p_next_clear
    )


def _choose_anticipatory_object_and_dest(
    env: SimpleGridImageEnv,
    *,
    anticipation_weight: float,
) -> Tuple[str, str]:
    target_objects = _objects_on_receptacle(env, env.target_receptacle)
    if not target_objects:
        raise RuntimeError(
            "Anticipatory clear policy asked to move from an already-empty target."
        )
    best_key: Tuple[float, int, str, str] | None = None
    best_pair: Tuple[str, str] | None = None
    for obj_name in target_objects:
        for dest_rec in _candidate_destinations(env):
            immediate_cost = _immediate_relocation_cost(
                env,
                obj_name,
                dest_rec,
                carrying=False,
            )
            future_score = _anticipated_future_score(
                env,
                obj_name,
                dest_rec,
                anticipation_weight=anticipation_weight,
            )
            total_score = future_score - float(immediate_cost)
            key = (total_score, -immediate_cost, obj_name, dest_rec)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (obj_name, dest_rec)
    if best_pair is None:
        raise RuntimeError("No anticipatory clear move candidates found.")
    return best_pair


def _choose_anticipatory_dest_for_carried(
    env: SimpleGridImageEnv,
    object_name: str,
    *,
    anticipation_weight: float,
) -> str:
    best_key: Tuple[float, int, str] | None = None
    best_dest: str | None = None
    for dest_rec in _candidate_destinations(env):
        immediate_cost = _immediate_relocation_cost(
            env,
            object_name,
            dest_rec,
            carrying=True,
        )
        future_score = _anticipated_future_score(
            env,
            object_name,
            dest_rec,
            anticipation_weight=anticipation_weight,
        )
        total_score = future_score - float(immediate_cost)
        key = (total_score, -immediate_cost, dest_rec)
        if best_key is None or key > best_key:
            best_key = key
            best_dest = dest_rec
    if best_dest is None:
        raise RuntimeError("No anticipatory destination candidates found.")
    return best_dest


def scripted_action(
    env: SimpleGridImageEnv,
    policy_label: str,
    *,
    anticipation_weight: float,
) -> int:
    if getattr(env, "_pending_auto_success", False):
        return SimpleGridImageEnv.MOVE_UP

    if env.task_type == "move":
        target_object = env.target_object
        if target_object is None:
            return SimpleGridImageEnv.MOVE_UP
        if env.state.carrying == target_object:
            goal = _best_tile_for_receptacle(
                env,
                env.target_receptacle,
                env.state.agent,
            )
            if env.state.agent == goal:
                return SimpleGridImageEnv.PLACE
            return _next_step_towards(env.state.agent, goal)
        target_pos = _object_position(env, target_object)
        if env.state.agent == target_pos and env.state.carrying is None:
            return SimpleGridImageEnv.PICK
        return _next_step_towards(env.state.agent, target_pos)

    carrying = env.state.carrying
    if carrying is not None:
        if policy_label == "anticipatory" and env.task_mode == "paired_clear_followup":
            dest_rec = env.paired_followup_targets.get(
                env.target_receptacle,
                env.target_receptacle,
            )
        elif policy_label == "myopic":
            dest_rec = _choose_myopic_dest_for_carried(env, carrying)
        else:
            dest_rec = _choose_anticipatory_dest_for_carried(
                env,
                carrying,
                anticipation_weight=anticipation_weight,
            )
        goal = _best_tile_for_receptacle(env, dest_rec, env.state.agent)
        if env.state.agent == goal:
            return SimpleGridImageEnv.PLACE
        return _next_step_towards(env.state.agent, goal)

    if policy_label == "anticipatory" and env.task_mode == "paired_clear_followup":
        dest_rec = env.paired_followup_targets.get(
            env.target_receptacle,
            env.target_receptacle,
        )
        obj_name = _choose_lowest_cost_object_for_dest(env, dest_rec)
    elif policy_label == "myopic":
        obj_name, _ = _choose_myopic_object_and_dest(env)
    else:
        obj_name, _ = _choose_anticipatory_object_and_dest(
            env,
            anticipation_weight=anticipation_weight,
        )
    obj_pos = _object_position(env, obj_name)
    if env.state.agent == obj_pos:
        return SimpleGridImageEnv.PICK
    return _next_step_towards(env.state.agent, obj_pos)


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    stats = report["stats"]
    anticipation = report.get("anticipation") or {}
    return {
        "label": report["run_label"],
        "num_tasks": int(stats["tasks_attempted"]),
        "success_rate": float(stats["success_rate"]),
        "avg_task_steps": float(stats["avg_task_steps"]),
        "avg_task_return": float(stats["avg_task_return"]),
        "auto_rate": float(anticipation.get("overall_auto_rate", 0.0)),
        "auto_success_rate": float(anticipation.get("auto_success_rate", 0.0)),
        "active_success_rate": float(anticipation.get("active_success_rate", 0.0)),
        "active_attempted": int(anticipation.get("n_active_tasks", 0)),
        "auto_attempted": int(anticipation.get("n_auto_tasks", 0)),
        "primitive_steps": int(stats["total_steps"]),
        "reward_per_step": float(stats["reward_per_step"]),
        "discounted_return": float(stats["discounted_return"]),
    }


def run_policy(
    args: argparse.Namespace,
    *,
    policy_label: str,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    env.reset(seed=args.seed)
    use_env_task_process = bool(args.use_env_task_process)

    if not use_env_task_process and args.tasks_per_sequence <= 0:
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

    current_task: Optional[SampledTask]
    if use_env_task_process:
        current_task = None
    else:
        current_task = dequeue_task()
        _, _ = apply_sampled_task(env, current_task)
    current_task_auto_satisfied = bool(getattr(env, "_pending_auto_success", False))

    stats = {
        "total_steps": 0,
        "tasks_requested": int(args.num_tasks),
        "tasks_attempted": 0,
        "successes": 0,
        "failures": 0,
    }
    tasks_since_reset = 0
    episode_index = 0
    task_step_counter = 0
    task_return = 0.0
    task_records: List[dict] = []
    total_reward = 0.0
    discounted_return = 0.0
    discount = 1.0
    max_steps = None if args.total_steps is None or args.total_steps <= 0 else int(args.total_steps)
    progress = tqdm(total=args.num_tasks, desc=f"Scripted [{policy_label}]", unit="task")

    step = 0
    while stats["tasks_attempted"] < args.num_tasks:
        if max_steps is not None and step >= max_steps:
            break
        task_snapshot = SampledTask(
            env.task_type if use_env_task_process else current_task.task_type,
            env.target_object if use_env_task_process else current_task.object_name,
            env.target_receptacle if use_env_task_process else current_task.receptacle_name,
        )
        task_auto_snapshot = current_task_auto_satisfied
        action = scripted_action(
            env,
            policy_label,
            anticipation_weight=float(args.anticipation_weight),
        )
        _, reward, success, truncated, _ = env.step(action)
        stats["total_steps"] += 1
        task_step_counter += 1
        task_return += float(reward)
        total_reward += float(reward)
        discounted_return += discount * float(reward)
        discount *= float(args.gamma)
        step += 1

        if success or truncated:
            stats["tasks_attempted"] += 1
            progress.update(1)
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            task_records.append(
                {
                    "task_number": stats["tasks_attempted"],
                    "task_type": task_snapshot.task_type,
                    "target_object": task_snapshot.object_name,
                    "target_receptacle": task_snapshot.receptacle_name,
                    "success": bool(success),
                    "steps": task_step_counter,
                    "return": task_return,
                    "auto_satisfied": task_auto_snapshot,
                    "episode_position": tasks_since_reset,
                }
            )
            task_step_counter = 0
            task_return = 0.0
            if success:
                tasks_since_reset += 1
            reset_required = bool(truncated)
            if args.tasks_per_reset > 0 and tasks_since_reset >= args.tasks_per_reset:
                reset_required = True
            if reset_required:
                episode_index += 1
                reset_seed = args.seed + 100_003 * episode_index
                env.reset(seed=reset_seed)
                tasks_since_reset = 0
            if not use_env_task_process:
                current_task = dequeue_task()
                _, _ = apply_sampled_task(env, current_task)
            current_task_auto_satisfied = bool(getattr(env, "_pending_auto_success", False))

    progress.close()
    stats["success_rate"] = stats["successes"] / max(1, stats["tasks_attempted"])
    task_steps = [rec["steps"] for rec in task_records]
    task_returns = [rec["return"] for rec in task_records]
    stats["avg_task_steps"] = float(np.mean(task_steps)) if task_steps else 0.0
    stats["median_task_steps"] = float(np.median(task_steps)) if task_steps else 0.0
    stats["avg_task_return"] = float(np.mean(task_returns)) if task_returns else 0.0
    stats["cumulative_reward"] = float(total_reward)
    stats["reward_per_step"] = float(total_reward) / max(1, stats["total_steps"])
    stats["discounted_return"] = float(discounted_return)

    anticipation = compute_anticipation_metrics(task_records, args.tasks_per_reset)
    report = {
        "run_label": policy_label,
        "stats": stats,
        "anticipation": anticipation,
        "tasks": task_records,
    }
    with (output_dir / f"{policy_label}_rollout.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    summary = summarize_report(report)
    with (output_dir / f"{policy_label}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    if stats["tasks_attempted"] < args.num_tasks:
        print(
            f"[{policy_label}] stopped early after {stats['total_steps']} primitive steps "
            f"with {stats['tasks_attempted']}/{args.num_tasks} tasks evaluated."
        )
    return report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    myopic = run_policy(args, policy_label="myopic", output_dir=output_dir)
    anticipatory = run_policy(args, policy_label="anticipatory", output_dir=output_dir)

    comparison = {
        "seed": int(args.seed),
        "num_tasks": int(args.num_tasks),
        "tasks_per_reset": int(args.tasks_per_reset),
        "anticipation_weight": float(args.anticipation_weight),
        "myopic": summarize_report(myopic),
        "anticipatory": summarize_report(anticipatory),
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    print()
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
