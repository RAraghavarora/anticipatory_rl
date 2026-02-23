"""Label abstract object-placement states with planner-derived costs."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from anticipatory_rl.tasks.build_problem_from_task import (
    CONFIG as PROBLEM_CONFIG,
    _parse_template,
    build_problem_text_for_task,
    load_tasks,
)
from anticipatory_rl.tasks.generator import OBJECTS, RECEPTACLES, OBJECT_SOURCE_DIST
from anticipatory_rl.tasks.planner_utils import plan_cost, run_planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label abstract states with averaged planner costs.")
    parser.add_argument("--num-states", type=int, default=200, help="Number of abstract states to sample.")
    parser.add_argument("--tasks-per-state", type=int, default=3, help="K tasks per state for averaging.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("runs") / "tasks_1000.json",
        help="Pre-sampled task dataset.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--domain",
        type=Path,
        default=Path("pddl") / "gridworld_domain.pddl",
        help="Domain PDDL path.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("pddl") / "gridworld_problem.pddl",
        help="Problem template path.",
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=Path("downward") / "fast-downward.py",
        help="Planner entry point.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="astar(lmcut())",
        help="Planner search configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "phi_labels.npz",
        help="Destination npz file for states/labels.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("runs") / "phi_labels_meta.json",
        help="Optional JSON metadata (placements per state).",
    )
    return parser.parse_args()


def sample_abstract_state(rng: random.Random) -> Dict[str, str]:
    placements: Dict[str, str] = {}
    for obj in OBJECTS:
        dist = OBJECT_SOURCE_DIST.get(obj, {})
        placements[obj] = weighted_choice(rng, dist, RECEPTACLES)
    return placements


def weighted_choice(rng: random.Random, dist: Dict[str, float], candidates: Sequence[str]) -> str:
    valid = [c for c in candidates if dist.get(c, 0.0) > 0]
    if valid:
        weights = [dist[c] for c in valid]
        return rng.choices(valid, weights=weights, k=1)[0]
    return rng.choice(list(candidates))


def state_to_vector(placements: Dict[str, str]) -> np.ndarray:
    vec = np.zeros(len(OBJECTS) * len(RECEPTACLES), dtype=np.float32)
    for obj_idx, obj in enumerate(OBJECTS):
        receptacle = placements[obj]
        rec_idx = RECEPTACLES.index(receptacle)
        vec[obj_idx * len(RECEPTACLES) + rec_idx] = 1.0
    return vec


def task_trivially_satisfied(task: dict, placements: Dict[str, str]) -> bool:
    """Return True if the task goal already holds under the sampled placements."""
    task_type = task.get("task_type")
    payload = task.get("payload", {})

    if task_type in {"bring_single", "bring_pair"}:
        target = payload.get("target")
        objects = payload.get("objects", [])
        if not target or not objects:
            return False
        return all(placements.get(obj) == target for obj in objects)

    if task_type == "clear_receptacle":
        source = payload.get("source")
        if not source:
            return False
        return all(region != source for region in placements.values())

    return False


def main() -> None:
    args = parse_args()
    args.domain = args.domain.resolve()
    args.template = args.template.resolve()
    args.planner = args.planner.resolve()
    rng = random.Random(args.seed)
    tasks = load_tasks(args.tasks_file)
    template = _parse_template(args.template)
    surface_dist = PROBLEM_CONFIG.get("surface_distribution", {})

    state_vectors: List[np.ndarray] = []
    labels: List[float] = []
    placements_dump: List[Dict[str, str]] = []

    from tqdm import tqdm
    for idx in tqdm(range(args.num_states), desc="Labeling states"):
        placements = sample_abstract_state(rng)
        costs: List[float] = []
        task_attempts = 0
        print(f"Sampled state {idx} with placements: {placements}")
        while len(costs) < args.tasks_per_state:
            task = rng.choice(tasks)
            task_name = f"state-{idx}-task-{len(costs)}"
            print(f"Labeling state {idx}, task attempt {task_attempts}, task {task}...")
            # If the sampled task is already satisfied by the abstract placements, treat
            # it as zero-cost without invoking the planner.
            if task_trivially_satisfied(task, placements):
                costs.append(0.0)
                continue
            try:
                problem_text = build_problem_text_for_task(
                    task,
                    template,
                    task_name,
                    surface_dist=surface_dist,
                    rng=random.Random(rng.randint(0, 10**9)),
                    placements=placements,
                )
            except ValueError:
                task_attempts += 1
                if task_attempts > args.tasks_per_state * 5:
                    raise RuntimeError(
                        f"Unable to find compatible tasks for state {idx}; "
                        "consider resampling placements or tasks."
                    )
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                problem_path = tmp_path / "problem.pddl"
                problem_path.write_text(problem_text)
                plan_path = run_planner(
                    args.planner, args.domain, problem_path, args.search, tmp_path
                )
                cost = plan_cost(plan_path)
                costs.append(float(cost))

        state_vectors.append(state_to_vector(placements))
        labels.append(sum(costs) / len(costs))
        placements_dump.append(placements)

    arr_states = np.stack(state_vectors, axis=0)
    arr_labels = np.asarray(labels, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        states=arr_states,
        labels=arr_labels,
        objects=np.array(OBJECTS),
        receptacles=np.array(RECEPTACLES),
    )
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(placements_dump, indent=2))
    print(f"Wrote {len(state_vectors)} labeled states to {args.output}")


if __name__ == "__main__":
    main()
