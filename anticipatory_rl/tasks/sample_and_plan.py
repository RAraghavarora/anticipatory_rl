"""Sample config-driven tasks, build 10x10 PDDL problems, and solve them."""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from anticipatory_rl.controllers.pddl_controller import FastDownwardPlanner

Coord = Tuple[int, int]

GRID_SIZE = 10
DEFAULT_OBJECTS = [
    "water_bottle",
    "tiffin_box",
    "apple",
    "soda_can",
    "drinking_glass",
    "banana",
    "book",
    "phone",
    "pen",
]
DEFAULT_RECEPTACLES = [
    "kitchen_table",
    "kitchen_counter",
    "dining_table",
    "study_table",
    "shelf",
]
RECEPTACLE_LAYOUTS: Dict[str, List[Coord]] = {
    "kitchen_table": [(x, y) for x in range(0, 3) for y in range(0, 3)],
    "kitchen_counter": [(x, y) for x in range(7, 10) for y in range(0, 4)],
    "dining_table": [(x, y) for x in range(7, 10) for y in range(7, 10)],
    "study_table": [(x, y) for x in range(0, 3) for y in range(7, 10)],
    "shelf": [(x, y) for x in range(4, 6) for y in range(3, 7)],
}
WALKWAY_TILES: List[Coord] = (
    [(x, 2) for x in range(GRID_SIZE)]
    + [(x, 7) for x in range(GRID_SIZE)]
    + [(2, y) for y in range(GRID_SIZE)]
    + [(7, y) for y in range(GRID_SIZE)]
    + [(5, y) for y in range(2, 7)]
    + [(6, y) for y in range(2, 7)]
)
ACCESSIBLE_TILES: List[Coord] = sorted(
    set(tile for tiles in RECEPTACLE_LAYOUTS.values() for tile in tiles) | set(WALKWAY_TILES)
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_config(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


CONFIG = _load_config(CONFIG_PATH)


def _loc_name(coord: Coord) -> str:
    x, y = coord
    return f"loc_{x}{y}"


def _all_locations() -> List[str]:
    return [_loc_name((x, y)) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]


def _adjacent_pairs() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    accessible = set(ACCESSIBLE_TILES)
    for x, y in ACCESSIBLE_TILES:
        loc = _loc_name((x, y))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in accessible:
                pairs.append((loc, _loc_name((nx, ny))))
    return pairs


ALL_LOCATIONS = _all_locations()
ADJACENT_PAIRS = _adjacent_pairs()


def _weighted_choice(rng: random.Random, distribution: Dict[str, float], candidates: Sequence[str]) -> str:
    valid = [c for c in candidates if distribution.get(c, 0.0) > 0]
    if valid:
        weights = [distribution[c] for c in valid]
        return rng.choices(valid, weights=weights, k=1)[0]
    if not candidates:
        raise ValueError("No candidates available for sampling.")
    return rng.choice(list(candidates))


def _assign_objects(
    rng: random.Random,
    object_names: Sequence[str],
    surfaces: Sequence[str],
    surface_dist: Dict[str, float],
) -> Tuple[Dict[Coord, List[str]], Dict[str, str]]:
    tile_stacks: Dict[Coord, List[str]] = {
        tile: []
        for surface in surfaces
        for tile in RECEPTACLE_LAYOUTS.get(surface, [])
    }
    object_regions: Dict[str, str] = {}
    for obj in object_names:
        placed = False
        for _ in range(100):
            surface = _weighted_choice(rng, surface_dist, surfaces)
            tiles = RECEPTACLE_LAYOUTS.get(surface)
            if not tiles:
                raise KeyError(f"No layout defined for receptacle '{surface}'")
            candidates = [tile for tile in tiles if len(tile_stacks[tile]) < 2]
            if not candidates:
                continue
            tile = rng.choice(candidates)
            tile_stacks[tile].append(obj)
            object_regions[obj] = surface
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Unable to place object '{obj}' without exceeding stack limits.")
    dense_stacks = {tile: stack for tile, stack in tile_stacks.items() if stack}
    return dense_stacks, object_regions


def _sample_move_task(
    rng: random.Random,
    object_dist: Dict[str, float],
    object_names: Sequence[str],
    object_regions: Dict[str, str],
    receptacles: Sequence[str],
    object_source_dist: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, str], bool]:
    obj = _weighted_choice(rng, object_dist, object_names)
    current_region = object_regions[obj]
    source_dist = object_source_dist.get(obj, {})
    candidates = [r for r in receptacles if r != current_region]
    if not candidates:
        raise ValueError("No alternative receptacles available for move task")
    target = _weighted_choice(rng, source_dist, candidates)
    task = {"task_type": "move", "object": obj, "target_receptacle": target}
    return task, True


def _sample_clear_task(
    rng: random.Random,
    surface_dist: Dict[str, float],
    receptacles: Sequence[str],
    object_regions: Dict[str, str],
) -> Tuple[Dict[str, str], bool]:
    occupied = {region for region in object_regions.values()}
    candidates = [r for r in receptacles if r in occupied]
    if not candidates:
        return {}, False
    target = _weighted_choice(rng, surface_dist, candidates)
    task = {"task_type": "clear", "source_receptacle": target}
    return task, True


def _sample_task(
    rng: random.Random,
    clear_prob: float,
    object_dist: Dict[str, float],
    surface_dist: Dict[str, float],
    object_names: Sequence[str],
    receptacles: Sequence[str],
    object_regions: Dict[str, str],
    object_source_dist: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    for _ in range(50):
        if rng.random() < clear_prob:
            task, ok = _sample_clear_task(rng, surface_dist, receptacles, object_regions)
        else:
            task, ok = _sample_move_task(
                rng, object_dist, object_names, object_regions, receptacles, object_source_dist
            )
        if ok:
            return task
    raise RuntimeError("Failed to sample a valid task after multiple attempts")


def _init_lines(
    tile_stacks: Dict[Coord, List[str]],
    object_regions: Dict[str, str],
    receptacles: Sequence[str],
) -> List[str]:
    lines = ["(agent-at klara loc_00)", "(handfree klara)"]
    for rec in receptacles:
        tiles = RECEPTACLE_LAYOUTS.get(rec, [])
        for coord in tiles:
            lines.append(f"(belongs {_loc_name(coord)} {rec})")
    for a, b in ADJACENT_PAIRS:
        lines.append(f"(adjacent {a} {b})")
    clear_locations = set(ALL_LOCATIONS)
    clear_objects: set[str] = set()
    for tile, stack in tile_stacks.items():
        loc = _loc_name(tile)
        if not stack:
            continue
        clear_locations.discard(loc)
        lines.append(f"(on {stack[0]} {loc})")
        lines.append(f"(in {stack[0]} {object_regions[stack[0]]})")
        for idx in range(1, len(stack)):
            lines.append(f"(on {stack[idx]} {stack[idx - 1]})")
            lines.append(f"(in {stack[idx]} {object_regions[stack[idx]]})")
        clear_objects.add(stack[-1])
    for loc in sorted(clear_locations):
        lines.append(f"(clear {loc})")
    for obj in sorted(clear_objects):
        lines.append(f"(clear {obj})")
    return lines


def _goal_lines(task: Dict[str, str]) -> List[str]:
    if task["task_type"] == "move":
        return [f"(in {task['object']} {task['target_receptacle']})"]
    if task["task_type"] == "clear":
        source = task["source_receptacle"]
        tiles = sorted(RECEPTACLE_LAYOUTS[source], key=_loc_name)
        return [f"(clear {_loc_name(tile)})" for tile in tiles]
    raise ValueError(f"Unsupported task type {task['task_type']}")


def _extract_plan_cost(plan_path: Path) -> int:
    text = plan_path.read_text().strip().splitlines()
    for line in reversed(text):
        stripped = line.strip().lower()
        if stripped.startswith("; cost"):
            parts = stripped.split()
            for idx, token in enumerate(parts):
                if token == "=" and idx + 1 < len(parts):
                    try:
                        return int(float(parts[idx + 1]))
                    except ValueError:
                        continue
            try:
                number = float(parts[2])
                return int(number)
            except (IndexError, ValueError):
                continue
    # Fallback: count action lines.
    return sum(1 for line in text if line.strip().startswith("("))


def _format_typed_block(names: Sequence[str], type_name: str) -> List[str]:
    lines: List[str] = []
    chunk = 8
    for idx in range(0, len(names), chunk):
        segment = " ".join(names[idx : idx + chunk])
        lines.append(f"    {segment} - {type_name}")
    return lines


def _build_problem_text(
    problem_name: str,
    object_names: Sequence[str],
    receptacles: Sequence[str],
    tile_stacks: Dict[Coord, List[str]],
    object_regions: Dict[str, str],
    task: Dict[str, str],
) -> str:
    objects_section = ["  (:objects", "    klara - agent"]
    objects_section.extend(_format_typed_block(ALL_LOCATIONS, "location"))
    objects_section.extend(_format_typed_block(receptacles, "receptacle"))
    objects_section.extend(_format_typed_block(object_names, "cargo"))
    objects_section.append("  )")

    init_body = "\n".join(f"    {line}" for line in _init_lines(tile_stacks, object_regions, receptacles))
    goal_facts = _goal_lines(task)
    goal_body = "\n".join(f"      {line}" for line in goal_facts)

    return "\n".join(
        [
            f"(define (problem {problem_name})",
            "  (:domain gridworld-rearrangement)",
            *objects_section,
            "  (:init",
            init_body,
            "  )",
            "  (:goal",
            "    (and",
            goal_body,
            "    )",
            "  )",
            ")",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample tasks from config, emit 10x10 PDDL problems, and plan them with Fast Downward.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of tasks to sample and solve.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for sampling placements and tasks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "sampled_plans",
        help="Directory where domain/problem/plan files will be written.",
    )
    parser.add_argument(
        "--domain",
        type=Path,
        default=Path("pddl") / "gridworld_domain.pddl",
        help="Domain PDDL template to copy per task.",
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=Path("downward") / "fast-downward.py",
        help="Fast Downward entry point.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="lazy_greedy([ff()], preferred=[ff()])",
        help="Fast Downward search configuration string.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel planner workers.",
    )
    return parser.parse_args()


def _solve_task_worker(payload: Dict[str, object]) -> Dict[str, object]:
    idx = payload["idx"]
    seed = payload["seed"]
    object_names = payload["object_names"]
    receptacles = payload["receptacles"]
    surface_dist = payload["surface_dist"]
    object_dist = payload["object_dist"]
    object_source_dist = payload["object_source_dist"]
    clear_prob = payload["clear_prob"]
    domain_text = payload["domain_text"]
    planner_path = Path(payload["planner_path"])
    search = payload["search"]
    output_dir = Path(payload["output_dir"])

    rng = random.Random(seed)
    tile_stacks, object_regions = _assign_objects(rng, object_names, receptacles, surface_dist)
    task = _sample_task(
        rng,
        clear_prob,
        object_dist,
        surface_dist,
        object_names,
        receptacles,
        object_regions,
        object_source_dist,
    )

    problem_name = f"task-{idx:04d}"
    task_dir = output_dir / f"task_{idx:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    domain_path = task_dir / "domain.pddl"
    domain_path.write_text(domain_text)
    problem_path = task_dir / "problem.pddl"
    plan_path = task_dir / "plan.soln"

    problem_text = _build_problem_text(problem_name, object_names, receptacles, tile_stacks, object_regions, task)
    problem_path.write_text(problem_text)

    planner = FastDownwardPlanner(planner_path, search=search)
    planner.plan(domain_path, problem_path, plan_path)
    cost = _extract_plan_cost(plan_path)

    rel_plan = plan_path.relative_to(output_dir)
    return {
        "problem": problem_name,
        "task": task,
        "object_regions": object_regions,
        "cost": cost,
        "plan_path": str(rel_plan),
        "idx": idx,
    }


def main() -> None:
    args = _parse_args()
    run_root = args.output
    run_root.mkdir(parents=True, exist_ok=True)

    object_dist = CONFIG.get("object_distribution", {})
    surface_dist = CONFIG.get("surface_distribution", {})
    object_source_dist = CONFIG.get("object_source_distribution", {})
    clear_prob = float(CONFIG.get("task_distribution", {}).get("clear_receptacle", 0.0))
    clear_prob = max(0.0, min(1.0, clear_prob))

    object_names = sorted(list(object_dist.keys()) or list(DEFAULT_OBJECTS))
    receptacles = sorted(list(surface_dist.keys()) or list(DEFAULT_RECEPTACLES))

    if len(object_names) != 9:
        raise ValueError(f"Expected 9 objects from config, found {len(object_names)}: {object_names}")
    if len(receptacles) != 5:
        raise ValueError(f"Expected 5 receptacles from config, found {len(receptacles)}: {receptacles}")

    for rec in receptacles:
        if rec not in RECEPTACLE_LAYOUTS:
            raise KeyError(f"Missing layout definition for receptacle '{rec}'")

    domain_text = args.domain.read_text()

    manifest: List[Dict[str, object]] = [None] * args.count
    total_cost = 0
    solved = 0

    payloads = [
        {
            "idx": idx,
            "seed": args.seed + idx,
            "object_names": object_names,
            "receptacles": receptacles,
            "surface_dist": surface_dist,
            "object_dist": object_dist,
            "object_source_dist": object_source_dist,
            "clear_prob": clear_prob,
            "domain_text": domain_text,
            "planner_path": str(args.planner),
            "search": args.search,
            "output_dir": str(run_root),
        }
        for idx in range(args.count)
    ]

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {executor.submit(_solve_task_worker, payload): payload["idx"] for payload in payloads}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - debugging aid
                print(f"[plan] task-{idx:04d}: FAILED with {exc}")
                raise
            manifest[idx] = {
                "problem": result["problem"],
                "task": result["task"],
                "object_regions": result["object_regions"],
                "cost": result["cost"],
            }
            total_cost += result["cost"]
            solved += 1
            print(f"[plan] {result['problem']}: wrote {result['plan_path']} (cost={result['cost']})", flush=True)

    missing = [idx for idx, entry in enumerate(manifest) if entry is None]
    if missing:
        raise RuntimeError(f"Missing manifest entries for tasks: {missing}")

    manifest_path = run_root / "tasks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    average = total_cost / solved if solved else 0.0
    print(f"Finished sampling {solved} tasks to {run_root.resolve()}")
    print(f"Total cost: {total_cost}")
    print(f"Average cost: {average:.3f}")


if __name__ == "__main__":
    main()
