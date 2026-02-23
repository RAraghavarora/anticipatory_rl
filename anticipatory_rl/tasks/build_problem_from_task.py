"""Generate a PDDL problem with randomized init and task-conditioned goals."""

from __future__ import annotations

import argparse
import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

OBJECTS = [
    "water_bottle",
    "tiffin_box",
    "apple",
    "soda_can",
    "drinking_glass",
]

RECEPTACLE_TILES: Dict[str, List[Tuple[int, int]]] = {
    "kitchen_table": [(0, 0), (1, 0)],
    "kitchen_counter": [(1, 1), (2, 1), (1, 2), (2, 2)],
    "dining_table": [(4, 1), (4, 2), (5, 1), (5, 2)],
    "study_table": [(3, 3), (4, 3), (3, 4), (4, 4)],
    "shelf": [(0, 5), (1, 5)],
}

ALL_TILES = [(x, y) for x in range(6) for y in range(6)]
TILE_TO_RECEPTACLE = {
    tile: name for name, tiles in RECEPTACLE_TILES.items() for tile in tiles
}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a PDDL problem tailored to a specific sampled task."
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("runs") / "tasks_1000.json",
        help="Path to the JSON file containing the sampled tasks.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="0-based index of the task inside the tasks JSON.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("pddl") / "gridworld_problem.pddl",
        help="Base PDDL problem template to customize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("problem.pddl"),
        help="Destination path for the generated problem file.",
    )
    parser.add_argument(
        "--problem-name",
        type=str,
        default=None,
        help="Optional override for the `(define (problem ...))` identifier.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Seed controlling randomized initial placements (defaults to task index).",
    )
    return parser.parse_args()


def _load_tasks(path: Path) -> Sequence[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of tasks in {path}, got {type(data)}")
    return data


def load_tasks(path: Path) -> Sequence[dict]:
    """Public helper to load the sampled tasks list."""
    return _load_tasks(path)


def _load_config(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


CONFIG = _load_config(CONFIG_PATH)


def _loc_name(tile: Tuple[int, int]) -> str:
    x, y = tile
    return f"loc_{x}{y}"


def _weighted_choice(
    rng: random.Random, distribution: Dict[str, float], candidates: Sequence[str]
) -> str:
    valid = [c for c in candidates if distribution.get(c, 0.0) > 0]
    if valid:
        weights = [distribution[c] for c in valid]
        return rng.choices(valid, weights=weights, k=1)[0]
    if not candidates:
        raise ValueError("No candidates available for weighted choice")
    return rng.choice(list(candidates))


def _extract_section(text: str, start_token: str, end_token: str) -> str:
    start = text.index(start_token)
    end = text.index(end_token, start)
    return text[start:end]


@dataclass
class TemplateParts:
    header: str
    objects_block: str
    static_init_lines: List[str]

    def assemble(self, dynamic_facts: Sequence[str], goal_block: str, problem_name: str) -> str:
        init_lines = list(dynamic_facts)
        if self.static_init_lines:
            init_lines.append("")
            init_lines.extend(self.static_init_lines)
        init_body = "\n".join(f"    {line}" for line in init_lines)
        text = "".join(
            [
                self.header,
                self.objects_block,
                "  (:init\n",
                f"{init_body}\n",
                "  )\n",
                goal_block,
                "\n)\n",
            ]
        )
        return _replace_problem_name(text, problem_name)


def _replace_problem_name(text: str, problem_name: str) -> str:
    pattern = re.compile(r"\(define \(problem [^)]+\)")
    replacement = f"(define (problem {problem_name})"
    if not pattern.search(text):
        raise ValueError("Could not locate problem name in template.")
    return pattern.sub(replacement, text, count=1)


def _parse_template(path: Path) -> TemplateParts:
    text = path.read_text()
    objects_block = _extract_section(text, "  (:objects", "  (:init")
    header = text[: text.index("  (:objects")]
    init_block = _extract_section(text, "  (:init", "  (:goal")
    static_lines: List[str] = []
    for raw in init_block.splitlines()[1:]:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("(belongs") or line.startswith("(adjacent") or (
            line.startswith(";;") and ("region" in line or "adjacency" in line)
        ):
            static_lines.append(line)
    return TemplateParts(header=header, objects_block=objects_block, static_init_lines=static_lines)


def _build_goal_block(facts: Sequence[str]) -> str:
    fact_lines = "\n".join(f"      {fact}" for fact in facts)
    return "\n".join(
        [
            "  (:goal",
            "    (and",
            fact_lines,
            "    )",
            "  )",
        ]
    )


def _assign_objects(
    rng: random.Random,
    surface_dist: Dict[str, float],
    placements: Dict[str, str] | None = None,
) -> Tuple[Dict[Tuple[int, int], List[str]], Dict[str, str]]:
    if rng is None and placements is None:
        rng = random.Random()
    tile_stacks: Dict[Tuple[int, int], List[str]] = {}
    object_regions: Dict[str, str] = {}

    if placements:
        counters = {name: 0 for name in RECEPTACLE_TILES.keys()}
        for obj in OBJECTS:
            if obj not in placements:
                raise ValueError(f"Placement missing object '{obj}'")
            surface = placements[obj]
            if surface not in RECEPTACLE_TILES:
                raise ValueError(f"Unknown receptacle '{surface}' in placements")
            tiles = RECEPTACLE_TILES[surface]
            idx = counters[surface] % len(tiles)
            coord = tiles[idx]
            counters[surface] += 1
            tile_stacks.setdefault(coord, []).append(obj)
            object_regions[obj] = surface
        return tile_stacks, object_regions

    surfaces = list(RECEPTACLE_TILES.keys())
    for obj in rng.sample(OBJECTS, len(OBJECTS)):
        surface = _weighted_choice(rng, surface_dist, surfaces)
        tile = rng.choice(RECEPTACLE_TILES[surface])
        tile_stacks.setdefault(tile, []).append(obj)
        object_regions[obj] = surface
    return tile_stacks, object_regions


def _build_dynamic_facts(
    tile_stacks: Dict[Tuple[int, int], List[str]],
    object_regions: Dict[str, str],
) -> List[str]:
    facts: List[str] = []
    facts.append("(agent-at klara loc_00)")
    facts.append("(handfree klara)")

    clear_locations = {_loc_name(tile) for tile in ALL_TILES}
    clear_objects: set[str] = set()

    for tile, stack in tile_stacks.items():
        if not stack:
            continue
        loc = _loc_name(tile)
        clear_locations.discard(loc)
        facts.append(f"(on {stack[0]} {loc})")
        for idx in range(1, len(stack)):
            facts.append(f"(on {stack[idx]} {stack[idx - 1]})")
        clear_objects.add(stack[-1])
        for obj in stack:
            region = TILE_TO_RECEPTACLE.get(tile, object_regions[obj])
            facts.append(f"(in {obj} {region})")

    for loc in sorted(clear_locations):
        facts.append(f"(clear {loc})")
    for obj in sorted(clear_objects):
        facts.append(f"(clear {obj})")
    return facts


def _goal_facts(task: dict, object_regions: Dict[str, str]) -> List[str]:
    payload = task.get("payload", {})
    task_type = task.get("task_type")
    if task_type in {"bring_single", "bring_pair"}:
        objects = payload.get("objects", [])
        target = payload.get("target")
        if not target:
            raise ValueError(f"Task {task_type} missing target")
        return [f"(in {obj} {target})" for obj in objects]

    if task_type == "clear_receptacle":
        source = payload.get("source")
        if not source:
            raise ValueError("clear_receptacle task missing source")
        if not any(region == source for region in object_regions.values()):
            raise ValueError(f"No objects located on source receptacle '{source}' in init state")
        tiles = RECEPTACLE_TILES[source]
        return [f"(clear {_loc_name(tile)})" for tile in tiles]

    raise ValueError(f"Unsupported task type '{task_type}'")


def build_problem_text_for_task(
    task: dict,
    template: TemplateParts | Path,
    problem_name: str,
    *,
    surface_dist: Dict[str, float] | None = None,
    rng: random.Random | None = None,
    placements: Dict[str, str] | None = None,
) -> str:
    if not isinstance(template, TemplateParts):
        template = _parse_template(Path(template))
    if surface_dist is None:
        surface_dist = CONFIG.get("surface_distribution", {})
    rng = rng or random.Random()

    tile_stacks, object_regions = _assign_objects(rng, surface_dist, placements)

    if task["task_type"] == "clear_receptacle":
        source = task["payload"]["source"]
        if not any(region == source for region in object_regions.values()):
            raise ValueError(f"No objects located on source receptacle '{source}' for clear task.")

    dynamic = _build_dynamic_facts(tile_stacks, object_regions)
    goal = _goal_facts(task, object_regions)
    goal_block = _build_goal_block(goal)
    return template.assemble(dynamic, goal_block, problem_name)


def main() -> None:
    args = _parse_args()
    tasks = _load_tasks(args.tasks_file)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(
            f"task_index {args.task_index} is out of range for {len(tasks)} tasks"
        )
    task = tasks[args.task_index]
    problem_name = args.problem_name or f"task-{args.task_index}"

    init_seed = args.init_seed if args.init_seed is not None else args.task_index
    rng = random.Random(init_seed)

    updated_text = build_problem_text_for_task(
        task,
        _parse_template(args.template),
        problem_name,
        surface_dist=CONFIG.get("surface_distribution", {}),
        rng=rng,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(updated_text)
    print(
        f"Wrote problem for task #{args.task_index} "
        f"({task['task_type']}) to {args.output}"
    )


if __name__ == "__main__":
    main()
