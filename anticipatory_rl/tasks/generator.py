"""Random task generator for symbolic rearrangement tasks.

Each sampled task specifies only the desired goal configuration (e.g., move
specific objects to a target receptacle, or clear a receptacle entirely).
Actual initial object placements are left unspecified so that downstream code
can randomize them independently when constructing problem files or simulator
episodes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

OBJECTS = [
    "water_bottle",
    "tiffin_box",
    "apple",
    "soda_can",
    "drinking_glass",
]

RECEPTACLES = [
    "kitchen_table",
    "kitchen_counter",
    "dining_table",
    "study_table",
    "shelf",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_config(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


CONFIG = _load_config(CONFIG_PATH)
OBJECT_SOURCE_DIST = CONFIG.get("object_source_distribution", {})


@dataclass
class Task:
    task_type: str
    instructions: str
    payload: Dict[str, List[str] | str]


def _weighted_choice(distribution: Dict[str, float], candidates: Sequence[str]) -> str:
    valid = [c for c in candidates if c in distribution and distribution[c] > 0]
    if valid:
        weights = [distribution[c] for c in valid]
        return random.choices(valid, weights=weights, k=1)[0]
    if not candidates:
        raise ValueError("No candidates available for weighted choice")
    return random.choice(list(candidates))


def _weighted_sample_without_replacement(
    distribution: Dict[str, float], candidates: Sequence[str], k: int
) -> List[str]:
    choices = []
    remaining = list(candidates)
    dist = distribution.copy()
    for _ in range(min(k, len(remaining))):
        pick = _weighted_choice(dist, remaining)
        choices.append(pick)
        remaining.remove(pick)
        dist.pop(pick, None)
    if len(choices) < k:
        raise ValueError("Not enough unique objects available to sample")
    return choices


def _pick_target_receptacle(exclude: Sequence[str] = ()) -> str:
    choices = [name for name in RECEPTACLES if name not in exclude]
    if not choices:
        raise ValueError("No valid receptacles left to choose from")
    return random.choice(choices)


def generate_bring_single_task(object_dist: Dict[str, float]) -> Task:
    obj = _weighted_choice(object_dist, OBJECTS)
    target = _pick_target_receptacle()
    instructions = f"Move {obj} to {target}."
    payload = {"objects": [obj], "target": target}
    return Task(task_type="bring_single", instructions=instructions, payload=payload)


def generate_bring_pair_task(object_dist: Dict[str, float]) -> Task:
    objs = _weighted_sample_without_replacement(object_dist, OBJECTS, k=2)
    target = _pick_target_receptacle()
    instructions = f"Move {objs[0]} and {objs[1]} to {target}."
    payload = {"objects": objs, "target": target}
    return Task(task_type="bring_pair", instructions=instructions, payload=payload)


def generate_clear_receptacle_task(surface_dist: Dict[str, float]) -> Task:
    source = _weighted_choice(surface_dist, RECEPTACLES)
    instructions = f"Clear {source} completely."
    payload = {"source": source}
    return Task(task_type="clear_receptacle", instructions=instructions, payload=payload)


def _sample_task_category(task_dist: Dict[str, float]) -> str:
    if not task_dist:
        return random.choice(["bring_single", "bring_pair", "clear_receptacle"])
    categories = list(task_dist.keys())
    weights = [task_dist[cat] for cat in categories]
    return random.choices(categories, weights=weights, k=1)[0]


def generate_task_sequence(count: int = 10, seed: int | None = None) -> List[Task]:
    if seed is not None:
        random.seed(seed)
    tasks: List[Task] = []
    task_dist = CONFIG.get("task_distribution", {})
    object_dist = CONFIG.get("object_distribution", {})
    surface_dist = CONFIG.get("surface_distribution", {})
    for _ in range(count):
        task_type = _sample_task_category(task_dist)
        if task_type == "bring_single":
            task = generate_bring_single_task(object_dist)
        elif task_type == "bring_pair":
            task = generate_bring_pair_task(object_dist)
        elif task_type == "clear_receptacle":
            task = generate_clear_receptacle_task(surface_dist)
        else:
            raise ValueError(f"Unsupported task type '{task_type}'")
        tasks.append(task)
    return tasks


def main() -> None:
    tasks = generate_task_sequence()
    for idx, task in enumerate(tasks, start=1):
        print(f"Task {idx}: {task.task_type}")
        print(f"  {task.instructions}")
        print(f"  Payload: {asdict(task)['payload']}")
        print()


if __name__ == "__main__":
    main()
