"""Random task generator for simple pick-and-place scenarios.

The script defines:
- A list of receptacles and the objects initially stored inside each one.
- Three canonical task types (bring one object, bring two objects, clear a receptacle).
- A generator that samples a sequence of 10 tasks drawn from these templates.

Run this file directly to print a freshly sampled task list.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

RECEPTACLES: Dict[str, List[str]] = {
    "kitchen_table": [],
    "kitchen_counter": ["water_bottle", "tiffin_box"],
    "dining_table": ["apple", "soda_can"],
    "study_table": ["drinking_glass"],
    "shelf": [],
}

TASK_TYPES = ["bring_single", "bring_pair", "clear_receptacle"]

# Precompute the home receptacle for each object for quick lookup.
OBJECT_HOME: Dict[str, str] = {}
for receptacle, objects in RECEPTACLES.items():
    for obj in objects:
        OBJECT_HOME[obj] = receptacle


@dataclass
class Task:
    task_type: str
    instructions: str
    payload: Dict[str, List[str] | str]


def _pick_target_receptacle(exclude: Sequence[str] = ()) -> str:
    choices = [name for name in RECEPTACLES.keys() if name not in exclude]
    if not choices:
        raise ValueError("No valid receptacles left to choose from")
    return random.choice(choices)


def generate_bring_single_task() -> Task:
    obj = random.choice(list(OBJECT_HOME.keys()))
    source = OBJECT_HOME[obj]
    target = _pick_target_receptacle(exclude=[source])
    instructions = f"Move {obj} from {source} to {target}."
    payload = {"objects": [obj], "source": source, "target": target}
    return Task(task_type="bring_single", instructions=instructions, payload=payload)


def generate_bring_pair_task() -> Task:
    if len(OBJECT_HOME) < 2:
        raise ValueError("Need at least two objects to create a pair task")
    objs = random.sample(list(OBJECT_HOME.keys()), 2)
    sources = [OBJECT_HOME[obj] for obj in objs]
    target = _pick_target_receptacle(exclude=set(sources))
    instructions = (
        f"Move {objs[0]} (from {sources[0]}) and {objs[1]} (from {sources[1]}) into {target}."
    )
    payload = {"objects": objs, "sources": sources, "target": target}
    return Task(task_type="bring_pair", instructions=instructions, payload=payload)


def generate_clear_receptacle_task() -> Task:
    filled = [name for name, items in RECEPTACLES.items() if items]
    if not filled:
        raise ValueError("No receptacles contain objects to clear")
    receptacle = random.choice(filled)
    target = _pick_target_receptacle(exclude=[receptacle])
    instructions = (
        f"Remove every object from {receptacle} and relocate them to {target}."
    )
    payload = {
        "source": receptacle,
        "target": target,
        "objects": RECEPTACLES[receptacle][:],
    }
    return Task(task_type="clear_receptacle", instructions=instructions, payload=payload)


TASK_GENERATORS = {
    "bring_single": generate_bring_single_task,
    "bring_pair": generate_bring_pair_task,
    "clear_receptacle": generate_clear_receptacle_task,
}


def generate_task_sequence(count: int = 10, seed: int | None = None) -> List[Task]:
    if seed is not None:
        random.seed(seed)
    tasks: List[Task] = []
    for _ in range(count):
        task_type = random.choice(TASK_TYPES)
        task = TASK_GENERATORS[task_type]()
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
