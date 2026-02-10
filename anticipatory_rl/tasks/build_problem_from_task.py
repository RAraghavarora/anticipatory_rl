"""Generate a PDDL problem file for a sampled task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a PDDL problem tailored to a specific sampled task."
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("runs") / "tasks_200.json",
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
    return parser.parse_args()


def _load_tasks(path: Path) -> Sequence[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of tasks in {path}, got {type(data)}")
    return data


def load_tasks(path: Path) -> Sequence[dict]:
    """Public helper to load the sampled tasks list."""
    return _load_tasks(path)


def _goal_facts(task: dict) -> List[str]:
    task_type = task.get("task_type")
    payload = task.get("payload", {})

    if task_type in {"bring_single", "bring_pair"}:
        objects: Sequence[str] = payload.get("objects", [])
        target: str = payload.get("target")
        if not target:
            raise ValueError(f"Task {task_type} missing 'target'")
        if not objects:
            raise ValueError(f"Task {task_type} missing 'objects'")
        return [f"(in {obj} {target})" for obj in objects]

    if task_type == "clear_receptacle":
        target: str = payload.get("target")
        objects: Sequence[str] = payload.get("objects", [])
        if not target:
            raise ValueError("clear_receptacle task missing 'target'")
        if not objects:
            raise ValueError("clear_receptacle task missing 'objects'")
        return [f"(in {obj} {target})" for obj in objects]

    raise ValueError(f"Unsupported task type '{task_type}'")


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


def _replace_problem_name(text: str, problem_name: str) -> str:
    pattern = re.compile(r"\(define \(problem [^)]+\)")
    replacement = f"(define (problem {problem_name})"
    if not pattern.search(text):
        raise ValueError("Could not locate problem name in template.")
    return pattern.sub(replacement, text, count=1)


def _find_matching_paren(text: str, start_idx: int) -> int:
    if text[start_idx] != "(":
        raise ValueError("Expected '(' at start index")
    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx + 1
    raise ValueError("Unbalanced parentheses in template goal block.")


def _replace_goal_block(text: str, new_block: str) -> str:
    marker = "  (:goal"
    start = text.find(marker)
    if start == -1:
        raise ValueError("Could not locate goal block in template.")
    start = text.index("(", start)
    end = _find_matching_paren(text, start)
    return text[:start] + new_block + text[end:]


def build_problem_text_for_task(
    task: dict, template_path: Path, problem_name: str
) -> str:
    goal_facts = _goal_facts(task)
    template_text = Path(template_path).read_text()
    updated_text = _replace_problem_name(template_text, problem_name)
    goal_block = _build_goal_block(goal_facts)
    return _replace_goal_block(updated_text, goal_block)


def main() -> None:
    args = _parse_args()
    tasks = _load_tasks(args.tasks_file)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(
            f"task_index {args.task_index} is out of range for {len(tasks)} tasks"
        )
    task = tasks[args.task_index]
    problem_name = args.problem_name or f"task-{args.task_index}"

    updated_text = build_problem_text_for_task(task, args.template, problem_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(updated_text)
    print(
        f"Wrote problem for task #{args.task_index} "
        f"({task['task_type']}) to {args.output}"
    )


if __name__ == "__main__":
    main()
