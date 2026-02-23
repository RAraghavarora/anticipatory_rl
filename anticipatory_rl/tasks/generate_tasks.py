"""CLI to dump sampled tasks into a JSON file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anticipatory_rl.tasks.generator import generate_task_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample tasks and write them to disk as JSON.")
    parser.add_argument("--count", type=int, default=1000, help="Number of tasks to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "tasks_1000.json",
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = generate_task_sequence(count=args.count, seed=args.seed)
    payload = [asdict(t) for t in tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {len(payload)} tasks to {args.output}")


if __name__ == "__main__":
    main()
