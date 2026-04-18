#!/usr/bin/env python3
"""Validate generated Paper2-scale restaurant layout corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("layouts"), list):
        raise ValueError("Corpus must be a dict with a 'layouts' list.")
    return payload


def _fail(errors: List[str]) -> None:
    if not errors:
        return
    raise SystemExit("\n".join(["Validation failed:"] + [f"- {e}" for e in errors]))


def validate(args: argparse.Namespace) -> None:
    corpus = _load(args.corpus_path)
    layouts = corpus["layouts"]
    errors: List[str] = []
    expected_object_count: int | None = None
    expected_location_count: int | None = None

    if len(layouts) < args.min_layouts:
        errors.append(f"layouts={len(layouts)} < min_layouts={args.min_layouts}")

    for idx, layout in enumerate(layouts):
        lid = str(layout.get("layout_id", f"idx_{idx}"))
        rooms = layout.get("rooms", [])
        categories = layout.get("categories", [])
        task_library = layout.get("task_library", [])
        locations = layout.get("locations", [])
        object_specs = layout.get("object_specs", [])
        paper2 = layout.get("paper2_cost", {})

        if not isinstance(rooms, list) or len(rooms) != 2:
            errors.append(f"{lid}: expected exactly 2 rooms")
        if not isinstance(categories, list) or len(categories) < args.min_categories:
            errors.append(f"{lid}: categories={len(categories) if isinstance(categories, list) else 'n/a'} < {args.min_categories}")
        if not isinstance(task_library, list) or not (args.min_tasks <= len(task_library) <= args.max_tasks):
            errors.append(f"{lid}: task_library size {len(task_library) if isinstance(task_library, list) else 'n/a'} outside [{args.min_tasks}, {args.max_tasks}]")
        if not isinstance(locations, list) or len(locations) < 8:
            errors.append(f"{lid}: expected at least 8 locations")
        if not isinstance(object_specs, list) or len(object_specs) == 0:
            errors.append(f"{lid}: expected non-empty object_specs")

        if isinstance(locations, list):
            if expected_location_count is None:
                expected_location_count = len(locations)
            elif len(locations) != expected_location_count:
                errors.append(f"{lid}: location count {len(locations)} != baseline {expected_location_count}")
        if isinstance(object_specs, list):
            if expected_object_count is None:
                expected_object_count = len(object_specs)
            elif len(object_specs) != expected_object_count:
                errors.append(f"{lid}: object count {len(object_specs)} != baseline {expected_object_count}")

        move_cfg = paper2.get("move", {}) if isinstance(paper2, dict) else {}
        grid = move_cfg.get("grid", {}) if isinstance(move_cfg, dict) else {}
        if int(grid.get("width", -1)) != args.grid_width or int(grid.get("height", -1)) != args.grid_height:
            errors.append(f"{lid}: paper2 grid must be {args.grid_width}x{args.grid_height}")
        fixed = paper2.get("fixed_costs", {}) if isinstance(paper2, dict) else {}
        for key in ("pick", "place", "fill", "wash", "brew", "fruit"):
            if float(fixed.get(key, -1.0)) != float(args.fixed_cost):
                errors.append(f"{lid}: fixed cost '{key}' != {args.fixed_cost}")

        if args.head > 0 and idx + 1 >= args.head:
            break

    _fail(errors)
    print(
        "Validation passed | "
        f"layouts={len(layouts)} "
        f"min_categories={args.min_categories} "
        f"task_range=[{args.min_tasks},{args.max_tasks}] "
        f"grid={args.grid_width}x{args.grid_height}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Paper2-scale restaurant layout corpus.")
    parser.add_argument("--corpus-path", type=Path, default=Path("data/restaurant_layouts/paper2_scale_layouts.json"))
    parser.add_argument("--min-layouts", type=int, default=1000)
    parser.add_argument("--min-categories", type=int, default=25)
    parser.add_argument("--min-tasks", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--grid-width", type=int, default=10)
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--fixed-cost", type=float, default=100.0)
    parser.add_argument("--head", type=int, default=0, help="Validate only first N layouts (0 => all).")
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
