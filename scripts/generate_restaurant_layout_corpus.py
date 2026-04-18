#!/usr/bin/env python3
"""Generate Paper2-scale symbolic restaurant layout corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _base_locations() -> List[Dict[str, Any]]:
    return [
        {"name": "kitchen_counter", "coord": [1, 1], "room": "kitchen"},
        {"name": "coffee_machine", "coord": [2, 1], "room": "kitchen"},
        {"name": "water_station", "coord": [2, 2], "room": "kitchen"},
        {"name": "fruit_station", "coord": [2, 3], "room": "kitchen"},
        {"name": "dish_rack", "coord": [1, 2], "room": "kitchen"},
        {"name": "sink", "coord": [1, 3], "room": "kitchen"},
        {"name": "pass_counter", "coord": [5, 5], "room": "serving"},
        {"name": "table_left", "coord": [7, 3], "room": "serving"},
        {"name": "bus_tub", "coord": [7, 5], "room": "serving"},
        {"name": "table_right", "coord": [7, 7], "room": "serving"},
    ]


def _category_to_kind(category: str) -> str:
    mapping = {
        "mug": "mug",
        "glass": "glass",
        "cup": "mug",
        "bowl": "bowl",
        "plate": "bowl",
        "saucer": "bowl",
        "jar": "glass",
        "pitcher": "glass",
        "carafe": "glass",
    }
    return mapping.get(category, "bowl")


def _build_categories() -> List[str]:
    return [
        "mug",
        "glass",
        "cup",
        "bowl",
        "plate",
        "saucer",
        "jar",
        "pitcher",
        "carafe",
        "teapot",
        "tray",
        "tumbler",
        "goblet",
        "flute",
        "ramekin",
        "soup_bowl",
        "dessert_bowl",
        "salad_bowl",
        "latte_mug",
        "espresso_cup",
        "water_glass",
        "coffee_mug",
        "fruit_bowl",
        "serving_bowl",
        "cleaning_bin",
    ]


def _jitter_locations(rng: np.random.Generator, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jittered: List[Dict[str, Any]] = []
    for loc in locations:
        x, y = int(loc["coord"][0]), int(loc["coord"][1])
        if loc["room"] == "kitchen":
            x = int(np.clip(x + int(rng.integers(-1, 2)), 0, 3))
            y = int(np.clip(y + int(rng.integers(-1, 2)), 0, 9))
        else:
            x = int(np.clip(x + int(rng.integers(-1, 2)), 5, 9))
            y = int(np.clip(y + int(rng.integers(-1, 2)), 0, 9))
        jittered.append({"name": loc["name"], "coord": [x, y], "room": loc["room"]})
    return jittered


def _build_object_specs(
    rng: np.random.Generator,
    categories: List[str],
    min_objects: int,
    max_objects: int,
) -> List[Dict[str, str]]:
    total = int(rng.integers(min_objects, max_objects + 1))
    specs: List[Dict[str, str]] = []
    for idx in range(total):
        category = categories[idx % len(categories)]
        kind = _category_to_kind(category)
        specs.append(
            {
                "name": f"{category}_{idx:03d}",
                "kind": kind,
                "category": category,
            }
        )
    rng.shuffle(specs)
    return specs


def _all_task_templates() -> List[Tuple[str, str | None, str | None]]:
    tasks: List[Tuple[str, str | None, str | None]] = []
    for loc in ("pass_counter", "table_left", "table_right"):
        tasks.append(("serve_water", loc, None))
        tasks.append(("make_coffee", loc, None))
        tasks.append(("serve_fruit_bowl", loc, None))
        tasks.append(("clear_containers", loc, None))
    for kind in ("mug", "glass", "bowl"):
        tasks.append(("wash_objects", None, kind))
    return tasks


def _build_task_library(
    rng: np.random.Generator,
    min_tasks: int,
    max_tasks: int,
) -> List[Dict[str, Any]]:
    templates = _all_task_templates()
    target = int(rng.integers(min_tasks, max_tasks + 1))
    out: List[Dict[str, Any]] = []
    for idx in range(target):
        task_type, target_location, target_kind = templates[idx % len(templates)]
        out.append(
            {
                "task_uid": f"task_{idx:03d}",
                "task_type": task_type,
                "target_location": target_location,
                "target_kind": target_kind,
            }
        )
    rng.shuffle(out)
    return out


def _paper2_grid_cfg() -> Dict[str, Any]:
    blocked = [[4, y] for y in range(10) if y != 5]
    return {
        "enabled": True,
        "fixed_costs": {"pick": 100.0, "place": 100.0, "fill": 100.0, "wash": 100.0, "brew": 100.0, "fruit": 100.0},
        "move": {
            "scale": 1.0,
            "grid": {"width": 10, "height": 10, "blocked_cells": blocked},
        },
    }


def generate_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    categories = _build_categories()
    base_locations = _base_locations()
    layouts: List[Dict[str, Any]] = []

    for i in range(args.num_layouts):
        layout_seed = int(rng.integers(0, 2**31 - 1))
        lrng = np.random.default_rng(layout_seed)
        locations = _jitter_locations(lrng, base_locations)
        location_cells = {loc["name"]: loc["coord"] for loc in locations}
        paper2 = _paper2_grid_cfg()
        paper2["move"]["location_cells"] = location_cells

        layout = {
            "layout_id": f"restaurant_layout_{i:04d}",
            "rooms": ["kitchen", "serving"],
            "num_categories": len(categories),
            "categories": categories,
            "locations": locations,
            "service_locations": ["pass_counter", "table_left", "table_right"],
            "wash_ready_locations": ["dish_rack", "kitchen_counter"],
            "dirty_drop_locations": ["sink", "bus_tub"],
            "stations": {"water": "water_station", "coffee": "coffee_machine", "fruit": "fruit_station", "wash": "sink"},
            "object_kinds": ["mug", "glass", "bowl"],
            "contents": ["empty", "water", "coffee", "fruit"],
            "task_types": ["serve_water", "make_coffee", "serve_fruit_bowl", "clear_containers", "wash_objects"],
            "object_specs": _build_object_specs(lrng, categories, args.min_objects, args.max_objects),
            "task_distribution": {
                "serve_water": 0.28,
                "make_coffee": 0.24,
                "serve_fruit_bowl": 0.18,
                "clear_containers": 0.15,
                "wash_objects": 0.15,
            },
            "service_location_distribution": {"pass_counter": 0.34, "table_left": 0.33, "table_right": 0.33},
            "wash_kind_distribution": {"mug": 0.34, "glass": 0.33, "bowl": 0.33},
            "reset_location_distribution": {
                "mug": {"kitchen_counter": 0.25, "dish_rack": 0.20, "sink": 0.10, "pass_counter": 0.15, "table_left": 0.15, "table_right": 0.15},
                "glass": {"kitchen_counter": 0.20, "dish_rack": 0.25, "sink": 0.10, "pass_counter": 0.15, "table_left": 0.15, "table_right": 0.10, "bus_tub": 0.05},
                "bowl": {"kitchen_counter": 0.20, "dish_rack": 0.20, "sink": 0.10, "fruit_station": 0.10, "pass_counter": 0.15, "table_left": 0.15, "table_right": 0.10},
            },
            "task_library": _build_task_library(lrng, args.min_tasks_per_layout, args.max_tasks_per_layout),
            "paper2_cost": paper2,
        }
        layouts.append(layout)

    return {
        "version": "paper2_scale_v1",
        "seed": int(args.seed),
        "num_layouts": int(args.num_layouts),
        "target_rooms_per_layout": 2,
        "target_categories": 25,
        "target_task_library_range": [int(args.min_tasks_per_layout), int(args.max_tasks_per_layout)],
        "layouts": layouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large-scale symbolic restaurant layout corpus.")
    parser.add_argument("--output-path", type=Path, default=Path("data/restaurant_layouts/paper2_scale_layouts.json"))
    parser.add_argument("--num-layouts", type=int, default=1000)
    parser.add_argument("--min-tasks-per-layout", type=int, default=50)
    parser.add_argument("--max-tasks-per-layout", type=int, default=100)
    parser.add_argument("--min-objects", type=int, default=60)
    parser.add_argument("--max-objects", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = generate_corpus(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, indent=2)
    print(f"Wrote {len(corpus['layouts'])} layouts -> {args.output_path}")


if __name__ == "__main__":
    main()
