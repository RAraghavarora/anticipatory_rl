#!/usr/bin/env python3
"""Combine RL and planner comparison outputs into unified apples-to-apples table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_rl(rl: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pol in ("myopic", "anticipatory"):
        s = rl[pol]["stats"]
        rows.append(
            {
                "family": "rl",
                "policy": pol,
                "paper2_cost": s.get("avg_task_paper2_cost"),
                "steps_per_task": s.get("avg_task_steps"),
                "task_wall_time_s": None,
                "success_rate": s.get("success_rate"),
            }
        )
    return rows


def _flatten_planner(planner: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pol in ("myopic", "anticipatory"):
        s = planner[pol]["stats"]
        rows.append(
            {
                "family": "planner",
                "policy": pol,
                "paper2_cost": s.get("avg_paper2_cost"),
                "steps_per_task": s.get("avg_steps_per_task"),
                "task_wall_time_s": s.get("avg_task_wall_time_s"),
                "success_rate": s.get("success_rate"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unified RL vs planner comparison table.")
    parser.add_argument("--rl-comparison", type=Path, required=True, help="Path to RL comparison.json.")
    parser.add_argument("--planner-comparison", type=Path, required=True, help="Path to planner_compare.json.")
    parser.add_argument("--output-json", type=Path, default=Path("runs/paper2_cross_family/unified_comparison.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("runs/paper2_cross_family/unified_comparison.csv"))
    args = parser.parse_args()

    rl = _read_json(args.rl_comparison)
    planner = _read_json(args.planner_comparison)
    rows = _flatten_rl(rl) + _flatten_planner(planner)
    payload = {
        "rows": rows,
        "delta_within_family": {
            "rl": rl.get("delta", {}),
            "planner": planner.get("delta", {}),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["family", "policy", "paper2_cost", "steps_per_task", "task_wall_time_s", "success_rate"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote unified comparison JSON -> {args.output_json}")
    print(f"Wrote unified comparison CSV  -> {args.output_csv}")


if __name__ == "__main__":
    main()
