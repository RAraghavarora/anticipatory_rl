"""Build planner-labeled restaurant dataset for APCostEstimator training."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from anticipatory_rl.envs.restaurant_symbolic_env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.tasks.restaurant_graph import build_restaurant_graph
from anticipatory_rl.tasks.restaurant_planner import (
    RestaurantPlannerState,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.tasks.restaurant.restaurant_utils import sample_task


def _load_layouts(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("layouts"), list):
        return [x for x in payload["layouts"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError(f"Unsupported layout corpus format: {path}")


def _build_one_row(job: Dict[str, Any]) -> Dict[str, Any]:
    idx = int(job["idx"])
    seed = int(job["seed"])
    rng = np.random.default_rng(seed)
    random.seed(seed)
    layout = job["layout"]
    env = RestaurantSymbolicEnv(config_path=Path(job["config_path"]), rng_seed=seed)
    env.reset(seed=seed + 1000, options={"layout": layout, "task_library": layout.get("task_library", [])})

    warmup_steps = int(rng.integers(0, int(job["max_warmup_steps"]) + 1))
    for _ in range(warmup_steps):
        valid = env._valid_action_mask()
        idxs = np.flatnonzero(valid > 0.0)
        if idxs.size == 0:
            break
        action = int(rng.choice(idxs))
        _, _, success, truncated, _ = env.step(action)
        if success or truncated:
            break

    state = RestaurantPlannerState.from_env(env)
    graph = build_restaurant_graph(state, locations=env.locations, location_coords=env.location_coords)

    label_costs: List[float] = []
    fail_count = 0
    for _k in range(int(job["followup_samples"])):
        task = sample_task(env, uniform_task_type_prob=0.2)
        result = solve_restaurant_task_with_fd(
            env,
            state,
            task,
            planner_path=Path(job["planner_path"]),
            domain_path=Path(job["domain_path"]),
            search=str(job["search"]),
            timeout_s=float(job["timeout_s"]),
        )
        value = float(result.plan_cost) if result.success else float(job["failure_penalty"])
        if value >= float(job["failure_penalty"]):
            fail_count += 1
        label_costs.append(value)

    split_probs = np.asarray(job["split_probs"], dtype=np.float64)
    split_names = ["train", "val", "test"]
    split = str(rng.choice(split_names, p=split_probs))
    return {
        "split": split,
        "layout_id": layout.get("layout_id", f"layout_{idx}"),
        "node_features": graph.node_features.tolist(),
        "edge_index": graph.edge_index.tolist(),
        "target_ap_cost": float(np.mean(label_costs)) if label_costs else float(job["failure_penalty"]),
        "solver_failures": int(fail_count),
        "followup_samples": int(job["followup_samples"]),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    layouts = _load_layouts(args.layout_corpus)
    planner = Path(args.planner_path)
    domain = Path(args.domain_path)

    rows: List[Dict[str, Any]] = []
    split_probs = np.array([args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio], dtype=np.float64)
    split_probs = split_probs / split_probs.sum()
    jobs: List[Dict[str, Any]] = []
    for i in range(args.num_states):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        jobs.append(
            {
                "idx": i,
                "seed": int(args.seed + i),
                "layout": layouts[int(rng.integers(0, len(layouts)))],
                "config_path": str(args.config_path),
                "planner_path": str(planner),
                "domain_path": str(domain),
                "search": args.search,
                "timeout_s": float(args.timeout_s),
                "followup_samples": int(args.followup_samples),
                "max_warmup_steps": int(args.max_warmup_steps),
                "failure_penalty": float(args.failure_penalty),
                "split_probs": split_probs.tolist(),
            }
        )
    progress_desc = f"planner-dataset shard {args.shard_index + 1}/{args.num_shards}"
    if args.jobs <= 1:
        rows = []
        for job in tqdm(jobs, desc=progress_desc, unit="state"):
            rows.append(_build_one_row(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            chunksize = max(1, len(jobs) // (args.jobs * 4) if args.jobs > 0 else 1)
            for row in tqdm(
                ex.map(_build_one_row, jobs, chunksize=chunksize),
                total=len(jobs),
                desc=progress_desc,
                unit="state",
            ):
                rows.append(row)

    return {
        "meta": {
            "num_rows": len(rows),
            "num_states_requested": int(args.num_states),
            "seed": int(args.seed),
            "search": args.search,
            "followup_samples": int(args.followup_samples),
            "failure_penalty": float(args.failure_penalty),
            "planner_path": str(planner),
            "domain_path": str(domain),
            "jobs": int(args.jobs),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build planner-labeled restaurant dataset for APCostEstimator.")
    parser.add_argument("--layout-corpus", type=Path, default=Path("data/restaurant_layouts/paper2_scale_layouts.json"))
    parser.add_argument("--config-path", type=Path, default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/restaurant_domain.pddl"))
    parser.add_argument("--search", type=str, default="astar(lmcut())")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--num-states", type=int, default=2000)
    parser.add_argument("--followup-samples", type=int, default=8)
    parser.add_argument("--max-warmup-steps", type=int, default=20)
    parser.add_argument("--failure-penalty", type=float, default=25000.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=1, help="CPU worker processes for parallel FD labeling.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard id in [0, num-shards).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for multi-node generation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("data/restaurant_planner_dataset/paper2_planner_labels.json"))
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    dataset = build_dataset(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"Wrote planner dataset with {dataset['meta']['num_rows']} rows -> {args.output_path}")


if __name__ == "__main__":
    main()
