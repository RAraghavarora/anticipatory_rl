#!/usr/bin/env python3
"""Run restaurant checkpoint comparisons across many seeds (parallel).

This is a thin orchestrator around `python -m anticipatory_rl.agents.restaurant_dqn_infer`.
It spawns one process per seed and pins each worker to either:
- CPU (by clearing CUDA visibility), or
- a single GPU id (via CUDA_VISIBLE_DEVICES).

Outputs:
- Per-seed: <output_dir>/seed_<seed>/comparison.json (+ plots from the infer script)
- Aggregate: <output_dir>/aggregate.json and <output_dir>/aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SeedJob:
    seed: int
    device: str  # "cpu" or "gpu:<id>"
    output_dir: Path


def _parse_int_list(spec: str) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _expand_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return _parse_int_list(args.seeds)
    if args.seed_from is None or args.seed_to is None:
        raise SystemExit("Pass either --seeds or both --seed-from and --seed-to.")
    if args.seed_to < args.seed_from:
        raise SystemExit("--seed-to must be >= --seed-from.")
    return list(range(int(args.seed_from), int(args.seed_to) + 1))


def _infer_cmd(
    *,
    anticipatory_weights: Path,
    myopic_weights: Path,
    config_path: str,
    layout_corpus: str,
    layout_id: str,
    sample_layout_per_reset: bool,
    task_library_per_layout: bool,
    output_dir: Path,
    seed: int,
    num_tasks: int,
    total_steps: int,
    eval_layout_count: int,
    task_sequence_length: int,
    hidden_dim: int,
    tasks_per_reset: int,
    max_task_steps: int,
    success_reward: float,
    invalid_action_penalty: float,
    travel_cost_scale: float,
    pick_cost: float,
    place_cost: float,
    wash_cost: float,
    fill_cost: float,
    brew_cost: float,
    fruit_cost: float,
    gamma: float,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "anticipatory_rl.agents.restaurant_dqn_infer",
        "--anticipatory-weights",
        str(anticipatory_weights),
        "--myopic-weights",
        str(myopic_weights),
        "--output-dir",
        str(output_dir),
        "--config-path",
        str(config_path),
        "--task-sequence-length",
        str(int(task_sequence_length)),
        "--eval-layout-count",
        str(int(eval_layout_count)),
        "--num-tasks",
        str(int(num_tasks)),
        "--total-steps",
        str(int(total_steps)),
        "--hidden-dim",
        str(int(hidden_dim)),
        "--tasks-per-reset",
        str(int(tasks_per_reset)),
        "--max-task-steps",
        str(int(max_task_steps)),
        "--success-reward",
        str(float(success_reward)),
        "--invalid-action-penalty",
        str(float(invalid_action_penalty)),
        "--travel-cost-scale",
        str(float(travel_cost_scale)),
        "--pick-cost",
        str(float(pick_cost)),
        "--place-cost",
        str(float(place_cost)),
        "--wash-cost",
        str(float(wash_cost)),
        "--fill-cost",
        str(float(fill_cost)),
        "--brew-cost",
        str(float(brew_cost)),
        "--fruit-cost",
        str(float(fruit_cost)),
        "--gamma",
        str(float(gamma)),
        "--seed",
        str(int(seed)),
        "--softmax-temperature",
        "0.0",
    ]
    if layout_corpus:
        cmd.extend(["--layout-corpus", str(layout_corpus)])
    if layout_id:
        cmd.extend(["--layout-id", str(layout_id)])
    if sample_layout_per_reset:
        cmd.append("--sample-layout-per-reset")
    cmd.extend(["--task-library-per-layout" if task_library_per_layout else "--no-task-library-per-layout"])
    return cmd


def _run_one_seed(
    job: SeedJob,
    *,
    anticipatory_weights: str,
    myopic_weights: str,
    config_path: str,
    layout_corpus: str,
    layout_id: str,
    sample_layout_per_reset: bool,
    task_library_per_layout: bool,
    num_tasks: int,
    total_steps: int,
    eval_layout_count: int,
    task_sequence_length: int,
    hidden_dim: int,
    tasks_per_reset: int,
    max_task_steps: int,
    success_reward: float,
    invalid_action_penalty: float,
    travel_cost_scale: float,
    pick_cost: float,
    place_cost: float,
    wash_cost: float,
    fill_cost: float,
    brew_cost: float,
    fruit_cost: float,
    gamma: float,
    quiet: bool,
) -> Tuple[int, Path]:
    job.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Avoid Intel OpenMP shared-memory usage (often blocked in sandboxes/containers).
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    if job.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # device is gpu:<id>
        gpu_id = job.device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = _infer_cmd(
        anticipatory_weights=Path(anticipatory_weights),
        myopic_weights=Path(myopic_weights),
        config_path=config_path,
        layout_corpus=layout_corpus,
        layout_id=layout_id,
        sample_layout_per_reset=sample_layout_per_reset,
        task_library_per_layout=task_library_per_layout,
        output_dir=job.output_dir,
        seed=job.seed,
        num_tasks=num_tasks,
        total_steps=total_steps,
        eval_layout_count=eval_layout_count,
        task_sequence_length=task_sequence_length,
        hidden_dim=hidden_dim,
        tasks_per_reset=tasks_per_reset,
        max_task_steps=max_task_steps,
        success_reward=success_reward,
        invalid_action_penalty=invalid_action_penalty,
        travel_cost_scale=travel_cost_scale,
        pick_cost=pick_cost,
        place_cost=place_cost,
        wash_cost=wash_cost,
        fill_cost=fill_cost,
        brew_cost=brew_cost,
        fruit_cost=fruit_cost,
        gamma=gamma,
    )

    log_path = job.output_dir / "infer.log"
    with log_path.open("wb") as log:
        # Always log subprocess output for later debugging.
        # If quiet=False we additionally mirror to the terminal.
        if quiet:
            subprocess.run(cmd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(8192), b""):
                log.write(chunk)
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except Exception:
                    pass
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)

    comparison_path = job.output_dir / "comparison.json"
    if not comparison_path.exists():
        raise RuntimeError(f"Missing output {comparison_path}; see {log_path}")
    return job.seed, comparison_path


def _safe_get(d: Dict[str, Any], path: Sequence[str]) -> Optional[float]:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _aggregate(comparison_paths: Iterable[Path]) -> Dict[str, Any]:
    # Aggregate key stats from per-seed comparison.json files.
    rows: List[Dict[str, Any]] = []
    for p in comparison_paths:
        d = json.loads(p.read_text())
        seed = int(d.get("seed", -1))
        row: Dict[str, Any] = {"seed": seed, "path": str(p)}

        for policy in ("anticipatory", "myopic"):
            for metric in (
                "success_rate",
                "avg_task_steps",
                "avg_task_return",
                "avg_task_paper2_cost",
                "paper2_cost_total",
                "reward_per_step",
                "auto_rate",
            ):
                v = _safe_get(d, [policy, "stats", metric])
                row[f"{policy}.{metric}"] = v

        for metric in ("Delta success rate", "Delta avg steps", "Delta avg return", "Delta reward/step", "Delta auto-rate"):
            # Stored in comparison.json under delta with snake-ish keys.
            # We tolerate older/newer naming by probing a few.
            pass

        # Prefer structured delta fields if present.
        if isinstance(d.get("delta"), dict):
            delta = d["delta"]
            # Current restaurant_dqn_infer uses these keys:
            # delta_success_rate, delta_avg_steps, delta_avg_return, delta_reward_per_step, delta_auto_rate
            for k in (
                "delta_success_rate",
                "delta_avg_steps",
                "delta_avg_return",
                "avg_task_paper2_cost",
                "paper2_cost_total",
                "delta_reward_per_step",
                "delta_auto_rate",
            ):
                if k in delta and isinstance(delta[k], (int, float)):
                    row[f"delta.{k}"] = float(delta[k])

        rows.append(row)

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        # Use population std; for small N users can compute SEM/CI externally.
        return {"mean": mean(values), "std": pstdev(values), "n": len(values)}

    # Compute aggregate per column (excluding seed/path).
    numeric_cols = sorted({k for r in rows for k, v in r.items() if k not in ("seed", "path") and isinstance(v, (int, float))})
    agg: Dict[str, Any] = {"n_seeds": len(rows), "by_metric": {}, "rows": rows}
    for col in numeric_cols:
        vals = [float(r[col]) for r in rows if isinstance(r.get(col), (int, float)) and r.get(col) == r.get(col)]
        agg["by_metric"][col] = stats(vals)
    return agg


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    cols = ["seed", "path"] + sorted({k for r in rows for k in r.keys() if k not in ("seed", "path")})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: int(x.get("seed", -1))):
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel multi-seed inference for restaurant DQN compare runs.")
    p.add_argument("--anticipatory-weights", type=str, required=True)
    p.add_argument("--myopic-weights", type=str, required=True)
    p.add_argument("--config-path", type=str, default="anticipatory_rl/configs/restaurant_symbolic.yaml")
    p.add_argument("--layout-corpus", type=str, default="", help="Optional layout corpus JSON path.")
    p.add_argument("--layout-id", type=str, default="", help="Optional fixed layout_id.")
    p.add_argument("--sample-layout-per-reset", action="store_true")
    p.add_argument(
        "--task-library-per-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-layout task library when present.",
    )
    p.add_argument("--output-dir", type=str, required=True)

    seed = p.add_mutually_exclusive_group(required=True)
    seed.add_argument("--seeds", type=str, default=None, help="Comma-separated list, e.g. 0,1,2,3.")
    seed.add_argument("--seed-from", type=int, default=None)
    p.add_argument("--seed-to", type=int, default=None, help="Inclusive.")

    p.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids to use, e.g. 0,1. Empty => CPU-only.")
    p.add_argument("--jobs", type=int, default=0, help="Parallel jobs. Default: len(gpus) if set else min(4, n_seeds).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-seed subprocess output.")

    # Eval parameters (kept aligned with restaurant_dqn_infer defaults)
    p.add_argument("--num-tasks", type=int, default=5000)
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--eval-layout-count", type=int, default=0)
    p.add_argument("--task-sequence-length", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tasks-per-reset", type=int, default=200)
    p.add_argument("--max-task-steps", type=int, default=24)
    p.add_argument("--success-reward", type=float, default=15.0)
    p.add_argument("--invalid-action-penalty", type=float, default=6.0)
    p.add_argument("--travel-cost-scale", type=float, default=1.0)
    p.add_argument("--pick-cost", type=float, default=1.0)
    p.add_argument("--place-cost", type=float, default=1.0)
    p.add_argument("--wash-cost", type=float, default=2.0)
    p.add_argument("--fill-cost", type=float, default=1.0)
    p.add_argument("--brew-cost", type=float, default=2.0)
    p.add_argument("--fruit-cost", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _expand_seeds(args)
    gpus = _parse_int_list(args.gpus)

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    if args.jobs and args.jobs > 0:
        jobs = int(args.jobs)
    else:
        jobs = len(gpus) if gpus else min(4, len(seeds))
        jobs = max(1, jobs)

    devices: List[str]
    if gpus:
        devices = [f"gpu:{g}" for g in gpus]
    else:
        devices = ["cpu"]

    seed_jobs: List[SeedJob] = []
    for i, seed in enumerate(seeds):
        dev = devices[i % len(devices)]
        seed_jobs.append(SeedJob(seed=seed, device=dev, output_dir=base_out / f"seed_{seed}"))

    comparison_paths: List[Path] = []
    failures: List[Tuple[int, str]] = []
    # ThreadPool is enough here because each worker runs an external python
    # process; this also avoids platform/sandbox semaphore limits.
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [
            ex.submit(
                _run_one_seed,
                job,
                anticipatory_weights=args.anticipatory_weights,
                myopic_weights=args.myopic_weights,
                config_path=args.config_path,
                layout_corpus=args.layout_corpus,
                layout_id=args.layout_id,
                sample_layout_per_reset=bool(args.sample_layout_per_reset),
                task_library_per_layout=bool(args.task_library_per_layout),
                num_tasks=args.num_tasks,
                total_steps=args.total_steps,
                eval_layout_count=args.eval_layout_count,
                task_sequence_length=args.task_sequence_length,
                hidden_dim=args.hidden_dim,
                tasks_per_reset=args.tasks_per_reset,
                max_task_steps=args.max_task_steps,
                success_reward=args.success_reward,
                invalid_action_penalty=args.invalid_action_penalty,
                travel_cost_scale=args.travel_cost_scale,
                pick_cost=args.pick_cost,
                place_cost=args.place_cost,
                wash_cost=args.wash_cost,
                fill_cost=args.fill_cost,
                brew_cost=args.brew_cost,
                fruit_cost=args.fruit_cost,
                gamma=args.gamma,
                quiet=bool(args.quiet),
            )
            for job in seed_jobs
        ]

        for f in as_completed(futs):
            try:
                seed, path = f.result()
                comparison_paths.append(path)
                print(f"[ok] seed={seed} -> {path}")
            except Exception as e:  # noqa: BLE001
                failures.append((-1, repr(e)))
                print(f"[fail] {e}", file=sys.stderr)

    comparison_paths = sorted(comparison_paths, key=lambda p: int(p.parent.name.split("_")[-1]))
    agg = _aggregate(comparison_paths)
    agg["seeds"] = seeds
    agg["gpus"] = gpus
    agg["jobs"] = jobs
    agg["failures"] = failures

    (base_out / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True))
    _write_csv(agg["rows"], base_out / "aggregate.csv")

    print(f"Wrote {base_out / 'aggregate.json'}")
    print(f"Wrote {base_out / 'aggregate.csv'}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
