#!/usr/bin/env python3
"""Plot restaurant multi-seed metrics with mean±std error bars.

Reads the `aggregate.json` produced by `scripts/restaurant_multi_seed_infer.py`.

Produces a 1x3 subplot figure (separate y-axes) for:
- avg steps / task
- avg task return
- reward / step
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_metric(agg: Dict, key: str) -> tuple[float, float]:
    by = agg.get("by_metric", {})
    if key not in by:
        raise KeyError(f"Missing metric '{key}' in aggregate.json")
    entry = by[key]
    return float(entry["mean"]), float(entry["std"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot restaurant multi-seed mean±std metrics.")
    p.add_argument(
        "--aggregate-json",
        type=Path,
        required=True,
        help="Path to aggregate.json from scripts/restaurant_multi_seed_infer.py",
    )
    p.add_argument("--out", type=Path, required=True, help="Output image path (.png recommended).")
    p.add_argument("--title", type=str, default="Restaurant (5 eval seeds, 5000 tasks each)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg = json.loads(args.aggregate_json.read_text())

    metrics = [
        ("Avg steps / task", "avg_task_steps", "Steps"),
        ("Avg task return", "avg_task_return", "Return"),
        ("Reward / step", "reward_per_step", "Reward"),
    ]
    labels = ["Anticipatory", "Myopic"]
    colors = ["#2563eb", "#f97316"]  # blue / orange

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (title, suffix, ylab) in zip(axes, metrics):
        ant_mean, ant_std = _get_metric(agg, f"anticipatory.{suffix}")
        myo_mean, myo_std = _get_metric(agg, f"myopic.{suffix}")

        x = [0, 1]
        means = [ant_mean, myo_mean]
        stds = [ant_std, myo_std]

        ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            alpha=0.9,
            capsize=6,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        # Tight-ish y-lims with padding.
        lo = min(m - s for m, s in zip(means, stds))
        hi = max(m + s for m, s in zip(means, stds))
        pad = (hi - lo) * 0.25 if hi > lo else 1.0
        ax.set_ylim(lo - pad * 0.15, hi + pad * 0.35)

        # Annotate mean value on each bar.
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.3g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)


if __name__ == "__main__":
    main()

