#!/usr/bin/env python3
"""Plot restaurant multiseed summary with per-metric y-axes.

Reads the JSON written by scripts/restaurant_multi_seed_infer.py (aggregate.json)
and produces a multi-panel figure with Anticipatory vs Myopic mean±std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("anticipatory.success_rate", "myopic.success_rate", "Success rate"),
    ("anticipatory.auto_rate", "myopic.auto_rate", "Auto-rate"),
    ("anticipatory.avg_task_steps", "myopic.avg_task_steps", "Avg steps / task"),
    ("anticipatory.avg_task_return", "myopic.avg_task_return", "Avg task return"),
    ("anticipatory.reward_per_step", "myopic.reward_per_step", "Reward / step"),
]


def _get_mean_std(agg: Dict[str, Any], key: str) -> Tuple[float, float]:
    bm = agg.get("by_metric", {})
    entry = bm.get(key)
    if not isinstance(entry, dict):
        raise KeyError(f"Missing metric {key} in by_metric")
    return float(entry["mean"]), float(entry["std"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Anticipatory vs Myopic mean±std from aggregate.json.")
    p.add_argument("--aggregate-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="e.g. runs/foo/restaurant_multiseed.png")
    p.add_argument("--title", type=str, default="Restaurant Evaluation (5 seeds, 5000 tasks each)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg = json.loads(args.aggregate_json.read_text())

    # 2x3 grid, last cell unused (5 metrics).
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    axes_list = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    colors = {"Anticipatory": "#1f77b4", "Myopic": "#ff7f0e"}
    for ax, (ant_key, myo_key, label) in zip(axes_list, METRICS, strict=False):
        ant_mean, ant_std = _get_mean_std(agg, ant_key)
        myo_mean, myo_std = _get_mean_std(agg, myo_key)

        x = [0, 1]
        means = [ant_mean, myo_mean]
        errs = [ant_std, myo_std]
        names = ["Anticipatory", "Myopic"]

        ax.bar(
            x,
            means,
            yerr=errs,
            capsize=4,
            color=[colors[n] for n in names],
            alpha=0.9,
            linewidth=0.0,
        )
        ax.set_xticks(x, names, rotation=0)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)

        # Tight y-lims around the bars, but not too tight.
        lo = min(m - e for m, e in zip(means, errs))
        hi = max(m + e for m, e in zip(means, errs))
        pad = (hi - lo) * 0.15 if hi > lo else (abs(hi) * 0.15 + 1e-3)
        ax.set_ylim(lo - pad, hi + pad)

        # Show numeric values above bars.
        for xi, m, e in zip(x, means, errs):
            ax.text(xi, m + e + pad * 0.05, f"{m:.4g}", ha="center", va="bottom", fontsize=9)

    # Hide unused last subplot (bottom-right).
    axes_list[-1].axis("off")

    fig.suptitle(args.title, y=1.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

