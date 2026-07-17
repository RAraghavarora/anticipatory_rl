"""Compare v3_new myopic vs anticipatory DQN training runs.

Loads sanitized CSVs (one contiguous run each) via awk pre-filtering,
validates train_args metadata and CSV integrity, plots 4-panel comparison
with raw+smooth curves and train_summary annotations, saves plot + manifest.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Myopic": ROOT / "runs/v3_new/v3_myopic_seed0",
    "Anticipatory": ROOT / "runs/v3_new/v3_anticipatory_seed0",
}
METRICS = [
    "success_rate_rolling__window_100",
    "avg_task_return_rolling__window_100",
    "q_selected_abs_max",
    "avg_loss_rolling__window_100",
]
TITLES = {
    "success_rate_rolling__window_100": "Success Rate (rolling 100)",
    "avg_task_return_rolling__window_100": "Avg Task Return (rolling 100)",
    "q_selected_abs_max": "Q-selected |max|",
    "avg_loss_rolling__window_100": "Train Loss (rolling 100)",
}
SUMMARY_METRIC_MAP = {
    "success_rate_rolling__window_100": "success_rate",
    "avg_task_return_rolling__window_100": "avg_task_return",
    "q_selected_abs_max": "max_abs_q_selected",
    "avg_loss_rolling__window_100": "mean_loss",
}
COLORS = {"Myopic": "#1f77b4", "Anticipatory": "#d62728"}
OUT_PLOT = ROOT / "runs/v3_new/comparison_plot.png"
OUT_MANIFEST = ROOT / "runs/v3_new/comparison_manifest.json"


def load_args(run_dir: Path) -> dict:
    path = run_dir / "train_args.json"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def validate_args(myopic: dict, anticipatory: dict):
    shared = {"seed", "config_path", "gamma", "env_reset_tasks"}
    for key in shared:
        mv = myopic[key]
        av = anticipatory[key]
        if mv != av:
            print(f"ERROR: {key} differs: myopic={mv}, anticipatory={av}", file=sys.stderr)
            sys.exit(1)

    mv = myopic["tasks_per_episode"]
    av = anticipatory["tasks_per_episode"]
    if mv == av:
        print(f"ERROR: tasks_per_episode must differ; both are {mv}", file=sys.stderr)
        sys.exit(1)
    if mv != 1:
        print(f"WARNING: myopic tasks_per_episode = {mv} (expected 1)")
    if av != 200:
        print(f"WARNING: anticipatory tasks_per_episode = {av} (expected 200)")

    ms = myopic["total_steps"]
    asteps = anticipatory["total_steps"]
    if ms != asteps:
        print(f"WARNING: total_steps differs: myopic={ms}, anticipatory={asteps} (proceeding)")

    print(f"Metadata OK: seed={myopic['seed']}, gamma={myopic['gamma']}, "
          f"config={myopic['config_path']}, env_reset_tasks={myopic['env_reset_tasks']}, "
          f"tasks_per_episode myopic={mv} anticipatory={av}")


def load_csv_awk(csv_path: Path, metrics: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)
    cond = " || ".join(f'$3=="{m}"' for m in metrics)
    proc = subprocess.run(
        ["awk", "-F,", f"NR>1 && ({cond}) {{print $2\",\"$3\",\"$5}}", str(csv_path)],
        capture_output=True, text=True, check=True,
    )
    df = pd.read_csv(StringIO(proc.stdout), header=None, names=["step", "metric", "value"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    if df.empty:
        print(f"ERROR: no data loaded from {csv_path}", file=sys.stderr)
        sys.exit(1)
    return df


def validate_csv(df: pd.DataFrame, label: str, expected_total: int):
    steps = sorted(df["step"].unique())
    if not steps:
        print(f"ERROR: no steps in {label} data", file=sys.stderr)
        sys.exit(1)

    max_step = max(steps)
    if max_step > expected_total:
        print(f"WARNING: {label} max step {max_step} > train_args total_steps {expected_total}")

    pct = max_step / expected_total * 100
    if pct < 90:
        print(f"WARNING: {label} max step {max_step} is {pct:.1f}% of total_steps {expected_total}")

    gaps = []
    for i in range(1, len(steps)):
        if steps[i] != steps[i - 1] + 1 and steps[i] > steps[i - 1]:
            gaps.append((steps[i - 1], steps[i]))
    if gaps:
        print(f"WARNING: {label} step sequence has gaps: first at step "
              f"{gaps[0][0]} -> {gaps[0][1]} ({gaps[0][1] - gaps[0][0]} missing)")

    has_reset = any(s == 0 for s in steps[1:])
    if has_reset:
        print(f"ERROR: {label} has step reset (step=0 after step>0) — CSV is not a single contiguous run",
              file=sys.stderr)
        sys.exit(1)

    print(f"  {label}: {len(steps)} unique steps, 0..{max_step}, max_pct={pct:.1f}%")


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "train_summary.json"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def smooth(df: pd.DataFrame, window: int = 50) -> tuple[pd.Series, pd.Series]:
    s = df.set_index("step")["value"].rolling(window, min_periods=1).mean()
    return s.index, s.values


def main():
    print("Validating train_args metadata...", flush=True)
    myopic_args = load_args(RUNS["Myopic"])
    antic_args = load_args(RUNS["Anticipatory"])
    validate_args(myopic_args, antic_args)

    print("\nLoading metrics (awk pre-filtered)...", flush=True)
    data = {}
    for label, run_dir in RUNS.items():
        df = load_csv_awk(run_dir / "metrics.csv", METRICS)
        validate_csv(df, label, load_args(run_dir)["total_steps"])
        per_metric = {}
        for m in METRICS:
            sub = df[df["metric"] == m][["step", "value"]].sort_values("step").reset_index(drop=True)
            per_metric[m] = sub
        data[label] = per_metric

    print("\nLoading train_summary...", flush=True)
    myopic_summary = load_summary(RUNS["Myopic"])
    antic_summary = load_summary(RUNS["Anticipatory"])

    print("\nPlotting...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for label in RUNS:
            df = data[label].get(metric)
            if df is None or df.empty:
                continue

            ax.plot(df["step"], df["value"], color=COLORS[label],
                    alpha=0.12, linewidth=0.5, label=f"{label} raw")

            st, sv = smooth(df, window=50)
            ax.plot(st, sv, color=COLORS[label], linewidth=1.5, label=f"{label} smooth")

        ax.set_title(TITLES.get(metric, metric), fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        summary_key = SUMMARY_METRIC_MAP[metric]
        mv = myopic_summary.get(summary_key, float("nan"))
        av = antic_summary.get(summary_key, float("nan"))
        text = f"Myopic: {mv:.3f}\nAnticipatory: {av:.3f}"
        ax.text(0.98, 0.06, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92))

        if i == 0:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("v3_new toy_level_3: Myopic (400k) vs Anticipatory (1M), gamma=0.95, seed=0",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=130)
    plt.close()
    print(f"  saved: {OUT_PLOT}")

    print("\nWriting manifest...", flush=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            "myopic": {
                "dir": str(RUNS["Myopic"].relative_to(ROOT)),
                "label": myopic_args["run_label"],
                "total_steps": myopic_args["total_steps"],
                "tasks_per_episode": myopic_args["tasks_per_episode"],
            },
            "anticipatory": {
                "dir": str(RUNS["Anticipatory"].relative_to(ROOT)),
                "label": antic_args["run_label"],
                "total_steps": antic_args["total_steps"],
                "tasks_per_episode": antic_args["tasks_per_episode"],
            },
        },
        "shared": {
            "config": myopic_args["config_path"],
            "gamma": myopic_args["gamma"],
            "seed": myopic_args["seed"],
            "env_reset_tasks": myopic_args["env_reset_tasks"],
        },
        "final_metrics": {
            "myopic": {
                "success_rate": myopic_summary["success_rate"],
                "avg_task_return": myopic_summary["avg_task_return"],
                "max_abs_q_selected": myopic_summary["max_abs_q_selected"],
                "mean_loss": myopic_summary["mean_loss"],
            },
            "anticipatory": {
                "success_rate": antic_summary["success_rate"],
                "avg_task_return": antic_summary["avg_task_return"],
                "max_abs_q_selected": antic_summary["max_abs_q_selected"],
                "mean_loss": antic_summary["mean_loss"],
            },
        },
        "plot": str(OUT_PLOT.relative_to(ROOT)),
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved: {OUT_MANIFEST}")

    print("\nFinal metrics from train_summary.json:")
    print(f"  Myopic:       success_rate={myopic_summary['success_rate']:.4f}, "
          f"avg_return={myopic_summary['avg_task_return']:.2f}, "
          f"max_abs_q={myopic_summary['max_abs_q_selected']:.2f}, "
          f"mean_loss={myopic_summary['mean_loss']:.4f}")
    print(f"  Anticipatory: success_rate={antic_summary['success_rate']:.4f}, "
          f"avg_return={antic_summary['avg_task_return']:.2f}, "
          f"max_abs_q={antic_summary['max_abs_q_selected']:.2f}, "
          f"mean_loss={antic_summary['mean_loss']:.4f}")

    print("\nDone.")
    return manifest


if __name__ == "__main__":
    main()
