"""Compare v3 myopic vs anticipatory DQN runs. Pre-filters the huge metrics.csv with awk."""
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Myopic": ROOT / "runs/v3/v3_myopic_seed0",
    "Anticipatory": ROOT / "runs/v3/v3_anticipatory_seed0",
}
METRICS = [
    "success_rate_rolling__window_100",
    "avg_task_return_rolling__window_100",
    "q_selected_abs_max",
    "avg_loss_rolling__window_100",
]
COLORS = {"Myopic": "#1f77b4", "Anticipatory": "#d62728"}


def load_all_metrics(run_dir: Path, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """Single awk pass over the giant CSV -> dict of small DataFrames, one per metric."""
    # Build an awk condition: $3=="m1" || $3=="m2" || ...
    cond = " || ".join(f'$3=="{m}"' for m in metrics)
    proc = subprocess.run(
        ["awk", "-F,", f'{cond} {{print $2","$3","$5}}', str(run_dir / "metrics.csv")],
        capture_output=True, text=True, check=True,
    )
    from io import StringIO
    df = pd.read_csv(StringIO(proc.stdout), header=None, names=["step", "metric", "value"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("step")
    return {m: df[df["metric"] == m][["step", "value"]].reset_index(drop=True) for m in metrics}


# Single awk pass per run
print("loading myopic...", flush=True)
data = {label: load_all_metrics(rd, METRICS) for label, rd in RUNS.items()}
print("loaded.", flush=True)


def smooth(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    if len(df) <= window:
        return df
    s = df.set_index("step")["value"].rolling(window, min_periods=1).mean()
    return s.reset_index()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

titles = {
    "success_rate_rolling__window_100": "Success Rate (rolling 100)",
    "avg_task_return_rolling__window_100": "Avg Task Return (rolling 100)",
    "q_selected_abs_max": "Q-selected |max|",
    "avg_loss_rolling__window_100": "Train Loss (rolling 100)",
}

for i, metric in enumerate(METRICS):
    ax = axes[i]
    for label in RUNS:
        df = data[label].get(metric)
        if df is None or df.empty:
            continue
        s = smooth(df, window=50)
        ax.plot(s["step"], s["value"], label=label, color=COLORS[label], linewidth=1.2, alpha=0.9)
    ax.set_title(titles.get(metric, metric), fontsize=10)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=9)

fig.suptitle("v3 toy_level_3: Myopic vs Anticipatory DQN (gamma=0.95, seed=0, 1M steps)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "runs/v3/comparison_plot.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")
