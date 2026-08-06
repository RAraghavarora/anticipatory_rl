"""Training reward (task return) vs. training step, myopic vs anticipatory, seed 0.

metrics.csv logs task_return/task_steps per completed task, indexed by task
count (not env step), split by task_type. This reconstructs a single raw
per-task return series in task order, cumsum's task_steps to get each task's
actual env-step position, then bins by env step (not a per-task sliding
window) and plots the per-bin mean +/- std -- a per-task rolling window stays
jagged here because task reward scale varies a lot by task type, and binning
across many steps is what actually smooths that out, matching how RL papers
render single-run variance bands.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {
    "Myopic DQN": ("runs/v3_myo_g0.97_peb/metrics.csv", "#56B4E9"),
    "Anticipatory DQN": ("runs/v3_ant_g0.97_peb/metrics.csv", "#009E73"),
}
OUT_PATH = "results/canonical_planner/figures/training_reward_myopic_vs_anticipatory.png"
BINNED_CSV_PATH = "results/canonical_planner/figures/training_reward_binned.csv"
N_BINS = 150


def load_binned_series(metrics_csv_path):
    chunks = []
    for chunk in pd.read_csv(metrics_csv_path, chunksize=2_000_000, usecols=["step", "metric", "value"]):
        mask = chunk["metric"].str.startswith(("task_return__task_type_", "task_steps__task_type_"))
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    df["kind"] = df["metric"].str.split("__").str[0]
    df = df.pivot_table(index="step", columns="kind", values="value", aggfunc="first").reset_index()
    df = df.sort_values("step").rename(columns={"step": "task_index"})
    df["env_step"] = df["task_steps"].cumsum()

    bin_edges = np.linspace(0, df["env_step"].max(), N_BINS + 1)
    df["bin"] = pd.cut(df["env_step"], bin_edges, include_lowest=True)
    binned = df.groupby("bin", observed=True)["task_return"].agg(["mean", "std", "count"])
    binned["env_step"] = (bin_edges[:-1] + bin_edges[1:]) / 2
    return binned.dropna(subset=["mean"])


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    binned_out = []
    for label, (path, color) in RUNS.items():
        b = load_binned_series(path)
        lo = b["mean"] - b["std"].fillna(0.0)
        hi = b["mean"] + b["std"].fillna(0.0)
        ax.fill_between(b["env_step"], lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot(b["env_step"], b["mean"], color=color, linewidth=2, label=label)
        b = b.reset_index(drop=True)
        b.insert(0, "label", label)
        binned_out.append(b)
    pd.concat(binned_out, ignore_index=True).to_csv(BINNED_CSV_PATH, index=False)
    print(f"wrote {BINNED_CSV_PATH}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Task return")
    ax.set_title("Training reward vs. steps (seed 0) -- binned mean ± std")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
