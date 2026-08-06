"""Per-seed binned task-return curves, myopic vs anticipatory, all 4 seeds.

Same binning approach as plot_training_reward.py (env-step bins, not a
per-task rolling window -- task reward scale varies too much by task type for
a sliding window to look clean). Output feeds a seed-aggregated learning
curve: at each bin, the "draws" across the 4 seeds become the distribution
ggdist::stat_lineribbon (or an equivalent) turns into a median + interval band
-- literature-standard between-seed variance, not within-run task noise.
"""
import pandas as pd
import numpy as np

RUNS = {
    ("Myopic DQN", 0): "runs/v3_myo_g0.97_peb/metrics.csv",
    ("Myopic DQN", 4): "runs/v3_myopic_g0.97_peb_s4/metrics.csv",
    ("Myopic DQN", 8): "runs/v3_myopic_g0.97_peb_s8/metrics.csv",
    ("Myopic DQN", 16): "runs/v3_myopic_g0.97_peb_s16/metrics.csv",
    ("Anticipatory DQN", 0): "runs/v3_ant_g0.97_peb/metrics.csv",
    ("Anticipatory DQN", 4): "runs/v3_ant_g0.97_peb_s4/metrics.csv",
    ("Anticipatory DQN", 8): "runs/v3_ant_g0.97_peb_s8/metrics.csv",
    ("Anticipatory DQN", 16): "runs/v3_ant_g0.97_peb_s16/metrics.csv",
}
OUT_CSVS = {
    "task_return": "results/canonical_planner/figures/training_reward_seeds_binned.csv",
    "task_success": "results/canonical_planner/figures/training_success_seeds_binned.csv",
}
N_BINS = 100
TOTAL_STEPS = 500_000


def load_binned(metrics_csv_path, value_metric):
    chunks = []
    for chunk in pd.read_csv(metrics_csv_path, chunksize=2_000_000, usecols=["step", "metric", "value"]):
        mask = chunk["metric"].str.startswith((f"{value_metric}__task_type_", "task_steps__task_type_"))
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    df["kind"] = df["metric"].str.split("__").str[0]
    df = df.pivot_table(index="step", columns="kind", values="value", aggfunc="first").reset_index()
    df = df.sort_values("step").rename(columns={"step": "task_index"})
    df["env_step"] = df["task_steps"].cumsum()

    bin_edges = np.linspace(0, TOTAL_STEPS, N_BINS + 1)
    df["bin"] = pd.cut(df["env_step"], bin_edges, include_lowest=True)
    binned = df.groupby("bin", observed=True)[value_metric].mean()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    out = pd.DataFrame({"bin_index": range(N_BINS), "env_step": bin_centers})
    out["mean_value"] = out["bin_index"].map(lambda i: binned.iloc[i] if i < len(binned) else np.nan)
    return out.dropna(subset=["mean_value"])


def main():
    for value_metric, out_csv in OUT_CSVS.items():
        rows = []
        for (label, seed), path in RUNS.items():
            b = load_binned(path, value_metric)
            b["label"] = label
            b["seed"] = seed
            rows.append(b)
            print(f"[{value_metric}] {label} seed={seed}: {len(b)} bins")
        pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
