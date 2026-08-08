"""Per-seed binned Q-value/target-value curves, myopic vs anticipatory, all 4 seeds.

Unlike task_return/task_success (logged per-task, needing env-step
reconstruction via cumsum(task_steps)), q_selected_mean/target_mean are
already logged flat per training step -- bin directly, no reconstruction.
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
OUT_CSV = "results/canonical_planner/figures/qvalue_seeds_binned.csv"
N_BINS = 100
TOTAL_STEPS = 500_000
METRICS = ["q_selected_mean", "target_mean"]


def load_binned(metrics_csv_path):
    chunks = []
    for chunk in pd.read_csv(metrics_csv_path, chunksize=2_000_000, usecols=["step", "metric", "value"]):
        mask = chunk["metric"].isin(METRICS)
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)

    bin_edges = np.linspace(0, TOTAL_STEPS, N_BINS + 1)
    df["bin"] = pd.cut(df["step"], bin_edges, include_lowest=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    out_frames = []
    for metric in METRICS:
        sub = df[df["metric"] == metric]
        binned = sub.groupby("bin", observed=True)["value"].mean()
        out = pd.DataFrame({"bin_index": range(N_BINS), "step": bin_centers})
        out["mean_value"] = out["bin_index"].map(lambda i: binned.iloc[i] if i < len(binned) else np.nan)
        out["metric"] = metric
        out_frames.append(out.dropna(subset=["mean_value"]))
    return pd.concat(out_frames, ignore_index=True)


def main():
    rows = []
    for (label, seed), path in RUNS.items():
        b = load_binned(path)
        b["label"] = label
        b["seed"] = seed
        rows.append(b)
        print(f"{label} seed={seed}: {len(b)} rows")
    pd.concat(rows, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
