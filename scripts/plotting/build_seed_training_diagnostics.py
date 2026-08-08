"""Per-seed binned training diagnostics, myopic vs anticipatory, all 4 seeds:
success rate, task return, auto-rate, selected Q-value, TD target.

Two different loading paths, both binned to the same 100-bin grid over
0-500k steps:
- task_success / task_return: raw per-task, split by task_type, logged at
  "step" = task index (not env step). Reconstruct env-step position by
  cumsum(task_steps) in task order, same as build_seed_learning_curves.py.
- auto_rate_rolling / q_selected_mean / target_mean: already flat, logged
  per env step directly -- no reconstruction, just bin by "step".
  auto_rate_rolling is the training script's own on-policy 100-task rolling
  window (there is no raw per-task auto-success metric to rebuild from the
  way there is for success/return) -- flagged in the figure caption, not
  hidden.
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
OUT_CSV = "results/canonical_planner/figures/training_diagnostics_seeds_binned.csv"
N_BINS = 100
TOTAL_STEPS = 500_000
FLAT_METRICS = {
    "auto_rate_rolling__window_100": "Auto-success rate",
    "q_selected_mean": "Selected Q-value",
    "target_mean": "TD target",
}
RECONSTRUCT_METRICS = {"task_success": "Success rate", "task_return": "Task return"}


def bin_series(step, value):
    bin_edges = np.linspace(0, TOTAL_STEPS, N_BINS + 1)
    bins = pd.cut(step, bin_edges, include_lowest=True)
    binned = pd.Series(value).groupby(bins, observed=True).mean()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    out = pd.DataFrame({"bin_index": range(N_BINS), "step": bin_centers})
    out["mean_value"] = out["bin_index"].map(lambda i: binned.iloc[i] if i < len(binned) else np.nan)
    return out.dropna(subset=["mean_value"])


def load_flat(metrics_csv_path):
    chunks = []
    for chunk in pd.read_csv(metrics_csv_path, chunksize=2_000_000, usecols=["step", "metric", "value"]):
        mask = chunk["metric"].isin(FLAT_METRICS)
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    frames = []
    for metric, label in FLAT_METRICS.items():
        sub = df[df["metric"] == metric]
        b = bin_series(sub["step"], sub["value"])
        b["metric"] = label
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


def load_reconstructed(metrics_csv_path):
    chunks = []
    for chunk in pd.read_csv(metrics_csv_path, chunksize=2_000_000, usecols=["step", "metric", "value"]):
        mask = chunk["metric"].str.startswith(("task_return__task_type_", "task_success__task_type_", "task_steps__task_type_"))
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    df["kind"] = df["metric"].str.split("__").str[0]
    df = df.pivot_table(index="step", columns="kind", values="value", aggfunc="first").reset_index()
    df = df.sort_values("step")
    df["env_step"] = df["task_steps"].cumsum()

    frames = []
    for metric, label in RECONSTRUCT_METRICS.items():
        b = bin_series(df["env_step"], df[metric])
        b["metric"] = label
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


def main():
    rows = []
    for (label, seed), path in RUNS.items():
        flat = load_flat(path)
        recon = load_reconstructed(path)
        b = pd.concat([flat, recon], ignore_index=True)
        b["label"] = label
        b["seed"] = seed
        rows.append(b)
        print(f"{label} seed={seed}: {len(b)} rows")
    pd.concat(rows, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
