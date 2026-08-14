"""Per-seed binned training diagnostics, v5, myopic vs anticipatory:
success rate, task return, selected Q-value, TD target. Auto-success rate
omitted (dropped on request; it was the one panel in the 2-room version
built from an on-policy rolling window rather than raw per-task data anyway).

Same two loading paths and same 100-bin/500k-step grid as
build_seed_training_diagnostics.py (the 2-room version):
- task_success / task_return: raw per-task, split by task_type, logged at
  "step" = task index (not env step). Reconstruct env-step position by
  cumsum(task_steps) in task order.
- q_selected_mean / target_mean: already flat, logged per env step
  directly -- no reconstruction, just bin by "step".

Both methods have 5 seeds now (0, 4, 8, 16, 42).
"""
import pandas as pd
import numpy as np

RUNS = {
    ("Myopic DQN", 0): "runs/v5-myopic-g097-s0/metrics.csv",
    ("Myopic DQN", 4): "runs/v5-myopic-g097-s4/metrics.csv",
    ("Myopic DQN", 8): "runs/v5-myopic-g097-s8/metrics.csv",
    ("Myopic DQN", 16): "runs/v5-myopic-g097-s16/metrics.csv",
    ("Myopic DQN", 42): "runs/v5-myopic-g097-s42/metrics.csv",
    ("Anticipatory DQN", 0): "runs/v5_ant_g0.97_s0/metrics.csv",
    ("Anticipatory DQN", 4): "runs/v5_ant_g0.97_s4/metrics.csv",
    ("Anticipatory DQN", 8): "runs/v5_ant_g0.97_s8/metrics.csv",
    ("Anticipatory DQN", 16): "runs/v5_ant_g0.97_s16/metrics.csv",
    ("Anticipatory DQN", 42): "runs/v5_ant_g0.97_s42/metrics.csv",
}
OUT_CSV = "results/v5/figures/training_diagnostics_seeds_binned.csv"
N_BINS = 100
TOTAL_STEPS = 500_000
FLAT_METRICS = {
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
