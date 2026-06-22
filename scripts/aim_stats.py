"""Export Aim metrics to CSV using query_metrics — the only SDK path that resolves
run.run_label correctly (meta_attrs_tree.collect() is buggy/cached).
"""

from __future__ import annotations

import pandas as pd
from aim import Repo


KNOWN_LABELS = [
    # "rest_v2_2_anticipatory_5",
    "rest_v2_2_ant_option3_seed0"
    # "myopic_toy_auto_complete"
]

METRICS = [
    "td_abs_mean",
    "loss",
    "greedy_success_rolling",
    "success_rate_rolling",
    "avg_task_return_rolling",
    "q_selected_mean",
    "q_selected_abs_max",
    "target_mean",
    "target_abs_max",
    "q_by_action_type",
    "avg_loss_rolling",
]

repo = Repo(".")

# Discover hash → label
hash_to_label: dict[str, str] = {}
for label in KNOWN_LABELS:
    q = repo.query_metrics(f'run.run_label == "{label}" and metric.name == "loss"')
    for rm in q.iter_runs():
        hash_to_label[rm.run.hash] = label

print(f"Found {len(hash_to_label)} run hashes across {len(set(hash_to_label.values()))} labels")
print(hash_to_label)
# Collect metrics only for discovered runs
metrics_list = '", "'.join(METRICS)
hashes_list = '", "'.join(hash_to_label)
query = repo.query_metrics(f'metric.name in ["{metrics_list}"] and run.hash in ["{hashes_list}"]')

raw_records: list[dict] = []
for run_metrics in query.iter_runs():
    rh = run_metrics.run.hash
    label = hash_to_label.get(rh, f"Run: {rh}")

    for metric in run_metrics:
        # Use dataframe() instead of sparse_numpy() — sparse_numpy() returns
        # Aim's internal storage IDs (huge int64 values, not training steps),
        # which made the old `rank(method="dense")` produce a non-chronological
        # "step" column and corrupt the export. dataframe() exposes the real
        # user-provided `step` (training step) alongside `value`.
        mdf = metric.dataframe()
        ctx = metric.context.to_dict()
        action = ctx.get("action_type", "")
        metric_label = f"{metric.name}_{action}" if action else metric.name

        for _, row in mdf.iterrows():
            raw_records.append({
                "run_label": label,
                "run_hash": rh,
                "step": int(row["step"]),
                "metric_label": metric_label,
                "value": float(row["value"]),
            })

# Pivot to wide: one column per metric_label
df = pd.DataFrame(raw_records)

df_wide = df.pivot_table(
    index=["run_label", "run_hash", "step"],
    columns="metric_label",
    values="value",
    aggfunc="first",
)

# Forward fill missing values so episodic metrics aren't mostly empty
df_wide = df_wide.groupby(level="run_label").ffill().reset_index()

df_wide.to_csv("aim_metrics_wide.csv", index=False)
print(f"Done – {len(df_wide):,} rows × {len(df_wide.columns)} columns → aim_metrics_wide.csv")
