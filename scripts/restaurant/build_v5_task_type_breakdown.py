# Per-task-type mean cost for v5 (3-room), matching the exact methodology
# of the existing 2-room figure (scripts/plotting/plot_task_type_breakdown.R):
# mean cost per (seed, task_type) first -- averaging over that type's
# occurrences across the 10 sequences -- THEN mean +/- SEM across the
# checkpoint seeds, so the error bar reflects between-seed variance, not
# within-task-type noise.
#
# Three methods:
#   - Anticipatory RL (guided): gamma=0.97 headline, cost_ratio=3.0, 5 seeds.
#   - One-task GNN (augmented): Talukder et al. as published (counterfactual
#     augmentation), 4 seeds -- the GNN sweep has no seed 42, only the DQN
#     ones do (results/v5/gnn/seeds/, not the superseded single-seed-42 run
#     -- see results/v5/gnn/raw/superseded_seed42/README.md).
#   - Myopic RL (guided): 5 seeds, same as the anticipatory arm.

import glob
import json

import numpy as np
import pandas as pd

OUT_CSV = "results/v5/figures/task_type_breakdown.csv"
# DQN arms have seed 42 now (5 seeds); the GNN arm doesn't (still 4).
DQN_SEEDS = [0, 4, 8, 16, 42]
GNN_SEEDS = [0, 4, 8, 16]


def per_task_type_summary(file_pattern, task_key_path, label, seeds):
    per_seed_rows = []
    for s in seeds:
        files = sorted(glob.glob(file_pattern.format(seed=s)))
        assert len(files) == 10, (label, s, len(files))
        costs_by_type = {}
        for f in files:
            d = json.load(open(f))
            for key in task_key_path:
                d = d[key]
            for t in d:
                costs_by_type.setdefault(t["task_type"], []).append(t["cost"])
        for task_type, costs in costs_by_type.items():
            per_seed_rows.append(dict(seed=s, task_type=task_type, mean_cost=np.mean(costs)))

    per_seed = pd.DataFrame(per_seed_rows)
    summary = per_seed.groupby("task_type").agg(
        mean_cost=("mean_cost", "mean"),
        sem=("mean_cost", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ).reset_index()
    summary.insert(0, "label", label)
    return summary


anticipatory = per_task_type_summary(
    "results/v5/planner/raw/anticipatory_dqn/v5_ant_g0.97_s{seed}_seq*_cr3.0.json",
    ["guided", "tasks"],
    "Anticipatory RL (guided)",
    DQN_SEEDS,
)
gnn_augmented = per_task_type_summary(
    "results/v5/gnn/seeds/gnn_aug_s{seed}_seq*.json",
    ["gnn_anticipatory", "tasks"],
    "One-task GNN (augmented)",
    GNN_SEEDS,
)
myopic_guided = per_task_type_summary(
    "results/v5/planner/raw/myopic_dqn/v5-myopic-g097-s{seed}_seq*_cr3.0.json",
    ["guided", "tasks"],
    "Myopic RL (guided)",
    DQN_SEEDS,
)

summary = pd.concat([anticipatory, gnn_augmented, myopic_guided], ignore_index=True)
summary.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(summary)} rows)")
