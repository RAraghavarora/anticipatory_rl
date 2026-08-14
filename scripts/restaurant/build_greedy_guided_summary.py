# Per-(restaurant, deployment, method) data across checkpoint seeds, for the
# greedy-vs-guided evaluation figure. Reproduces exactly the numbers in the
# thesis table (tab:greedy-guided) from the same source files that table was
# built from -- see results/v5/RESULTS.md and results/canonical_planner/ for
# provenance.
#
# 3-room DQN arms (Myopic + Anticipatory) now have a 5-seed sweep (seed 42
# added). The 2-room rows stay at their own 4 checkpoint seeds.
#
# Writes two CSVs: the per-seed long table (one row per seed, the ground
# truth for the paired slope plot) and its mean/sd collapse (kept for the
# thesis table and any bar-style reuse).

import glob
import json

import numpy as np
import pandas as pd

PER_SEED_CSV = "results/v5/figures/greedy_guided_per_seed.csv"
SUMMARY_CSV = "results/v5/figures/greedy_guided_grid.csv"
# v5 DQN seeds -- 5 now (seed 42 added). The 2-room rows read their own
# checkpoint_seed straight from canonical_planner and are unaffected.
V5_SEEDS = [0, 4, 8, 16, 42]
per_seed_rows = []

# --- 2-room (level_3) greedy: seed_summary.csv is already per-seed ---
greedy_label = {"myopic_dqn_greedy": "Myopic RL", "anticipatory_dqn_greedy": "Anticipatory RL"}
gseed = pd.read_csv("results/canonical_planner/greedy_rl/seed_summary.csv")
for _, r in gseed.iterrows():
    method = greedy_label[r.method_id]
    for metric, col, scale in [
        ("Success", "success_rate", 100.0), ("Auto%", "auto_success_rate", 100.0),
        ("Steps/task", "mean_steps", 1.0), ("Cost/task", "mean_cost_pddl", 1.0),
    ]:
        per_seed_rows.append(dict(restaurant="2-room", deployment="Greedy", method=method,
                                   seed=int(r.checkpoint_seed), metric=metric, value=r[col] * scale))

# --- 2-room guided: per-seed aggregate over its 10 sequences ---
# 2-room guided both at beta=1.25 now (myopic re-evaluated to match the
# anticipatory arm's search bound; the old myopic_dqn_beta1 arm was beta=1.00).
guided_label = {"myopic_dqn_beta1_25": "Myopic RL", "anticipatory_dqn_beta1_25": "Anticipatory RL"}
planner = pd.read_csv("results/canonical_planner/planner/run_summary.csv")
sub = planner[planner.method_id.isin(guided_label)].copy()
sub["steps_per_task"] = sub.total_actions / sub.task_count
per_seed = sub.groupby(["method_id", "checkpoint_seed"]).agg(
    auto=("auto_success_rate", "mean"), steps=("steps_per_task", "mean"), cost=("mean_cost_pddl", "mean"),
).reset_index()
for mid, method in guided_label.items():
    d = per_seed[per_seed.method_id == mid]
    for _, r in d.iterrows():
        for metric, col, scale in [("Auto%", "auto", 100.0), ("Steps/task", "steps", 1.0), ("Cost/task", "cost", 1.0)]:
            per_seed_rows.append(dict(restaurant="2-room", deployment="Guided", method=method,
                                       seed=int(r.checkpoint_seed), metric=metric, value=r[col] * scale))
        per_seed_rows.append(dict(restaurant="2-room", deployment="Guided", method=method,
                                   seed=int(r.checkpoint_seed), metric="Success", value=100.0))

# --- 3-room (v5) greedy: parse raw per-seed, per-sequence trajectory summaries ---
for s in V5_SEEDS:
    files = sorted(glob.glob(f"results/v5/greedy/raw/greedy_s{s}_seq*.json"))
    assert len(files) == 10, (s, len(files))
    succ, auto, steps, cost = [], [], [], []
    for f in files:
        d = json.load(open(f))["anticipatory"]["summary"]
        succ.append(d["success_rate"]); auto.append(d["auto_rate"])
        steps.append(d["mean_steps"]); cost.append(d["mean_pddl_cost"])
    for metric, vals, scale in [("Success", succ, 100.0), ("Auto%", auto, 100.0),
                                 ("Steps/task", steps, 1.0), ("Cost/task", cost, 1.0)]:
        per_seed_rows.append(dict(restaurant="3-room", deployment="Greedy", method="Anticipatory RL",
                                   seed=s, metric=metric, value=np.mean(vals) * scale))

# --- 3-room guided: gamma=0.97 (headline arm), cost_ratio=3.0, all 4 seeds ---
for s in V5_SEEDS:
    files = sorted(glob.glob(f"results/v5/planner/raw/anticipatory_dqn/v5_ant_g0.97_s{s}_seq*_cr3.0.json"))
    assert len(files) == 10, (s, len(files))
    auto, steps, cost = [], [], []
    for f in files:
        d = json.load(open(f))["guided"]["summary"]
        auto.append(d["auto_rate"]); steps.append(d["total_actions"] / d["attempted"]); cost.append(d["mean_pddl_cost"])
    for metric, vals, scale in [("Auto%", auto, 100.0), ("Steps/task", steps, 1.0), ("Cost/task", cost, 1.0)]:
        per_seed_rows.append(dict(restaurant="3-room", deployment="Guided", method="Anticipatory RL",
                                   seed=s, metric=metric, value=np.mean(vals) * scale))
    per_seed_rows.append(dict(restaurant="3-room", deployment="Guided", method="Anticipatory RL",
                               seed=s, metric="Success", value=100.0))

# --- 3-room Myopic RL: 5-seed sweep ---
MYOPIC_GREEDY_SEEDS = [0, 4, 8, 16, 42]
for s in MYOPIC_GREEDY_SEEDS:
    files = sorted(glob.glob(f"results/v5/greedy/raw/greedy_v5-myopic-g097-s{s}_seq*.json"))
    assert len(files) == 10, (s, len(files))
    succ, auto, steps, cost = [], [], [], []
    for f in files:
        d = json.load(open(f))["anticipatory"]["summary"]
        succ.append(d["success_rate"]); auto.append(d["auto_rate"])
        steps.append(d["mean_steps"]); cost.append(d["mean_pddl_cost"])
    for metric, vals, scale in [("Success", succ, 100.0), ("Auto%", auto, 100.0),
                                 ("Steps/task", steps, 1.0), ("Cost/task", cost, 1.0)]:
        per_seed_rows.append(dict(restaurant="3-room", deployment="Greedy", method="Myopic RL",
                                   seed=s, metric=metric, value=np.mean(vals) * scale))

MYOPIC_GUIDED_SEEDS = [0, 4, 8, 16, 42]
for s in MYOPIC_GUIDED_SEEDS:
    files = sorted(glob.glob(f"results/v5/planner/raw/myopic_dqn/v5-myopic-g097-s{s}_seq*_cr3.0.json"))
    assert len(files) == 10, (s, len(files))
    auto, steps, cost = [], [], []
    for f in files:
        d = json.load(open(f))["guided"]["summary"]
        auto.append(d["auto_rate"]); steps.append(d["total_actions"] / d["attempted"]); cost.append(d["mean_pddl_cost"])
    for metric, vals, scale in [("Auto%", auto, 100.0), ("Steps/task", steps, 1.0), ("Cost/task", cost, 1.0)]:
        per_seed_rows.append(dict(restaurant="3-room", deployment="Guided", method="Myopic RL",
                                   seed=s, metric=metric, value=np.mean(vals) * scale))
    per_seed_rows.append(dict(restaurant="3-room", deployment="Guided", method="Myopic RL",
                               seed=s, metric="Success", value=100.0))

per_seed_df = pd.DataFrame(per_seed_rows)
per_seed_df.to_csv(PER_SEED_CSV, index=False)
print(f"wrote {PER_SEED_CSV} ({len(per_seed_df)} rows)")

summary = per_seed_df.groupby(["restaurant", "deployment", "method", "metric"]).agg(
    mean=("value", "mean"), sd=("value", lambda x: x.std(ddof=1)),
).reset_index()
summary["sd"] = summary["sd"].fillna(0.0)
summary.to_csv(SUMMARY_CSV, index=False)
print(f"wrote {SUMMARY_CSV} ({len(summary)} rows)")
