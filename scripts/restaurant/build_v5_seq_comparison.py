# Sequence-by-sequence total PDDL cost, Anticipatory RL (guided) vs a
# clairvoyant oracle (K=3 by default, --k 4 for the K=4 arm), for the 10
# canonical v5 sequences -- everything needed is already in
# results/v5/planner/run_summary.csv, no raw JSON parsing.
#
# Anticipatory RL (guided): gamma=0.97 (headline arm), cost_ratio=3.0, all 4
# checkpoint seeds -- reports mean/min/max total cost per sequence across
# seeds (min/max, not SD, same n=4 rule as every other figure this session).
#
# Clairvoyant Oracle: a single deterministic search per sequence (no seed),
# so just one point -- but it's a satisficing UPPER BOUND, and
# windows_at_cap/windows (how many of that sequence's rolling K-task windows
# hit the 600s search budget) varies a lot per sequence, and is worse
# overall at K=4 (266/500 windows at cap) than K=3 (168/500) -- carried
# through so the plot can flag sequences where that value is a weaker bound.

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=3, choices=[3, 4])
args = parser.parse_args()

METHOD_ID = f"clairvoyant_k{args.k}_lama"
LABEL = f"K={args.k} Clairvoyant Oracle"
OUT_CSV = "results/v5/figures/seq_comparison.csv" if args.k == 3 else f"results/v5/figures/seq_comparison_k{args.k}.csv"

df = pd.read_csv("results/v5/planner/run_summary.csv")

ours = df[
    (df.method_id == "anticipatory_dqn")
    & (df.cost_ratio == 3.0)
    & (df.checkpoint.str.startswith("v5_ant_g0.97_s", na=False))
]
ours_summary = ours.groupby("sequence_id").agg(
    mean_cost=("total_cost_pddl", "mean"),
    min_cost=("total_cost_pddl", "min"),
    max_cost=("total_cost_pddl", "max"),
    n_seeds=("total_cost_pddl", "size"),
).reset_index()
ours_summary.insert(1, "label", "Anticipatory RL (guided)")

oracle = df[df.method_id == METHOD_ID][
    ["sequence_id", "total_cost_pddl", "windows_at_cap", "windows"]
].rename(columns={"total_cost_pddl": "mean_cost"})
oracle["min_cost"] = oracle["mean_cost"]
oracle["max_cost"] = oracle["mean_cost"]
oracle["cap_rate"] = oracle["windows_at_cap"] / oracle["windows"]
oracle.insert(1, "label", LABEL)

out = pd.concat([
    ours_summary.assign(cap_rate=float("nan")),
    oracle.drop(columns=["windows_at_cap", "windows"]),
], ignore_index=True)

out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(out)} rows)")
print(out)
