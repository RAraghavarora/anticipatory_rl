# Per-sequence mean PDDL cost for v5 (3-room), one row per (method, seed,
# sequence): the FD oracle contributes 10 sequence-level points (no seed),
# the DQN arms 50 (5 checkpoint seeds x 10 sequences), the GNN arms 40 (4
# seeds -- GNN has no seed 42, only the DQN sweeps do).
#
# Method scope mirrors the 2-room panel's *roles* (oracle, greedy RL, guided
# RL, GNN faithful/augmented) so the two panels compare like-for-like.
# v5-only arms with no 2-room counterpart (K2/K4 clairvoyant, no-demos
# ablation, GNN steelman) are left out of this figure -- they're analyzed in
# RESULTS.md/ANALYSIS.md, not duplicated here.
#
# Clairvoyant Oracle (clairvoyant_k3_lama) is satisficing with a 600s/window
# budget and 168/500 windows hit that cap (see RESULTS.md) -- its per-sequence
# costs are UPPER BOUNDS, not exact optima. Plotted anyway (that's the
# reported arm everywhere else in this archive) but worth remembering before
# reading its spread as a tight distribution.

import glob
import json

import pandas as pd

OUT_CSV = "results/v5/figures/cost_raincloud.csv"
# DQN arms have seed 42 now (5 seeds); the GNN arms don't (still 4).
DQN_SEEDS = [0, 4, 8, 16, 42]
GNN_SEEDS = [0, 4, 8, 16]
TASKS_PER_SEQ = 50
rows = []

# --- Oracles: exact per-sequence total from run_summary.csv, no seed ---
run_summary = pd.read_csv("results/v5/planner/run_summary.csv")
for method_id in ["myopic_fd_optimal", "clairvoyant_k3_lama"]:
    sub = run_summary[run_summary.method_id == method_id]
    for _, r in sub.iterrows():
        rows.append(dict(method_id=method_id, seed=None, seq=r.sequence_id,
                          mean_cost_pddl=r.total_cost_pddl / TASKS_PER_SEQ))

# --- Myopic RL guided/greedy: 5-seed sweep ---
for method_id in ["myopic_dqn", "myopic_dqn_greedy"]:
    for s in DQN_SEEDS:
        sub = run_summary[(run_summary.method_id == method_id) & (run_summary.checkpoint == f"v5-myopic-g097-s{s}")]
        assert len(sub) == 10, (method_id, s, len(sub))
        for _, r in sub.iterrows():
            rows.append(dict(method_id=method_id, seed=s, seq=r.sequence_id,
                              mean_cost_pddl=r.total_cost_pddl / TASKS_PER_SEQ))


def add_from_json(method_id, file_pattern, key_path, seeds):
    for s in seeds:
        files = sorted(glob.glob(file_pattern.format(seed=s)))
        assert len(files) == 10, (method_id, s, len(files))
        for f in files:
            d = json.load(open(f))
            for key in key_path:
                d = d[key]
            rows.append(dict(method_id=method_id, seed=s, seq=f, mean_cost_pddl=d))


add_from_json(
    "anticipatory_dqn_guided",
    "results/v5/planner/raw/anticipatory_dqn/v5_ant_g0.97_s{seed}_seq*_cr3.0.json",
    ["guided", "summary", "mean_pddl_cost"], DQN_SEEDS,
)
add_from_json(
    "anticipatory_dqn_greedy",
    "results/v5/greedy/raw/greedy_s{seed}_seq*.json",
    ["anticipatory", "summary", "mean_pddl_cost"], DQN_SEEDS,
)
add_from_json(
    "gnn_faithful",
    "results/v5/gnn/seeds/gnn_faithful_s{seed}_seq*.json",
    ["gnn_anticipatory", "summary", "mean_cost"], GNN_SEEDS,
)
add_from_json(
    "gnn_counterfactual",
    "results/v5/gnn/seeds/gnn_aug_s{seed}_seq*.json",
    ["gnn_anticipatory", "summary", "mean_cost"], GNN_SEEDS,
)

out = pd.DataFrame(rows)

# Excluded: myopic_dqn_greedy, seed 16, iid-eval-seq-02 (6,171/task) -- 55%
# above the next-highest point in that method's own distribution (3,983) and
# nearly 3x its median. With only 3 greedy myopic seeds so far, one window
# this extreme stretches the shared x-axis enough to flatten every other
# method's raincloud in the same panel. Dropped from this figure only (not
# from the other 4 figures built off the same underlying data) -- excluded,
# not silently absent: this comment plus the print below is the record.
outlier = (out.method_id == "myopic_dqn_greedy") & (out.seed == 16) & (out.seq == "iid-eval-seq-02")
print(f"dropping {outlier.sum()} outlier row(s):\n{out[outlier]}")
out = out[~outlier]

out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(out)} rows)")
print(out.groupby("method_id").size())
