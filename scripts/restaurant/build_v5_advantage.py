# Advantage over Myopic Oracle (%), all v5 methods -- (myopic_total -
# method_total) / myopic_total * 100, where each total is the summed PDDL
# cost over one method's 10-sequence x 50-task canonical run. Positive =
# cheaper than pure myopic (K=1) planning; negative = worse.
#
# Everything needed is in results/v5/planner/run_summary.csv, including the
# 4-seed GNN sweep (gnn_counterfactual/gnn_faithful, checkpoints
# gnn_v5_{aug,faithful}_s{0,4,8,16}) -- NOT the superseded single-seed-42
# run RESULTS.md's headline table still shows (650,775); this recomputes
# from the current per-seed source so it can't silently drift from the
# other v5 figures built this session, which already use the 4-seed sweep.
#
# Anticipatory RL (guided): gamma=0.97 (headline arm), cost_ratio=3.0.
# K=2/K=3/K=4 oracles are each a SINGLE run over all 10 sequences (no seed
# sweep) -- one total, one advantage value, not 10 fake "seeds" from
# grouping by sequence_id (that was a bug caught before this file was used:
# grouping single-run methods by sequence_id summed nothing -- each group
# was already just 1 row -- and compared a single sequence's cost against
# the all-10-sequences myopic total, inflating "advantage" to ~90% instead
# of the correct ~28%).
#
# GNN steelman and the no-demos ablation are omitted entirely (dropped on
# request -- they're ablations for the ANALYSIS.md narrative, not part of
# the headline method comparison this chart is for).

import pandas as pd

OUT_CSV = "results/v5/figures/advantage_over_myopic.csv"

df = pd.read_csv("results/v5/planner/run_summary.csv")

myopic_total = df[df.method_id == "myopic_fd_optimal"]["total_cost_pddl"].sum()


def advantage_single(label, sub):
    total = sub["total_cost_pddl"].sum()
    adv = (myopic_total - total) / myopic_total * 100
    return dict(label=label, mean_adv=adv, min_adv=adv, max_adv=adv, n=1)


def advantage_multi_seed(label, sub):
    totals = sub.groupby("checkpoint")["total_cost_pddl"].sum()
    advantages = (myopic_total - totals) / myopic_total * 100
    return dict(label=label, mean_adv=advantages.mean(), min_adv=advantages.min(),
                max_adv=advantages.max(), n=len(advantages))


rows = [dict(label="Myopic Oracle", mean_adv=0.0, min_adv=0.0, max_adv=0.0, n=1)]

rows.append(advantage_single("K=2 Optimal", df[df.method_id == "k2_fd_optimal"]))
rows.append(advantage_multi_seed("One-task GNN", df[df.method_id == "gnn_faithful"]))
rows.append(advantage_multi_seed("One-task GNN (augmented)", df[df.method_id == "gnn_counterfactual"]))
rows.append(advantage_multi_seed(
    "Anticipatory RL (guided)",
    df[(df.method_id == "anticipatory_dqn") & (df.cost_ratio == 3.0)
       & (df.checkpoint.str.startswith("v5_ant_g0.97_s", na=False))],
))
rows.append(advantage_single("K=3 Clairvoyant Oracle", df[df.method_id == "clairvoyant_k3_lama"]))
rows.append(advantage_single("K=4 Clairvoyant Oracle", df[df.method_id == "clairvoyant_k4_lama"]))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}")
print(out.round(2))
print(f"myopic_total = {myopic_total}")
