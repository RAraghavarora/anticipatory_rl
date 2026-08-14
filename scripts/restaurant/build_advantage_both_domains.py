# Advantage over Myopic Oracle (%), 2-room and 3-room (v5) merged into one
# table -- (domain_myopic_total - method_total) / domain_myopic_total * 100,
# each total summed over that domain's own 10-sequence x 50-task canonical
# run, against THAT DOMAIN's OWN myopic total (670,025 for 2-room, 674,500
# for v5 -- two different numbers, never cross-divided). The percentage
# itself is what makes merging legitimate: it's relative to each domain's
# own baseline, not a raw cost that would conflate two different-sized
# domains.
#
# Method rosters differ only in the oracle arms now -- 2-room has K=2/K=3,
# v5 additionally has K=4. Every RL/GNN method now exists in both domains
# (v5's Myopic RL is a partial seed sweep, not padded to match). Methods
# present in both domains keep the same base label so the plot can
# group/pair them; the domain is a separate column, not baked into the label.
#
# 2-room greedy DQN rows (Myopic RL greedy, Anticipatory RL greedy) use
# best-(seed,checkpoint_variant)-only, matching every other 2-room figure in
# this archive (results/canonical_planner/greedy_rl/seed_summary.csv) --
# myopic-greedy seed 16's final checkpoint diverged late in training, so
# pooling all seeds isn't the established convention there. v5's greedy
# rows have no such divergence and pool all 5 seeds (see the raincloud
# figure's build script).

import glob
import json

import numpy as np
import pandas as pd

OUT_CSV = "results/v5/figures/advantage_both_domains.csv"
# v5 DQN seeds -- 5 now (seed 42). 2-room reads its own checkpoint_seed and
# GNN has only 4 seeds; both are handled by their own groupby, not this list.
V5_DQN_SEEDS = [0, 4, 8, 16, 42]
rows = []

# ============================== 2-room ==============================
guided = pd.read_csv("results/canonical_planner/planner/run_summary.csv")
myopic_total_2room = guided[guided.method_id == "myopic_fd_optimal"]["total_cost_pddl"].sum()


def adv_2room(total):
    return (myopic_total_2room - total) / myopic_total_2room * 100


rows.append(dict(domain="2-room", label="Myopic Oracle", mean_adv=0.0, min_adv=0.0, max_adv=0.0, n=1))

for method_id, label in [("k2_fd_optimal", "K=2 Optimal"), ("clairvoyant_k3_lama", "K=3 Clairvoyant Oracle")]:
    total = guided[guided.method_id == method_id]["total_cost_pddl"].sum()
    rows.append(dict(domain="2-room", label=label, mean_adv=adv_2room(total),
                      min_adv=adv_2room(total), max_adv=adv_2room(total), n=1))

# Both 2-room guided arms at beta=1.25 (myopic re-evaluated to match the
# anticipatory search bound; myopic_dqn_beta1 was the old beta=1.00 arm).
for method_id, label in [("myopic_dqn_beta1_25", "Myopic RL (guided)"),
                          ("anticipatory_dqn_beta1_25", "Anticipatory RL (guided)")]:
    totals = guided[guided.method_id == method_id].groupby("checkpoint_seed")["total_cost_pddl"].sum()
    advantages = adv_2room(totals)
    rows.append(dict(domain="2-room", label=label, mean_adv=advantages.mean(),
                      min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

gnn_seed = pd.read_csv("results/canonical_planner/gnn/seed_summary.csv")
for method_id, label in [("gnn_faithful", "One-task GNN"), ("gnn_counterfactual", "One-task GNN (augmented)")]:
    totals = gnn_seed[gnn_seed.method_id == method_id].groupby("seed")["total_cost_pddl"].sum()
    advantages = adv_2room(totals)
    rows.append(dict(domain="2-room", label=label, mean_adv=advantages.mean(),
                      min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

# Greedy DQN: best (seed, variant) only, same selection rule as
# plot_canonical_cost_raincloud.R (min mean_cost_pddl per method_id).
greedy_all = pd.read_csv("results/canonical_planner/greedy_rl/seed_summary.csv")
for method_id, label in [("myopic_dqn_greedy", "Myopic RL (greedy)"),
                         ("anticipatory_dqn_greedy", "Anticipatory RL (greedy)")]:
    sub = greedy_all[greedy_all.method_id == method_id]
    best = sub.loc[sub.mean_cost_pddl.idxmin()]
    total = best.mean_cost_pddl * best.task_count  # task_count = 500 = 10 seqs x 50 tasks
    adv = adv_2room(total)
    rows.append(dict(domain="2-room", label=label, mean_adv=adv, min_adv=adv, max_adv=adv, n=1))

# ============================== 3-room (v5) ==============================
v5 = pd.read_csv("results/v5/planner/run_summary.csv")
myopic_total_v5 = v5[v5.method_id == "myopic_fd_optimal"]["total_cost_pddl"].sum()


def adv_v5(total):
    return (myopic_total_v5 - total) / myopic_total_v5 * 100


rows.append(dict(domain="3-room", label="Myopic Oracle", mean_adv=0.0, min_adv=0.0, max_adv=0.0, n=1))

for method_id, label in [("k2_fd_optimal", "K=2 Optimal"),
                         ("clairvoyant_k3_lama", "K=3 Clairvoyant Oracle"),
                         ("clairvoyant_k4_lama", "K=4 Clairvoyant Oracle")]:
    total = v5[v5.method_id == method_id]["total_cost_pddl"].sum()
    adv = adv_v5(total)
    rows.append(dict(domain="3-room", label=label, mean_adv=adv, min_adv=adv, max_adv=adv, n=1))

for method_id, label in [("gnn_faithful", "One-task GNN"), ("gnn_counterfactual", "One-task GNN (augmented)")]:
    totals = v5[v5.method_id == method_id].groupby("checkpoint")["total_cost_pddl"].sum()
    advantages = adv_v5(totals)
    rows.append(dict(domain="3-room", label=label, mean_adv=advantages.mean(),
                      min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

anticipatory_guided = v5[(v5.method_id == "anticipatory_dqn") & (v5.cost_ratio == 3.0)
                         & (v5.checkpoint.str.startswith("v5_ant_g0.97_s", na=False))]
totals = anticipatory_guided.groupby("checkpoint")["total_cost_pddl"].sum()
advantages = adv_v5(totals)
rows.append(dict(domain="3-room", label="Anticipatory RL (guided)", mean_adv=advantages.mean(),
                  min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

# Myopic RL, v5: 5-seed sweep for both guided and greedy. Both already have
# per-sequence total_cost_pddl directly in run_summary.csv, same as the
# anticipatory rows.
MYOPIC_CHECKPOINTS = [f"v5-myopic-g097-s{s}" for s in V5_DQN_SEEDS]
myopic_guided = v5[(v5.method_id == "myopic_dqn") & (v5.checkpoint.isin(MYOPIC_CHECKPOINTS))]
totals = myopic_guided.groupby("checkpoint")["total_cost_pddl"].sum()
advantages = adv_v5(totals)
rows.append(dict(domain="3-room", label="Myopic RL (guided)", mean_adv=advantages.mean(),
                  min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

myopic_greedy = v5[(v5.method_id == "myopic_dqn_greedy") & (v5.checkpoint.isin(MYOPIC_CHECKPOINTS))]
totals = myopic_greedy.groupby("checkpoint")["total_cost_pddl"].sum()
advantages = adv_v5(totals)
rows.append(dict(domain="3-room", label="Myopic RL (greedy)", mean_adv=advantages.mean(),
                  min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

# Anticipatory RL (greedy), v5: no per-domain run_summary.csv row exists for
# this arm -- parse the raw per-seed trajectory JSONs (same source as
# build_v5_cost_raincloud.py), pooling all 5 seeds (no known divergence to
# exclude, unlike 2-room's myopic-greedy seed 16).
seed_totals = []
for s in V5_DQN_SEEDS:
    files = sorted(glob.glob(f"results/v5/greedy/raw/greedy_s{s}_seq*.json"))
    assert len(files) == 10, (s, len(files))
    total = sum(json.load(open(f))["anticipatory"]["summary"]["total_pddl_cost"] for f in files)
    seed_totals.append(total)
advantages = adv_v5(np.array(seed_totals))
rows.append(dict(domain="3-room", label="Anticipatory RL (greedy)", mean_adv=advantages.mean(),
                  min_adv=advantages.min(), max_adv=advantages.max(), n=len(advantages)))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(out)} rows)")
print(out.round(2).to_string())
print(f"myopic_total_2room={myopic_total_2room}  myopic_total_v5={myopic_total_v5}")
