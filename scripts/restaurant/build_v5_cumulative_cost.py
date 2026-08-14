# Per-task PDDL cost, v5, long format (method_id, seed, seq, task_index,
# cost) -- the raw material for a v5 cumulative-cost-vs-tasks curve to
# match scripts/plotting/plot_canonical_cost_ggdist.R's 2-room figure.
#
# Six methods, three JSON schemas:
#   - myopic_fd_optimal / clairvoyant_k3_lama (oracles, no seed): raw FD
#     search output has a `windows` list; `first_segment_cost` is the cost
#     of the ONE task actually executed at that window (index i), which is
#     what a task-by-task cumulative curve needs -- not `window_plan_cost`,
#     which is the full K-task lookahead plan's cost.
#   - gnn_faithful / gnn_counterfactual (4-seed sweep): `tasks[i].cost`
#     under the `gnn_anticipatory` key.
#   - anticipatory_dqn guided (gamma=0.97, cost_ratio=3.0, 4 seeds):
#     `tasks[i].cost` under the `guided` key.
#   - anticipatory_dqn greedy (4 seeds): `tasks[i].pddl_cost` under the
#     `anticipatory` key (direct policy rollout, different field name).
#   - myopic_dqn guided / myopic_dqn_greedy: same schemas as their
#     anticipatory counterparts above.
#
# DQN arms use 5 seeds {0,4,8,16,42}; GNN arms use 4 (no seed 42).

import glob
import json

import pandas as pd

OUT_CSV = "results/v5/figures/cumulative_cost_per_task.csv"
# DQN arms have seed 42 now (5 seeds); the GNN arms don't (still 4).
DQN_SEEDS = [0, 4, 8, 16, 42]
GNN_SEEDS = [0, 4, 8, 16]
rows = []


def add_oracle(method_id, dir_name):
    for f in sorted(glob.glob(f"results/v5/planner/raw/{dir_name}/*.json")):
        d = json.load(open(f))
        seq = d["sequence_id"]
        for w in d["windows"]:
            rows.append(dict(method_id=method_id, seed=None, seq=seq,
                              task_index=w["index"], cost=w["first_segment_cost"]))


def add_seeded(method_id, file_pattern, key_path, cost_field, seeds, index_field="index"):
    for s in seeds:
        files = sorted(glob.glob(file_pattern.format(seed=s)))
        assert len(files) == 10, (method_id, s, len(files))
        for f in files:
            d = json.load(open(f))
            for key in key_path:
                d = d[key]
            seq = f.split("seq")[1][:2]
            for t in d:
                rows.append(dict(method_id=method_id, seed=s, seq=f"seq{seq}",
                                  task_index=t[index_field], cost=t[cost_field]))


add_oracle("myopic_fd_optimal", "myopic_fd_optimal")
add_oracle("clairvoyant_k3_lama", "clairvoyant_k3_lama")
add_seeded("gnn_faithful", "results/v5/gnn/seeds/gnn_faithful_s{seed}_seq*.json",
           ["gnn_anticipatory", "tasks"], "cost", GNN_SEEDS)
add_seeded("gnn_counterfactual", "results/v5/gnn/seeds/gnn_aug_s{seed}_seq*.json",
           ["gnn_anticipatory", "tasks"], "cost", GNN_SEEDS)
add_seeded("anticipatory_dqn_guided", "results/v5/planner/raw/anticipatory_dqn/v5_ant_g0.97_s{seed}_seq*_cr3.0.json",
           ["guided", "tasks"], "cost", DQN_SEEDS)
add_seeded("anticipatory_dqn_greedy", "results/v5/greedy/raw/greedy_s{seed}_seq*.json",
           ["anticipatory", "tasks"], "pddl_cost", DQN_SEEDS, index_field="task_idx")
add_seeded("myopic_dqn_guided", "results/v5/planner/raw/myopic_dqn/v5-myopic-g097-s{seed}_seq*_cr3.0.json",
           ["guided", "tasks"], "cost", DQN_SEEDS)
add_seeded("myopic_dqn_greedy", "results/v5/greedy/raw/greedy_v5-myopic-g097-s{seed}_seq*.json",
           ["anticipatory", "tasks"], "pddl_cost", DQN_SEEDS, index_field="task_idx")

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(out)} rows)")
print(out.groupby("method_id").size())
