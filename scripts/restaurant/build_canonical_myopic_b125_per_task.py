# Per-task rows for the 2-room Myopic RL (guided) arm re-evaluated at
# beta = 1.25 (method_id myopic_dqn_beta1_25), extracted from the raw plan
# JSONs. This exists because the new beta=1.25 arm was only added to
# run_summary.csv and the raw JSONs -- the canonical per-task
# task_results.csv (a hand-managed file, no generator) still carries only
# the old beta=1.00 arm. The figures that need per-task granularity
# (cumulative_cost_ggdist, task_type_breakdown) read this supplement and
# union it alongside task_results.csv rather than mutating that canonical
# file.
#
# Schema matches the columns those R scripts actually use from
# task_results.csv: method_id, checkpoint_seed, task_index, task_type,
# task_cost_pddl. 4 seeds x 10 sequences x 50 tasks = 2000 rows.

import glob
import json
import re

import pandas as pd

OUT_CSV = "results/canonical_planner/planner/myopic_b125_per_task.csv"
RAW_GLOB = "results/canonical_planner/planner/raw/myopic_dqn_beta1_25/myopic_b125_seed*_seq*.json"

rows = []
for f in sorted(glob.glob(RAW_GLOB)):
    m = re.search(r"seed(\d+)_seq(\d+)", f)
    seed = int(m.group(1))
    for t in json.load(open(f))["tasks"]:
        rows.append(dict(
            method_id="myopic_dqn_beta1_25",
            checkpoint_seed=seed,
            task_index=t["index"],
            task_type=t["task_type"],
            task_cost_pddl=t["cost"],
        ))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(out)} rows, {out.checkpoint_seed.nunique()} seeds)")
