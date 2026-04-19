#!/usr/bin/env bash
set -euo pipefail

RL_COMPARISON="${RL_COMPARISON:-runs/paper2_scale_full/infer/comparison.json}"
PLANNER_COMPARISON="${PLANNER_COMPARISON:-runs/paper2_planner_compare/planner_compare.json}"
OUT_ROOT="${OUT_ROOT:-runs/paper2_cross_family}"

python paper_restaurant/scripts/eval_restaurant_rl_vs_planner.py \
  --rl-comparison "${RL_COMPARISON}" \
  --planner-comparison "${PLANNER_COMPARISON}" \
  --output-json "${OUT_ROOT}/unified_comparison.json" \
  --output-csv "${OUT_ROOT}/unified_comparison.csv"
