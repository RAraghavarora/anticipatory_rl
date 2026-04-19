#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
AP_WEIGHTS="${AP_WEIGHTS:-runs/restaurant_apcost_gnn/apcost_estimator.pt}"
OUT_PATH="${OUT_PATH:-runs/paper2_planner_compare/planner_compare.json}"

python -m anticipatory_rl.tasks.restaurant_planner_policy_eval \
  --layout-corpus "${CORPUS_PATH}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --planner-path downward/fast-downward.py \
  --domain-path pddl/restaurant_domain.pddl \
  --apcost-weights "${AP_WEIGHTS}" \
  --eval-layout-count 500 \
  --task-sequence-length 40 \
  --sample-layout-per-reset \
  --anticipatory-followups 8 \
  --output-path "${OUT_PATH}" \
  --seed 0
