#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
OUT_PATH="${OUT_PATH:-data/restaurant_planner_dataset/paper2_planner_labels.json}"
JOBS="${JOBS:-8}"

python scripts/validate_restaurant_layout_corpus.py --corpus-path "${CORPUS_PATH}"

python -m anticipatory_rl.tasks.build_restaurant_planner_dataset \
  --layout-corpus "${CORPUS_PATH}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --planner-path downward/fast-downward.py \
  --domain-path pddl/restaurant_domain.pddl \
  --num-states 2000 \
  --followup-samples 8 \
  --jobs "${JOBS}" \
  --output-path "${OUT_PATH}" \
  --seed 0
