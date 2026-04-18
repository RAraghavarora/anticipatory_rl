#!/usr/bin/env bash
set -euo pipefail

# Full protocol target: 500 layout sequences x 40 tasks = 20,000 tasks/seed.
# Provide trained checkpoints via ANT_WEIGHTS and MYO_WEIGHTS.

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
OUT_ROOT="${OUT_ROOT:-runs/paper2_scale_full}"
SEED_FROM="${SEED_FROM:-0}"
SEED_TO="${SEED_TO:-4}"
ANT_WEIGHTS="${ANT_WEIGHTS:-runs/paper2_full_ant_seed0/restaurant_dqn.pt}"
MYO_WEIGHTS="${MYO_WEIGHTS:-runs/paper2_full_myo_seed0/restaurant_dqn.pt}"

python scripts/validate_restaurant_layout_corpus.py \
  --corpus-path "${CORPUS_PATH}"

python paper_restaurant/scripts/restaurant_multi_seed_infer.py \
  --anticipatory-weights "${ANT_WEIGHTS}" \
  --myopic-weights "${MYO_WEIGHTS}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus "${CORPUS_PATH}" \
  --sample-layout-per-reset \
  --eval-layout-count 500 \
  --task-sequence-length 40 \
  --num-tasks 20000 \
  --tasks-per-reset 40 \
  --seed-from "${SEED_FROM}" \
  --seed-to "${SEED_TO}" \
  --output-dir "${OUT_ROOT}/infer"
