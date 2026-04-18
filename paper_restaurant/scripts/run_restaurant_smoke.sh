#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
OUT_ROOT="${OUT_ROOT:-runs/paper2_scale_smoke}"
SEED="${SEED:-0}"

python scripts/generate_restaurant_layout_corpus.py \
  --output-path "${CORPUS_PATH}" \
  --num-layouts 1000 \
  --seed 0

python scripts/validate_restaurant_layout_corpus.py \
  --corpus-path "${CORPUS_PATH}" \
  --head 50

python -m anticipatory_rl.agents.restaurant_dqn \
  --run-label paper2_smoke_myo_seed"${SEED}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus "${CORPUS_PATH}" \
  --sample-layout-per-reset \
  --task-sequence-length 40 \
  --tasks-per-reset 1 \
  --env-reset-tasks 40 \
  --total-steps 150000 \
  --seed "${SEED}"

python -m anticipatory_rl.agents.restaurant_dqn \
  --run-label paper2_smoke_ant_seed"${SEED}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus "${CORPUS_PATH}" \
  --sample-layout-per-reset \
  --task-sequence-length 40 \
  --tasks-per-reset 40 \
  --env-reset-tasks 40 \
  --total-steps 150000 \
  --seed "${SEED}"

python paper_restaurant/scripts/restaurant_multi_seed_infer.py \
  --anticipatory-weights runs/paper2_smoke_ant_seed"${SEED}"/restaurant_dqn.pt \
  --myopic-weights runs/paper2_smoke_myo_seed"${SEED}"/restaurant_dqn.pt \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus "${CORPUS_PATH}" \
  --sample-layout-per-reset \
  --eval-layout-count 10 \
  --task-sequence-length 40 \
  --num-tasks 400 \
  --tasks-per-reset 40 \
  --seed-from "${SEED}" \
  --seed-to "${SEED}" \
  --output-dir "${OUT_ROOT}/infer"
