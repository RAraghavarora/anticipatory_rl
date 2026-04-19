#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
OUT_ROOT="${OUT_ROOT:-runs/paper2_scale_medium}"
SEED_FROM="${SEED_FROM:-0}"
SEED_TO="${SEED_TO:-1}"

python scripts/generate_restaurant_layout_corpus.py \
  --output-path "${CORPUS_PATH}" \
  --num-layouts 1000 \
  --seed 0

python scripts/validate_restaurant_layout_corpus.py \
  --corpus-path "${CORPUS_PATH}"

for SEED in $(seq "${SEED_FROM}" "${SEED_TO}"); do
  python -m anticipatory_rl.agents.restaurant_dqn \
    --run-label paper2_medium_myo_seed"${SEED}" \
    --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
    --layout-corpus "${CORPUS_PATH}" \
    --sample-layout-per-reset \
    --task-sequence-length 200 \
    --tasks-per-episode 1 \
    --total-steps 500000 \
    --seed "${SEED}"

  python -m anticipatory_rl.agents.restaurant_dqn \
    --run-label paper2_medium_ant_seed"${SEED}" \
    --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
    --layout-corpus "${CORPUS_PATH}" \
    --sample-layout-per-reset \
    --task-sequence-length 200 \
    --tasks-per-episode 200 \
    --total-steps 500000 \
    --seed "${SEED}"
done

python paper_restaurant/scripts/restaurant_multi_seed_infer.py \
  --anticipatory-weights runs/paper2_medium_ant_seed0/restaurant_dqn.pt \
  --myopic-weights runs/paper2_medium_myo_seed0/restaurant_dqn.pt \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus "${CORPUS_PATH}" \
  --sample-layout-per-reset \
  --eval-layout-count 500 \
  --task-sequence-length 200 \
  --num-tasks 20000 \
  --seed-from "${SEED_FROM}" \
  --seed-to "${SEED_TO}" \
  --output-dir "${OUT_ROOT}/infer"
