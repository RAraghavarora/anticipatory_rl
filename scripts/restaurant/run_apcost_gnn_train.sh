#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/restaurant_planner_dataset/paper2_planner_labels.json}"
OUT_DIR="${OUT_DIR:-runs/restaurant_apcost_gnn}"

python -m anticipatory_rl.agents.train_restaurant_apcost_gnn \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUT_DIR}" \
  --epochs 10 \
  --hidden-dim 128 \
  --layers 4 \
  --seed 0
