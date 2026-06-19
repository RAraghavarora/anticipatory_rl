#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
CONFIG_PATH="${CONFIG_PATH:-configs/restaurant/toy_restaurant.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/toy_restaurant}"
OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/inference/persistent_sweep_seed0}"
NUM_TASKS="${NUM_TASKS:-1000}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
TASKS_PER_RESET="${TASKS_PER_RESET:-200}"
MAX_TASK_STEPS="${MAX_TASK_STEPS:-64}"
SEED="${SEED:-0}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-0.0}"

run_single() {
  local label="$1"
  local gamma="$2"
  local checkpoint="$3"
  local config_path="${4:-${CONFIG_PATH}}"
  local output_dir="${OUT_ROOT}/${label}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping ${label}: missing checkpoint ${checkpoint}" >&2
    return 0
  fi

  echo "Running persistent inference: ${label}"
  conda run -n thesis python -m anticipatory_rl.agents.restaurant.restaurant_dqn_infer \
    --state-dict "${checkpoint}" \
    --output-dir "${output_dir}" \
    --num-tasks "${NUM_TASKS}" \
    --total-steps "${TOTAL_STEPS}" \
    --tasks-per-reset "${TASKS_PER_RESET}" \
    --max-task-steps "${MAX_TASK_STEPS}" \
    --config-path "${config_path}" \
    --seed "${SEED}" \
    --gamma "${gamma}" \
    --softmax-temperature "${SOFTMAX_TEMPERATURE}"
}

run_compare() {
  local label="$1"
  local gamma="$2"
  local anticipatory_checkpoint="$3"
  local myopic_checkpoint="$4"
  local config_path="${5:-${CONFIG_PATH}}"
  local output_dir="${OUT_ROOT}/${label}"

  if [[ ! -f "${anticipatory_checkpoint}" ]]; then
    echo "Skipping ${label}: missing anticipatory checkpoint ${anticipatory_checkpoint}" >&2
    return 0
  fi
  if [[ ! -f "${myopic_checkpoint}" ]]; then
    echo "Skipping ${label}: missing myopic checkpoint ${myopic_checkpoint}" >&2
    return 0
  fi

  echo "Running persistent comparison: ${label}"
  conda run -n thesis python -m anticipatory_rl.agents.restaurant.restaurant_dqn_infer \
    --anticipatory-weights "${anticipatory_checkpoint}" \
    --myopic-weights "${myopic_checkpoint}" \
    --output-dir "${output_dir}" \
    --num-tasks "${NUM_TASKS}" \
    --total-steps "${TOTAL_STEPS}" \
    --tasks-per-reset "${TASKS_PER_RESET}" \
    --max-task-steps "${MAX_TASK_STEPS}" \
    --config-path "${config_path}" \
    --seed "${SEED}" \
    --gamma "${gamma}" \
    --softmax-temperature "${SOFTMAX_TEMPERATURE}"
}

mkdir -p "${OUT_ROOT}/single" "${OUT_ROOT}/compare"

run_single "single/anticipatory_gamma0.80_seed0" "0.80" "${RUN_ROOT}/anticipatory/gamma0.80_seed0/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.85_seed0" "0.85" "${RUN_ROOT}/anticipatory/gamma0.85_seed0/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.90_seed0" "0.90" "${RUN_ROOT}/anticipatory/gamma0.90_seed0/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.95_seed0" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed0/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.95_seed42" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed42/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.95_seed64" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed64/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.95_seed128" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed128/restaurant_dqn.pt"
run_single "single/anticipatory_gamma0.99_seed0" "0.99" "${RUN_ROOT}/anticipatory/gamma0.99_seed0/restaurant_dqn.pt"

run_single "single/myopic_gamma0.80_seed0" "0.80" "${RUN_ROOT}/myopic/gamma0.80_seed0/restaurant_dqn.pt"
run_single "single/myopic_gamma0.85_seed0" "0.85" "${RUN_ROOT}/myopic/gamma0.85_seed0/restaurant_dqn.pt"
run_single "single/myopic_gamma0.90_seed0" "0.90" "${RUN_ROOT}/myopic/gamma0.90_seed0/restaurant_dqn.pt"
run_single "single/myopic_gamma0.95_seed0" "0.95" "${RUN_ROOT}/myopic/gamma0.95_seed0/restaurant_dqn.pt"
run_single "single/myopic_gamma0.99_seed0" "0.99" "${RUN_ROOT}/myopic/gamma0.99_seed0/restaurant_dqn.pt"

run_compare "compare/gamma0.80_seed0" "0.80" "${RUN_ROOT}/anticipatory/gamma0.80_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.80_seed0/restaurant_dqn.pt"
run_compare "compare/gamma0.85_seed0" "0.85" "${RUN_ROOT}/anticipatory/gamma0.85_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.85_seed0/restaurant_dqn.pt"
run_compare "compare/gamma0.90_seed0" "0.90" "${RUN_ROOT}/anticipatory/gamma0.90_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.90_seed0/restaurant_dqn.pt"
run_compare "compare/gamma0.95_seed0" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.95_seed0/restaurant_dqn.pt"
run_compare "compare/anticipatory_gamma0.95_vs_myopic_gamma0.99_seed0" "0.95" "${RUN_ROOT}/anticipatory/gamma0.95_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.99_seed0/restaurant_dqn.pt"
run_compare "compare/gamma0.99_seed0" "0.99" "${RUN_ROOT}/anticipatory/gamma0.99_seed0/restaurant_dqn.pt" "${RUN_ROOT}/myopic/gamma0.99_seed0/restaurant_dqn.pt"

run_single "v1.5/single/anticipatory_gamma0.95_seed0" "0.95" "${RUN_ROOT}/v1.5/anticipatory/gamma0.95_seed0/restaurant_dqn.pt" "configs/restaurant/toy_level_1_5.yaml"
run_single "v1.5/single/myopic_gamma0.95_seed0" "0.95" "${RUN_ROOT}/v1.5/myopic/gamma0.95_seed0/restaurant_dqn.pt" "configs/restaurant/toy_level_1_5.yaml"
run_compare "v1.5/compare/gamma0.95_seed0" "0.95" "${RUN_ROOT}/v1.5/anticipatory/gamma0.95_seed0/restaurant_dqn.pt" "${RUN_ROOT}/v1.5/myopic/gamma0.95_seed0/restaurant_dqn.pt" "configs/restaurant/toy_level_1_5.yaml"

echo "Persistent inference sweep complete: ${OUT_ROOT}"
