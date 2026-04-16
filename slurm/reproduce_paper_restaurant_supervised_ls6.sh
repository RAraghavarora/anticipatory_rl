#!/bin/bash

#================================================
# SBATCH
#================================================
#SBATCH -J paper_rest_eval
#SBATCH -A CCR25013
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p runs/paper_restaurant

module load cuda/12.2

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script." >&2
  exit 1
fi
conda activate thesis

CHECKPOINT=paper_restaurant/checkpoints/paper_restaurant_anticipatory_gnn.pt
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

srun python -m paper_restaurant.reproduce_restaurant_supervised \
  --paper-settings \
  --tasks-per-environment 72 \
  --candidate-goal-limit 24 \
  --estimator learned \
  --device cuda \
  --gnn-checkpoint "${CHECKPOINT}" \
  --output-json runs/paper_restaurant/paper_restaurant_supervised_eval.json \
  --output-plot runs/paper_restaurant/paper_restaurant_supervised_cost_curve.png \
  --seed 0
