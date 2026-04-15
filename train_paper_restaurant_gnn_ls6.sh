#!/bin/bash

#================================================
# SBATCH
#================================================
#SBATCH -J paper_rest_gnn
#SBATCH -A CCR25013
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p paper_restaurant/checkpoints

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
export PYTHONUNBUFFERED=1

NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}

srun --ntasks=1 python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision no \
  --dynamo_backend no \
  -m paper_restaurant.train_gnn \
  --num-train-envs 96 \
  --num-val-envs 16 \
  --states-per-env 64 \
  --tasks-per-environment 72 \
  --future-task-sample all \
  --epochs 10 \
  --batch-size 8 \
  --lr 0.01 \
  --hidden-dim 256 \
  --num-layers 4 \
  --heads 4 \
  --dataset-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda \
  --output-dir paper_restaurant/checkpoints \
  --seed 0
