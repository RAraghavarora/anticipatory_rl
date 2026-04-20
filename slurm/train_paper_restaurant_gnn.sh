#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=paper_rest_gnn
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p paper_restaurant/checkpoints

DATASET_WORKERS=${SLURM_CPUS_PER_TASK:-1}

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"

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

if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=${SLURM_GPUS_ON_NODE}
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  IFS=',' read -ra CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0}"
  NUM_GPUS=${#CUDA_DEVICES[@]}
fi

MASTER_PORT=${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
ACCELERATE_ARGS=(
  --num_processes "${NUM_GPUS}"
  --num_machines 1
  --main_process_port "${MASTER_PORT}"
  --mixed_precision no
  --dynamo_backend no
)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

srun --ntasks=1 python -m accelerate.commands.launch "${ACCELERATE_ARGS[@]}" -m restaurant.paper_restaurant.train_gnn \
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
  --dataset-workers "${DATASET_WORKERS}" \
  --device cuda \
  --output-dir paper_restaurant/checkpoints \
  --seed 0

echo "Job finished at $(date -Is)"
