#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=paper1_gnn
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
mkdir -p paper1_blockworld/checkpoints

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script." >&2
  exit 1
fi
conda activate thesis

python -m paper1_blockworld.train_gnn \
  --num-train-envs 250 \
  --num-val-envs 0 \
  --num-test-envs 150 \
  --states-per-env 200 \
  --tasks-per-environment 24 \
  --future-task-sample all \
  --epochs 10 \
  --batch-size 8 \
  --lr 0.01 \
  --hidden-dim 128 \
  --num-layers 3 \
  --device cuda \
  --output-dir paper1_blockworld/checkpoints \
  --seed 0

echo "Job finished at $(date -Is)"
