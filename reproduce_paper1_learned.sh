#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=paper1_eval
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p runs/paper1_blockworld_exact

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

CHECKPOINT=paper1_blockworld/checkpoints/paper1_anticipatory_gnn.pt
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

python -m paper1_blockworld.reproduce_paper1 \
  --paper-settings \
  --tasks-per-environment 24 \
  --preparation-iterations 200 \
  --candidate-goal-limit 24 \
  --estimator learned \
  --device cuda \
  --gnn-checkpoint "${CHECKPOINT}" \
  --output-json runs/paper1_blockworld_exact/paper1_learned_eval.json \
  --seed 0

echo "Job finished at $(date -Is)"
