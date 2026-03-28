#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J paper1_eval_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A IRI23005
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p runs/paper1_blockworld_exact

module load cuda/12.2
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

CHECKPOINT=paper1_blockworld/checkpoints/paper1_anticipatory_gnn.pt
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

srun python -m paper1_blockworld.reproduce_paper1 \
  --paper-settings \
  --tasks-per-environment 24 \
  --preparation-iterations 200 \
  --candidate-goal-limit 24 \
  --estimator learned \
  --device cuda \
  --gnn-checkpoint "${CHECKPOINT}" \
  --output-json runs/paper1_blockworld_exact/paper1_learned_eval.json \
  --seed 0

date
echo "Job finished"
