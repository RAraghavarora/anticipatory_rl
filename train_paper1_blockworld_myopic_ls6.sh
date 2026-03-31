#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J p1bw_myo_tpr1_ls6
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

module load cuda/12.2
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate llm

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

srun python -m anticipatory_rl.agents.paper1_blockworld_image_dqn \
  --run-label myopic_blockworld \
  --total-steps 500000 \
  --replay-size 10000 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tau 0.01 \
  --num-envs 8 \
  --render-tile-px 12 \
  --tasks-per-reset 1 \
  --env-reset-tasks 10 \
  --episode-step-limit 4000 \
  --task-library-size 24 \
  --max-task-steps 64 \
  --success-reward 12 \
  --step-penalty 1.0 \
  --invalid-action-penalty 5.0 \
  --correct-pick-bonus 1.0 \
  --seed 0

date
echo "Job finished"
