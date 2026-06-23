#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J rest_ant_g09_s42_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A ASC26023
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

module load cuda/12.2
PARENT_DIR="$(dirname "$PWD")"
source "$PARENT_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate thesis

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

srun python -m anticipatory_rl.agents.restaurant.dqn \
  --run-label restaurant_anticipatory_gamma09_seed42 \
  --tasks-per-episode 200 \
  --env-reset-tasks 200 \
  --total-steps 1000000 \
  --seed 42 \
  --gamma 0.9 \
  --config-path configs/restaurant/toy_level_2_2.yaml

date
echo "Job finished"
