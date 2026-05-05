#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J rest_sf_myopic_ls6
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
source /work/11373/raghavaurora2/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

# Run training with myopic mode (no bootstrapping at task success)
srun python -m anticipatory_rl.agents.restaurant.sf_dqn \
    --total-steps 500000 \
    --sf-dim 64 \
    --myopic \
    --max-steps-per-task 24 \
    --output-name "restaurant_sf_myopic.pt" \
    --run-label "sf_myopic_$(date +%Y%m%d_%H%M%S)"

date
echo "Job finished"
