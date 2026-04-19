#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J rest_myo_tpr1_ls6
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
conda activate thesis 

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

CONFIG_PATH=anticipatory_rl/configs/restaurant_symbolic.yaml

srun python -m anticipatory_rl.agents.restaurant_dqn \
  --config-path "${CONFIG_PATH}" \
  --run-label restaurant_myopic \
  --total-steps 500000 \
  --replay-size 50000 \
  --batch-size 128 \
  --hidden-dim 256 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tasks-per-episode 1 \
  --task-sequence-length 200 \
  --episode-step-limit 3000 \
  --max-steps-per-task 100 \
  --success-reward 15 \
  --invalid-action-penalty 6 \
  --travel-cost-scale 1.0 \
  --pick-cost 1.0 \
  --place-cost 1.0 \
  --wash-cost 2.0 \
  --fill-cost 1.0 \
  --brew-cost 2.0 \
  --fruit-cost 2.0 \
  --seed 0

date
echo "Job finished"