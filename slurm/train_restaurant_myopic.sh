#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=rest_myo_cap_tpr1
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

set -euo pipefail
echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

source /u/rarora1/ant_env/bin/activate
CONFIG_PATH=anticipatory_rl/configs/restaurant_symbolic.yaml

python -m anticipatory_rl.agents.restaurant.restaurant_dqn \
  --config-path "${CONFIG_PATH}" \
  --run-label restaurant_capacity_myopic \
  --total-steps 500000 \
  --replay-size 50000 \
  --batch-size 128 \
  --hidden-dim 256 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tasks-per-reset 1 \
  --env-reset-tasks 200 \
  --episode-step-limit 3000 \
  --max-task-steps 24 \
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

echo "Job finished at $(date -Is)"
