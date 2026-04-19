#!/bin/bash
# Submit from the repo root so paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=infer_restaurant
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

python -m anticipatory_rl.agents.restaurant_dqn_infer \
  --anticipatory-weights ./runs/restaurant_anticipatory/restaurant_dqn.pt \
  --myopic-weights ./runs/restaurant_myopic/restaurant_dqn.pt \
  --output-dir ./runs/compare_restaurant_dqn \
  --config-path "${CONFIG_PATH}" \
  --num-tasks 5000 \
  --total-steps 250000 \
  --task-sequence-length 200 \
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
  --gamma 0.99 \
  --seed 0

echo "Job finished at $(date -Is)"