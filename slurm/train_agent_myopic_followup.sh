#!/bin/bash
# Submit from the repo root (anticipatory_rl/) so paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=imgdq_myo_paired_tpr1
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
CONFIG_PATH=anticipatory_rl/configs/config_5x5_3r4o_paired_clear_followup.yaml

python -m anticipatory_rl.agents.simple_grid_image_dqn \
  --grid-size 5 \
  --num-objects 4 \
  --max-task-steps 200 \
  --run-label myopic_paired_followup \
  --total-steps 700000 \
  --replay-size 50000 \
  --batch-size 64 \
  --lr 3e-4 \
  --tasks-per-reset 1 \
  --env-reset-tasks 3 \
  --episode-step-limit 600 \
  --config-path "${CONFIG_PATH}" \
  --ensure-receptacle-coverage \
  --tau 0.01 \
  --gamma 0.97 \
  --success-reward 12 \
  --correct-pick-bonus 0.0 \
  --distance-reward-scale 0.0 \
  --clear-receptacle-shaping-scale 3.0 \
  --seed 0

echo "Job finished at $(date -Is)"
