#!/bin/bash
# Submit from the repo root (anticipatory_rl/) so paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=p1bw_myo_tpr1
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

python -m anticipatory_rl.agents.paper1_blockworld_image_dqn \
  --run-label myopic_blockworld \
  --total-steps 500000 \
  --replay-size 50000 \
  --batch-size 64 \
  --hidden-dim 256 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tau 0.01 \
  --num-envs 8 \
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

echo "Job finished at $(date -Is)"
