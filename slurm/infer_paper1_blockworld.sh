#!/bin/bash
# Submit from the repo root (anticipatory_rl/) so paths resolve.

#================================================
# SBATCH
#================================================
#SBATCH --job-name=p1bw_infer
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

set -euo pipefail
echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

source /u/rarora1/ant_env/bin/activate

python -m anticipatory_rl.agents.blockworld.blockworld_dqn_infer \
  --anticipatory-weights ./runs/anticipatory_blockworld_paper1_blockworld_image_dqn_tpr10/paper1_blockworld_image_dqn.pt \
  --myopic-weights ./runs/myopic_blockworld_paper1_blockworld_image_dqn_tpr1/paper1_blockworld_image_dqn.pt \
  --output-dir ./runs/compare_blockworld_image_dqn_infer \
  --num-sequences 100 \
  --tasks-per-reset 10 \
  --total-steps 50000 \
  --render-tile-px 12 \
  --max-task-steps 64 \
  --seed 0

echo "Job finished at $(date -Is)"
