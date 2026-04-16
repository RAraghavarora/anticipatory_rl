#!/bin/bash
# Submit from the repo root (anticipatory_rl/) so paths resolve.
# Log directory is committed as slurm_logs/; if missing: mkdir -p slurm_logs

#================================================
# SBATCH — logs under slurm_logs/ with readable job name + unique job id
#================================================
#SBATCH --job-name=inference
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/infer_%x.%j.out
#SBATCH --error=slurm_logs/infer_%x.%j.err

#================================================
# Environment and job
#================================================
set -euo pipefail
echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"

source /u/rarora1/ant_env/bin/activate
CONFIG_PATH=./anticipatory_rl/configs/config_5x5_3r4o.yaml

python -m anticipatory_rl.agents.simple_grid_image_dqn_infer \
  --anticipatory-weights ./runs/5_anticipatory_image_dqn_tpr200/simple_grid_image_dqn.pt \
  --myopic-weights ./runs/5_myopic_image_dqn_tpr1/simple_grid_image_dqn.pt \
  --output-dir ./runs/compare_ant_vs_myo \
  --grid-size 5 \
  --num-objects 4 \
  --max-task-steps 200 \
  --config-path "${CONFIG_PATH}" \
  --ensure-receptacle-coverage \
  --tasks-per-reset 200 \
  --success-reward 12 \
  --clear-receptacle-shaping-scale 3.0 \
  --num-tasks 1000 \
  --total-steps 200000 \
  --gamma 0.97 \
  --no-save-frames \
  --seed 0

echo "Job finished at $(date -Is)"
