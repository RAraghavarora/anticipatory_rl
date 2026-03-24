#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=anticipatory_agent
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g
#SBATCH --time=48:00:00
#SBATCH --output=anticipatory_agent%j.out
#SBATCH --error=anticipatory_agent%j.err

#================================================
# Environment and Job Execution
#================================================
echo "Job started on $(hostname) at $(date)"

source /u/rarora1/ant_env/bin/activate

python -m anticipatory_rl.agents.simple_grid_image_dqn \
  --grid-size 5 \
  --num-objects 4 \
  --total-steps 700000 \
  --replay-size 50000 \
  --batch-size 64 \
  --lr 3e-4 \
  --tasks-per-reset 200 \
  --env-reset-tasks 200 \
  --episode-step-limit 4000 \
  --config-path anticipatory_rl/configs/config_5x5_3r4o.yaml \
  --tau 0.01 \
  --gamma 0.97 \
  --success-reward 12 \
  --clear-receptacle-shaping-scale 3.0

echo "Job finished at $(date)"
