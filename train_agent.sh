#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=train_3box_agent      # Name of the job
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1                     # Total number of nodes to request
#SBATCH --tasks=1                     # Total number of tasks (processes) to run
#SBATCH --gpus-per-node=1             # Number of GPUs per node
#SBATCH --cpus-per-task=16             # Cores per task
#SBATCH --mem=20g                     # Main memory requested
#SBATCH --time=48:00:00               # Duration of job
#SBATCH --output=train_3box_job_%j.out     # Standard output log file
#SBATCH --error=train_3box_job_%j.err      # Standard error log file

#================================================
# Environment and Job Execution
#================================================
echo "Job started on $(hostname) at $(date)"

source /u/rarora1/ant_env/bin/activate
# python -m anticipatory_rl.agents.three_box_dqn --step-cost 1.0 --render-tile-px 4 --total-steps 500_000
python -m anticipatory_rl.agents.simple_grid_image_dqn \
  --grid-size 5 \
  --num-objects 3 \
  --total-steps 300000 \
  --replay-size 50000 \
  --batch-size 64 \
  --lr 3e-4 \
  --tasks-per-reset 20 \
  --episode-step-limit 4000 \
  --num-envs 4 \
  --output runs/simple_grid_image_dqn_5x5_3r3o.pt
  --config-file configs/simple_grid_image_dqn_5x5_3r3o.yaml
echo "Job finished at $(date)"
