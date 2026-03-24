#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=clear_rec      # Name of the job
#SBATCH --account=bger-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1                     # Total number of nodes to request
#SBATCH --tasks=1                     # Total number of tasks (processes) to run
#SBATCH --gpus-per-node=1             # Number of GPUs per node
#SBATCH --cpus-per-task=16             # Cores per task
#SBATCH --mem=20g                     # Main memory requested
#SBATCH --time=48:00:00               # Duration of job
#SBATCH --output=clear_rec%j.out     # Standard output log file
#SBATCH --error=clear_rec%j.err      # Standard error log file

#================================================
# Environment and Job Execution
#================================================
echo "Job started on $(hostname) at $(date)"

source /u/rarora1/ant_env/bin/activate
# python -m anticipatory_rl.agents.three_box_dqn --step-cost 1.0 --render-tile-px 4 --total-steps 500_000
python -m anticipatory_rl.agents.simple_grid_image_dqn \
  --grid-size 5 \
  --num-objects 4 \
  --total-steps 700000 \
  --replay-size 50000 \
  --batch-size 64 \
  --lr 3e-4 \
  --tasks-per-reset 1000 \
  --episode-step-limit 4000 \
  --config-path anticipatory_rl/configs/config_5x5_3r4o.yaml \
  --tau 0.01 \
  --gamma 0.97 \
  --success-reward 12 \
  --clear-receptacle-shaping-scale 3.0
echo "Job finished at $(date)"
