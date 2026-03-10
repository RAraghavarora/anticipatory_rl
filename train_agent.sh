#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=train_3box_agent      # Name of the job
#SBATCH --account=bewg-delta-gpu
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
python -m anticipatory_rl.agents.three_box_dqn --step-cost 1.0 --render-tile-px 4 --total-steps 500_000
echo "Job finished at $(date)"