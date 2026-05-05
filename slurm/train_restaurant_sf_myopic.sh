#!/bin/bash
#SBATCH --job-name=rest_sf_myopic
#SBATCH --output=slurm_logs/rest_sf_myopic_%j.out
#SBATCH --error=slurm_logs/rest_sf_myopic_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gypsum

# Load modules
module load anaconda3/2021.05
module load cuda/11.8

# Activate conda environment
source /home/aurora/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Navigate to code directory
cd /home/aurora/raghav/raghav/anticipatory_rl

# Run training with myopic mode (no bootstrapping at task success)
python -m anticipatory_rl.agents.restaurant.sf_dqn \
    --total-steps 500000 \
    --sf-dim 64 \
    --myopic \
    --max-steps-per-task 24 \
    --output-name "restaurant_sf_myopic.pt" \
    --run-label "sf_myopic_$(date +%Y%m%d_%H%M%S)"

echo "Training completed at $(date)"