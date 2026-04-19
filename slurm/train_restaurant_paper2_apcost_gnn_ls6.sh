#!/bin/bash
# Train planner-side APCostEstimator on LS6.

#SBATCH -J rest_p2_gnn_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -A IRI23005
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
module load cuda/12.2
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

DATASET_PATH="${DATASET_PATH:-data/restaurant_planner_dataset/paper2_planner_labels.json}"
OUT_DIR="${OUT_DIR:-runs/restaurant_apcost_gnn}"

srun python -m anticipatory_rl.agents.train_restaurant_apcost_gnn \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUT_DIR}" \
  --epochs 10 \
  --hidden-dim 128 \
  --layers 4 \
  --seed 0
