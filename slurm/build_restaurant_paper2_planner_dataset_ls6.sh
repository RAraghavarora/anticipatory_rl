#!/bin/bash
# Build planner-labeled restaurant dataset on LS6.

#SBATCH -J rest_p2_dataset_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -A IRI23005
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
OUT_PATH="${OUT_PATH:-data/restaurant_planner_dataset/paper2_planner_labels.json}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-32}}"

srun python -m anticipatory_rl.tasks.build_restaurant_planner_dataset \
  --layout-corpus "${CORPUS_PATH}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --planner-path downward/fast-downward.py \
  --domain-path pddl/restaurant_domain.pddl \
  --num-states 2000 \
  --followup-samples 8 \
  --jobs "${JOBS}" \
  --output-path "${OUT_PATH}" \
  --seed 0
