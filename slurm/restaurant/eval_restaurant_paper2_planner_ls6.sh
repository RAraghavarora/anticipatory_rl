#!/bin/bash
# Evaluate planner myopic vs anticipatory baseline on LS6.

#SBATCH -J rest_p2_planner_eval_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A IRI23005
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
module load cuda/12.2
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

CORPUS_PATH="${CORPUS_PATH:-data/restaurant_layouts/paper2_scale_layouts.json}"
AP_WEIGHTS="${AP_WEIGHTS:-runs/restaurant_apcost_gnn/apcost_estimator.pt}"
OUT_PATH="${OUT_PATH:-runs/paper2_planner_compare/planner_compare.json}"

srun python -m anticipatory_rl.tasks.restaurant_planner_policy_eval \
  --layout-corpus "${CORPUS_PATH}" \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --planner-path downward/fast-downward.py \
  --domain-path pddl/restaurant_domain.pddl \
  --apcost-weights "${AP_WEIGHTS}" \
  --eval-layout-count 500 \
  --task-sequence-length 40 \
  --sample-layout-per-reset \
  --anticipatory-followups 8 \
  --output-path "${OUT_PATH}" \
  --seed 0
