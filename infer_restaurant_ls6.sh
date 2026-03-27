#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J rest_infer_ls6
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

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

CONFIG_PATH=anticipatory_rl/configs/restaurant_symbolic.yaml

srun python -m anticipatory_rl.agents.restaurant_dqn_infer \
  --anticipatory-weights ./runs/restaurant_anticipatory/restaurant_dqn.pt \
  --myopic-weights ./runs/restaurant_myopic/restaurant_dqn.pt \
  --output-dir ./runs/compare_restaurant_dqn \
  --config-path "${CONFIG_PATH}" \
  --num-tasks 5000 \
  --total-steps 250000 \
  --tasks-per-reset 200 \
  --max-task-steps 24 \
  --success-reward 15 \
  --invalid-action-penalty 6 \
  --travel-cost-scale 1.0 \
  --pick-cost 1.0 \
  --place-cost 1.0 \
  --wash-cost 2.0 \
  --fill-cost 1.0 \
  --brew-cost 2.0 \
  --fruit-cost 2.0 \
  --gamma 0.99 \
  --seed 0

date
echo "Job finished"
