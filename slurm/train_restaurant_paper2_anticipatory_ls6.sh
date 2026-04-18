#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J rest_paper2_ant_ls6
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
LAYOUT_CORPUS=data/restaurant_layouts/paper2_scale_layouts.json

if [[ ! -f "${LAYOUT_CORPUS}" ]]; then
  echo "Missing layout corpus: ${LAYOUT_CORPUS}"
  echo "Generate it before submitting this job."
  exit 2
fi

srun python -m anticipatory_rl.agents.restaurant_dqn \
  --config-path "${CONFIG_PATH}" \
  --layout-corpus "${LAYOUT_CORPUS}" \
  --sample-layout-per-reset \
  --task-sequence-length 40 \
  --run-label restaurant_paper2_anticipatory_seed0 \
  --total-steps 700000 \
  --tasks-per-reset 40 \
  --env-reset-tasks 40

date
echo "Job finished"
