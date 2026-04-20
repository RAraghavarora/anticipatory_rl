#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J p1bw_infer_ls6
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
conda activate llm

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

srun python -m anticipatory_rl.agents.blockworld.blockworld_dqn_infer \
  --anticipatory-weights ./runs/anticipatory_blockworld_paper1_blockworld_image_dqn_tpr10/paper1_blockworld_image_dqn.pt \
  --myopic-weights ./runs/myopic_blockworld_paper1_blockworld_image_dqn_tpr1/paper1_blockworld_image_dqn.pt \
  --output-dir ./runs/compare_blockworld_image_dqn_infer \
  --num-sequences 100 \
  --tasks-per-reset 10 \
  --total-steps 50000 \
  --render-tile-px 12 \
  --max-task-steps 64 \
  --seed 0

date
echo "Job finished"
