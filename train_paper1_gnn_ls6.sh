#!/bin/bash
# Submit from repo root so relative paths resolve.

#================================================
# SBATCH (TACC Lonestar6)
#================================================
#SBATCH -J paper1_gnn_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 48:00:00
#SBATCH -A IRI23005
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs
mkdir -p paper1_blockworld/checkpoints

module load cuda/12.2
source /work/10110/raghavaurora/ls6/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Job: ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-local}  node=$(hostname)  started=$(date -Is)"
echo "Stdout: slurm_logs/${SLURM_JOB_NAME}.o${SLURM_JOB_ID}"
echo "Stderr: slurm_logs/${SLURM_JOB_NAME}.e${SLURM_JOB_ID}"

srun python -m paper1_blockworld.train_gnn \
  --num-train-envs 250 \
  --num-val-envs 0 \
  --num-test-envs 150 \
  --states-per-env 200 \
  --tasks-per-environment 24 \
  --future-task-sample all \
  --epochs 10 \
  --batch-size 8 \
  --lr 0.01 \
  --hidden-dim 128 \
  --num-layers 3 \
  --device cuda \
  --output-dir paper1_blockworld/checkpoints \
  --seed 0

date
echo "Job finished"
