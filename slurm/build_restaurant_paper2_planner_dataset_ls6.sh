#!/bin/bash
# Build planner-labeled restaurant dataset on LS6.

#SBATCH -J rest_p2_dataset_ls6
#SBATCH -o slurm_logs/%x.o%j
#SBATCH -e slurm_logs/%x.e%j
#SBATCH -p normal
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
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
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-64}}"
NUM_STATES="${NUM_STATES:-2000}"
FOLLOWUP_SAMPLES="${FOLLOWUP_SAMPLES:-8}"
SEED="${SEED:-0}"
SHARD_DIR="${SHARD_DIR:-data/restaurant_planner_dataset/shards_${SLURM_JOB_ID}}"
mkdir -p "${SHARD_DIR}"
export OUT_PATH SHARD_DIR

echo "Launching ${SLURM_NTASKS} shards across ${SLURM_JOB_NUM_NODES} nodes"
srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  python -m anticipatory_rl.tasks.build_restaurant_planner_dataset \
    --layout-corpus "'"${CORPUS_PATH}"'" \
    --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
    --planner-path downward/fast-downward.py \
    --domain-path pddl/restaurant_domain.pddl \
    --num-states "'"${NUM_STATES}"'" \
    --followup-samples "'"${FOLLOWUP_SAMPLES}"'" \
    --jobs "'"${JOBS}"'" \
    --num-shards "'"${SLURM_NTASKS}"'" \
    --shard-index "${SLURM_PROCID}" \
    --output-path "'"${SHARD_DIR}"'/shard_${SLURM_PROCID}.json" \
    --seed "'"${SEED}"'"
'

python - <<'PY'
import glob
import json
import os
from pathlib import Path

out_path = Path(os.environ["OUT_PATH"])
shard_dir = Path(os.environ["SHARD_DIR"])
shard_paths = sorted(glob.glob(str(shard_dir / "shard_*.json")))
if not shard_paths:
    raise RuntimeError(f"No shard outputs found in {shard_dir}")

rows = []
meta_rows = []
for p in shard_paths:
    payload = json.loads(Path(p).read_text(encoding="utf-8"))
    rows.extend(payload.get("rows", []))
    meta_rows.append(payload.get("meta", {}))

merged = {
    "meta": {
        "num_rows": len(rows),
        "num_shards": len(shard_paths),
        "source_shards": [str(Path(p).name) for p in shard_paths],
        "shard_meta": meta_rows,
    },
    "rows": rows,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"Wrote merged planner dataset -> {out_path} ({len(rows)} rows)")
PY
