#!/usr/bin/env bash
# Generate GNN training data for v5, sharded by seed.
#
# One state costs ~250 FD calls on v5 (1 chain solve + ~10 augmented solves, then a V_AP
# label per surviving candidate, each needing all 48 enumerated tasks solved). Matching
# level_3's 2000-state dataset is ~500k FD calls -- ~28h serial. The script is a sequential
# loop, so shard by seed and merge.
#
#   REPO=/path/to/wt-gnn CONDA=/path/to/conda.sh SHARDS=20 PER=100 ./run_gnn_v5_data.sh
set -eu

# NOTE: these must be EXPORTED or set inline on the same line as the command.
#   VAR=x on its own line is a shell variable and is NOT inherited by this script.
#   good:  REPO=$PWD CONDA=/path/conda.sh ./scripts/run_gnn_v5_data.sh
#   good:  export REPO=$PWD; ./scripts/run_gnn_v5_data.sh
REPO=${REPO:-$PWD}
CONDA=${CONDA:-$HOME/miniconda3/etc/profile.d/conda.sh}
ENVNAME=${ENVNAME:-ant_rl}
SHARDS=${SHARDS:-20}
PER=${PER:-100}
TIMEOUT=${TIMEOUT:-30}
MAXAUGS=${MAXAUGS:-10}

# Fail loudly rather than generating hours of garbage in the wrong environment.
[ -d "$REPO" ]  || { echo "ERROR: REPO not a directory: $REPO"; exit 1; }
[ -f "$CONDA" ] || { echo "ERROR: conda profile not found: $CONDA"; exit 1; }
cd "$REPO"
source "$CONDA"
conda activate "$ENVNAME" || { echo "ERROR: cannot activate env: $ENVNAME"; exit 1; }
export PYTHONPATH=.
for f in configs/restaurant/toy_level_5.yaml pddl/toy_restaurant_domain.pddl \
         downward/builds/release/bin/downward scripts/gnn/generate_data_aug.py; do
  [ -e "$f" ] || { echo "ERROR: missing $f (is Fast Downward built?)"; exit 1; }
done
python -c 'import torch_geometric, sentence_transformers' \
  || { echo "ERROR: torch_geometric / sentence_transformers missing in $ENVNAME"; exit 1; }
echo "repo=$REPO  env=$ENVNAME  python=$(which python)"

OUT=runs/v5_gnn_data
LOG=logs/v5_gnn_data
mkdir -p "$OUT" "$LOG"

CFG=configs/restaurant/toy_level_5.yaml
DOM=pddl/toy_restaurant_domain.pddl
FD=downward/fast-downward.py

echo "sharding ${SHARDS} x ${PER} states = $((SHARDS*PER)) total, $(nproc) cores"
date

JOBS=$OUT/jobs.txt; : > "$JOBS"
for i in $(seq 0 $((SHARDS-1))); do
  echo "python scripts/gnn/generate_data_aug.py \
--config-path $CFG --domain-path $DOM --planner-path $FD \
--seed $((1000+i)) --num-states $PER --max-augs $MAXAUGS --timeout-s $TIMEOUT \
--log-interval 10 --output-path $OUT/shard_${i}.pt > $LOG/shard_${i}.log 2>&1" >> "$JOBS"
done

# Leave 2 cores free; FD is single-threaded per job.
PAR=$(( $(nproc) - 2 )); [ "$PAR" -lt 1 ] && PAR=1
[ "$PAR" -gt "$SHARDS" ] && PAR=$SHARDS
echo "running ${PAR} concurrent"
xargs -a "$JOBS" -d '\n' -P "$PAR" -I{} bash -c '{}'

echo "shards done, merging"
python - <<'PY'
import glob, torch, os
shards = sorted(glob.glob("runs/v5_gnn_data/shard_*.pt"))
data, meta = [], None
for f in shards:
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  skip {f}: {e}"); continue
    items = d["dataset"] if isinstance(d, dict) and "dataset" in d else d
    data.extend(items)
    print(f"  {os.path.basename(f)}: {len(items)}")
print(f"merged {len(data)} samples from {len(shards)} shards")
torch.save(data, "runs/v5_gnn_data/train_data_v5_aug.pt")
# companion npz so the v_ap distribution can be eyeballed the same way as toy3_2k_aug.npz
import numpy as np
vs = [float(s["v_ap"]) for s in data if isinstance(s, dict) and "v_ap" in s]
if vs:
    np.savez("runs/v5_gnn_data/train_data_v5_aug.npz", v_ap=np.array(vs))
    a = np.array(vs)
    print(f"v_ap min/mean/max = {a.min():.0f}/{a.mean():.0f}/{a.max():.0f}  (level_3 aug was 384/1148/1967)")
PY
echo "ALL DONE"
date
