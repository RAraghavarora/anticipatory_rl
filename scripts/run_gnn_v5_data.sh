#!/usr/bin/env bash
# Generate faithful or augmented GNN training data for v5, sharded by seed.
#
# An augmented chain state costs ~250 FD calls (candidate solves plus 48-task V_AP labels);
# a faithful state costs 49. Matching level_3's 2000 chain states is therefore ~500k or
# ~98k FD calls respectively. Each shard is serial, so run shards concurrently and merge.
#
#   MODE=aug REPO=/path/to/repo SHARDS=40 PER=50 ./scripts/run_gnn_v5_data.sh
#   MODE=faithful REPO=/path/to/repo SHARDS=40 PER=50 ./scripts/run_gnn_v5_data.sh
set -eu

# Activate the uv environment before running this script.
REPO=${REPO:-$PWD}
SHARDS=${SHARDS:-20}
PER=${PER:-100}
TIMEOUT=${TIMEOUT:-30}
MAXAUGS=${MAXAUGS:-10}
MODE=${MODE:-aug}

case "$MODE" in
  aug)
    GENERATOR=scripts/gnn/generate_data_aug.py
    GENERATOR_ARGS="--max-augs $MAXAUGS"
    ;;
  faithful)
    GENERATOR=scripts/gnn/generate_data.py
    GENERATOR_ARGS=""
    ;;
  *)
    echo "ERROR: MODE must be 'aug' or 'faithful', got: $MODE"
    exit 1
    ;;
esac

# Fail loudly rather than generating hours of garbage in the wrong environment.
[ -d "$REPO" ]  || { echo "ERROR: REPO not a directory: $REPO"; exit 1; }
cd "$REPO"
export PYTHONPATH=.
for f in configs/restaurant/toy_level_5.yaml pddl/toy_restaurant_domain.pddl \
         downward/builds/release/bin/downward "$GENERATOR"; do
  [ -e "$f" ] || { echo "ERROR: missing $f (is Fast Downward built?)"; exit 1; }
done
python "$GENERATOR" --help >/dev/null \
  || { echo "ERROR: generator import preflight failed in the active environment"; exit 1; }
echo "repo=$REPO  python=$(command -v python)"

OUT=runs/v5_gnn_data/$MODE
LOG=logs/v5_gnn_data/$MODE
MERGED=runs/v5_gnn_data/train_data_v5_$MODE
export OUT MERGED SHARDS
mkdir -p "$OUT" "$LOG"

CFG=configs/restaurant/toy_level_5.yaml
DOM=pddl/toy_restaurant_domain.pddl
FD=downward/fast-downward.py

echo "sharding ${SHARDS} x ${PER} states = $((SHARDS*PER)) total, $(nproc) cores"
date

JOBS=$OUT/jobs.txt; : > "$JOBS"
for i in $(seq 0 $((SHARDS-1))); do
  LOG_FILE=$LOG/shard_${i}.log
  echo "python $GENERATOR \
--config-path $CFG --domain-path $DOM --planner-path $FD \
--seed $((1000+i)) --num-states $PER $GENERATOR_ARGS --timeout-s $TIMEOUT \
--log-interval 10 --output-path $OUT/shard_${i}.pt > $LOG_FILE 2>&1; \
rc=\$?; if [ \$rc -ne 0 ]; then \
echo \"ERROR: $MODE shard $i failed (exit \$rc): $LOG_FILE\" >&2; \
tail -n 30 \"$LOG_FILE\" >&2; exit \$rc; fi; \
echo \"completed $MODE shard $((i+1))/$SHARDS\"" >> "$JOBS"
done

# Leave 2 cores free; FD is single-threaded per job.
PAR=$(( $(nproc) - 2 )); [ "$PAR" -lt 1 ] && PAR=1
[ "$PAR" -gt "$SHARDS" ] && PAR=$SHARDS
echo "running ${PAR} concurrent"
if ! xargs -a "$JOBS" -d '\n' -P "$PAR" -I{} bash -c '{}'; then
  echo "ERROR: one or more $MODE shards failed; inspect $LOG"
  exit 1
fi

echo "shards done, merging"
python - <<'PY'
import glob, torch, os
out = os.environ["OUT"]
merged = os.environ["MERGED"]
shards = sorted(glob.glob(os.path.join(out, "shard_*.pt")))
expected = int(os.environ["SHARDS"])
if len(shards) != expected:
    raise SystemExit(f"ERROR: expected {expected} shards, found {len(shards)} in {out}")
data, meta = [], None
for f in shards:
    d = torch.load(f, map_location="cpu", weights_only=False)
    items = d["dataset"] if isinstance(d, dict) and "dataset" in d else d
    data.extend(items)
    print(f"  {os.path.basename(f)}: {len(items)}")
print(f"merged {len(data)} samples from {len(shards)} shards")
torch.save(data, merged + ".pt")
# companion npz so the v_ap distribution can be eyeballed the same way as toy3_2k_aug.npz
import numpy as np
vs = [float(s["v_ap"]) for s in data if isinstance(s, dict) and "v_ap" in s]
if vs:
    np.savez(merged + ".npz", v_ap=np.array(vs))
    a = np.array(vs)
    print(f"v_ap min/mean/max = {a.min():.0f}/{a.mean():.0f}/{a.max():.0f}")
PY
echo "ALL DONE"
date
