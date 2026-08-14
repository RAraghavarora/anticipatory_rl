#!/usr/bin/env bash
# Train + evaluate 3 additional seeds for a v5 GNN baseline, so the baseline can be
# reported as mean +/- std over 4 seeds like the RL arms are.
#
# Data generation is NOT repeated. That is the expensive part (~98k FD calls faithful,
# ~500k augmented) and the dataset is fixed across seeds -- only the model seed changes.
# train_gnn.py --seed drives init, batch shuffling AND the train/val split (train_gnn.py:71),
# so this is a real seed sweep and not just a different initialisation.
#
# Hyperparameters are READ from the original run's metadata.json rather than retyped, so a
# new seed differs from the archived checkpoint in exactly one argument. Retyping them is
# how you silently get a 4-seed spread that is really a 2-configuration spread.
#
#   MODE=faithful RUN_DIR=runs/gnn_v5_faithful_e40 DATA=runs/v5_gnn_data/train_data_v5_faithful.pt \
#     REPO=/var/local/aurora/anticipatory_rl ./scripts/run_gnn_v5_seeds.sh
#   MODE=aug      RUN_DIR=runs/gnn_v5_aug          DATA=runs/v5_gnn_data/train_data_v5_aug.pt      \
#     REPO=/var/local/aurora/anticipatory_rl ./scripts/run_gnn_v5_seeds.sh
#
# Activate the uv environment before running.
set -eu

REPO=${REPO:-$PWD}
MODE=${MODE:?set MODE=faithful or MODE=aug}
RUN_DIR=${RUN_DIR:-runs/__no_such_run__}   # original run dir, if its metadata.json is here
DATA=${DATA:?set DATA to the merged training .pt used by that run}
# Target set, matching the RL arms so the main table is one seed protocol throughout.
# Whichever of these the archived checkpoint already used is reused, not retrained.
SEEDS=${SEEDS:-"0 4 8 16"}
CFG=${CFG:-configs/restaurant/toy_level_5.yaml}
OUTROOT=${OUTROOT:-runs/v5_gnn_seeds}

# Eval knobs. These are NOT recoverable from the archived eval JSONs (eval_sequence.py
# writes no provenance), so they are set explicitly here. They must match whatever the
# archived gnn_faithful/gnn_counterfactual evaluations used or seed 1 is not comparable
# to seeds 2-4. Verify against your eval command before trusting the spread.
MAXAUGS=${MAXAUGS:-10}
GAMMA=${GAMMA:-1.0}
FD_TIMEOUT=${FD_TIMEOUT:-20}
SEARCH=${SEARCH:-"astar(ff())"}

# Filename stem of the archived evaluations for this mode, under runs/v5_gnn_eval/.
case "$MODE" in
  aug)      ARCHIVE_PREFIX=${ARCHIVE_PREFIX:-gnn_aug} ;;
  faithful) ARCHIVE_PREFIX=${ARCHIVE_PREFIX:-gnn_faithful_e40} ;;
esac

run_batch() {   # $1 = job file, $2 = concurrency
  local i=0
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    bash -c "$c" &
    i=$((i+1)); [ $((i % $2)) -eq 0 ] && wait
  done < "$1"
  wait
}

cd "$REPO"
export PYTHONPATH=.
# Unpinned torch spawned 3,401 threads on this box and produced load 1003 on 96 cores.
# Single-threaded was also 1.7x faster per task. Do not remove.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=

case "$MODE" in faithful|aug) ;; *) echo "ERROR: MODE must be faithful or aug"; exit 1 ;; esac
for f in "$DATA" "$RUN_DIR/metadata.json" "$CFG" \
         pddl/toy_restaurant_domain.pddl downward/fast-downward.py \
         scripts/gnn/train_gnn.py scripts/gnn/eval_sequence.py; do
  [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

OUT=$OUTROOT/$MODE
LOG=logs/$(basename $OUTROOT)/$MODE
mkdir -p "$OUT" "$LOG"

# ---- recover the original hyperparameters --------------------------------------------
# Preferred: read them from the original run's metadata.json, so a new seed differs in
# exactly one argument. The v5 faithful/aug runs were trained on another box and their
# metadata.json is not here, so each mode also carries a recipe fallback:
#
#   aug      epochs 12, step 1000. 8054 samples -> 805 batches/epoch. Confirmed by the
#            steelman's --scheduler-step 1439, which was chosen to give the steelman's
#            1151 batches/epoch the same number of lr halvings: 1151/805 * 1000 ~= 1430.
#   faithful epochs 40, step 1000. 2000 samples -> 200 batches/epoch. The archived
#            checkpoint is labelled gnn_v5_faithful_e40.
#
# Both original runs used seed 42 (train_gnn.py's default, as every other run here did).
if [ -f "$RUN_DIR/metadata.json" ]; then
  read -r ORIG_SEED HP <<EOF
$(python - "$RUN_DIR/metadata.json" <<'PY'
import json, sys
hp = json.load(open(sys.argv[1]))["hparams"]
skip = {"seed", "data_path", "output_dir", "run_label"}
flags = " ".join(f"--{k.replace('_','-')} {v}" for k, v in hp.items()
                 if k not in skip and v is not None)
print(hp.get("seed", "?"), flags)
PY
)
EOF
  echo "hparams source: $RUN_DIR/metadata.json"
else
  ORIG_SEED=42
  case "$MODE" in
    aug)      EPOCHS=12 ;;
    faithful) EPOCHS=40 ;;
  esac
  HP="--hidden-dim 64 --lr 0.01 --batch-size 8 --epochs $EPOCHS \
--scheduler-step 1000 --scheduler-gamma 0.5 --train-frac 0.8"
  echo "hparams source: built-in $MODE recipe (no metadata.json at $RUN_DIR)"
fi
echo "original seed: $ORIG_SEED"
echo "reused hparams: $HP"
echo "target seeds:  $SEEDS"

# If the archived checkpoint's seed is in the target set, reuse it as that seed and train
# only the rest. If it is NOT, all four are trained fresh and the archived gnn_faithful /
# gnn_counterfactual numbers in the archive for THIS domain are SUPERSEDED -- they came
# from a seed no longer in the reported set, so the generated tables must be regenerated.
TRAIN_SEEDS=""; REUSED=""
for s in $SEEDS; do
  if [ "$s" = "$ORIG_SEED" ]; then REUSED=$s; else TRAIN_SEEDS="$TRAIN_SEEDS $s"; fi
done
if [ -n "$REUSED" ]; then
  echo "reusing archived checkpoint as seed $REUSED; training:$TRAIN_SEEDS"
else
  echo "WARNING: archived seed $ORIG_SEED is not in the target set."
  echo "WARNING: training all of$TRAIN_SEEDS -- the archived $MODE totals for this"
  echo "WARNING: domain are superseded; regenerate the tables that cite them."
fi

# ---- phase 1: train (cheap; dataset already exists) -----------------------------------
echo "=== phase 1: training ==="; date
T=$OUT/train_jobs.txt; : > "$T"
for s in $TRAIN_SEEDS; do
  [ -s "$OUT/s${s}/best_model.pt" ] && { echo "skip train seed $s (exists)"; continue; }
  echo "python scripts/gnn/train_gnn.py --data-path $DATA --output-dir $OUT/s${s} \
--run-label gnn_v5_${MODE}_s${s} --seed $s $HP > $LOG/train_s${s}.log 2>&1 \
|| { echo 'ERROR: train seed $s failed' >&2; tail -30 $LOG/train_s${s}.log >&2; exit 1; }" >> "$T"
done
[ -s "$T" ] && run_batch "$T" 3
for s in $TRAIN_SEEDS; do
  [ -s "$OUT/s${s}/best_model.pt" ] || { echo "ERROR: no checkpoint for seed $s"; exit 1; }
  grep -h "Best val loss" "$LOG/train_s${s}.log" | sed "s/^/  s${s}: /"
done

# ---- phase 2: evaluate ----------------------------------------------------------------
# --policy gnn_anticipatory, not `both`: the myopic arm is deterministic and identical
# across seeds, so recomputing it 30 times doubles the bill for nothing. The flat output
# is then wrapped under "gnn_anticipatory" so build_v5_results.py (which does
# d.get("gnn_anticipatory")) reads these files unchanged.
echo "=== phase 2: evaluation ==="; date

# The reused seed already has ten archived evaluations; copy them into this run's naming
# so the spread below is computed over all four seeds uniformly. The archived files were
# produced with --policy both, whose output already carries a "gnn_anticipatory" key.
if [ -n "$REUSED" ]; then
  for q in 00 01 02 03 04 05 06 07 08 09; do
    A=runs/v5_gnn_eval/${ARCHIVE_PREFIX}_seq${q}.json
    [ -e "$A" ] || { echo "ERROR: archived eval missing: $A (set ARCHIVE_PREFIX)"; exit 1; }
    cp -n "$A" "$OUT/gnn_${MODE}_s${REUSED}_seq${q}.json"
  done
  echo "copied 10 archived evaluations in as seed $REUSED"
fi

E=$OUT/eval_jobs.txt; : > "$E"
for s in $TRAIN_SEEDS; do
  for q in 00 01 02 03 04 05 06 07 08 09; do
    P=$OUT/gnn_${MODE}_s${s}_seq${q}.json
    [ -s "$P" ] && continue
    echo "python scripts/gnn/eval_sequence.py \
--sequence-path experiments/sequences/iid-eval-seq-${q}.json \
--gnn-model $OUT/s${s}/best_model.pt --config-path $CFG \
--policy gnn_anticipatory --max-augs $MAXAUGS --gamma $GAMMA \
--fd-timeout-s $FD_TIMEOUT --search '$SEARCH' --seed 0 \
--output-path $P > $LOG/eval_s${s}_seq${q}.log 2>&1 \
&& python -c \"import json,sys; p=sys.argv[1]; d=json.load(open(p)); \
d='gnn_anticipatory' in d and d or {'gnn_anticipatory': d}; \
json.dump(d, open(p,'w'), indent=2, default=str)\" $P \
|| echo 'FAILED: $MODE s${s} seq${q}' >&2" >> "$E"
  done
done
PAR=${PAR:-24}
echo "$(wc -l < "$E") eval jobs, $PAR concurrent"
run_batch "$E" "$PAR"

echo "=== totals ==="
python - "$OUT" "$MODE" "$SEEDS" <<'PY'
import json, sys, glob, os, statistics
out, mode, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
tot = {}
for s in seeds:
    fs = sorted(glob.glob(os.path.join(out, f"gnn_{mode}_s{s}_seq*.json")))
    # eval_sequence.py writes "total_cost"; "total_cost_pddl" is build_v5_results.py's
    # column name for the same quantity. Using the latter here aborted the script under
    # `set -e` AFTER all evaluations had already been written.
    c = [json.load(open(f))["gnn_anticipatory"]["summary"]["total_cost"] for f in fs]
    tot[s] = (sum(c), len(fs))
    print(f"  s{s}: {sum(c):,.0f} over {len(fs)}/10 sequences")
full = [v for v, n in tot.values() if n == 10]
if len(full) > 1:
    print(f"mean {statistics.mean(full):,.0f} +/- {statistics.stdev(full):,.0f} "
          f"over {len(full)} complete seeds")
PY
echo "ALL DONE"; date
