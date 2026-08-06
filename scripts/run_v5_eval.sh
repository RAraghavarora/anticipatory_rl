#!/usr/bin/env bash
# v5 evaluation matrix on 5080. Builds a job list, runs it through a bounded-parallel
# queue, and writes one JSON per job.
#
#   sweep    : guided planner on seq-01 across cost_ratio  (DECISIVE -- queued first)
#   baselines: K=1/K=2 optimal + K=3 satisficing on all 10 sequences
#
# Concurrency is capped below core count on purpose: planner timeouts are wall-clock,
# so oversubscription can manufacture fake timeouts in the exact measurement we care about.
set -eu

# Portable: override REPO/CONDA/ENVNAME per machine. These must be EXPORTED or set
# inline on the SAME line as the command -- `VAR=x` on its own line is a shell
# variable and is not inherited by this script.
#   REPO=$PWD CONDA=/path/conda.sh ./scripts/run_v5_eval.sh
REPO=${REPO:-$PWD}
CONDA=${CONDA:-$HOME/miniconda3/etc/profile.d/conda.sh}
ENVNAME=${ENVNAME:-ant_rl}
[ -d "$REPO" ]  || { echo "ERROR: REPO not a directory: $REPO"; exit 1; }
[ -f "$CONDA" ] || { echo "ERROR: conda profile not found: $CONDA"; exit 1; }
cd "$REPO"
source "$CONDA"
conda activate "$ENVNAME" || { echo "ERROR: cannot activate env: $ENVNAME"; exit 1; }
export PYTHONPATH=.
echo "repo=$REPO  env=$ENVNAME  python=$(which python)"

CFG=configs/restaurant/toy_level_5.yaml
Q=runs/v5_ant/restaurant_dqn_best.pt
GAMMA=0.99
RSTAR=74.01255445461501
OUT=runs/v5eval
LOG=logs/v5eval
JOBS=$OUT/jobs.txt
PAR=${PAR:-20}          # < 24 cores, leaves headroom
mkdir -p "$OUT" "$LOG"
: > "$JOBS"

SEQS="00 01 02 03 04 05 06 07 08 09"

# --- sweep first so it is never starved behind the 4h K=3 jobs -------------------
# cost_ratio is a budget RELATIVE to each task's myopic cost, in RL units. The jar
# investment is ~30 RL units; an average task is ~13.4 (ratio ~3.2) but a cheap task
# is ~4.5 (ratio ~7.7). So the sweep has to reach 8, not stop at 4.
# The smoke test also hit expansions=5000, so test whether the search is
# expansion-limited rather than budget-limited.
for CR in 1.25 2.0 3.0 4.0 6.0 8.0; do
  echo "python scripts/restaurant/evaluate_bellman_novelty_sequence.py \
--policy both --q-weights $Q --config-path $CFG \
--sequence-path experiments/sequences/iid-eval-seq-01.json \
--gamma $GAMMA --success-reward $RSTAR \
--cost-ratio $CR --max-depth 20 --max-expansions 5000 \
--fd-timeout-s 60 --seed 0 \
> $OUT/guided_seq01_cr${CR}.json 2> $LOG/guided_seq01_cr${CR}.log" >> "$JOBS"
done
for CR in 4.0 8.0; do
  echo "python scripts/restaurant/evaluate_bellman_novelty_sequence.py \
--policy both --q-weights $Q --config-path $CFG \
--sequence-path experiments/sequences/iid-eval-seq-01.json \
--gamma $GAMMA --success-reward $RSTAR \
--cost-ratio $CR --max-depth 20 --max-expansions 20000 \
--fd-timeout-s 60 --seed 0 \
> $OUT/guided_seq01_cr${CR}_exp20k.json 2> $LOG/guided_seq01_cr${CR}_exp20k.log" >> "$JOBS"
done

# --- baselines -------------------------------------------------------------------
# K=1/K=2 use astar(hmax()): optimal, so a refusal to fetch the jar is a proof.
# A failed optimal solve ABORTS the sequence, hence the generous timeouts.
# K=3 uses satisficing lama: exact K=3 is intractable here (dies mid-sequence).
for S in $SEQS; do
  for K in 1 2; do
    TO=900; [ "$K" = "2" ] && TO=1800     # K=2's worst window took 851s locally
    echo "python scripts/restaurant/toy_clairvoyant_oracle.py \
--sequence-path experiments/sequences/iid-eval-seq-${S}.json --config-path $CFG \
--K $K --seed 0 --timeout-s $TO --search 'astar(hmax())' \
--output-path $OUT/base_seq${S}_k${K}_opt.json > $LOG/base_seq${S}_k${K}_opt.log 2>&1" >> "$JOBS"
  done
  echo "python scripts/restaurant/toy_clairvoyant_oracle.py \
--sequence-path experiments/sequences/iid-eval-seq-${S}.json --config-path $CFG \
--K 3 --seed 0 --timeout-s 600 \
--output-path $OUT/base_seq${S}_k3_sat.json > $LOG/base_seq${S}_k3_sat.log 2>&1" >> "$JOBS"
done

echo "$(wc -l < "$JOBS") jobs, ${PAR} concurrent, $(nproc) cores"
date
xargs -a "$JOBS" -d '\n' -P "$PAR" -I{} bash -c '{}'
echo "ALL DONE"
date
