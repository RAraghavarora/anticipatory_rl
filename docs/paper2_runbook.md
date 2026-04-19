# Paper2 Training and Inference Runbook

Run all commands from the repository root.

## 1) Generate and validate the Paper2 layout dataset

Generate the layout corpus:

```bash
python scripts/generate_restaurant_layout_corpus.py \
  --output-path data/restaurant_layouts/paper2_scale_layouts.json \
  --num-layouts 1000 \
  --seed 0
```

Validate the generated corpus:

```bash
python scripts/validate_restaurant_layout_corpus.py \
  --corpus-path data/restaurant_layouts/paper2_scale_layouts.json
```

## 2) Train RL agents on Lonestar6 (SLURM)

Submit myopic training:

```bash
sbatch slurm/train_restaurant_paper2_myopic_ls6.sh
```

Submit anticipatory training:

```bash
sbatch slurm/train_restaurant_paper2_anticipatory_ls6.sh
```

Expected checkpoints:

- `runs/restaurant_paper2_myopic_seed0/restaurant_dqn.pt`
- `runs/restaurant_paper2_anticipatory_seed0/restaurant_dqn.pt`

## 3) Run multi-seed inference (post-training)

Example full Paper2 protocol (`500 x 40 = 20,000` tasks per eval seed):

```bash
python paper_restaurant/scripts/restaurant_multi_seed_infer.py \
  --anticipatory-weights runs/restaurant_paper2_anticipatory_seed0/restaurant_dqn.pt \
  --myopic-weights runs/restaurant_paper2_myopic_seed0/restaurant_dqn.pt \
  --config-path anticipatory_rl/configs/restaurant_symbolic.yaml \
  --layout-corpus data/restaurant_layouts/paper2_scale_layouts.json \
  --sample-layout-per-reset \
  --eval-layout-count 500 \
  --task-sequence-length 40 \
  --num-tasks 20000 \
  --seed-from 0 \
  --seed-to 4 \
  --output-dir runs/paper2_scale_full/infer
```

## RL semantics (current)

- `--tasks-per-episode` (training only): controls replay bootstrapping boundary.
- `--task-sequence-length`: controls physical world reset cadence.
- `--max-steps-per-task`: timeout threshold per task.
- On timeout, the system samples a new task **without** resetting the world.

## 4) Build planner-labeled dataset (FD labels)

Local:

```bash
bash paper_restaurant/scripts/run_planner_dataset.sh
```

To increase local CPU parallelism:

```bash
JOBS=16 bash paper_restaurant/scripts/run_planner_dataset.sh
```

LS6:

```bash
sbatch slurm/build_restaurant_paper2_planner_dataset_ls6.sh
```

This LS6 job is CPU-oriented and defaults to multi-node sharded generation:
- partition: `normal`
- nodes/tasks: `4` nodes, `4` shards (one per task/rank)
- CPUs per shard: `64` (`JOBS` defaults to `SLURM_CPUS_PER_TASK`)

Override resources/parallelism if needed:

```bash
sbatch -N 4 -n 4 --cpus-per-task=64 --export=ALL,JOBS=64,NUM_STATES=4000 slurm/build_restaurant_paper2_planner_dataset_ls6.sh
```

## 5) Train planner-side APCostEstimator (GNN)

Local:

```bash
bash paper_restaurant/scripts/run_apcost_gnn_train.sh
```

LS6:

```bash
sbatch slurm/train_restaurant_paper2_apcost_gnn_ls6.sh
```

## 6) Evaluate planner myopic vs anticipatory

Local:

```bash
bash paper_restaurant/scripts/run_planner_compare.sh
```

LS6:

```bash
sbatch slurm/eval_restaurant_paper2_planner_ls6.sh
```

## 7) Create unified RL-vs-planner comparison table

```bash
bash paper_restaurant/scripts/run_unified_rl_vs_planner.sh
```

## Planner-only LS6 order (quick reference)

If you are running only the planner-side pipeline on LS6, submit in this order:

```bash
sbatch slurm/build_restaurant_paper2_planner_dataset_ls6.sh
sbatch slurm/train_restaurant_paper2_apcost_gnn_ls6.sh
sbatch slurm/eval_restaurant_paper2_planner_ls6.sh
```

Do not launch the next stage until the previous job finishes successfully.

## Optional unified RL wrapper

Run dataset generation + validation + RL training + multi-seed inference:

```bash
bash paper_restaurant/scripts/run_restaurant_rl.sh
```

Override defaults if needed:

```bash
SEED_FROM=0 SEED_TO=4 OUT_ROOT=runs/paper2_scale_full \
bash paper_restaurant/scripts/run_restaurant_rl.sh
```

## Notes

- RL defaults now use action costs and travel scale of `25`; rewards remain
  unchanged unless overridden.
- `paper2_cost` is logged as an auxiliary planner-style metric and does not
  replace RL reward for training.
