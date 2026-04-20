# Paper2 Training and Inference Runbook

Run all commands from the repository root.

## Paper2 alignment summary

- Layout corpus scale target: `1000` restaurant layouts.
- Two-room structure per layout: kitchen + serving room.
- Category scale target: `25` category labels per layout.
- Task-library scale target: `50-100` feasible tasks per layout. (Randomly chosen from 50-100)
- Evaluation protocol target: `500` layout sequences x `40` tasks
  (`20,000` task executions per evaluation seed).
- Separate planner-style logging via `paper2_cost`:
  - non-move fixed costs: `pick/place/fill/wash/brew/fruit = 100`
  - move cost: Dijkstra shortest path on a `10x10` occupancy grid.
- Locations:10 
```
  kitchen_counter
  coffee_machine
  water_station
  fruit_station
  dish_rack
  sink
  pass_counter
  table_left
  bus_tub
  table_right
```
- Objects: 60
- When we sample an environment we assign each location to a (x,y) in a 10x10 grid. We use these locations to calculate move costs. Kitchen locations are on the left (x<=4), and serving locations are on the right.

### What is intentionally different

- RL reward is unchanged and remains the optimization objective.
- `paper2_cost` is an auxiliary metric and is not used for policy optimization.
- Symbolic macro-actions approximate planner actions and do not replicate full
  embodied planning internals.

## Restaurant environment specification

### State space (symbolic)

The observation encodes:

- agent location (one-hot over locations),
- held object (one-hot over objects plus empty),
- per-object features:
  - location (including held slot),
  - dirty flag,
  - contents (`empty`, `water`, `coffee`, `fruit`),
  - object kind (`mug`, `glass`, `bowl`),
- current task descriptor:
  - task type,
  - target location (or none),
  - target kind (or none).

### Action space (macro actions)

- `pick:<object>` for each object in the current layout.
- `place:<location>` for each location in the current layout.
- `wash_held`
- `fill_water_held`
- `make_coffee_held`
- `fill_fruit_held`

### Task categories and completion conditions

The core task families are:

- `serve_water(target_location)`
  - success if a `mug` or `glass` containing `water` is at `target_location`.
- `make_coffee(target_location)`
  - success if a `mug` containing `coffee` is at `target_location`.
- `serve_fruit_bowl(target_location)`
  - success if a `bowl` containing `fruit` is at `target_location`.
- `clear_containers(target_location)`
  - success if no container object remains at `target_location`.
- `wash_objects(target_kind)`
  - success if at least one object of `target_kind` is clean, empty, and at a
    wash-ready location.

Task libraries generated for Paper2-scale layouts are built from these families
with per-layout targets and feasibility filtering.

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
