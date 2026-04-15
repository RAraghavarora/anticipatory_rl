# Paper 1 Blockworld Reproduction

This directory contains a self-contained reproduction of the small 2D Blockworld setting from `paper1.pdf`:

- `7x7` 2D grid world with `7` colored goal regions and `3` white parking regions
- `8` named blocks (`a`-`h`) with per-environment color assignments: `5` colored and `3` white
- uniform task distribution over one-block and two-block placement tasks
- Fast Downward planning
- graph encoder and a PyTorch Geometric GraphSAGE anticipatory-cost regressor
- myopic, anticipatory, preparation + myopic, and preparation + anticipatory baselines

## What Is Reproduced

This code recreates the **environment structure** and the **evaluation protocol** from the paper:

- procedurally generated 2D blockworlds
- 20 uniformly sampled tasks per environment
- 10-task sequences
- 32 environments and 100 sequences per environment by default

The original paper uses a learned GNN estimator of anticipatory cost. This directory now includes that training path with:

- 3 GraphSAGE layers
- LeakyReLU activations
- mean graph pooling
- MAE loss
- Adam
- batch size 8
- learning rate `0.01`
- 10 epochs

For convenience, the evaluator still also supports an **exact / sampled one-step future-cost estimator** computed by additional planning calls:

`V_AP(s) = E_tau [ V_tau(s) ]`

## Main Files

- `world.py`: world generator, state representation, task library
- `planner.py`: PDDL problem generation and Fast Downward wrapper
- `gnn.py`: graph encoder and PyTorch Geometric GraphSAGE regressor
- `estimator.py`: oracle and learned future-cost estimators
- `train_gnn.py`: dataset generation and GNN training
- `reproduce_paper1.py`: baselines, preparation, evaluation loop
- `pddl/blockworld_domain.pddl`: planning domain

## Running

Use the `thesis` conda environment:

```bash
conda run -n thesis python -m paper1_blockworld.reproduce_paper1 --smoke-test
```

Train the anticipatory GNN:

```bash
conda run -n thesis python -m paper1_blockworld.train_gnn --smoke-test
```

Distributed training with `accelerate`:

```bash
conda run -n thesis accelerate launch --multi_gpu --num_processes 4 \
  -m paper1_blockworld.train_gnn \
  --batch-size 8
```

`--batch-size` is treated as the global batch size, so the paper setting of `8` stays unchanged when training is split across multiple GPUs.
The planner-labeled dataset is cached as a `.pt` file under the output directory by default; override it with `--dataset-cache /path/to/file.pt`.

A paper-aligned training run:

```bash
conda run -n thesis accelerate launch --multi_gpu --num_processes 4 \
  -m paper1_blockworld.train_gnn \
  --num-train-envs 250 \
  --num-val-envs 0 \
  --num-test-envs 150 \
  --states-per-env 200 \
  --tasks-per-environment 20 \
  --future-task-sample all \
  --epochs 10 \
  --batch-size 8 \
  --lr 0.01 \
  --dataset-workers 16
```

Evaluate with the trained checkpoint:

```bash
conda run -n thesis python -m paper1_blockworld.reproduce_paper1 \
  --paper-settings \
  --estimator learned \
  --gnn-checkpoint paper1_blockworld/checkpoints/paper1_anticipatory_gnn.pt
```

A larger run closer to the paper setup:

```bash
conda run -n thesis python -m paper1_blockworld.reproduce_paper1 \
  --num-envs 32 \
  --num-sequences 100 \
  --sequence-length 10 \
  --tasks-per-environment 20 \
  --preparation-iterations 200 \
  --future-task-sample 8 \
  --estimator oracle
```

Use `--future-task-sample all` for the exact one-step expectation, but it is much slower.

## Assumptions

The paper leaves some implementation details implicit. This reproduction makes the following concrete choices:

- regions are single-cell colored goal slots embedded in a 2D grid
- white regions act as legal parking locations
- only region cells can store blocks
- move cost is proportional to grid travel through unit-cost adjacency
- pick and place have fixed costs
- multiple blocks cannot occupy the same location
- preparation uses hill-climbing over sampled tasks, as described in the paper

These assumptions are documented so the code is inspectable and easy to adjust.

- regions are sampled as `10` single-capacity placement locations in a `7x7` grid
- white regions act as legal parking locations
- objects are sampled onto regions only, never onto arbitrary floor cells
- each environment assigns `5` colored object labels and `3` white object labels from `a`-`h`
- the task library is fixed at `20` unique tasks per environment by default

## Notes On The Learned Estimator

- The implementation now uses PyTorch Geometric `SAGEConv`, matching the paper’s model family more closely.
- Training labels are generated from planner-computed one-step anticipatory cost under the environment's uniform task library.
- Distributed training uses Hugging Face `accelerate`; planner-labeled datasets are cached once on the main process and reused by all workers.
- Planner label generation can be parallelized across CPU workers with `--dataset-workers`; this is the main lever for reducing end-to-end wall time.
- The learned estimator is intended to match the paper's role in the pipeline, while the oracle estimator remains useful for debugging and upper-bound comparisons.

## Remaining Ambiguity

The paper does not fully specify every implementation detail, so some assumptions remain necessary:

- the exact procedural world generator is not published
- the hidden dimension of the GNN is not stated
- the exact alternate-goal enumeration details are only described at a high level

This code now matches the explicit details in the paper as closely as possible from the published description, while keeping all remaining assumptions local and inspectable.
