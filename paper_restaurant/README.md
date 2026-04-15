# Paper Restaurant Supervised Reproduction

This package implements a paper-style supervised anticipatory planner for the
restaurant domain, parallel to the existing `paper1_blockworld/` reproduction.

## What It Covers

- planner-centric evaluation, not Gym RL rollouts
- two baselines only in v1:
  - `myopic`
  - `anticipatory`
- weighted one-step anticipatory planning cost regression
- graph estimator with:
  - frozen SBERT node embeddings by default
  - `TransformerConv`
  - MAE
  - Adagrad
  - StepLR
- matched-seed evaluation with average cost per task

## Main Files

- `world.py`: paper-style restaurant world, layout generator, task library sampler
- `planner.py`: exact symbolic macro-action planner with paper-style cost accounting
- `candidates.py`: bounded candidate goal expansion for anticipatory planning
- `gnn.py`: graph encoder, text embeddings, TransformerConv estimator
- `estimator.py`: oracle and learned future-cost estimators
- `train_gnn.py`: planner-labeled dataset generation and supervised training
- `reproduce_restaurant_supervised.py`: myopic vs anticipatory evaluation

## Reproduction Assumptions

The published restaurant benchmark does not release the exact planning domain,
ontology, or all feature details. This implementation makes the following
explicit assumptions and keeps them fixed across experiments:

- the restaurant is a two-room layout (`kitchen`, `serving_room`)
- the task families reuse our current setting:
  - `ServeWater`
  - `MakeCoffee`
  - `ServeFruitBowl`
  - `ClearContainers`
  - `WashObjects`
  - plus generic `pick_place` tasks
- containers have no capacity limits
- the planner is an exact symbolic macro-action solver with:
  - movement cost from shortest-path distance over a grid occupancy map
  - constant manipulation/fill/clear/wash costs
- node attributes use this fixed 9-bit schema:
  - `dirty`
  - `empty`
  - `contains_water`
  - `contains_coffee`
  - `contains_fruit`
  - `wash_source`
  - `water_source`
  - `coffee_source`
  - `fruit_source`

The planner is not Fast Downward over a published restaurant PDDL domain,
because that domain is not available. The solver here is exact over the repo’s
symbolic restaurant action model, which makes the reproduction runnable and
inspectable.

## Running

Smoke-test the oracle evaluator:

```bash
conda run -n thesis python -m paper_restaurant.reproduce_restaurant_supervised \
  --smoke-test \
  --estimator oracle
```

Smoke-test training without downloading SBERT:

```bash
conda run -n thesis python -m paper_restaurant.train_gnn \
  --smoke-test \
  --text-encoder hash
```

Paper-style training:

```bash
conda run -n thesis accelerate launch --multi_gpu --num_processes 4 \
  -m paper_restaurant.train_gnn \
  --num-train-envs 96 \
  --num-val-envs 16 \
  --states-per-env 64 \
  --tasks-per-environment 72 \
  --future-task-sample all \
  --epochs 10 \
  --batch-size 8 \
  --lr 0.01 \
  --hidden-dim 256 \
  --num-layers 4
```

Evaluate a learned checkpoint:

```bash
conda run -n thesis python -m paper_restaurant.reproduce_restaurant_supervised \
  --paper-settings \
  --estimator learned \
  --gnn-checkpoint paper_restaurant/checkpoints/paper_restaurant_anticipatory_gnn.pt
```
