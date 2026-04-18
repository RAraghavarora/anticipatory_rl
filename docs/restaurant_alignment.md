# Restaurant Alignment Notes

- Layout corpus scale: target `1000` restaurant layouts.
- Two-room structure per layout: kitchen + serving room.
- Category scale: `25` category labels per layout.
- Task-library scale: `50-100` feasible tasks per layout.
- Evaluation protocol target: `500` layout sequences of `40` tasks
  (`20,000` task executions total).
- Separate planner-style cost logging via `paper2_cost`:
  - non-move fixed costs: `pick/place/fill/wash/brew/fruit = 100`
  - move cost: Dijkstra shortest path on a `10x10` occupancy grid.

## What is intentionally different

- RL reward remains unchanged and is still used for optimization.
- `paper2_cost` is logged as an auxiliary metric, not used as training reward.
- Symbolic macro-actions approximate planner actions and do not replicate
  full embodied motion/planning internals.