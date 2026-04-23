# Restaurant Environment Assumptions vs Paper Clarifications

This document summarizes (1) assumptions made while recreating the restaurant environment, and (2) new information provided by the authors (PDDL + answers) that should drive updates.

## Assumptions we made

- ~~Layout generation: assumed a fixed set of 10 symbolic locations with manual coordinates, no explicit two-room sampling process.~~
- ~~Object inventory: assumed 3 object kinds (mug, glass, bowl) with 5 total objects.~~
- ~~Task types: assumed only 5 high-level tasks and no additional pick-and-place tasks.~~
- ~~Task distribution: assumed non-uniform per-task and per-location distributions.~~
- Task library: assumed a simple list or weighted sampling with feasibility not formally checked.
- ~~Action model: assumed a compact macro-action set (`pick`, `place`, `wash_held`, `fill_water_held`, `make_coffee_held`, `fill_fruit_held`) with simplified preconditions.~~
- ~~Action costs: assumed uniform per-action constants and Manhattan travel distance for RL reward.~~
- Initial state sampling: assumed simple distributions over locations and stochastic object contents/dirty status.
- Timeout/reset handling: assumed task timeout simply advances the task without resetting the physical world.
- Planner cost bridge: assumed a separate Dijkstra-based planner cost used only for logging.

## New information from the authors

From the shared PDDL domain in [restaurant/domain.py](restaurant/domain.py):

- Action set is richer than our macro-action set (e.g., `apply-spread`, `make-fruit-bowl`, `pour`, `refill_water`, `drain`).
- Types include a broader inventory of items and locations (e.g., `knife`, `plate`, `spread`, `apple`, `jar`, `fountain`, `dishwasher`, `countertop`).
- Predicates and preconditions are more detailed (e.g., `is-dirty`, `is-spreadable`, `is-slicable`, `filled-with`, `restrict-move-to`).
- Costs vary by action (e.g., `fill` is 1000, `wash` is 200, `make-coffee` is 50), rather than uniform constants.
- Motion cost uses a `known-cost` function for travel between locations, rather than Manhattan grid distance.

From the author answers:

- Dijkstra cost uses ProcTHOR layouts and occupancy grids, with resolution set to 0.1 units per cell.
- Task distribution $P(\tau)$ is uniform.

## Changes implemented

### ✅ Phase 1: Domain Alignment Layer
- Created `anticipatory_rl/envs/restaurant/pddl_domain.py` with centralized PDDL-aligned specification
- Defined PDDL action costs matching author's domain (fill: 1000, wash: 200, make-coffee: 50)
- Added mapping between PDDL action names and RL macro-action names

### ✅ Phase 2: Uniform Task Distribution
- Updated `RestaurantSymbolicEnv._resample_task()` to always use uniform sampling
- Updated `RestaurantWorldGenerator._sample_task_weights()` to always return uniform weights
- Task distribution P(τ) is uniform per author clarification

### ✅ Phase 3: Cost Semantics Alignment
- Separated three cost channels: RL reward (unchanged), PDDL action costs, paper2 logging costs
- Updated `RestaurantSymbolicEnv._configure_paper2_cost()` to always use PDDL costs
- Updated `RestaurantWorldConfig.sample()` to use PDDL costs
- Action costs match PDDL domain (fill: 1000, wash: 200, make-coffee: 50, etc.)

### ✅ Phase 4: Movement Cost Documentation
- Added documentation in `_dijkstra_distance()` explaining 0.1 resolution mapping assumption
- Documented that full ProcTHOR occupancy grid integration is deferred
- Noted that symbolic grid provides reasonable approximation for paper2_cost evaluation

### ✅ Phase 5: Expanded Object Inventory
- Updated object kinds to match PDDL domain exactly: cup, mug, jar, coffeegrinds, water, bread, knife, plate, bowl, spread, apple
- Total: 11 object kinds
- Added 20 object instances for a total of 20 objects
- Updated reset location distributions for all object kinds

### ✅ Phase 6: Expanded Location Set
- Added 5 new locations: prep_counter, pantry_shelf, service_shelf, host_stand, table_center
- Total: 15 locations
- Updated service locations to include table_center
- Updated wash-ready locations to include prep_counter
- Updated paper2_cost grid to 16x8 (was 10x10)

### ✅ Phase 7: Expanded Task Types
- Updated task types to match PDDL: serve_water, make_coffee, make_fruit_bowl, clear_containers, wash_objects, pick_place
- Total: 6 task types
- Updated task satisfaction checking for make_fruit_bowl (uses apples in bowls)
- Updated task sampling to include pick_place
- Added object_name field to RestaurantTask

### ✅ Phase 8: Expanded Action Set
- Added 3 new actions: pour_held, refill_water, drain
- Total: 42 actions (20 pick + 15 place + 7 macro)
- Updated action space, action meanings, and action execution
- Added action validity checking for new actions
- Action costs match PDDL domain (pour: 200, refill_water: 50, drain: 50)

## Remaining gaps

- **Action set**: Still missing `apply-spread` action (requires spreadable objects and spread contents)
- **Task library**: Feasibility checking not formally implemented
- **ProcTHOR integration**: Full occupancy grid integration with 0.1 resolution is deferred

## References

- PDDL domain: [restaurant/domain.py](restaurant/domain.py)
- PDDL specification: [anticipatory_rl/envs/restaurant/pddl_domain.py](anticipatory_rl/envs/restaurant/pddl_domain.py)
- Runbook: [docs/paper2_runbook.md](docs/paper2_runbook.md)
- Symbolic environment: [anticipatory_rl/envs/restaurant/restaurant_symbolic_env.py](anticipatory_rl/envs/restaurant/restaurant_symbolic_env.py)
- DQN trainer: [anticipatory_rl/agents/restaurant/restaurant_dqn.py](anticipatory_rl/agents/restaurant/restaurant_dqn.py)
- Planner world: [restaurant/paper_restaurant/world.py](restaurant/paper_restaurant/world.py)
- Planner: [restaurant/paper_restaurant/planner.py](restaurant/paper_restaurant/planner.py)
