# Blockworld Environment Documentation

## Overview

This document describes our implementation of the **Blockworld environment** for evaluating anticipatory reinforcement learning. The environment is based on the benchmark introduced by Dhakal et al. (2023) in their work on anticipatory planning for long-lived robots.

## What This Environment Represents

The Blockworld environment simulates a **long-lived service robot** operating in a persistent 2D world where:

- The robot must complete a **sequence of pick-and-place tasks** drawn from a task distribution
- Tasks involve moving colored blocks to specific colored regions
- The environment **persists between tasks** — objects remain where they were placed
- How one task is completed directly affects the cost of future tasks

This creates the core challenge: **myopic planning** (solving only the current task) can create **side effects** that make future tasks more expensive, while **anticipatory planning** can trade off immediate cost for better preparedness.

### Example Scenario

```
Task 1: Move block A to red region
Task 2: Move block B to blue region

Myopic behavior:
- Places block F on blue region (cheapest for Task 1)
- Must move F away before completing Task 2 (expensive!)

Anticipatory behavior:
- Places block F on white region (slightly more costly for Task 1)
- Blue region is free for Task 2 (cheaper overall!)
```

## Why We Use This Environment

### 1. **Reproducing Prior Work**
Dhakal et al. demonstrated that anticipatory planning reduces overall task costs in persistent environments. We recreate their Blockworld benchmark to:
- Validate their findings with a different approach (RL instead of symbolic planning)
- Enable direct comparison between symbolic planning and RL methods
- Provide a controlled testbed for anticipatory behavior

### 2. **Testing Anticipatory RL**
The environment is ideal for evaluating our reinforcement learning approach because:
- **Task interdependence**: White blocks/regions serve as "parking" resources, creating strategic trade-offs
- **Persistent state**: The world state carries over between tasks, making preparedness measurable
- **Measurable anticipation**: We can quantify benefits through reduced task costs and auto-satisfaction rates
- **Computational tractability**: Small enough for rapid experimentation, complex enough to show meaningful differences

### 3. **Controlled Complexity**
Unlike large-scale embodied simulators:
- Fast iteration cycles enable systematic ablation studies
- Deterministic dynamics allow reproducible experiments
- Interpretable state space makes debugging easier
- Planner-based cost evaluation provides ground truth metrics

## Environment Specification

### World Geometry

```
Grid:           10 × 10 cells
Region size:    2 × 2 cells each
Block size:     1 × 1 cell each
Total regions:  10 (7 colored + 3 white)
Total blocks:   8 (5 colored + 3 white)
```

**Colored Regions**: `red`, `blue`, `green`, `orange`, `teal`, `purple`, `yellow`  
**White Regions**: `white_1`, `white_2`, `white_3`  
**Blocks**: `a`, `b`, `c`, `d`, `e`, `f`, `g`, `h`  
**Colored Block Colors**: `red`, `blue`, `green`, `orange`, `teal` (assigned to 5 of the 8 blocks)

### Task Space

Tasks are pick-and-place directives sampled from a library of ~20 unique tasks per environment:

- **1-block tasks**: `a → red` (move block `a` to the red region)
- **2-block tasks**: `a → red, b → blue` (move two blocks to specific regions)

**Key constraints**:
- Only **non-white blocks** appear in task goals
- Only **non-white regions** appear in task goals
- At most **one block per region**
- Tasks are sampled uniformly from the library

This design makes white blocks and white regions valuable as "temporary storage" that doesn't interfere with likely future goals.

### State Representation

The complete state consists of:
- **Robot position**: `(x, y)` coordinates in the 10×10 grid
- **Block placements**: For each block, its `(x, y)` position
- **Held object**: `None` or the name of the block being held
- **Current task**: The task assignment(s) to be satisfied

For the **image-based RL agent**, the state is encoded as an **8-channel image** (see Figure in report):
1. **Channel 0**: Robot position (white dot)
2. **Channel 1**: Block positions (colored squares)
3. **Channel 2**: Region boundaries (colored 2×2 squares)
4. **Channels 3-4**: Target block masks (highlights which blocks need to be moved)
5. **Channels 5-7**: Target region masks (highlights destination regions)

### Action Space

The RL agent uses a **primitive 6-action space**:

```python
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3
PICK       = 4
PLACE      = 5
```

**Movement rules**:
- Robot can move to any **unoccupied cell**
- Robot **cannot** move onto cells occupied by blocks
- Robot **can** move onto region tiles (regions are traversable floor tiles)

**Pick/Place rules**:
- **Pick**: Must be adjacent to exactly one block (4-connected neighbors)
- **Place**: Must be adjacent to an empty region tile or empty cell
- Cannot pick if already holding a block
- Cannot place if not holding a block

## Implementation Assumptions

Since Dhakal et al.'s paper does not specify exact implementation details, we made the following **justified assumptions**:

### 1. Grid Geometry ✅
- **Assumption**: 10×10 grid with 2×2 regions
- **Justification**: Allows 10 non-overlapping regions with free space, consistent with visual figures in the paper

### 2. Movement Model ✅
- **Assumption**: Agent can traverse region tiles but not occupied block tiles
- **Justification**: Consistent with LazyPRM collision-aware motion planning mentioned in the paper
  - Regions are colored floor areas (traversable)
  - Blocks are physical obstacles (not traversable)

### 3. Manipulation Rules ✅
- **Assumption**: Pick/place requires adjacency (not same-cell operation)
- **Assumption**: Must be adjacent to exactly one block to pick
- **Justification**: Standard robotics manipulation model; avoids ambiguity when multiple blocks are adjacent

### 4. Cost Model ✅
- **Movement cost**: 25 × Euclidean path length (via Lazy PRM)
- **Pick cost**: 100 (fixed)
- **Place cost**: 100 (fixed)
- **Justification**: Matches the cost model described in Dhakal et al.

### 5. Task Generation ✅
- **Assumption**: 20 tasks randomly generated per environment
- **Justification**: Paper states "20-25 hand-coded tasks" — we interpret this as "pre-generated per environment"

## How We Use RL for Anticipatory vs. Myopic Behavior

### The Core Difference

Both agents operate in the **same persistent environment** with the same action space, rewards, and task sequences. They differ **only** in how they handle task completion during learning:

| Aspect | Myopic Agent | Anticipatory Agent |
|--------|--------------|-------------------|
| **Formulation** | Sequence of episodic tasks | Continuing-task MDP |
| **At task success** | Terminal state (no bootstrap) | Bootstrap through next task |
| **Value function** | $V_{\text{myo}}(s,\tau) = \mathbb{E}[\sum_{t=0}^{T_\tau-1} \gamma^t r_t]$ | $V(s,\tau) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t]$ |
| **Preparedness** | No incentive to prepare | Values persistent state under future tasks |

### Bellman Targets

The agents use different Bellman targets at successful task boundaries:

```python
# Within-task transition (both agents)
y = r + γ * max Q(s', τ, a')

# Task completion transition
if myopic:
    y = r  # Terminal, no bootstrap
elif anticipatory:
    y = r + γ * max Q(s', τ', a')  # Bootstrap through next task τ'
```

where `τ'` is the next task sampled from the task distribution.

### What This Means in Practice

**Myopic agent**:
- Treats each task as an isolated episode
- No reason to care about the post-task state `s'`
- Optimizes: "Complete current task as cheaply as possible"

**Anticipatory agent**:
- Learns continuing value across task boundaries
- Values the post-task state `s'` through future tasks
- Optimizes: "Complete current task while leaving the world well-prepared"

### The Role of Discount Factor γ

The discount factor controls the **anticipation horizon**:

- **Low γ** (e.g., 0.9): Emphasizes near-term tasks (~10 tasks ahead)
- **High γ** (e.g., 0.99): Values long-term preparedness (~100 tasks ahead)
- **γ → 0**: Reduces to myopic behavior

From our report (Theorem 1):
```
|Q*_γ(x,a) - E[r(x,a,x')]| ≤ (γ/(1-γ)) R_max
```

This formalizes how γ controls the weight placed on future tasks vs. immediate reward.

### Measuring Anticipatory Behavior

We evaluate anticipation through multiple metrics:

1. **Average task cost** (planner-evaluated using LazyPRM + pick/place costs)
   - Lower cost indicates better planning
   
2. **Auto-satisfaction rate** (tasks already satisfied at start)
   - Higher rate indicates better preparedness
   
3. **Steps per task**
   - Fewer steps indicate efficient organization

4. **Average task return** (RL reward)
   - Higher return indicates better policy

## Training Details

### Network Architecture
- **Input**: 8-channel image (render_size × render_size)
- **Architecture**: Convolutional Q-Network
  ```
  Conv2D(8 → 32, kernel=3, stride=2)
  Conv2D(32 → 64, kernel=3, stride=2)
  Conv2D(64 → 64, kernel=3, stride=2)
  Flatten → Dense(512) → Dense(|A|=6)
  ```

### Hyperparameters
- **Replay buffer**: 100k transitions
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epsilon decay**: Linear from 1.0 to 0.05
- **Target network update**: Every 1000 steps
- **Discount factor (γ)**: 0.95 (anticipatory), 0.95 (myopic within-task)

### Reward Function
```python
success_reward = +12.0
step_penalty = -1.0
invalid_action_penalty = -5.0
correct_pick_bonus = +1.0  # Picking a target block
```

## Results Summary

From our experiments (see full report for details):

### Blockworld Results
| Metric | Anticipatory | Myopic | Improvement |
|--------|-------------|--------|-------------|
| Avg task cost | 17,765 | 22,751 | **21.9% reduction** |

### Restaurant Results (for comparison)
| Metric | Anticipatory | Myopic | Improvement |
|--------|-------------|--------|-------------|
| Steps/task | 1.893 | 1.973 | **4.0% reduction** |
| Task return | 10.96 | 10.37 | **5.6% increase** |
| Reward/step | 5.79 | 5.26 | **10.1% increase** |

**Key finding**: Anticipatory RL learns to organize the environment over time, reducing costs for the continuing task stream.

## Code Structure

```
anticipatory_rl/
├── envs/blockworld/
│   └── blockworld_env.py          # Main environment implementation
├── agents/blockworld/
│   ├── blockworld_image_dqn.py    # Image-based DQN training
│   └── blockworld_dqn_infer.py    # Inference and evaluation
└── blockworld/
    ├── motion.py                   # LazyPRM motion planner
    ├── planner.py                  # PDDL-based symbolic planner
    └── world.py                    # Core world state definitions
```

## Usage Example

```python
from anticipatory_rl.envs.blockworld.blockworld_env import Paper1BlockworldImageEnv

# Create environment
env = Paper1BlockworldImageEnv(
    task_library_size=20,
    max_task_steps=64,
    procedural_layout=True,
)

# Reset with new task
obs, info = env.reset(seed=42)

# Step through actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Access state information
robot_pos = info['robot']
placements = info['placements']
current_task = info['task_assignments']
```

## Comparison with Dhakal et al.

| Aspect | Dhakal et al. (2023) | Our Implementation |
|--------|---------------------|-------------------|
| **Method** | Symbolic planning + GNN cost estimator | Reinforcement learning (DQN) |
| **Cost estimation** | Offline supervised learning | Online value learning |
| **Anticipation** | One-step lookahead | Full discounted horizon |
| **Task knowledge** | Assumes known task distribution | Learns from experience |
| **Computation** | Requires many planner calls | Amortized through learned policy |

*Last updated: 20 April 2026*