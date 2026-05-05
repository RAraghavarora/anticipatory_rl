# Successor Features (SF) Restaurant Branch

## Overview
This branch implements Successor Features (SFs) for the symbolic restaurant domain as an alternative to the conditional branched dueling Q-network currently on the main `restaurant` branch. The SF approach is designed to directly formalize the anticipatory RL concept where "preparedness" corresponds to value under future task distributions. This aligns better with the core research on anticipatory RL compared to the previous branching DQN architecture.

## Motivation
- The conditional branched dueling Q-network showed near-zero success rates on simple pick_place tasks (~0.00 to 0.02), suggesting a fundamental architectural issue rather than just task complexity
- The previous architecture handled structured action spaces but didn't interact meaningfully with the anticipatory nature of the learning objective  
- Successor Features naturally support the theoretical connection to anticipatory RL via closed-form anticipatory value estimates

## Architecture

### Successor Feature Network
- **Shared Encoder**: Two-layer MLP mapping observations to hidden representations
- **Action Embeddings**: Learned embeddings for action types (11), object1 (20+1), location (16+1), object2 (20+1)
- **Conditional SF Heads**: Cascade of four heads:
  - `psi_t_head`: Encodes state + action-type features into SF space
  - `psi_x_head`: Encodes state + action-type + object1 features 
  - `psi_y_head`: Encodes state + action-type + object1 + location features
  - `psi_z_head`: Encodes state + action-type + object1 + location + object2 features
- **Output**: Concatenated SF vectors (default 64-dimensional total)

### Task Weight Network  
- **Task Encoder**: Maps task specifications (type, location, kind, name) to one-hot vectors
- **Task Head**: MLP maps one-hot task vectors to same dimension as SF space
- **Output**: Task-dependent weight vectors used in Q-value computation

### Q-value Computation
```python
Q(s, a, τ) = psi(s, a) · w(τ)
```
where `·` denotes dot product, resulting in the bilinear SF parameterization.

## Training Modes

### Myopic Mode (`--myopic`)  
- At task success: bootstrap value is zero (no continuation to next task in value estimate)
- Effective Bellman target: `y = r` (only immediate reward)
- Suitable for non-continuing task learning

### Anticipatory Mode (default)
- At task success: bootstrap value based on Q-value of next task sampled from distribution
- Effective Bellman target: `y = r + γ max_a' Q(s', a', τ')`  
- Enables learning of anticipatory behaviors and cross-task preparation

## Usage

### Training
```bash
# Myopic SF training:
python -m anticipatory_rl.agents.restaurant.sf_dqn \
    --myopic --sf-dim 64 --total-steps 500000

# Anticipatory SF training:  
python -m anticipatory_rl.agents.restaurant.sf_dqn \
    --sf-dim 64 --total-steps 500000
```

### SLURM Scripts
- `slurm/train_restaurant_sf_myopic.sh`: SLURM script for myopic training
- `slurm/train_restaurant_sf_anticipatory.sh`: SLURM script for anticipatory training

### Key Hyperparameters
- `--sf-dim`: Dimension of the successor feature and task weight vectors (default: 64)
- `--hidden-dim`: Size of hidden layers in the network (default: 256)
- `--myopic`: Toggle for myopic vs anticipatory learning
- `--tasks-per-episode`: Number of tasks before hard episode boundary (-1 for unbounded)
- `--max-steps-per-task`: Episode truncation for individual tasks (default: 24)

## Key Design Decisions

### Bilinear Approach
- Using bi-linear SF parameterization where both `psi(s,a)` and `w(τ)` are trained end-to-end with TD loss only
- Accepts the scale identifiability ambiguity inherent in `Q(s,a,τ) = psi^T w` decomposition
- Chosen for getting a working implementation quickly given 0% success rates in previous approach

### Conditional SF Architecture  
- Preserves structured action space with typed components (action_type, object1, location, object2) instead of generic arguments
- Computes conditional successor features similar to the dueling SF architecture: psi_t + psi_x + psi_y + psi_z
- Maintains conditional masking relationships between valid actions

### Action Selection
- Uses full action enumeration from environmental validity masks to ensure action validity
- Computes Q-value for each valid action combination and selects greedy action
- Epsilon-greedy exploration during training

## Results Comparison

This approach should be compared against:
- Current `restaurant` branch: Conditional branched dueling Q-network
- Future improvements: More sophisticated factorizations that mitigate identification ambiguities  

The anticipatory mode aims to achieve measurably better success rates on pick_place tasks (>0.5) compared to the baseline near-zero performance, validating the structural change toward anticipatory methods.