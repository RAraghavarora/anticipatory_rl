# Plan: Fix SF-DQN Training Failures

## Problems Identified

| # | Problem | Symptom |
|---|---------|---------|
| 1 | Reward-to-cost ratio ≈ 0.2% | Signal (success +15) drowned by noise (costs up to -400/step) |
| 2 | `LayerNorm` on task weights `w(τ)` | Caps `‖w‖ ≈ 8`, forcing ψ to carry all magnitude burden |
| 3 | Gradient clipping `max_grad_norm=1.0` | Prevents weights from growing to needed scale |
| 4 | `sf_dim=64` too large | Degenerate, low-rank SF space with collisions |
| 5 | Gradient symmetry across ψ heads | All four heads receive identical gradient `2(Q - y) · w`, collapse into redundancy |

## Changes

### 1. Rebalance reward/cost scale (`env.py` defaults and `sf_dqn.py` CLI)

The success reward is numerically irrelevant compared to action costs. Rescale to make success meaningful:

- Change `success_reward` default from `15.0` → **`100.0`**
- Change `travel_cost_scale` default from `25.0` → **`1.0`**
- Change all action costs (`pick_cost`, `place_cost`, `wash_cost`, etc.) from `25.0` → **`1.0`**
- Change `invalid_action_penalty` from `6.0` → **`0.5`**

This shifts the Q-value range from ~[-8000, +15] to ~[-100, +100] — signal-to-noise ratio ≈ 1:1 instead of 1:500. The optimal policy is unchanged; only the relative magnitudes are corrected.

Apply these defaults in both `sf_dqn.py`'s `build_parser()` and the SLURM scripts if they override any of these values.

### 2. Remove `LayerNorm` from task weight head

Delete `nn.LayerNorm(sf_dim)` from `self.task_head` in the network constructor. The LayerNorm was added to prevent scale drift, but with the rebalanced rewards (fix #1), the target Q-values are within a reasonable range and the bilinear form needs room to learn the appropriate scale. The scale ambiguity (`ψ` and `w` can scale inversely) is benign with gradient descent — the network self-stabilizes.

### 3. Reduce `sf_dim` from 64 → 16 and relax grad clip

- Change `--sf-dim` default from `64` → **`16`**
- Change `--max-grad-norm` default from `1.0` → **`10.0`**

The task space has only 6 task types × ~15 locations × ~11 object kinds × 20 objects. The effective dimensionality is far below 64. A 16-dim SF provides enough capacity to represent distinct Q-values per task/action without degeneracy. The relaxed grad clip lets the network grow weights to the needed scale now that LayerNorm is removed.

### 4. Break gradient symmetry with staggered zero-initialization

All four ψ heads receive the identical gradient `∂L/∂ψᵢ = 2(Q - y) · w` because they're summed. Fix by **zero-initializing the output layers of ψₓ, ψ_y, ψ_z**:

```python
# After constructing the heads, zero-init the final Linear's weight and bias
# for all heads EXCEPT psi_t_head.
for head in [self.psi_x_head, self.psi_y_head, self.psi_z_head]:
    final_linear = head[-1]  # nn.Linear(hidden_dim//2, sf_dim)
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)
```

This creates a **curriculum through initialization**:
- At step 0: ψ = ψ_t only (other heads contribute zero). The base head learns the coarse "cost of acting" signal.
- As training progresses: the other heads gradually turn on via gradient updates, each learning task-specific refinements on top of ψ_t's base.
- This prevents all four heads from competing to learn the same signal from the start.

No new parameters, no new losses, no architectural changes — just an initialization strategy that breaks the symmetry.

## Files to Modify

1. **`anticipatory_rl/agents/restaurant/sf_dqn.py`**
   - Change defaults in `build_parser()` (reward, costs, sf_dim, grad_norm)
   - Remove `nn.LayerNorm(sf_dim)` from `task_head`
   - Add zero-initialization for ψₓ, ψ_y, ψ_z output layers in `__init__`

2. **`slurm/train_restaurant_sf_myopic.sh`**
   - Remove any CLI args that override the old defaults (they'll pick up the new ones)
   - No need to pass `--sf-dim 64` anymore — 16 will be the new default

## What Not to Do

- **Don't** change the conditional branching architecture itself — the factorization ψ = ψ_t + ψ_x + ψ_y + ψ_z is theoretically correct for the structured action domain.
- **Don't** add auxiliary losses or orthogonality constraints — adds complexity and hyperparameters without addressing the root symmetry cause.
- **Don't** normalize ψ to unit norm — this just moves the scale problem from w to ψ.
- **Don't** change the task specification or observation encoding — only the reward/cost ratios and the optimization dynamics need fixing.

## Verification

After applying changes:
1. Run `python -m py_compile anticipatory_rl/agents/restaurant/sf_dqn.py` for syntax check
2. Run the unit tests: `python -m unittest tests.restaurant.test_restaurant_factored_env -v`
3. Submit a short training run (50K steps) to verify success rate improves from ~0%
