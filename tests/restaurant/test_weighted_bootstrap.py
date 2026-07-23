"""Theory-verification tests for zero-variance boundary bootstrap.

Properties under test:
  1. Encoding roundtrip: _obs()[offset:] == _task_obs_encoding(task).
  2. Distribution completeness: sum(P) == 1.0, |support| == 42.
  3. Distribution matches sample_task: 100k empirical ≈ enumerated (+-0.01).
  4. _task_already_satisfied task=… param equivalence.
  5. Auto-satisfaction mask correctness at a known world state.
  6. Weighted-expectation point-mass degeneracy.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.task_sampling import sample_task
from anticipatory_rl.agents.restaurant.dqn import (
    _build_task_expectation_context,
    _compute_weighted_next_q,
    TaskExpectationContext,
    RestaurantQNetwork,
    ACTION_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_to_str(t: RestaurantTask) -> str:
    """Stable string key for binning."""
    return f"{t.task_type}|{t.target_location}|{t.target_kind}|{t.object_name}"


def _count_enumerated(env: RestaurantSymbolicEnv) -> int:
    return len(env.enumerate_task_distribution())


# ---------------------------------------------------------------------------
# 1. Encoding roundtrip
# ---------------------------------------------------------------------------

def test_task_encoding_roundtrip(env: RestaurantSymbolicEnv):
    """_obs() tail equals _task_obs_encoding(task) for any task."""
    offset = env.task_obs_offset

    # Test with the current task after reset.
    obs = env._obs()
    tail = obs[offset:]
    expected = env._task_obs_encoding(env.task)
    assert np.array_equal(tail, expected), (
        f"Roundtrip failed for reset task {env.task}"
    )

    # Test with several different tasks.
    test_tasks: list[RestaurantTask] = [
        RestaurantTask(task_type="serve_water", target_location="servingtable"),
        RestaurantTask(task_type="make_coffee", target_location="servingtable"),
        RestaurantTask(task_type="make_fruit_bowl", target_location="servingtable"),
        RestaurantTask(task_type="clear_containers", target_location="servingtable"),
        RestaurantTask(task_type="wash_objects", target_kind="bowl"),
        RestaurantTask(task_type="wash_objects", target_kind="knife"),
        RestaurantTask(task_type="pick_place", target_location="countertop", object_name="cup_0"),
        RestaurantTask(task_type="pick_place", target_location="fountain", object_name="jar_0"),
        RestaurantTask(task_type="pick_place", target_location="servingtable", object_name="plate_0"),
    ]
    for task in test_tasks:
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )
        obs = env._obs()
        tail = obs[offset:]
        expected = env._task_obs_encoding(task)
        assert np.array_equal(tail, expected), (
            f"Roundtrip failed for task {task}"
        )


# ---------------------------------------------------------------------------
# 2. Distribution completeness
# ---------------------------------------------------------------------------

def test_enumerate_task_distribution_sum_to_one(env: RestaurantSymbolicEnv):
    """Probabilities from enumerate_task_distribution sum to exactly 1.0."""
    dist = env.enumerate_task_distribution()
    total = sum(p for _, p in dist)
    assert abs(total - 1.0) < 1e-10, f"Probabilities sum to {total}, not 1.0"


def test_enumerate_task_distribution_count(env: RestaurantSymbolicEnv):
    """Toy level 3 supports exactly 42 task variants:
    1 serve_water + 1 make_coffee + 1 make_fruit_bowl + 1 clear_containers
    + 2 wash_objects + 36 pick_place.
    """
    dist = env.enumerate_task_distribution()
    assert len(dist) == 42, f"Expected 42, got {len(dist)}"


# ---------------------------------------------------------------------------
# 3. Distribution matches sample_task (CRITICAL)
# ---------------------------------------------------------------------------

def test_distribution_matches_sample_task(env: RestaurantSymbolicEnv, monkeypatch):
    """Draw 100k iid tasks; assert empirical frequencies match enumerated P.

    This is the critical test: if sample_task diverges from
    enumerate_task_distribution, the weighted expectation computes the wrong E.
    """
    env.reset(seed=0)
    dist = env.enumerate_task_distribution()
    enumerated_probs: dict[str, float] = {}
    for task, p in dist:
        enumerated_probs[_task_to_str(task)] = p

    n_samples = 100_000
    empirical: dict[str, float] = {}
    for _ in range(n_samples):
        t = sample_task(env)
        key = _task_to_str(t)
        empirical[key] = empirical.get(key, 0.0) + 1.0
    for key in empirical:
        empirical[key] /= n_samples

    # Check every key present in the distribution.
    failures: list[str] = []
    for key, expected in enumerated_probs.items():
        got = empirical.get(key, 0.0)
        if abs(got - expected) > 0.01:
            failures.append(f"  {key}: expected={expected:.4f} got={got:.4f}")

    # Check no extra keys in empirical.
    for key in empirical:
        if key not in enumerated_probs:
            failures.append(f"  {key}: in empirical only (p={empirical[key]:.4f})")

    assert not failures, (
        f"sample_task / enumerate_task_distribution mismatch "
        f"after {n_samples} samples:\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 4. _task_already_satisfied param equivalence
# ---------------------------------------------------------------------------

def test_task_already_satisfied_param_equivalence(env: RestaurantSymbolicEnv):
    """_task_already_satisfied(task=X) matches set_task + _task_already_satisfied()."""
    # After reset, cup_0 is clean at countertop (wash_ready location),
    # so wash_objects(cup) would be auto-satisfied. But cup kind is NOT
    # in wash_kind_distribution (only bowl/knife), so wash_objects requires
    # bowl or knife. They start dirty with 60% probability.
    # We need tasks where the answer is deterministic after a known setup.

    # Reproducible setup: seed 0 => cup_0 at countertop (clean),
    # cup_1 at dishwasher (dirty), water_machine at coffeemachine,
    # plate_0 at servingtable, etc.
    env.reset(seed=0)

    # After reset, let's check a few tasks.
    # wash_objects(bowl): depends on whether bowl started clean/dirty.
    # seed=0 makes this deterministic. Let's just verify the param matches.
    test_cases = [
        RestaurantTask(task_type="serve_water", target_location="servingtable"),
        RestaurantTask(task_type="make_coffee", target_location="servingtable"),
        RestaurantTask(task_type="clear_containers", target_location="servingtable"),
        RestaurantTask(task_type="wash_objects", target_kind="bowl"),
        RestaurantTask(task_type="pick_place", target_location="countertop", object_name="cup_0"),
    ]

    for task in test_cases:
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )
        by_setting = env._task_already_satisfied()
        by_param = env._task_already_satisfied(task=task)
        assert by_setting == by_param, (
            f"Mismatch for {task}: set_task={by_setting}, task=param={by_param}"
        )

        # Also test with a different task set on env (not-matching).
        # Set a different task and verify the param still gives the right answer.
        env.set_task("serve_water", target_location="servingtable")
        by_param_again = env._task_already_satisfied(task=task)
        assert by_param_again == by_param, (
            f"Param failed when env.task differs: {task}"
        )


# ---------------------------------------------------------------------------
# 5. Auto-satisfaction mask correctness at a known world state
# ---------------------------------------------------------------------------

def test_auto_satisfaction_mask_correctness(env: RestaurantSymbolicEnv):
    """At a known world state, verify which tasks are/aren't auto-satisfied.

    Seed 0 (deterministic):
      - cup_0: countertop, clean
      - cup_1: dishwasher, dirty
      - water_fountain: fountain (permanent water)
      - water_machine: coffeemachine (water present)
      - plate_0: servingtable
      - jar_0: shelf
      - bowl_0, knife_0: countertop (60% dirty at seed 0 → both dirty)
      - apple_0: countertop
    """
    env.reset(seed=0)

    dist = env.enumerate_task_distribution()
    auto_tasks: list[str] = []
    not_auto_tasks: list[str] = []

    for task, _ in dist:
        sat = env._task_already_satisfied(task=task)
        key = _task_to_str(task)
        if sat:
            auto_tasks.append(key)
        else:
            not_auto_tasks.append(key)

    # --- Tasks that MUST be auto-satisfied ---
    # water_machine has water at coffeemachine, cup_0 is clean at countertop
    #
    # We cannot assert many "must be auto" without knowing the exact dirty
    # state of bowl_0/knife_0 at seed 0. Let's check the actual state first.
    bowl_dirty = env.state.objects["bowl_0"].dirty
    knife_dirty = env.state.objects["knife_0"].dirty

    # wash_objects(knife) should be auto ONLY if knife is clean at wash_ready location.
    # knife_0 at countertop (wash_ready). Auto iff not dirty.
    kn_key = "wash_objects|None|knife|None"
    if knife_dirty:
        assert kn_key in not_auto_tasks, f"{kn_key} should NOT be auto (knife is dirty)"
    else:
        assert kn_key in auto_tasks, f"{kn_key} should be auto (knife is clean)"

    bowl_key = "wash_objects|None|bowl|None"
    if bowl_dirty:
        assert bowl_key in not_auto_tasks, f"{bowl_key} should NOT be auto (bowl is dirty)"
    else:
        assert bowl_key in auto_tasks, f"{bowl_key} should be auto (bowl is clean)"

    # serve_water(servingtable): no cup/mug with water at servingtable → NOT auto.
    assert "serve_water|servingtable|None|None" in not_auto_tasks

    # make_coffee(servingtable): no cup/mug with coffee → NOT auto.
    assert "make_coffee|servingtable|None|None" in not_auto_tasks

    # make_fruit_bowl(servingtable): no bowl at servingtable → NOT auto.
    assert "make_fruit_bowl|servingtable|None|None" in not_auto_tasks

    # clear_containers(servingtable): plate_0 IS at servingtable → NOT auto.
    assert "clear_containers|servingtable|None|None" in not_auto_tasks

    # pick_place(cup_0, countertop): cup_0 IS at countertop, hand is free → auto.
    assert "pick_place|countertop|None|cup_0" in auto_tasks

    # pick_place(cup_0, servingtable): cup_0 NOT at servingtable → NOT auto.
    assert "pick_place|servingtable|None|cup_0" in not_auto_tasks

    # pick_place(cup_1, dishwasher): cup_1 IS at dishwasher, hand free → auto.
    assert "pick_place|dishwasher|None|cup_1" in auto_tasks


# ---------------------------------------------------------------------------
# 6. Weighted-expectation point-mass degeneracy
# ---------------------------------------------------------------------------

def test_weighted_expectation_point_mass(env: RestaurantSymbolicEnv):
    """When P is a point mass on a single tau', the weighted expectation
    equals the single-sample DDQN Q(s', tau', a').
    """
    device = torch.device("cpu")

    # Build the full context, then overwrite probabilities to a point mass.
    ctx = _build_task_expectation_context(env, device)
    n_tasks = ctx.n_tasks
    obs_dim = ctx.obs_offset + ctx.obs_slices.shape[1]
    world_dim = ctx.obs_offset

    # Pick target task k* = 0 (first in enumeration = serve_water, servingtable).
    k_star = 0
    point_probs = torch.zeros(n_tasks, device=device)
    point_probs[k_star] = 1.0

    # Build a custom TaskExpectationContext with point-mass probabilities.
    point_ctx = TaskExpectationContext(
        n_tasks=n_tasks,
        probabilities=point_probs,
        obs_slices=ctx.obs_slices,
        obs_offset=ctx.obs_offset,
        auto_complete_action=ctx.auto_complete_action,
        auto_complete_masks=ctx.auto_complete_masks,
    )

    # Create a DQN (small hidden dim to keep test fast).
    q_net = RestaurantQNetwork(
        input_dim=obs_dim,
        action_type_dim=len(ACTION_TYPES),
        object_dim=env.num_objects + 1,
        location_dim=env.num_locations + 1,
        hidden_dim=64,
    )
    target_net = RestaurantQNetwork(
        input_dim=obs_dim,
        action_type_dim=len(ACTION_TYPES),
        object_dim=env.num_objects + 1,
        location_dim=env.num_locations + 1,
        hidden_dim=64,
    )
    target_net.load_state_dict(q_net.state_dict())
    q_net.eval()
    target_net.eval()

    # Collect several world states (next_states) by stepping the env.
    n_boundary = 4
    world_obs_list: list[np.ndarray] = []
    next_atm_list: list[np.ndarray] = []
    next_o1m_list: list[np.ndarray] = []
    next_lm_list: list[np.ndarray] = []
    next_o2m_list: list[np.ndarray] = []

    # Also record auto masks for the batch (all zeros for a clean test).
    next_auto_masks: list[np.ndarray] = []

    env.reset(seed=0)
    for _ in range(n_boundary):
        # Give observation with a specific task so we can get next_masks.
        env.set_task("serve_water", target_location="servingtable")
        obs = env._obs()
        next_masks = env._compute_action_masks()
        world_obs_list.append(obs[:world_dim])
        next_atm_list.append(next_masks["valid_action_type_mask"])
        next_o1m_list.append(next_masks["valid_object1_mask"])
        next_lm_list.append(next_masks["valid_location_mask"])
        next_o2m_list.append(next_masks["valid_object2_mask"])
        next_auto_masks.append(np.zeros(n_tasks, dtype=np.float32))

        # Step randomly to get a different world.
        at_mask = next_masks["valid_action_type_mask"]
        valid_types = np.flatnonzero(at_mask > 0.5)
        if len(valid_types) == 0:
            continue
        at = int(env._rng.choice(valid_types))
        o1_mask = next_masks["valid_object1_mask"][at]
        valid_o1 = np.flatnonzero(o1_mask > 0.5)
        o1 = int(env._rng.choice(valid_o1))
        loc_mask = next_masks["valid_location_mask"][at]
        valid_loc = np.flatnonzero(loc_mask > 0.5)
        loc = int(env._rng.choice(valid_loc))
        o2_mask = next_masks["valid_object2_mask"][at, o1]
        valid_o2 = np.flatnonzero(o2_mask > 0.5)
        o2 = int(env._rng.choice(valid_o2))
        env.step({"action_type": at, "object1": o1, "location": loc, "object2": o2})

    # Build a synthetic TensorDict batch.
    from tensordict import TensorDict
    td = TensorDict({
        "next_state": torch.tensor(np.stack([
            np.concatenate([wo, np.zeros(obs_dim - world_dim, dtype=np.float32)])
            for wo in world_obs_list
        ]), dtype=torch.float32),
        "next_action_type_mask": torch.tensor(np.stack(next_atm_list), dtype=torch.float32),
        "next_object1_mask": torch.tensor(np.stack(next_o1m_list), dtype=torch.float32),
        "next_location_mask": torch.tensor(np.stack(next_lm_list), dtype=torch.float32),
        "next_object2_mask": torch.tensor(np.stack(next_o2m_list), dtype=torch.float32),
        "next_auto_satisfied_mask": torch.tensor(np.stack(next_auto_masks), dtype=torch.float32),
        "task_boundary": torch.ones(n_boundary, dtype=torch.float32),
        "done": torch.zeros(n_boundary, dtype=torch.float32),
    }, batch_size=torch.Size([n_boundary]))

    boundary_mask = torch.ones(n_boundary, dtype=torch.bool)

    weighted_q = _compute_weighted_next_q(
        q_net, target_net, td, point_ctx, boundary_mask, device,
    )

    # Single-sample: for each boundary, run DDQN on (world + task_k*) only.
    task_enc = ctx.obs_slices[k_star].unsqueeze(0)  # (1, task_obs_dim)
    single_qs: list[float] = []
    with torch.no_grad():
        for i in range(n_boundary):
            wo = torch.tensor(world_obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            state_obs = torch.cat([wo, task_enc], dim=-1)
            atm = torch.tensor(next_atm_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            o1m = torch.tensor(next_o1m_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            lm = torch.tensor(next_lm_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            o2m = torch.tensor(next_o2m_list[i], dtype=torch.float32, device=device).unsqueeze(0)

            # DDQN: greedy action from q_net, Q value from target_net.
            action_type, o1, loc, o2 = q_net(state_obs, action_type_masks=atm,
                                             object1_masks=o1m, location_masks=lm,
                                             object2_masks=o2m, decode_greedy=True)
            single_q = target_net(state_obs, action_types=action_type,
                                  object1=o1, location=loc, object2=o2,
                                  action_type_masks=atm, object1_masks=o1m,
                                  location_masks=lm, object2_masks=o2m)
            single_qs.append(single_q.item())

    weighted_vals = weighted_q.squeeze(1).tolist()
    for i in range(n_boundary):
        assert abs(weighted_vals[i] - single_qs[i]) < 1e-5, (
            f"Boundary {i}: weighted={weighted_vals[i]:.6f} != single={single_qs[i]:.6f}"
        )
