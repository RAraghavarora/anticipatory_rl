"""Tests for the dedicated `env._task_rng` stream.

Covers:
- same-seed determinism of the complete task sequence (including pick_place.object_name)
- full task-tuple pairing across divergent action streams
- isolation of task_rng from the global action RNG
- pick_place.object_name is state-independent (sampled from pick_place_object_distribution via task_rng)
- supported-object restriction: only configured objects appear
- state independence: relocating/containing objects does not change object_name
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import RestaurantPlannerState
from anticipatory_rl.envs.restaurant.task_sampling import sample_task
from scripts.restaurant.toy_anticipatory_oracle import _enumerate_future_tasks

CONFIG = "configs/restaurant/toy_level_3.yaml"


def _noop_move(env: RestaurantSymbolicEnv) -> dict:
    return {
        "action_type": env.action_type_index["move"],
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }


def _drive_to_boundary(env: RestaurantSymbolicEnv, action) -> None:
    _, _, success, truncated, _ = env.step(action)
    while not (success or truncated):
        _, _, success, truncated, _ = env.step(action)


def _full_tuple(task) -> tuple:
    return (task.task_type, task.target_location, task.target_kind, task.object_name)


def _triplet(task) -> tuple:
    return (task.task_type, task.target_location, task.target_kind)


def test_task_rng_determinism_same_seed():
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    seq_a = [_triplet(sample_task(env_a)) for _ in range(20)]
    seq_b = [_triplet(sample_task(env_b)) for _ in range(20)]
    assert seq_a == seq_b, f"task_rng not deterministic: {seq_a} vs {seq_b}"


def test_task_rng_paired_across_action_streams():
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    env_b.action_space.seed(202)

    noop = _noop_move(env_a)
    tasks_a, tasks_b = [], []
    for _ in range(20):
        tasks_a.append(_triplet(env_a.task))
        tasks_b.append(_triplet(env_b.task))
        _drive_to_boundary(env_a, noop)
        _drive_to_boundary(env_b, env_b.action_space.sample())

    assert tasks_a == tasks_b, (
        f"task sequence diverged across action streams:\n  a: {tasks_a}\n  b: {tasks_b}"
    )


def test_task_rng_independent_from_action_rng():
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    for _ in range(500):
        random.random()
        np.random.random()

    seq_a = [_triplet(sample_task(env_a)) for _ in range(20)]
    seq_b = [_triplet(sample_task(env_b)) for _ in range(20)]
    assert seq_a == seq_b, "task_rng leaked dependency on global random/np.random"


def _force_pick_place(env: RestaurantSymbolicEnv) -> None:
    env.task_distribution = {"pick_place": 1.0}
    env.task_types = ("pick_place",)


def test_pick_place_object_name_is_paired():
    """pick_place.object_name is drawn from task_rng via the fixed distribution,
    so two envs on the same seed must produce identical complete task tuples."""
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    _force_pick_place(env_a)
    _force_pick_place(env_b)

    # Divergent worlds: put a configured pick_place object inside something in env_b.
    cup_1 = "cup_1"
    assert cup_1 in env_b.pick_place_object_distribution
    env_b.state.objects[cup_1].contained_in = "_blocked"

    seq_a = [_full_tuple(sample_task(env_a)) for _ in range(30)]
    seq_b = [_full_tuple(sample_task(env_b)) for _ in range(30)]
    assert seq_a == seq_b, (
        f"Complete task tuples diverged despite shared task_rng:\n  a: {seq_a}\n  b: {seq_b}"
    )


def test_pick_place_object_name_state_independent():
    """Relocating or containing objects must NOT change which object_name is drawn.
    The distribution is fixed in config and sampled via task_rng."""
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    _force_pick_place(env_a)
    _force_pick_place(env_b)

    for name in env_b.pick_place_object_distribution:
        env_b.state.objects[name].location = "fountain"
        env_b.state.objects[name].contained_in = "bowl_0"

    seq_a = [_full_tuple(sample_task(env_a)) for _ in range(50)]
    seq_b = [_full_tuple(sample_task(env_b)) for _ in range(50)]
    assert seq_a == seq_b


def test_pick_place_object_support_restriction():
    """Only objects in pick_place_object_distribution should ever be sampled."""
    env = RestaurantSymbolicEnv(config_path=CONFIG)
    env.reset(seed=0)
    _force_pick_place(env)

    allowed = set(env.pick_place_object_distribution.keys())
    assert allowed == {"cup_0", "cup_1", "bowl_0", "knife_0", "jar_0", "plate_0"}
    # Run many samples; every one must be from the configured support.
    for _ in range(200):
        task = sample_task(env)
        assert task.object_name in allowed, (
            f"pick_place object_name '{task.object_name}' not in configured distribution "
            f"support {allowed}"
        )


@pytest.mark.parametrize(
    "distribution, message",
    [
        ({"missing": 1.0}, "unknown object"),
        ({"water_fountain": 1.0}, "not pickable"),
        ({"apple_0": 1.0}, "not eligible"),
        ({"cup_0": -1.0}, "negative weight"),
        ({"cup_0": 0.0}, "total weight"),
    ],
)
def test_pick_place_object_distribution_validation(distribution, message):
    env = RestaurantSymbolicEnv(config_path=CONFIG)
    with pytest.raises(ValueError, match=message):
        env._parse_pick_place_object_distribution({"pick_place_object_distribution": distribution})


def test_pick_place_object_distribution_deterministic():
    """Same seed → identical object_name sequence for pick_place tasks."""
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=42)
    env_b.reset(seed=42)
    _force_pick_place(env_a)
    _force_pick_place(env_b)

    seq_a = [sample_task(env_a).object_name for _ in range(50)]
    seq_b = [sample_task(env_b).object_name for _ in range(50)]
    assert seq_a == seq_b, (
        f"pick_place object_name sequence not deterministic:\n  a: {seq_a}\n  b: {seq_b}"
    )


def test_complete_task_tuples_paired_after_pick_place():
    """After a pick_place task, subsequent tasks must remain fully paired
    across agents with divergent worlds (task_rng stays in sync)."""
    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=11)
    env_b.reset(seed=11)
    real_dist = dict(env_a.task_distribution)
    real_types = tuple(env_a.task_types)

    _force_pick_place(env_a)
    _force_pick_place(env_b)

    # Different world state should NOT affect task sampling.
    plate_0 = "plate_0"
    assert plate_0 in env_b.pick_place_object_distribution
    env_b.state.objects[plate_0].contained_in = "_blocked"

    # Draw one task on each; object_name should be identical (task_rng-driven).
    task_a = sample_task(env_a)
    task_b = sample_task(env_b)
    assert _full_tuple(task_a) == _full_tuple(task_b), (
        f"pick_place task tuple diverged:\n  a: {_full_tuple(task_a)}\n  b: {_full_tuple(task_b)}"
    )

    # Restore full distribution and verify subsequent tasks are paired.
    for env in (env_a, env_b):
        env.task_distribution = real_dist
        env.task_types = real_types

    seq_a = [_triplet(sample_task(env_a)) for _ in range(15)]
    seq_b = [_triplet(sample_task(env_b)) for _ in range(15)]
    assert seq_a == seq_b, (
        f"task_rng desynced after pick_place:\n  a: {seq_a}\n  b: {seq_b}"
    )


def test_future_task_enumeration_uses_fixed_pick_place_distribution():
    env = RestaurantSymbolicEnv(config_path=CONFIG)
    env.reset(seed=0)
    state_a = RestaurantPlannerState.from_env(env)
    state_b = state_a.copy()
    state_b.objects["cup_0"].contained_in = "bowl_0"

    tasks_a = _enumerate_future_tasks(env, state_a)
    tasks_b = _enumerate_future_tasks(env, state_b)
    assert tasks_a == tasks_b

    pick_place = [
        (task, probability)
        for task, probability in tasks_a
        if task.task_type == "pick_place"
    ]
    expected_support = set(env.pick_place_object_distribution)
    assert {task.object_name for task, _ in pick_place} == expected_support
    assert sum(probability for _, probability in pick_place) == pytest.approx(
        env.task_distribution["pick_place"] / sum(env.task_distribution.values())
    )
