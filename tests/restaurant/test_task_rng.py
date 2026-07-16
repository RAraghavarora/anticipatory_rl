"""Tests for the dedicated `env._task_rng` stream (replaces the pre-generated library).

Covers:
- same-seed determinism of the task sequence
- pairing across divergent action streams (key replacement for the library pairing test)
- isolation of task_rng from the global action RNG
- pick_place.object_name is state-conditioned (NOT paired) while task_type/target_location are
- task_rng stays paired after a pick_place object-name divergence (stream-split proof)
"""
from __future__ import annotations

import random

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.task_sampling import sample_task

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


def test_pick_place_object_name_not_paired():
    probe = RestaurantSymbolicEnv(config_path=CONFIG)
    probe.reset(seed=7)
    _force_pick_place(probe)
    first_obj = sample_task(probe).object_name
    assert first_obj is not None

    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    _force_pick_place(env_a)
    _force_pick_place(env_b)
    env_b.state.objects[first_obj].contained_in = "_blocked"

    task_a = sample_task(env_a)
    task_b = sample_task(env_b)

    assert task_a.object_name == first_obj
    assert task_a.task_type == task_b.task_type == "pick_place"
    assert task_a.target_location == task_b.target_location
    assert task_a.object_name != task_b.object_name


def test_task_rng_paired_after_pick_place_divergence():
    probe = RestaurantSymbolicEnv(config_path=CONFIG)
    probe.reset(seed=11)
    _force_pick_place(probe)
    first_obj = sample_task(probe).object_name
    assert first_obj is not None

    env_a = RestaurantSymbolicEnv(config_path=CONFIG)
    env_b = RestaurantSymbolicEnv(config_path=CONFIG)
    env_a.reset(seed=11)
    env_b.reset(seed=11)
    real_dist = dict(env_a.task_distribution)
    real_types = tuple(env_a.task_types)

    _force_pick_place(env_a)
    _force_pick_place(env_b)
    env_b.state.objects[first_obj].contained_in = "_blocked"
    task_a = sample_task(env_a)
    task_b = sample_task(env_b)
    assert task_a.object_name != task_b.object_name

    for env in (env_a, env_b):
        env.task_distribution = real_dist
        env.task_types = real_types

    seq_a = [_triplet(sample_task(env_a)) for _ in range(15)]
    seq_b = [_triplet(sample_task(env_b)) for _ in range(15)]
    assert seq_a == seq_b, (
        f"task_rng desynced after pick_place object-name divergence:\n  a: {seq_a}\n  b: {seq_b}"
    )
