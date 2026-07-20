"""Tests for task-library index persistence, leading-auto skip, truncation, and
seeded layout determinism on toy_level_3."""
from __future__ import annotations

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.task_sampling import sample_task


def test_task_library_index_persists_across_reset(env):
    library = [
        {"task_type": "serve_water", "target_location": "servingtable"},
        {"task_type": "make_coffee", "target_location": "servingtable"},
        {"task_type": "make_fruit_bowl", "target_location": "servingtable"},
        {"task_type": "pick_place", "object_name": "apple_0", "target_location": "shelf"},
    ]
    library.extend(
        {"task_type": "serve_water", "target_location": "servingtable"} for _ in range(16)
    )

    env.reset(seed=0, options={"task_library": library})
    env._resample_task()
    env._resample_task()
    continued_index = env._task_library_index

    env.reset(seed=1)

    assert env._task_library_index == continued_index + 1
    expected = library[continued_index]
    assert env.task.task_type == expected["task_type"]
    assert env.task.target_location == expected.get("target_location")
    assert env.task.target_kind == expected.get("target_kind")
    assert env.task.object_name == expected.get("object_name")
    assert env.task.task_type != library[0]["task_type"]


def test_library_skips_leading_auto_tasks(env):
    # cup_0 starts clean/empty on countertop, so wash_objects(cup) is auto at reset.
    auto_task = {"task_type": "wash_objects", "target_kind": "cup"}
    non_auto = {"task_type": "serve_water", "target_location": "servingtable"}
    library = [auto_task, auto_task, auto_task, auto_task, non_auto]

    env.reset(seed=0, options={"task_library": library})

    assert env.task.task_type == "serve_water"
    assert env.task.task_type != "wash_objects"
    assert env._task_library_index == len(library)
    assert env.task.target_location == non_auto["target_location"]


def test_env_truncation_preserves_world(env):
    env.set_task("serve_water", target_location="servingtable")
    agent_before = env.state.agent_location
    holding_before = env.state.holding
    locations_before = {n: o.location for n, o in env.state.objects.items()}

    noop = {
        "action_type": env.action_type_index["move"],
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }
    truncated = False
    for _ in range(env.max_steps_per_task):
        _, _, _, truncated, _ = env.step(noop)
        if truncated:
            break
    assert truncated

    assert env.state.agent_location == agent_before
    assert env.state.holding == holding_before
    for name, loc in locations_before.items():
        assert env.state.objects[name].location == loc


def test_sample_object_layout_reset_seed_determinism():
    env_a = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    env_b = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    for _ in range(20):
        sample_task(env_a)
    for _ in range(5):
        sample_task(env_b)

    env_a.reset(seed=42)
    env_b.reset(seed=42)

    assert env_a.state.agent_location == env_b.state.agent_location
    assert set(env_a.state.objects) == set(env_b.state.objects)
    for name in env_a.state.objects:
        oa = env_a.state.objects[name]
        ob = env_b.state.objects[name]
        assert oa.location == ob.location, f"{name} location"
        assert oa.dirty == ob.dirty, f"{name} dirty"
        assert oa.filled_with == ob.filled_with, f"{name} filled_with"
