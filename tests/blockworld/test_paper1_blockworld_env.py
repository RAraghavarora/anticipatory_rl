from __future__ import annotations

import numpy as np

from anticipatory_rl.envs import Paper1BlockworldImageEnv
from blockworld.world import Task, WorldConfig, WorldState


def make_env(**kwargs) -> Paper1BlockworldImageEnv:
    defaults = {
        "procedural_layout": False,
        "render_tile_px": 8,
        "max_task_steps": 4,
    }
    defaults.update(kwargs)
    return Paper1BlockworldImageEnv(**defaults)


def canonical_config() -> WorldConfig:
    return WorldConfig()


def test_reset_invariants() -> None:
    env = Paper1BlockworldImageEnv(procedural_layout=True, render_tile_px=8)
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert len(env.task_library) == 20
    assert len(env.state.placements) == len(env.config.all_blocks)
    assert len(set(env.state.placements.values())) == len(env.state.placements)
    assert all(coord in env.config.region_cells for coord in env.state.placements.values())
    for task in env.task_library:
        assert 1 <= len(task.assignments) <= 2
        for block, region in task.assignments:
            assert block in env.config.nonwhite_blocks
            assert region in env.config.region_coords
            assert not region.startswith("white")
    assert info["task_size"] in {1, 2}


def test_boundary_move_penalized_noop() -> None:
    env = make_env()
    config = canonical_config()
    obs, _ = env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {"a": config.region_coords["red"]},
            "robot_pos": (0, 0),
            "task_library": [Task((("a", "blue"),))],
            "task": Task((("a", "blue"),)),
        },
    )
    assert obs.shape[0] == 8
    _, reward, success, horizon, _ = env.step(env.MOVE_LEFT)
    assert reward == -6.0
    assert not success
    assert not horizon
    assert env.state.robot == (0, 0)


def test_pick_and_place_dynamics() -> None:
    env = make_env()
    config = canonical_config()
    task = Task((("a", "blue"),))
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {
                "a": config.region_coords["red"],
                "c": config.region_coords["green"],
            },
            "robot_pos": config.region_coords["red"],
            "task_library": [task],
            "task": task,
        },
    )

    _, reward, success, horizon, info = env.step(env.PICK)
    assert reward == 0.0
    assert not success
    assert not horizon
    assert env.state.holding == "a"
    assert "a" not in env.state.placements
    assert not info["can_pick"]

    env.state.robot = (3, 0)
    _, reward, success, horizon, info = env.step(env.PLACE)
    assert reward == env.success_reward
    assert success
    assert not horizon
    assert env.state.holding is None
    assert env.state.placements["a"] == config.region_coords["blue"]
    assert info["success"] is True


def test_place_rejects_occupied_region() -> None:
    env = make_env()
    config = canonical_config()
    task = Task((("a", "blue"),))
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "state": WorldState(
                robot=(3, 0),
                placements={
                    "b": config.region_coords["blue"],
                    "c": config.region_coords["green"],
                },
                holding="a",
            ),
            "task_library": [task],
            "task": task,
        },
    )
    _, reward, success, horizon, _ = env.step(env.PLACE)
    assert reward == -6.0
    assert not success
    assert not horizon
    assert env.state.holding == "a"
    assert env.state.placements["b"] == config.region_coords["blue"]


def test_double_task_requires_both_assignments() -> None:
    env = make_env()
    config = canonical_config()
    task = Task((("a", "red"), ("b", "blue")))
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {
                "a": config.region_coords["red"],
                "b": config.region_coords["green"],
            },
            "task_library": [task],
            "task": task,
        },
    )
    assert env._task_already_satisfied() is False  # noqa: SLF001
    env.state.placements["b"] = config.region_coords["blue"]
    assert env._task_already_satisfied() is True  # noqa: SLF001


def test_non_task_clutter_move_does_not_break_satisfied_task() -> None:
    env = make_env()
    config = canonical_config()
    task = Task((("a", "red"),))
    white_region = config.white_regions[0]
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {
                "a": config.region_coords["red"],
                "f": config.region_coords["blue"],
            },
            "task_library": [task],
            "task": task,
        },
    )
    assert env._task_already_satisfied() is True  # noqa: SLF001
    env.state.placements["f"] = config.region_coords[white_region]
    assert env._task_already_satisfied() is True  # noqa: SLF001


def test_success_resamples_without_reset_and_sets_auto_followup() -> None:
    env = make_env()
    config = canonical_config()
    first = Task((("a", "red"),))
    second = Task((("b", "blue"),))
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {
                "a": config.region_coords["red"],
                "b": config.region_coords["blue"],
            },
            "task_library": [second],
            "task": first,
        },
    )
    placements_before = dict(env.state.placements)
    _, reward, success, horizon, info = env.step(env.MOVE_UP)
    assert reward == env.success_reward
    assert success
    assert not horizon
    assert env.state.placements == placements_before
    assert env.current_task.assignments == second.assignments
    assert info["next_auto_satisfied"] is True


def test_horizon_resamples_without_reset() -> None:
    env = make_env(max_task_steps=1)
    config = canonical_config()
    first = Task((("a", "blue"),))
    second = Task((("b", "red"),))
    env.reset(
        seed=0,
        options={
            "world_config": config,
            "placements": {
                "a": config.region_coords["red"],
                "b": config.region_coords["green"],
            },
            "robot_pos": (0, 0),
            "task_library": [second],
            "task": first,
        },
    )
    placements_before = dict(env.state.placements)
    _, reward, success, horizon, _ = env.step(env.MOVE_RIGHT)
    assert reward == -1.0
    assert not success
    assert horizon
    assert env.state.placements == placements_before
    assert env.current_task.assignments == second.assignments


def test_observation_masks_and_single_task_slot2_zero() -> None:
    env = make_env()
    config = canonical_config()
    task = Task((("a", "red"),))
    obs, _ = env.reset(
        seed=0,
        options={
            "world_config": config,
            "state": WorldState(
                robot=(2, 2),
                placements={"b": config.region_coords["blue"]},
                holding="a",
            ),
            "task_library": [task],
            "task": task,
        },
    )
    x0, y0, x1, y1 = env._tile_bounds((2, 2))  # noqa: SLF001
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0
    assert float(obs.max()) <= 1.0
    assert np.all(obs[5] == 0.0)
    assert np.all(obs[6] == 0.0)
    assert np.all(obs[7] == 0.0)
    assert np.all(obs[3, y0:y1, x0:x1] == 1.0)
    rx0, ry0, rx1, ry1 = env._tile_bounds(config.region_coords["red"])  # noqa: SLF001
    assert np.all(obs[4, ry0:ry1, rx0:rx1] == 1.0)
