from __future__ import annotations

from typing import Callable

import numpy as np

from anticipatory_rl.agents.restaurant.dqn import (
    _auto_complete_replay_action_and_masks,
    _decide_task_transition,
)
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.utils import extract_masks


def _no_op(env: RestaurantSymbolicEnv) -> dict[str, int]:
    return {
        "action_type": env.action_type_index["move"],
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }


def _auto_step(
    env: RestaurantSymbolicEnv,
    resample: Callable[[], None],
) -> tuple[bool, bool, dict]:
    before = env._action_mask_state_key()
    env._resample_task = resample
    _, _, success, truncated, info = env.step(_no_op(env))
    assert success and not truncated and info["auto_success"]
    return success, env._action_mask_state_key() == before, info


def _bootstrap_done(
    *,
    success: bool = True,
    tasks: int = 0,
    env_tasks: int = 0,
    tasks_per_episode: int = 200,
    env_reset_tasks: int = 50,
) -> bool:
    return _decide_task_transition(
        success,
        False,
        tasks,
        env_tasks,
        1,
        tasks_per_episode,
        env_reset_tasks,
        0,
    ).bootstrap_done


def _prepare_clean_cup(env: RestaurantSymbolicEnv) -> None:
    cup = env.state.objects["cup_0"]
    cup.location = "countertop"
    cup.dirty = False
    cup.filled_with = None
    env.set_task("wash_objects", target_kind="cup")


def test_same_task_unchanged_world_auto_success_bootstraps(env: RestaurantSymbolicEnv):
    _prepare_clean_cup(env)
    success, world_unchanged, info = _auto_step(
        env,
        lambda: env.set_task("wash_objects", target_kind="cup", task_source="iid"),
    )

    assert world_unchanged
    assert info["task"]["task_type"] == "wash_objects"
    assert not _bootstrap_done(success=success)


def test_cross_task_unchanged_world_auto_success_bootstraps(env: RestaurantSymbolicEnv):
    _prepare_clean_cup(env)
    success, world_unchanged, info = _auto_step(
        env,
        lambda: env.set_task("clear_containers", target_location="servingtable", task_source="iid"),
    )

    assert world_unchanged
    assert info["task"]["task_type"] == "clear_containers"
    assert not _bootstrap_done(success=success)


def test_consecutive_unchanged_world_auto_successes_bootstrap(env: RestaurantSymbolicEnv):
    for obj in env.state.objects.values():
        if obj.location == "servingtable":
            obj.location = "shelf"
    _prepare_clean_cup(env)

    first_success, first_unchanged, _ = _auto_step(
        env,
        lambda: env.set_task("clear_containers", target_location="servingtable", task_source="iid"),
    )
    second_success, second_unchanged, _ = _auto_step(
        env,
        lambda: env.set_task("wash_objects", target_kind="cup", task_source="iid"),
    )

    assert first_unchanged and second_unchanged
    assert not _bootstrap_done(success=first_success)
    assert not _bootstrap_done(success=second_success, tasks=1, env_tasks=1)


def test_world_changing_auto_success_bootstraps(env: RestaurantSymbolicEnv):
    cup = env.state.objects["cup_0"]
    cup.location = "servingtable"
    cup.dirty = False
    cup.filled_with = "coffee"
    env.set_task("make_coffee", target_location="servingtable")

    success, world_unchanged, _ = _auto_step(
        env,
        lambda: env.set_task("wash_objects", target_kind="cup", task_source="iid"),
    )

    assert not world_unchanged
    assert cup.filled_with is None
    assert not _bootstrap_done(success=success)


def test_myopic_auto_success_is_terminal_at_credit_boundary(env: RestaurantSymbolicEnv):
    _prepare_clean_cup(env)
    success, _, _ = _auto_step(
        env,
        lambda: env.set_task("wash_objects", target_kind="cup", task_source="iid"),
    )

    assert _bootstrap_done(success=success, tasks_per_episode=1)


def test_50th_auto_success_bootstraps_at_world_boundary(env: RestaurantSymbolicEnv):
    _prepare_clean_cup(env)
    success, _, _ = _auto_step(
        env,
        lambda: env.set_task("wash_objects", target_kind="cup", task_source="iid"),
    )

    assert not _bootstrap_done(success=success, tasks=49, env_tasks=49)


def test_auto_success_uses_replay_only_auto_complete(env: RestaurantSymbolicEnv):
    masks = extract_masks(env._info(success=False))
    action, replay_masks = _auto_complete_replay_action_and_masks(env, masks)
    auto_idx = env.action_type_index["auto_complete"]

    assert action == {
        "action_type": auto_idx,
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }
    assert np.flatnonzero(replay_masks["valid_action_type_mask"]).tolist() == [auto_idx]
    assert not masks["valid_action_type_mask"][auto_idx]
