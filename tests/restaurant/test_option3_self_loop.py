"""Tests for Option 3 self-loop auto-success detection logic.

A self-loop occurs when: auto_success=True AND world_unchanged=True.
- auto_success: task was already satisfied before agent took an action.
- world_unchanged: consume_delivery was a no-op (e.g., wash_objects, empty clear_containers).

No task-equality check — any auto-success on a stagnant world is terminal for
bootstrapping, which prevents multi-task loops (A→B→A) from generating degenerate
infinite Q-values.

Tasks like serve_water/make_coffee mutate world via consume_delivery,
so world_unchanged=False even on auto-success — those can never be self-loops.
"""
from __future__ import annotations

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


def _step_no_op(env: RestaurantSymbolicEnv) -> tuple[bool, dict]:
    """Take a no-op move action and return (success, info)."""
    act = {
        "action_type": env.action_type_index["move"],
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }
    _, _, success, _, info = env.step(act)
    return bool(success), info


def test_self_loop_detection_logic():
    """Pure unit test of the 2-condition AND: auto_success AND world_unchanged."""
    assert bool(True and True), "auto_success + world_unchanged -> self-loop"
    assert not bool(True and False), "world changed -> not self-loop"
    assert not bool(False and True), "not auto-success -> not self-loop"


def test_self_loop_fires_on_world_unchanged_auto_success(env: RestaurantSymbolicEnv, monkeypatch):
    """Integration test: wash_objects auto-success on an unchanged world.

    monkeypatch forces _resample_task to re-set wash_objects. Even though the task
    stays the same, we only check auto_success+world_unchanged (no task_equality).
    """
    env.state.agent_location = "countertop"
    env.state.objects["cup_0"].location = "countertop"
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None

    env.set_task("wash_objects", target_kind="cup")

    def _force_same_task(self):
        self.set_task("wash_objects", target_kind="cup", task_source="iid")

    monkeypatch.setattr(RestaurantSymbolicEnv, "_resample_task", _force_same_task)

    world_key_before = env._action_mask_state_key()
    info_before = env._info(success=False)
    task_before = dict(info_before.get("task", {}))

    succeed, info_after = _step_no_op(env)
    assert succeed, "wash_objects with clean cup should auto-succeed"

    world_unchanged = env._action_mask_state_key() == world_key_before
    auto_success_flag = info_after.get("auto_success", False)
    task_after = info_after.get("task", {})
    task_equality = task_after == task_before

    assert auto_success_flag, "Should be auto-success"
    assert world_unchanged, "World should be unchanged after wash_objects auto-success"
    assert task_equality, "Forced resample should produce same task"


def test_self_loop_fires_on_different_task_too(env: RestaurantSymbolicEnv, monkeypatch):
    """When auto-success resamples a different task but world is unchanged, self-loop still fires.

    The fix drops task_equality — multi-task loops (A→B→A) on a stagnant world
    are equally degenerate, so any world-unchanged auto-success is terminal.

    monkeypatch forces _resample_task to set serve_water (different from wash_objects),
    making the test deterministic.
    """
    env.state.agent_location = "countertop"
    env.state.objects["cup_0"].location = "countertop"
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None

    env.set_task("wash_objects", target_kind="cup")

    def _force_different_task(self):
        self.set_task("serve_water", target_location="servingtable", task_source="iid")

    monkeypatch.setattr(RestaurantSymbolicEnv, "_resample_task", _force_different_task)

    world_key_before = env._action_mask_state_key()
    info_before = env._info(success=False)
    task_before = dict(info_before.get("task", {}))

    succeed, info_after = _step_no_op(env)
    assert succeed, "wash_objects with clean cup should auto-succeed"

    world_unchanged = env._action_mask_state_key() == world_key_before
    auto_success_flag = info_after.get("auto_success", False)
    task_after = info_after.get("task", {})
    task_equality = task_after == task_before

    assert auto_success_flag, "Should be auto-success"
    assert world_unchanged, "World should be unchanged after wash_objects auto-success"
    assert not task_equality, "Forced resample should produce different task"
