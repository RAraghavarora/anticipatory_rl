"""Regression tests for plate reset state."""
from __future__ import annotations

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


CONFIG_PATH = "configs/restaurant/toy_level_3.yaml"


def test_clear_containers_non_auto_at_reset():
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
    env.reset(seed=0)
    env.set_task("clear_containers", target_location="servingtable")
    assert env._task_already_satisfied() is False
    assert env._is_pickable_kind("plate") is True
