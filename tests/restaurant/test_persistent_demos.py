"""Regression tests for persistent myopic oracle demo rollouts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from anticipatory_rl.agents.restaurant import dqn
from anticipatory_rl.agents.restaurant.dqn import _persistent_oracle_rollout
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import PlannerResult


CONFIG_PATH = "configs/restaurant/toy_level_3.yaml"


class _Collector:
    def __init__(self) -> None:
        self.items: List[Any] = []

    def add(self, td: Any) -> None:
        self.items.append(td)


def _make_move_plan(actions):
    """Return a fixed plan of move actions; each element is a (src, dst) pair."""
    def solver(env, state, task, *, planner_path, domain_path, alias, timeout_s):
        del state, task, planner_path, domain_path, alias, timeout_s
        plan = [("move", [src, dst]) for src, dst in actions]
        return PlannerResult(success=True, plan_actions=plan, plan_cost=1.0, solve_time_s=0.0)
    return solver


def _make_adaptive_move_solver():
    """Return a 1-step move plan to a location different from the agent's current one."""
    def solver(env, state, task, *, planner_path, domain_path, alias, timeout_s):
        del state, task, planner_path, domain_path, alias, timeout_s
        src = env.state.agent_location
        dst = "coffeemachine" if src != "coffeemachine" else "countertop"
        return PlannerResult(success=True, plan_actions=[("move", [src, dst])], plan_cost=1.0, solve_time_s=0.0)
    return solver


def _make_failing_solver():
    def solver(env, state, task, *, planner_path, domain_path, alias, timeout_s):
        del env, state, task, planner_path, domain_path, alias, timeout_s
        return PlannerResult(success=False, plan_actions=[], plan_cost=0.0, solve_time_s=0.0, error="mock")
    return solver


def _force_non_auto_task(env, task_type="clear_containers", target_location="servingtable"):
    env.task_distribution = {name: 0.0 for name in env.task_types}
    env.task_distribution[task_type] = 1.0

    def _resample():
        env.set_task(task_type, target_location=target_location, task_source="test")
        env._task_steps = 0

    env._resample_task = _resample


def test_persistent_rollout_counts_outcomes(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=2, rng_seed=0)
    env.reset(seed=0)
    _force_non_auto_task(env)

    # Move out and back so the persistent agent ends where it started.
    plan = [("countertop", "coffeemachine"), ("coffeemachine", "countertop")]
    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_move_plan(plan))

    collector = _Collector()
    result = _persistent_oracle_rollout(
        env,
        n_outcomes=3,
        max_steps=2,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=200,
        transition_store=collector,
    )
    assert result["outcomes"] == 3
    assert result["stored"] == 6
    assert len(collector.items) == 6


def test_auto_success_skips_transition(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=64, rng_seed=0)
    env.reset(seed=0)

    # Make serve_water auto-satisfied at servingtable.
    cup = env.state.objects["cup_0"]
    cup.location = "servingtable"
    cup.filled_with = "water"
    cup.dirty = False
    env.set_task("serve_water", target_location="servingtable", task_source="test")
    assert env._pending_auto_success is True

    _force_non_auto_task(env, "clear_containers", "servingtable")
    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_failing_solver())

    collector = _Collector()
    result = _persistent_oracle_rollout(
        env,
        n_outcomes=1,
        max_steps=64,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=200,
        transition_store=collector,
    )
    assert result["outcomes"] == 1
    assert result["stored"] == 0
    assert len(collector.items) == 0
    # Env advanced past the auto task.
    assert not (env.task.task_type == "serve_water" and env.task.target_location == "servingtable")


def test_planner_failure_preserves_world(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=64, rng_seed=0)
    env.reset(seed=0)
    _force_non_auto_task(env, "clear_containers", "servingtable")

    world_before = env._action_mask_state_key()
    task_before = env.task
    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_failing_solver())

    collector = _Collector()
    result = _persistent_oracle_rollout(
        env,
        n_outcomes=1,
        max_steps=64,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=200,
        transition_store=collector,
    )
    assert result["outcomes"] == 1
    assert result["stored"] == 0
    assert len(collector.items) == 0
    assert env._action_mask_state_key() == world_before
    assert env.task != task_before


def test_plan_terminal_flags(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=2, rng_seed=0)
    env.reset(seed=0)
    _force_non_auto_task(env, "clear_containers", "servingtable")

    plan = [("countertop", "coffeemachine"), ("coffeemachine", "countertop")]
    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_move_plan(plan))

    collector = _Collector()
    _persistent_oracle_rollout(
        env,
        n_outcomes=1,
        max_steps=2,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=200,
        transition_store=collector,
    )
    assert len(collector.items) == 2
    assert float(collector.items[0]["done"]) == 0.0
    assert float(collector.items[1]["done"]) == 0.0


def test_world_reset_every_n_outcomes(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=1, rng_seed=0)
    env.reset(seed=0)
    _force_non_auto_task(env, "clear_containers", "servingtable")

    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_adaptive_move_solver())

    reset_calls = []
    original_reset = env.reset

    def counting_reset(*, seed=None, options=None):
        reset_calls.append(seed)
        return original_reset(seed=seed, options=options)

    monkeypatch.setattr(env, "reset", counting_reset)

    collector = _Collector()
    _persistent_oracle_rollout(
        env,
        n_outcomes=5,
        max_steps=1,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=2,
        transition_store=collector,
    )
    # One reset at start plus after every 2 outcomes (outcomes 2 and 4).
    assert len(reset_calls) == 3


def test_long_plan_treated_as_failure(monkeypatch):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH, max_steps_per_task=64, rng_seed=0)
    env.reset(seed=0)
    _force_non_auto_task(env, "clear_containers", "servingtable")

    world_before = env._action_mask_state_key()
    plan = [("countertop", "coffeemachine"), ("coffeemachine", "countertop")]
    monkeypatch.setattr(dqn, "solve_restaurant_task_with_fd", _make_move_plan(plan))

    collector = _Collector()
    result = _persistent_oracle_rollout(
        env,
        n_outcomes=1,
        max_steps=1,
        seed_base=0,
        planner_path=Path("dummy"),
        domain_path=Path("dummy"),
        env_reset_tasks=200,
        transition_store=collector,
    )
    assert result["outcomes"] == 1
    assert result["stored"] == 0
    assert len(collector.items) == 0
    assert env._action_mask_state_key() == world_before
