"""Tests for the training-loop task-transition fix (Task 4).

Guards the research-critical myopic-vs-anticipatory distinction:
- _decide_task_transition: timeouts count toward both horizons but never wipe the
  world on per-task truncation; only the step-limit safety resets the world.
- bootstrap_done: truncated alone NEVER sets it. For tpe=1 every boundary is
  terminal; for tpe=200 only the 200-task reset / step-limit is terminal.
- train(): a per-task timeout does NOT call env.reset() (world persists).
- env.step(): truncated and auto_success are mutually exclusive by environment contract.
"""
from __future__ import annotations

from pathlib import Path

import torch

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.utils import extract_masks, random_valid_index


def _transition(
    success: bool,
    truncated: bool,
    *,
    tasks: int = 0,
    env_tasks: int = 0,
    steps: int = 0,
    tpe: int = 1,
    env_reset: int = 200,
    step_limit: int = 0,
):
    from anticipatory_rl.agents.restaurant.dqn import _decide_task_transition
    return _decide_task_transition(
        success, truncated, tasks, env_tasks, steps, tpe, env_reset, step_limit,
    )


def test_decide_myopic_success_terminal():
    t = _transition(success=True, truncated=False, tasks=0, tpe=1)
    assert t.episode_done_flag
    assert t.bootstrap_done
    assert not t.trunc_reset_flag
    assert t.tasks_since_reset == 0


def test_decide_myopic_timeout_terminal_no_world_wipe():
    t = _transition(success=False, truncated=True, tasks=0, tpe=1)
    assert t.episode_done_flag
    assert t.bootstrap_done
    assert not t.trunc_reset_flag
    assert t.tasks_since_reset == 0


def test_decide_anticipatory_timeout_bootstraps_through():
    t = _transition(success=False, truncated=True, tasks=5, env_tasks=5, tpe=200)
    assert not t.episode_done_flag
    assert not t.bootstrap_done
    assert not t.trunc_reset_flag


def test_decide_env_reset_fires():
    t = _transition(success=True, truncated=False, tasks=199, env_tasks=199, tpe=200, env_reset=200)
    assert t.env_reset_flag
    assert t.episode_done_flag
    assert t.bootstrap_done
    assert t.env_tasks_since_reset == 0


def test_decide_step_limit_fires():
    t = _transition(
        success=False, truncated=False, tasks=5, env_tasks=5, steps=100, tpe=200, step_limit=100
    )
    assert t.trunc_reset_flag
    assert t.bootstrap_done
    assert t.tasks_since_reset == 0
    assert t.env_tasks_since_reset == 0


def test_decide_timeout_drives_env_reset():
    t = _transition(success=False, truncated=True, tasks=199, env_tasks=199, tpe=200, env_reset=200)
    assert t.env_reset_flag


def test_train_loop_timeout_no_world_wipe(monkeypatch, tmp_path):
    from anticipatory_rl.agents.restaurant.dqn import build_parser, train

    config_abs = Path("configs/restaurant/toy_level_3.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "anticipatory_rl.agents.restaurant.dqn.select_device", lambda: torch.device("cpu")
    )

    reset_count = 0
    original_reset = RestaurantSymbolicEnv.reset

    def counting_reset(self, *, seed=None, options=None):
        nonlocal reset_count
        reset_count += 1
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(RestaurantSymbolicEnv, "reset", counting_reset)

    args = build_parser().parse_args([
        "--total-steps", "40",
        "--tasks-per-episode", "200",
        "--max-steps-per-task", "4",
        "--config-path", str(config_abs),
        "--run-label", "_test_timeout_integration",
    ])
    train(args)

    # baseline = 1: just the initial reset (task library removed from training).
    # With the fix, per-task timeouts do NOT call env.reset (world persists across tasks).
    # With the bug, this would be ~baseline + 10 (one reset per timeout).
    assert reset_count == 1, f"expected baseline 1 reset, got {reset_count}"


def _random_valid_action(env: RestaurantSymbolicEnv, masks):
    at = random_valid_index(masks["valid_action_type_mask"])
    o1 = random_valid_index(masks["valid_object1_mask"][at])
    loc = random_valid_index(masks["valid_location_mask"][at])
    o2 = random_valid_index(masks["valid_object2_mask"][at, o1])
    return {"action_type": at, "object1": o1, "location": loc, "object2": o2}


def test_truncated_and_auto_success_mutually_exclusive(env: RestaurantSymbolicEnv):
    env.reset(seed=0)
    info = env._info(success=False)
    for _ in range(500):
        masks = extract_masks(info)
        action = _random_valid_action(env, masks)
        _, _, _, truncated, info = env.step(action)
        auto_success = bool(info.get("auto_success", False))
        assert not (truncated and auto_success), "truncated and auto_success both True on same step"


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
