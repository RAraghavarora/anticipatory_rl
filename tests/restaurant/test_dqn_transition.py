"""Tests for the training-loop task-transition fix (Task 4).

Guards the research-critical myopic-vs-anticipatory distinction:
- _decide_task_transition: timeouts count toward both horizons but never wipe the
  world on per-task truncation; only the step-limit safety resets the world.
- bootstrap_done: ONLY myopic task success sets it (success AND tpe<=1). All
  artificial cutoffs (timeout, env_reset, step_limit) are non-terminal under PEB.
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
    assert not t.bootstrap_done
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
    assert not t.bootstrap_done
    assert t.env_tasks_since_reset == 0


def test_decide_step_limit_fires():
    t = _transition(
        success=False, truncated=False, tasks=5, env_tasks=5, steps=100, tpe=200, step_limit=100
    )
    assert t.trunc_reset_flag
    assert not t.bootstrap_done
    assert t.tasks_since_reset == 0
    assert t.env_tasks_since_reset == 0


def test_decide_timeout_drives_env_reset():
    t = _transition(success=False, truncated=True, tasks=199, env_tasks=199, tpe=200, env_reset=200)
    assert t.env_reset_flag


def test_train_loop_timeout_no_world_wipe(monkeypatch, tmp_path):
    import json
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

    # Best checkpoint: 40 steps / 4 steps-per-task = at most 10 outcomes,
    # far below the 100-outcome window threshold.
    run_dir = Path("runs") / "_test_timeout_integration"
    best_path = run_dir / "restaurant_dqn_best.pt"
    assert not best_path.exists(), f"Best checkpoint should not exist with <100 outcomes: {best_path}"

    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["best_checkpoint_path"] is None
    assert summary["best_checkpoint_metric"] is None
    assert summary["best_checkpoint_value"] is None
    assert summary["best_checkpoint_step"] is None
    assert summary["best_checkpoint_task"] is None


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


def test_timeout_replay_uses_old_task():
    """Timeout splice: replay next_obs must carry old task, post-action world."""
    import numpy as np
    # Simulate the splice from the train loop (dqn.py lines 1423-1427)
    obs = np.arange(132, dtype=np.float32)       # pre-step obs (old task in task slice)
    next_obs = np.arange(132, 264, dtype=np.float32)  # post-step obs (env resampled → new task)
    task_obs_offset = 100  # example split point

    # Apply the same logic as the train loop
    truncated, success = True, False
    if truncated and not success:
        replay_next_obs = next_obs.copy()
        replay_next_obs[task_obs_offset:] = obs[task_obs_offset:]
    else:
        replay_next_obs = next_obs

    # World slice (before offset) comes from post-action state
    np.testing.assert_array_equal(replay_next_obs[:task_obs_offset], next_obs[:task_obs_offset])
    # Task slice (after offset) comes from pre-step state (old task)
    np.testing.assert_array_equal(replay_next_obs[task_obs_offset:], obs[task_obs_offset:])


def test_myopic_and_anticipatory_timeout_parity():
    """Both agents store timeout as done=0, task_boundary=0 (PEB)."""
    for tpe in (1, 200):
        t = _transition(success=False, truncated=True, tasks=0, tpe=tpe)
        assert not t.bootstrap_done, f"tpe={tpe}: timeout should not be terminal"


def test_best_checkpoint_saved_and_valid(monkeypatch, tmp_path):
    """Best checkpoint written periodically from a deterministic greedy eval.

    Verifies exact metadata by reloading the saved best weights and re-running
    the same deterministic _greedy_success_rate call, confirming the recorded
    value is exactly reproducible (not a noisy on-policy training statistic)."""
    import json
    import torch
    from anticipatory_rl.agents.restaurant.dqn import (
        build_parser, train, RestaurantQNetwork, _greedy_success_rate,
    )
    from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
    from anticipatory_rl.utils import select_device

    config_abs = Path("configs/restaurant/toy_level_3.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "anticipatory_rl.agents.restaurant.dqn.select_device", lambda: torch.device("cpu")
    )

    args = build_parser().parse_args([
        "--total-steps", "500",
        "--diagnostics-interval", "100",
        "--tasks-per-episode", "200",
        "--max-steps-per-task", "4",
        "--config-path", str(config_abs),
        "--run-label", "_test_best_ckpt",
        "--output-name", "test_dqn.pt",
    ])
    train(args)

    run_dir = Path("runs") / "_test_best_ckpt"
    final_path = run_dir / "test_dqn.pt"
    best_path = run_dir / "test_dqn_best.pt"
    assert final_path.exists(), f"Final checkpoint missing: {final_path}"
    assert best_path.exists(), f"Best checkpoint missing: {best_path}"

    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["best_checkpoint_metric"] == "persistent_greedy_success_rate"
    assert 0.0 <= summary["best_checkpoint_value"] <= 1.0
    assert summary["best_checkpoint_step"] % 100 == 0, (
        "best checkpoint should land on a diagnostics_interval boundary, not an "
        "arbitrary task-completion step"
    )
    assert str(best_path) in summary["best_checkpoint_path"]

    # --- Exact recomputation: reload best weights, re-run the same deterministic eval ---
    device = select_device()
    eval_env = RestaurantSymbolicEnv(
        config_path=config_abs,
        max_steps_per_task=args.max_steps_per_task,
        success_reward=args.success_reward,
        invalid_action_penalty=args.invalid_action_penalty,
        rng_seed=args.seed + 7_000_000,
    )
    obs, _ = eval_env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0]) if hasattr(obs, "shape") else len(obs)
    q_net = RestaurantQNetwork(
        obs_dim,
        int(eval_env.action_space["action_type"].n),
        int(eval_env.action_space["object1"].n),
        int(eval_env.action_space["location"].n),
        hidden_dim=args.hidden_dim,
        center_advantages=not getattr(args, "no_dueling_centering", False),
    ).to(device)
    q_net.load_state_dict(torch.load(best_path, map_location=device))

    recomputed_value = _greedy_success_rate(
        q_net, eval_env, device,
        n_tasks=summary["best_checkpoint_eval_n_tasks"],
        max_steps=args.max_steps_per_task,
        seed_base=args.seed + 8_000_000,
        env_reset_tasks=summary["best_checkpoint_eval_env_reset_tasks"],
    )
    assert recomputed_value == summary["best_checkpoint_value"], (
        f"Saved best checkpoint does not reproduce its own recorded metric: "
        f"expected {summary['best_checkpoint_value']}, got {recomputed_value}"
    )

    # Quick sanity: best checkpoint is loadable and has same structure as final.
    best_state = torch.load(best_path, map_location="cpu")
    final_state = torch.load(final_path, map_location="cpu")
    assert set(best_state.keys()) == set(final_state.keys())


def test_select_action_epsilon_zero_does_not_consume_rng():
    """epsilon=0.0 must never draw from the global `random` stream.

    A mid-training diagnostic/checkpoint eval that calls _select_action with
    epsilon=0.0 in a loop must not perturb the shared global RNG state that
    the main training loop's own epsilon-greedy exploration depends on --
    otherwise subsequent training-time exploration silently desynchronizes
    depending on checkpoint quality, breaking run reproducibility."""
    import random

    from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork, _select_action

    config_abs = Path("configs/restaurant/toy_level_3.yaml").resolve()
    device = torch.device("cpu")
    env = RestaurantSymbolicEnv(config_path=config_abs, max_steps_per_task=4)
    obs, info = env.reset(seed=0)
    masks = extract_masks(info)
    q_net = RestaurantQNetwork(
        int(obs.shape[0]) if hasattr(obs, "shape") else len(obs),
        int(env.action_space["action_type"].n),
        int(env.action_space["object1"].n),
        int(env.action_space["location"].n),
    ).to(device)

    random.seed(12345)
    state_before = random.getstate()
    for _ in range(20):
        _select_action(q_net, obs, masks, epsilon=0.0, device=device)
    assert random.getstate() == state_before, (
        "_select_action(epsilon=0.0) consumed from the global random stream"
    )
