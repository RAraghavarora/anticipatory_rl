"""Verify training (dqn.py) and inference (restaurant_dqn_infer.py) are consistent.

Guards against the "accidentally myopic" class of bug: training with one credit
horizon / gamma / env config and evaluating with another.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES, RestaurantSymbolicEnv


def _build_q_net(env, seed=0):
    from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork

    obs_dim = env.observation_space.shape[0]
    q_net = RestaurantQNetwork(
        input_dim=obs_dim,
        action_type_dim=len(ACTION_TYPES),
        object_dim=env.num_objects + 1,
        location_dim=env.num_locations + 1,
        hidden_dim=64,
    )
    torch.manual_seed(seed)
    for p in q_net.parameters():
        p.data.uniform_(-0.1, 0.1)
    q_net.eval()
    return q_net


def test_greedy_action_selection_matches(env):
    """Training _select_action(epsilon=0) and inference _sample_structured_action(temperature=0)
    must return identical actions on the same state + masks + weights."""
    from anticipatory_rl.agents.restaurant.dqn import _select_action
    from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import _sample_structured_action
    from anticipatory_rl.utils import extract_masks

    device = torch.device("cpu")
    q_net = _build_q_net(env, seed=42)

    obs = env._obs()
    info = env._info(success=False)
    masks = extract_masks(info)

    train_action = _select_action(q_net, obs, masks, epsilon=0.0, device=device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    infer_action = _sample_structured_action(
        q_net, obs, info, temperature=0.0, generator=gen, device=device,
    )

    assert train_action == infer_action, (
        f"Training/inference greedy actions diverged:\n"
        f"  train: {train_action}\n  infer: {infer_action}"
    )


def test_env_construction_params_match(monkeypatch):
    """The env built in train() and make_env() must use the same cost/config params."""
    from anticipatory_rl.agents.restaurant.dqn import build_parser
    from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import make_env, parse_args

    train_args = build_parser().parse_args([
        "--config-path", "configs/restaurant/toy_level_3.yaml",
    ])
    monkeypatch.setattr("sys.argv", [
        "restaurant_dqn_infer.py",
        "--state-dict", "dummy.pt",
        "--config-path", "configs/restaurant/toy_level_3.yaml",
    ])
    infer_args = parse_args()

    train_env = RestaurantSymbolicEnv(
        config_path=train_args.config_path,
        max_steps_per_task=train_args.max_steps_per_task,
        success_reward=train_args.success_reward,
        invalid_action_penalty=train_args.invalid_action_penalty,
        rng_seed=train_args.seed,
    )
    infer_env = make_env(infer_args)

    assert train_env.max_steps_per_task == infer_env.max_steps_per_task
    assert train_env.success_reward == infer_env.success_reward
    assert train_env.invalid_action_penalty == infer_env.invalid_action_penalty
    assert list(train_env.locations) == list(infer_env.locations)
    assert list(train_env.object_names) == list(infer_env.object_names)

    # Costs are loaded from rl_costs.yaml, not constructor arguments.
    assert train_env.travel_cost_scale == 0.25
    assert train_env.pick_cost == 1.0
    assert train_env.place_cost == 1.0
    assert train_env.wash_cost == 2.0
    assert train_env.fill_cost == 10.0
    assert train_env.brew_cost == 0.5
    assert train_env.fruit_cost == 1.0
    assert train_env.spread_cost == train_env.fruit_cost
    assert train_env.pour_cost == 2.0
    assert train_env.refill_cost == 0.5
    assert train_env.drain_cost == 0.5


def test_default_hyperparams_consistent(monkeypatch):
    """Default gamma, success_reward, travel_cost_scale, max_steps, config must match
    between training and inference parsers."""
    from anticipatory_rl.agents.restaurant.dqn import build_parser
    from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import parse_args

    train_defaults = vars(build_parser().parse_args([]))
    monkeypatch.setattr("sys.argv", ["restaurant_dqn_infer.py", "--state-dict", "dummy.pt"])
    infer_defaults = vars(parse_args())

    pairs = [
        ("gamma", "gamma"),
        ("success_reward", "success_reward"),
        ("max_steps_per_task", "max_task_steps"),
    ]
    for train_key, infer_key in pairs:
        assert train_defaults[train_key] == infer_defaults[infer_key], (
            f"Default {train_key} mismatch: train={train_defaults[train_key]} "
            f"infer={infer_defaults[infer_key]}"
        )


def test_credit_horizon_documented(monkeypatch):
    """Guard against the 'accidentally myopic' bug.

    Training tasks_per_episode controls the credit horizon (1=myopic, >1=anticipatory).
    Inference tasks_per_reset controls world reset cadence.

    For anticipatory evaluation, inference tasks_per_reset must be > 1 to preserve
    the persistent world the agent was trained on. A mismatch here was the root
    cause of the invalid 'anticipatory advantage' claim on v2.2.
    """
    from anticipatory_rl.agents.restaurant.dqn import build_parser
    from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import parse_args

    train_defaults = vars(build_parser().parse_args([]))
    monkeypatch.setattr("sys.argv", ["restaurant_dqn_infer.py", "--state-dict", "dummy.pt"])
    infer_defaults = vars(parse_args())

    # Document the defaults explicitly.
    assert train_defaults["tasks_per_episode"] == 1, (
        "Training default tasks_per_episode changed; update this test and "
        "verify SLURM scripts still pass --tasks-per-episode explicitly."
    )
    assert infer_defaults["tasks_per_reset"] == 200, (
        "Inference default tasks_per_reset changed; anticipatory eval requires > 1."
    )
    assert train_defaults["env_reset_tasks"] == 200, (
        "env_reset_tasks default changed; paired myopic/anticipatory runs rely on a shared reset cadence."
    )

    # The invariant: when training anticipatory (tasks_per_episode > 1),
    # inference must NOT reset every task.
    assert infer_defaults["tasks_per_reset"] > 1, (
        "Inference resets every task by default; this suppresses anticipatory effects."
    )


def test_generate_task_library_deterministic():
    """generate_task_library must be a pure function of (env config, seed, n_tasks):
    two calls with the same seed produce identical task libraries."""
    from anticipatory_rl.envs.restaurant.task_sampling import generate_task_library

    env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    first = generate_task_library(env, seed=0, n_tasks=50)
    second = generate_task_library(env, seed=0, n_tasks=50)
    assert first == second, "generate_task_library is not deterministic across calls with the same seed"


def test_inference_pairing(tmp_path, monkeypatch):
    """run_compare must produce paired task sequences across two different-weight
    agents. End-to-end inference pairing: both agents reseed env._task_rng on the
    same reset cadence, so different action streams (different step counts per
    task) still yield identical task_type/target_location/target_kind sequences.
    pick_place.object_name is state-conditioned and not asserted."""
    import json

    from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
    from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import (
        make_env,
        parse_args,
        run_compare,
    )

    monkeypatch.setattr(
        "anticipatory_rl.agents.restaurant.restaurant_dqn_infer.select_device",
        lambda: torch.device("cpu"),
    )

    ant_path = tmp_path / "ant.pt"
    myo_path = tmp_path / "myo.pt"
    monkeypatch.setattr("sys.argv", [
        "restaurant_dqn_infer.py",
        "--anticipatory-weights", str(ant_path),
        "--myopic-weights", str(myo_path),
        "--num-tasks", "20",
        "--tasks-per-reset", "200",
        "--seed", "0",
        "--config-path", "configs/restaurant/toy_level_3.yaml",
        "--output-dir", str(tmp_path / "compare_out"),
    ])
    args = parse_args()

    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.asarray(obs).shape[0])

    for path, seed in [(ant_path, 42), (myo_path, 64)]:
        torch.manual_seed(seed)
        net = RestaurantQNetwork(
            input_dim=obs_dim,
            action_type_dim=len(ACTION_TYPES),
            object_dim=env.action_space["object1"].n,
            location_dim=env.action_space["location"].n,
            hidden_dim=args.hidden_dim,
        )
        torch.save(net.state_dict(), path)

    run_compare(args)

    with (tmp_path / "compare_out" / "comparison.json").open() as f:
        comparison = json.load(f)
    ant_tasks = comparison["anticipatory"]["tasks"]
    myo_tasks = comparison["myopic"]["tasks"]
    assert len(ant_tasks) == len(myo_tasks), (
        f"Task list length mismatch: ant={len(ant_tasks)} myo={len(myo_tasks)}"
    )
    n = len(ant_tasks)
    assert n > 0, "No tasks were evaluated"
    for i in range(n):
        assert ant_tasks[i]["task_type"] == myo_tasks[i]["task_type"], (
            f"Task {i} task_type mismatch: ant={ant_tasks[i]['task_type']} myo={myo_tasks[i]['task_type']}"
        )
        assert ant_tasks[i]["target_location"] == myo_tasks[i]["target_location"], (
            f"Task {i} target_location mismatch: ant={ant_tasks[i]['target_location']} "
            f"myo={myo_tasks[i]['target_location']}"
        )
        assert ant_tasks[i]["target_kind"] == myo_tasks[i]["target_kind"], (
            f"Task {i} target_kind mismatch: ant={ant_tasks[i]['target_kind']} "
            f"myo={myo_tasks[i]['target_kind']}"
        )
