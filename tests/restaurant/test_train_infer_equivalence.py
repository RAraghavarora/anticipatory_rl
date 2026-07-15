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
        travel_cost_scale=train_args.travel_cost_scale,
        pick_cost=train_args.pick_cost,
        place_cost=train_args.place_cost,
        wash_cost=train_args.wash_cost,
        fill_cost=train_args.fill_cost,
        brew_cost=train_args.brew_cost,
        fruit_cost=train_args.fruit_cost,
        rng_seed=train_args.seed,
    )
    infer_env = make_env(infer_args)

    assert train_env.max_steps_per_task == infer_env.max_steps_per_task
    assert train_env.success_reward == infer_env.success_reward
    assert train_env.invalid_action_penalty == infer_env.invalid_action_penalty
    assert train_env.travel_cost_scale == infer_env.travel_cost_scale
    assert train_env.pick_cost == infer_env.pick_cost
    assert train_env.place_cost == infer_env.place_cost
    assert train_env.wash_cost == infer_env.wash_cost
    assert train_env.fill_cost == infer_env.fill_cost
    assert train_env.brew_cost == infer_env.brew_cost
    assert train_env.fruit_cost == infer_env.fruit_cost
    assert list(train_env.locations) == list(infer_env.locations)
    assert list(train_env.object_names) == list(infer_env.object_names)


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
        ("travel_cost_scale", "travel_cost_scale"),
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

    # The invariant: when training anticipatory (tasks_per_episode > 1),
    # inference must NOT reset every task.
    assert infer_defaults["tasks_per_reset"] > 1, (
        "Inference resets every task by default; this suppresses anticipatory effects."
    )
