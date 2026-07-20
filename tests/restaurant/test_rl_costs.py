"""Regression tests for RL cost loading and parity."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anticipatory_rl.agents.restaurant.dqn import build_parser
from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import parse_args
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.pddl_domain import PDDL_ACTION_COSTS


CONFIG_PATH = Path("configs/restaurant/toy_level_3.yaml")


@pytest.fixture(scope="module")
def costs() -> dict[str, float]:
    with open("configs/restaurant/rl_costs.yaml") as f:
        return yaml.safe_load(f)


def test_rl_costs_match_yaml(costs):
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
    assert env.travel_cost_scale == costs["travel_scale"]
    assert env.pick_cost == costs["pick"]
    assert env.place_cost == costs["place"]
    assert env.wash_cost == costs["wash"]
    assert env.fill_cost == costs["fill"]
    assert env.brew_cost == costs["make_coffee"]
    assert env.fruit_cost == costs["make_fruit_bowl"]
    assert env.spread_cost == costs["apply_spread"]
    assert env.pour_cost == costs["pour"]
    assert env.refill_cost == costs["refill_water"]
    assert env.drain_cost == costs["drain"]


def test_pddl_cost_ratios():
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
    assert env.pick_cost / PDDL_ACTION_COSTS["pick"] == pytest.approx(0.01)
    assert env.place_cost / PDDL_ACTION_COSTS["place"] == pytest.approx(0.01)
    assert env.wash_cost / PDDL_ACTION_COSTS["wash"] == pytest.approx(0.01)
    assert env.fill_cost / PDDL_ACTION_COSTS["fill"] == pytest.approx(0.01)
    assert env.brew_cost / PDDL_ACTION_COSTS["make-coffee"] == pytest.approx(0.01)
    assert env.fruit_cost / PDDL_ACTION_COSTS["make-fruit-bowl"] == pytest.approx(0.01)
    assert env.spread_cost / PDDL_ACTION_COSTS["apply-spread"] == pytest.approx(0.01)
    assert env.pour_cost / PDDL_ACTION_COSTS["pour"] == pytest.approx(0.01)
    assert env.refill_cost / PDDL_ACTION_COSTS["refill_water"] == pytest.approx(0.01)
    assert env.drain_cost / PDDL_ACTION_COSTS["drain"] == pytest.approx(0.01)


def test_move_cost_normalized():
    env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
    assert env._travel_cost("countertop", "fountain") == pytest.approx(9 * 0.25)
    assert env._travel_cost("countertop", "coffeemachine") == pytest.approx(1 * 0.25)


def test_train_infer_cost_parity(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"])
    train_args = build_parser().parse_args()
    train_env = RestaurantSymbolicEnv(
        config_path=train_args.config_path,
        max_steps_per_task=train_args.max_steps_per_task,
        success_reward=train_args.success_reward,
        invalid_action_penalty=train_args.invalid_action_penalty,
        rng_seed=train_args.seed,
    )

    monkeypatch.setattr("sys.argv", ["infer", "--state-dict", "dummy.pt"])
    infer_args = parse_args()
    infer_env = RestaurantSymbolicEnv(
        config_path=infer_args.config_path,
        max_steps_per_task=infer_args.max_task_steps,
        success_reward=infer_args.success_reward,
        invalid_action_penalty=infer_args.invalid_action_penalty,
        rng_seed=infer_args.seed,
    )

    attrs = [
        "travel_cost_scale", "pick_cost", "place_cost", "wash_cost", "fill_cost",
        "brew_cost", "fruit_cost", "spread_cost", "pour_cost", "refill_cost", "drain_cost",
    ]
    for attr in attrs:
        assert getattr(train_env, attr) == getattr(infer_env, attr), attr

