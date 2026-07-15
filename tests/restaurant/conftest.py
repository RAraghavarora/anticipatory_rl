from __future__ import annotations

from pathlib import Path

import pytest

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import RestaurantPlannerState


@pytest.fixture(scope="session")
def fd_path() -> Path:
    p = Path("downward/fast-downward.py")
    assert p.exists(), f"Fast Downward not built at {p.resolve()}. Build it first."
    return p


@pytest.fixture(scope="session")
def domain_path() -> Path:
    return Path("pddl/toy_restaurant_domain.pddl")


@pytest.fixture(scope="function")
def env() -> RestaurantSymbolicEnv:
    e = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    e.reset(seed=0)
    return e


@pytest.fixture(scope="function")
def planner_state(env: RestaurantSymbolicEnv) -> RestaurantPlannerState:
    return RestaurantPlannerState.from_env(env)
