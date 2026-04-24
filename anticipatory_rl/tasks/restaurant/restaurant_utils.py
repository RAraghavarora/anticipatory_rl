"""Shared utilities for restaurant task sampling and planning."""

from __future__ import annotations

import random

from anticipatory_rl.envs.restaurant_symbolic_env import RestaurantSymbolicEnv, RestaurantTask


def sample_task(
    env: RestaurantSymbolicEnv,
    *,
    uniform_task_type_prob: float = 0.0,
) -> RestaurantTask:
    """Sample a restaurant task from the environment's task distribution.

    Args:
        env: The restaurant environment.
        uniform_task_type_prob: Probability of sampling task type uniformly instead of
            using the weighted distribution. Default 0.0 (always weighted).

    Returns:
        A sampled RestaurantTask.
    """
    if random.random() < uniform_task_type_prob:
        ttype = random.choice(list(env.task_types))
    else:
        ttype = env._weighted_choice(env.task_distribution, env.task_types)

    if ttype in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
        target_location = env._weighted_choice(env.service_location_distribution, env.service_locations)
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None)

    if ttype == "pick_place":
        object_name = random.choice(env.object_names)
        target_location = random.choice(env.locations)
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None, object_name=object_name)

    target_kind = env._weighted_choice(env.wash_kind_distribution, env.object_kinds)
    return RestaurantTask(task_type=ttype, target_location=None, target_kind=target_kind)
