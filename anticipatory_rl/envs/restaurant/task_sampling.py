"""Shared utilities for restaurant task sampling and planning."""

from __future__ import annotations

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask


def sample_task(
    env: RestaurantSymbolicEnv,
    *,
    uniform_task_type_prob: float = 0.0,
) -> RestaurantTask:
    """Sample a restaurant task from the environment's task distribution.

    All random draws (task_type, target_location, target_kind,
    pick_place.target_location, pick_place.object_name) come from
    ``env._task_rng`` so agents with different action streams see identical
    complete task sequences. ``pick_place.object_name`` is sampled from
    ``env.pick_place_object_distribution`` (a state-independent config) and
    is therefore fully paired across agents.

    Args:
        env: The restaurant environment.
        uniform_task_type_prob: Probability of sampling task type uniformly instead of
            using the weighted distribution. Default 0.0 (always weighted).

    Returns:
        A sampled RestaurantTask.
    """
    trng = env._task_rng
    if trng.random() < uniform_task_type_prob:
        ttype = trng.choice(list(env.task_types))
    else:
        ttype = env._weighted_choice(env.task_distribution, env.task_types, rng=trng)

    if ttype in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
        target_location = env._weighted_choice(env.service_location_distribution, env.service_locations, rng=trng)
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None)

    if ttype == "pick_place":
        object_name = env._weighted_choice(
            env.pick_place_object_distribution,
            tuple(env.pick_place_object_distribution),
            rng=trng,
        )
        target_location = trng.choice(list(env.locations))
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None, object_name=object_name)

    target_kind = env._weighted_choice(env.wash_kind_distribution, env.object_kinds, rng=trng)
    return RestaurantTask(task_type=ttype, target_location=None, target_kind=target_kind)
