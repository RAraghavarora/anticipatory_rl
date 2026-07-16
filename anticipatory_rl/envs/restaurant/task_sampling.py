"""Shared utilities for restaurant task sampling and planning."""

from __future__ import annotations

from typing import Any, Dict, List

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask


def sample_task(
    env: RestaurantSymbolicEnv,
    *,
    uniform_task_type_prob: float = 0.0,
) -> RestaurantTask:
    """Sample a restaurant task from the environment's task distribution.

    Paired draws (task_type, target_location, target_kind, pick_place.target_location)
    come from ``env._task_rng`` so agents with different action streams see identical
    task sequences. ``pick_place.object_name`` is state-conditioned and drawn from
    ``env._rng`` (variable retry count must not desync ``_task_rng``).

    Args:
        env: The restaurant environment.
        uniform_task_type_prob: Probability of sampling task type uniformly instead of
            using the weighted distribution. Default 0.0 (always weighted).

    Returns:
        A sampled RestaurantTask.
    """
    trng = env._task_rng
    # ponytail: trng.random() always drawn (even when uniform_task_type_prob=0.0) and pick_place
    # object_name kept on env._rng so task_rng consumes a fixed 3 draws/task regardless of world state.
    if trng.random() < uniform_task_type_prob:
        ttype = trng.choice(list(env.task_types))
    else:
        ttype = env._weighted_choice(env.task_distribution, env.task_types, rng=trng)

    if ttype in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
        target_location = env._weighted_choice(env.service_location_distribution, env.service_locations, rng=trng)
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None)

    if ttype == "pick_place":
        for _ in range(50):
            object_name = env._rng.choice(list(env.object_names))
            obj = env.state.objects[object_name]
            if obj.kind in {"water", "coffeegrinds"} or obj.contained_in is not None:
                continue
            break
        target_location = trng.choice(list(env.locations))
        return RestaurantTask(task_type=ttype, target_location=target_location, target_kind=None, object_name=object_name)

    target_kind = env._weighted_choice(env.wash_kind_distribution, env.object_kinds, rng=trng)
    return RestaurantTask(task_type=ttype, target_location=None, target_kind=target_kind)


def generate_task_library(
    env: RestaurantSymbolicEnv,
    seed: int,
    n_tasks: int,
) -> List[Dict[str, Any]]:
    """Pre-generate a fixed task sequence from a seeded RNG.

    Resets the env (seeding env._rng and env._task_rng) then samples n_tasks tasks
    via sample_task(env). Returns dicts with keys task_type, target_location,
    target_kind, object_name — the shape env._parse_task_library consumes.

    Test-only utility after the task_rng refactor (train/infer use online task_rng).
    """
    env.reset(seed=seed)
    library: List[Dict[str, Any]] = []
    for _ in range(n_tasks):
        task = sample_task(env)
        library.append({
            "task_type": task.task_type,
            "target_location": task.target_location,
            "target_kind": task.target_kind,
            "object_name": task.object_name,
        })
    return library
