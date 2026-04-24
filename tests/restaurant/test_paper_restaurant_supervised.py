from __future__ import annotations

import random

from restaurant.paper_restaurant.candidates import expand_goal_candidates
from restaurant.paper_restaurant.estimator import OracleFutureCostEstimator
from restaurant.paper_restaurant.planner import FastDownwardRestaurantPlanner
from restaurant.paper_restaurant.reproduce_restaurant_supervised import AnticipatoryExperiment
from restaurant.paper_restaurant.world import (
    PaperRestaurantTask,
    RestaurantObjectState,
    RestaurantTaskLibrary,
    RestaurantWorldConfig,
    RestaurantWorldGenerator,
    RestaurantWorldState,
)


def test_world_generation_is_deterministic() -> None:
    config_a = RestaurantWorldConfig.sample(random.Random(7))
    config_b = RestaurantWorldConfig.sample(random.Random(7))
    assert tuple(sorted(config_a.containers)) == tuple(sorted(config_b.containers))
    assert config_a.container("pass_counter").coord == config_b.container("pass_counter").coord


def test_clear_candidate_expansion_includes_baseline_and_no_duplicates() -> None:
    rng = random.Random(0)
    config = RestaurantWorldConfig.sample(rng)
    generator = RestaurantWorldGenerator(config)
    state = RestaurantWorldState(
        robot_location="pass_counter",
        holding=None,
        objects={"cup_short": RestaurantObjectState("cup_short", "cup", "cup", "table_left", True, "water")},
    )
    for name, spec in config.object_specs.items():
        if name not in state.objects:
            state.objects[name] = RestaurantObjectState(name, spec.kind, spec.category, "pantry_shelf", False, "empty")
    planner = FastDownwardRestaurantPlanner(config)
    task = PaperRestaurantTask("clear_containers", target_location="table_left")
    expanded = expand_goal_candidates(
        generator,
        state,
        task,
        planner.default_goal_candidates(state, task),
        candidate_goal_limit=24,
    )
    assert expanded
    assert expanded[0].note == "baseline"
    signatures = [candidate.signature() for candidate in expanded]
    assert len(signatures) == len(set(signatures))


def test_service_counterfactual_prefers_cup_for_future_coffee() -> None:
    rng = random.Random(0)
    config = RestaurantWorldConfig.sample(rng)
    state = RestaurantWorldState(robot_location="prep_counter", holding=None, objects={})
    for name, spec in config.object_specs.items():
        location = "pantry_shelf"
        if name in {"mug_red", "cup_short"}:
            location = "prep_counter"
        state.objects[name] = RestaurantObjectState(name, spec.kind, spec.category, location, False, "empty")

    planner = FastDownwardRestaurantPlanner(config)
    library = RestaurantTaskLibrary(
        tasks=[PaperRestaurantTask("make_coffee", target_location="table_left")],
        weights={"make_coffee:table_left": 1.0},
    )
    estimator = OracleFutureCostEstimator(planner, future_task_sample=None, estimator_seed=0)
    experiment = AnticipatoryExperiment(
        config,
        planner,
        estimator=estimator,
        candidate_goal_limit=24,
    )
    current_task = PaperRestaurantTask("serve_water", target_location="table_left")
    myopic = experiment.solve_myopic(state, current_task)
    anticipatory = experiment.solve_anticipatory(state, current_task, library)

    assert myopic.candidate.bound_object == "mug_red"
    assert anticipatory.candidate.bound_object == "cup_short"
    assert estimator.estimate(anticipatory.final_state, library) < estimator.estimate(myopic.final_state, library)
