"""Core non-FD semantics of the finite-K clairvoyant sequence model."""

import pytest

from anticipatory_rl.envs.restaurant.env import RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    _current_task_satisfied,
    _split_sequence_plan,
    apply_planner_action,
    build_restaurant_sequence_problem_text,
)


def test_problem_encodes_task_chain_concrete_pick_place_and_water(env):
    tasks = [
        RestaurantTask("serve_water", target_location="servingtable"),
        RestaurantTask("pick_place", target_location="shelf", object_name="plate_0"),
    ]
    text = build_restaurant_sequence_problem_text(
        env, RestaurantPlannerState.from_env(env), tasks,
    )

    assert "(next-task t0 t1)" in text
    assert "(next-task t1 t-end)" in text
    assert "(task-object t1 plate_0)" in text
    assert "(is-at water fountain)" in text
    assert "(machine-water-available coffeemachine)" in text
    assert "(is-current-task t-end)" in text


def test_domain_gates_physical_actions_and_completion_is_zero_cost():
    text = Path("pddl/toy_restaurant_sequence_domain.pddl").read_text()
    physical = text.split("PHYSICAL ACTIONS", 1)[1].split("COMPLETION ACTIONS", 1)[0]
    completions = text.split("COMPLETION ACTIONS", 1)[1]

    for action in (
        "move", "pick", "place", "wash", "fill", "drain", "pour",
        "make-coffee", "make-fruit-bowl", "refill_water",
    ):
        action_text = physical.split(f"(:action {action}", 1)[1].split("(:action", 1)[0]
        assert "(not (current-task-satisfied))" in action_text
    assert completions.count("(:action complete-") == 6
    assert "increase (total-cost)" not in completions


def test_immediate_completion_gate_and_physical_cost(env):
    env.state.agent_location = "countertop"
    cup = env.state.objects["cup_0"]
    cup.location = "countertop"
    cup.filled_with = "water"
    state = RestaurantPlannerState.from_env(env)
    task = RestaurantTask("serve_water", target_location="servingtable")
    actions = [
        ("pick", ["cup_0", "countertop"]),
        ("move", ["countertop", "servingtable"]),
        ("place", ["cup_0", "servingtable"]),
        ("complete-serve-water", ["t0", "t-end", "cup_0"]),
    ]

    segment = _split_sequence_plan(env, state, [task], actions)[0]
    assert segment.paper2_cost == 400.0
    assert segment.completion_action[0] == "complete-serve-water"

    actions.insert(-1, ("move", ["servingtable", "countertop"]))
    with pytest.raises(ValueError, match="after task .* already satisfied"):
        _split_sequence_plan(env, state, [task], actions)


def test_deterministic_consumption_advances_persistent_state(env):
    for name in ("cup_0", "cup_1"):
        cup = env.state.objects[name]
        cup.location = "servingtable"
        cup.filled_with = "water"
        cup.dirty = False
    state = RestaurantPlannerState.from_env(env)
    tasks = [
        RestaurantTask("serve_water", target_location="servingtable"),
        RestaurantTask("serve_water", target_location="servingtable"),
    ]
    actions = [
        ("complete-serve-water", ["t0", "t1", "cup_0"]),
        ("complete-serve-water", ["t1", "t-end", "cup_1"]),
    ]

    segments = _split_sequence_plan(env, state, tasks, actions)
    assert [segment.auto_success for segment in segments] == [True, True]

    actions[0] = ("complete-serve-water", ["t0", "t1", "cup_1"])
    with pytest.raises(ValueError, match="deterministic first-eligible"):
        _split_sequence_plan(env, state, tasks, actions)


def test_machine_water_consumption_and_restoration_match_env_order(env):
    state = RestaurantPlannerState.from_env(env)
    state.agent_location = "coffeemachine"
    cup = state.objects["cup_0"]
    cup.location = "coffeemachine"
    cup.dirty = False
    cup.filled_with = None

    apply_planner_action(state, ("make-coffee", ["cup_0", "coffeemachine"]))
    assert state.objects["water_machine"].location is None
    assert state.objects["water_fountain"].location == "fountain"

    cup.location = None
    state.holding = "cup_0"
    cup.filled_with = "water"
    apply_planner_action(state, ("pour", ["cup_0", "water", "coffeemachine"]))
    assert state.objects["water_machine"].location == "coffeemachine"
    assert cup.filled_with is None


def test_all_six_task_satisfaction_rules_match_active_env(env):
    state = RestaurantPlannerState.from_env(env)
    tasks = [
        RestaurantTask("serve_water", target_location="servingtable"),
        RestaurantTask("make_coffee", target_location="servingtable"),
        RestaurantTask("make_fruit_bowl", target_location="servingtable"),
        RestaurantTask("clear_containers", target_location="shelf"),
        RestaurantTask("wash_objects", target_kind="bowl"),
        RestaurantTask("pick_place", target_location="shelf", object_name="plate_0"),
    ]

    for task in tasks:
        env.task = task
        assert _current_task_satisfied(state, task, env) == env._task_already_satisfied()


def test_pick_place_requires_concrete_object_and_free_hand(env):
    env.state.objects["plate_0"].location = "shelf"
    state = RestaurantPlannerState.from_env(env)
    task = RestaurantTask("pick_place", target_location="shelf", object_name="plate_0")
    action = [("complete-pick-place", ["t0", "t-end"])]

    assert _split_sequence_plan(env, state, [task], action)[0].auto_success
    state.holding = "cup_0"
    with pytest.raises(ValueError, match="before current-task-satisfied"):
        _split_sequence_plan(env, state, [task], action)
