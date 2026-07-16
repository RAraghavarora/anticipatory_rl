"""Tests for env-planner state equivalence on toy_level_3."""
from __future__ import annotations

import pytest

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    RestaurantTask,
    task_goal_clauses,
    apply_planner_action,
)

LOCATIONS = ("countertop", "coffeemachine", "dishwasher", "shelf", "servingtable", "fountain")


def _planner_action_from_env_action(env, action_dict, *, src_location=None, pre_holding=None):
    action_type_name = ACTION_TYPES[action_dict["action_type"]]
    if action_type_name == "move":
        dst_name = env.locations[action_dict["location"]]
        return ("move", [src_location, dst_name])
    if action_type_name == "pick":
        obj_name = env.object_names[action_dict["object1"]]
        return ("pick", [obj_name])
    if action_type_name == "place":
        loc_name = env.locations[action_dict["location"]]
        return ("place", [pre_holding, loc_name])
    if action_type_name == "wash":
        obj_name = env.object_names[action_dict["object1"]]
        return ("wash", [obj_name])
    if action_type_name == "fill":
        obj_name = env.object_names[action_dict["object1"]]
        return ("fill", [obj_name])
    if action_type_name == "make_coffee":
        obj_name = env.object_names[action_dict["object1"]]
        return ("make-coffee", [obj_name, env.state.agent_location])
    if action_type_name == "pour":
        obj_name = env.object_names[action_dict["object1"]]
        return ("pour", [obj_name, env.state.agent_location])
    if action_type_name == "refill_water":
        obj_name = env.object_names[action_dict["object1"]]
        return ("refill_water", [obj_name])
    if action_type_name == "drain":
        obj_name = env.object_names[action_dict["object1"]]
        return ("drain", [obj_name])
    pytest.skip(f"No planner equivalent for {action_type_name}")


def _assert_states_equal(env_state, planner_state):
    assert env_state.agent_location == planner_state.agent_location
    assert env_state.holding == planner_state.holding
    assert env_state.bread_spread == planner_state.bread_spread
    assert set(env_state.objects.keys()) == set(planner_state.objects.keys())
    for name in env_state.objects:
        eo = env_state.objects[name]
        po = planner_state.objects[name]
        assert eo.location == po.location, f"{name} location"
        assert eo.dirty == po.dirty, f"{name} dirty"
        assert eo.filled_with == po.filled_with, f"{name} filled_with"
        assert eo.contained_in == po.contained_in, f"{name} contained_in"


def _build_env_action(env, action_type, raw_params):
    d = {
        "action_type": env.action_type_index[action_type],
        "object1": env.none_object_index,
        "location": env.none_location_index,
        "object2": env.none_object_index,
    }
    for k, v in raw_params.items():
        if k == "object1":
            d["object1"] = env.object_name_index[v]
        elif k == "location":
            d["location"] = env.location_index[v]
        elif k == "object2":
            d["object2"] = env.object_name_index[v]
    return d


def test_canonical_locations_match(env):
    assert tuple(env.locations) == LOCATIONS


def test_make_coffee_rejects_jar(env, planner_state):
    env.state.agent_location = "coffeemachine"
    env.state.objects["jar_0"].location = "coffeemachine"

    action_spec = {
        "action_type": "make_coffee",
        "object1_name": "jar_0",
        "location_name": None,
        "object2_name": None,
    }
    assert not env._is_action_valid(action_spec)

    for task_type in ("serve_water", "make_coffee"):
        clauses = task_goal_clauses(
            planner_state,
            RestaurantTask(task_type=task_type, target_location="servingtable"),
            service_locations=env.service_locations,
            wash_ready_locations=env.wash_ready_locations,
        )
        for clause in clauses:
            assert "jar_0" not in clause


def test_refill_water_from_jar(env):
    env.state.agent_location = "shelf"
    env.state.holding = "cup_0"
    env.state.objects["cup_0"].location = None
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None
    env.state.objects["jar_0"].location = "shelf"
    env.state.objects["jar_0"].filled_with = "water"

    ps = RestaurantPlannerState.from_env(env)

    action = _build_env_action(env, "refill_water", {"object1": "cup_0", "object2": "jar_0"})
    env.step(action)

    assert env.state.objects["cup_0"].filled_with == "water"
    assert env.state.objects["jar_0"].filled_with == "water"

    apply_planner_action(ps, ("refill_water", ["cup_0"]))
    assert ps.objects["cup_0"].filled_with == "water"
    assert ps.objects["jar_0"].filled_with == "water"

    _assert_states_equal(env.state, ps)


def test_machine_water_consumed_on_make_coffee(env):
    env.state.agent_location = "coffeemachine"
    env.state.objects["cup_0"].location = "coffeemachine"
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None

    ps = RestaurantPlannerState.from_env(env)

    action = _build_env_action(env, "make_coffee", {"object1": "cup_0"})
    env.step(action)

    assert env.state.objects["water_machine"].location is None

    apply_planner_action(ps, ("make-coffee", ["cup_0", "coffeemachine"]))
    assert ps.objects["water_machine"].location is None

    _assert_states_equal(env.state, ps)


def test_machine_water_restored_on_pour(env):
    env.state.agent_location = "coffeemachine"
    env.state.holding = "cup_0"
    env.state.objects["cup_0"].location = None
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = "water"
    env.state.objects["water_machine"].location = None

    ps = RestaurantPlannerState.from_env(env)

    action = _build_env_action(env, "pour", {"object1": "cup_0"})
    env.step(action)

    assert env.state.objects["water_machine"].location == "coffeemachine"
    assert env.state.objects["cup_0"].filled_with is None

    apply_planner_action(ps, ("pour", ["cup_0", "coffeemachine"]))

    _assert_states_equal(env.state, ps)
    assert ps.objects["water_machine"].location == "coffeemachine"
    assert ps.objects["cup_0"].filled_with is None


def test_pour_validity(env):
    action_spec = {
        "action_type": "pour",
        "object1_name": "cup_0",
        "location_name": None,
        "object2_name": None,
    }

    env.state.agent_location = "fountain"
    env.state.holding = "cup_0"
    env.state.objects["cup_0"].location = None
    env.state.objects["cup_0"].filled_with = "water"
    assert not env._is_action_valid(action_spec)

    env.state.agent_location = "coffeemachine"
    env.state.objects["cup_0"].filled_with = None
    assert not env._is_action_valid(action_spec)

    # An empty held cup at the coffeemachine has nothing to pour -> invalid.
    ps = RestaurantPlannerState.from_env(env)
    ps_before = ps.copy()
    apply_planner_action(ps, ("pour", ["cup_0", "coffeemachine"]))
    assert ps.objects["cup_0"] == ps_before.objects["cup_0"]
    assert ps.objects["water_machine"] == ps_before.objects["water_machine"]
    assert ps.agent_location == ps_before.agent_location
    assert ps.holding == ps_before.holding


def test_fountain_water_permanent(env):
    assert env.state.objects["water_fountain"].location == "fountain"

    env.state.agent_location = "coffeemachine"
    env.state.objects["cup_0"].location = "coffeemachine"
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None
    env.step(_build_env_action(env, "make_coffee", {"object1": "cup_0"}))
    assert env.state.objects["water_fountain"].location == "fountain"
    ps = RestaurantPlannerState.from_env(env)
    assert ps.objects["water_fountain"].location == "fountain"

    env.state.objects["cup_1"].location = "coffeemachine"
    env.state.objects["cup_1"].filled_with = "water"
    env.state.objects["cup_1"].dirty = False
    env.state.holding = "cup_1"
    env.step(_build_env_action(env, "pour", {"object1": "cup_1"}))
    assert env.state.objects["water_fountain"].location == "fountain"
    ps = RestaurantPlannerState.from_env(env)
    assert ps.objects["water_fountain"].location == "fountain"

    env.state.agent_location = "shelf"
    env.state.holding = "cup_0"
    env.state.objects["cup_0"].location = None
    env.state.objects["cup_0"].filled_with = None
    env.state.objects["jar_0"].location = "shelf"
    env.state.objects["jar_0"].filled_with = "water"
    env.step(_build_env_action(env, "refill_water", {"object1": "cup_0", "object2": "jar_0"}))
    assert env.state.objects["water_fountain"].location == "fountain"
    ps = RestaurantPlannerState.from_env(env)
    assert ps.objects["water_fountain"].location == "fountain"


def test_water_not_pickable(env):
    assert not env._is_pickable_kind("water")

    info = env._info(success=False)
    pick_idx = env.action_type_index["pick"]
    pick_obj_mask = info["valid_object1_mask"][pick_idx]
    for name in ("water_fountain", "water_machine"):
        assert pick_obj_mask[env.object_name_index[name]] == 0.0


SEQUENCES = [
    ("move", [("move", {"location": "coffeemachine"})]),
    ("pick_place", [
        ("pick", {"object1": "cup_0"}),
        ("place", {"location": "countertop"}),
    ]),
    ("pick_fill_place", [
        ("pick", {"object1": "cup_0"}),
        ("move", {"location": "fountain"}),
        ("fill", {"object1": "cup_0"}),
        ("move", {"location": "countertop"}),
        ("place", {"location": "countertop"}),
    ]),
    ("make_coffee", [
        ("pick", {"object1": "cup_0"}),
        ("move", {"location": "coffeemachine"}),
        ("place", {"location": "coffeemachine"}),
        ("make_coffee", {"object1": "cup_0"}),
    ]),
    ("refill_water", [
        ("refill_water", {"object1": "cup_0", "object2": "jar_0"}),
    ]),
    ("wash", [
        ("wash", {"object1": "cup_0"}),
    ]),
]


@pytest.mark.parametrize("seq_name,raw_steps", SEQUENCES, ids=[s[0] for s in SEQUENCES])
def test_apply_planner_action_mirrors_env(env, seq_name, raw_steps):
    env.set_task("serve_water", target_location="servingtable")

    if seq_name == "refill_water":
        env.state.agent_location = "shelf"
        env.state.holding = "cup_0"
        env.state.objects["cup_0"].location = None
        env.state.objects["cup_0"].dirty = False
        env.state.objects["cup_0"].filled_with = None
        env.state.objects["jar_0"].location = "shelf"
        env.state.objects["jar_0"].filled_with = "water"

    if seq_name == "wash":
        env.state.agent_location = "dishwasher"
        env.state.objects["cup_0"].location = "dishwasher"
        env.state.objects["cup_0"].dirty = True
        env.state.objects["cup_0"].filled_with = "water"

    ps = RestaurantPlannerState.from_env(env)

    for raw_type, raw_params in raw_steps:
        src_location = env.state.agent_location
        pre_holding = env.state.holding

        env_action = _build_env_action(env, raw_type, raw_params)
        planner_action = _planner_action_from_env_action(
            env, env_action, src_location=src_location, pre_holding=pre_holding
        )

        env.step(env_action)
        apply_planner_action(ps, planner_action)

        _assert_states_equal(env.state, ps)

    if seq_name == "wash":
        assert env.state.objects["cup_0"].dirty is False
        assert env.state.objects["cup_0"].filled_with == "water"
        assert ps.objects["cup_0"].dirty is False
        assert ps.objects["cup_0"].filled_with == "water"


def test_task_goal_clauses(env, planner_state):
    sl = env.service_locations
    wl = env.wash_ready_locations

    sw = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="serve_water", target_location="servingtable"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(sw, list)
    assert len(sw) == 1
    assert "(or" in sw[0]
    assert "filled-with water" in sw[0]

    mc = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="make_coffee", target_location="servingtable"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(mc, list)
    assert len(mc) == 1
    assert "(or" in mc[0]
    assert "filled-with coffee" in mc[0]

    mfb = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="make_fruit_bowl", target_location="servingtable"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(mfb, list)
    assert len(mfb) == 1
    assert "(or" in mfb[0]

    cc = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="clear_containers", target_location="servingtable"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(cc, list)
    assert all(clause.startswith("(not") for clause in cc)

    wo = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="wash_objects", target_kind="cup"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(wo, list)
    assert len(wo) == 1
    assert "(or" in wo[0]

    pp = task_goal_clauses(
        planner_state,
        RestaurantTask(task_type="pick_place", object_name="cup_0", target_location="countertop"),
        service_locations=sl,
        wash_ready_locations=wl,
    )
    assert isinstance(pp, list)
    assert len(pp) == 1
    assert "cup_0" in pp[0]
    assert "hand-is-free" in pp[0]
