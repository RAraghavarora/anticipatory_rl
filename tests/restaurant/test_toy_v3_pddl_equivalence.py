"""Tests for toy_level_3 PDDL-equivalent liquid semantics.

Covers the machine-water trap: make_coffee consumes water at the coffee machine,
and pour (held container -> current location) re-supplies it. Pour is location-gated
to the coffee machine and generic over the held liquid.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_planner_action,
    build_restaurant_problem_text,
    solve_restaurant_task_with_fd,
)


CONFIG = Path("configs/restaurant/toy_level_3.yaml")
DOMAIN_PATH = Path("pddl/toy_restaurant_domain.pddl")
PLANNER_PATH = Path("downward/fast-downward.py")


def _water_at(env: RestaurantSymbolicEnv, loc: str) -> bool:
    return any(o.kind == "water" and o.location == loc for o in env.state.objects.values())


def _prep_clean_cup_at_machine(env: RestaurantSymbolicEnv, cup: str) -> None:
    s = env.state
    s.agent_location = "coffeemachine"
    s.holding = None
    obj = s.objects[cup]
    obj.location = "coffeemachine"
    obj.dirty = False
    obj.filled_with = None
    obj.contained_in = None


class ToyV3EnvSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RestaurantSymbolicEnv(config_path=CONFIG)
        self.env.reset(seed=0)

    def test_locations_are_canonical_without_roles(self) -> None:
        self.assertEqual(
            set(self.env.locations),
            {"countertop", "coffeemachine", "dishwasher", "fountain", "servingtable", "shelf"},
        )
        # No role indirection: canonical names carry their own role.
        self.assertTrue(self.env._is_location("fountain", "fountain"))
        self.assertTrue(self.env._is_location("coffeemachine", "coffeemachine"))

    def test_jar_always_clean_at_reset(self) -> None:
        jar = self.env.state.objects["jar_0"]
        self.assertEqual(jar.kind, "jar")
        self.assertFalse(jar.dirty)

    def test_make_coffee_rejects_jar(self) -> None:
        s = self.env.state
        s.agent_location = "coffeemachine"
        s.objects["jar_0"].location = "coffeemachine"
        s.objects["jar_0"].dirty = False
        s.objects["jar_0"].filled_with = None
        self.assertFalse(
            self.env._is_action_valid(
                {"action_type": "make_coffee", "object1_name": "jar_0", "location_name": None, "object2_name": None}
            )
        )

    def test_refill_water_fills_held_cup_and_preserves_jar(self) -> None:
        s = self.env.state
        s.agent_location = "shelf"
        s.holding = "cup_0"
        s.objects["cup_0"].location = None
        s.objects["cup_0"].dirty = False
        s.objects["cup_0"].filled_with = None
        s.objects["jar_0"].location = "shelf"
        s.objects["jar_0"].filled_with = "water"
        _, ok = self.env._execute_action(
            {
                "action_type": "refill_water",
                "object1_name": "cup_0",
                "location_name": None,
                "object2_name": "jar_0",
            }
        )
        self.assertTrue(ok)
        self.assertEqual(s.objects["cup_0"].filled_with, "water")
        self.assertEqual(s.objects["jar_0"].filled_with, "water")
        self.assertEqual(s.objects["jar_0"].location, "shelf")

    def test_reset_seeds_machine_water_and_permanent_fountain(self) -> None:
        self.assertTrue(_water_at(self.env, "coffeemachine"))
        self.assertTrue(_water_at(self.env, "fountain"))

    def test_no_coffeegrinds_kind(self) -> None:
        self.assertNotIn("coffeegrinds", self.env.object_kinds)
        self.assertFalse(any(o.kind == "coffeegrinds" for o in self.env.state.objects.values()))

    def test_first_make_coffee_valid_without_pour(self) -> None:
        _prep_clean_cup_at_machine(self.env, "cup_0")
        self.assertTrue(
            self.env._is_action_valid(
                {"action_type": "make_coffee", "object1_name": "cup_0", "location_name": None, "object2_name": None}
            )
        )

    def test_make_coffee_consumes_machine_water(self) -> None:
        _prep_clean_cup_at_machine(self.env, "cup_0")
        self.assertTrue(_water_at(self.env, "coffeemachine"))
        _, ok = self.env._execute_action(
            {"action_type": "make_coffee", "object1_name": "cup_0", "location_name": None, "object2_name": None}
        )
        self.assertTrue(ok)
        self.assertEqual(self.env.state.objects["cup_0"].filled_with, "coffee")
        self.assertTrue(self.env.state.objects["cup_0"].dirty)
        self.assertFalse(_water_at(self.env, "coffeemachine"))

    def test_second_make_coffee_invalid_after_consumption(self) -> None:
        _prep_clean_cup_at_machine(self.env, "cup_0")
        self.env._execute_action(
            {"action_type": "make_coffee", "object1_name": "cup_0", "location_name": None, "object2_name": None}
        )
        # A fresh clean cup at the machine still cannot brew: no water remains.
        s = self.env.state
        s.objects["cup_1"].location = "coffeemachine"
        s.objects["cup_1"].dirty = False
        s.objects["cup_1"].filled_with = None
        self.assertFalse(
            self.env._is_action_valid(
                {"action_type": "make_coffee", "object1_name": "cup_1", "location_name": None, "object2_name": None}
            )
        )

    def test_pour_restores_machine_water_and_rearms_coffee(self) -> None:
        _prep_clean_cup_at_machine(self.env, "cup_0")
        self.env._execute_action(
            {"action_type": "make_coffee", "object1_name": "cup_0", "location_name": None, "object2_name": None}
        )
        self.assertFalse(_water_at(self.env, "coffeemachine"))
        # Carry a water-filled cup to the machine and pour.
        s = self.env.state
        s.holding = "cup_1"
        s.objects["cup_1"].location = None
        s.objects["cup_1"].filled_with = "water"
        self.assertTrue(
            self.env._is_action_valid(
                {"action_type": "pour", "object1_name": "cup_1", "location_name": None, "object2_name": None}
            )
        )
        self.env._execute_action(
            {"action_type": "pour", "object1_name": "cup_1", "location_name": None, "object2_name": None}
        )
        self.assertIsNone(s.objects["cup_1"].filled_with)
        self.assertTrue(_water_at(self.env, "coffeemachine"))

    def test_pour_invalid_off_machine(self) -> None:
        s = self.env.state
        s.agent_location = "fountain"
        s.holding = "cup_0"
        s.objects["cup_0"].location = None
        s.objects["cup_0"].filled_with = "water"
        self.assertFalse(
            self.env._is_action_valid(
                {"action_type": "pour", "object1_name": "cup_0", "location_name": None, "object2_name": None}
            )
        )

    def test_pour_generic_over_liquid(self) -> None:
        s = self.env.state
        s.agent_location = "coffeemachine"
        s.holding = "cup_0"
        s.objects["cup_0"].location = None
        s.objects["cup_0"].filled_with = "coffee"
        self.assertTrue(
            self.env._is_action_valid(
                {"action_type": "pour", "object1_name": "cup_0", "location_name": None, "object2_name": None}
            )
        )
        self.env._execute_action(
            {"action_type": "pour", "object1_name": "cup_0", "location_name": None, "object2_name": None}
        )
        self.assertIsNone(s.objects["cup_0"].filled_with)

    def test_fountain_water_never_depletes(self) -> None:
        # Filling a held cup at the fountain does not remove the fountain source.
        s = self.env.state
        s.agent_location = "fountain"
        s.holding = "cup_0"
        s.objects["cup_0"].location = None
        s.objects["cup_0"].dirty = False
        s.objects["cup_0"].filled_with = None
        self.env._execute_action(
            {"action_type": "fill", "object1_name": "cup_0", "location_name": None, "object2_name": None}
        )
        self.assertEqual(s.objects["cup_0"].filled_with, "water")
        self.assertTrue(_water_at(self.env, "fountain"))



    def test_water_objects_not_pickable(self) -> None:
        # Raw liquid sources must be moved via fill/pour only, not pick/place.
        s = self.env.state
        s.holding = None
        for loc in ("fountain", "coffeemachine"):
            s.agent_location = loc
            water_obj = next(o.name for o in s.objects.values() if o.kind == "water" and o.location == loc)
            self.assertFalse(
                self.env._is_action_valid(
                    {"action_type": "pick", "object1_name": water_obj, "location_name": None, "object2_name": None}
                ),
                f"should not be able to pick {water_obj} at {loc}",
            )

class ToyV3PlannerMirror(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RestaurantSymbolicEnv(config_path=CONFIG)
        self.env.reset(seed=0)
        self.state = RestaurantPlannerState.from_env(self.env)

    @staticmethod
    def _water_at(state: RestaurantPlannerState, loc: str) -> bool:
        return any(o.kind == "water" and o.location == loc for o in state.objects.values())

    def test_planner_make_coffee_consumes_water(self) -> None:
        st = self.state
        st.agent_location = "coffeemachine"
        st.objects["cup_0"].location = "coffeemachine"
        st.objects["cup_0"].dirty = False
        st.objects["cup_0"].filled_with = None
        self.assertTrue(self._water_at(st, "coffeemachine"))
        apply_planner_action(st, ("make-coffee", ["cup_0", "coffeemachine", "water_machine"]))
        self.assertEqual(st.objects["cup_0"].filled_with, "coffee")
        self.assertFalse(self._water_at(st, "coffeemachine"))

    def test_planner_pour_restores_water(self) -> None:
        st = self.state
        st.agent_location = "coffeemachine"
        # Consume first.
        st.objects["cup_0"].location = "coffeemachine"
        st.objects["cup_0"].dirty = False
        st.objects["cup_0"].filled_with = None
        apply_planner_action(st, ("make-coffee", ["cup_0", "coffeemachine", "water_machine"]))
        self.assertFalse(self._water_at(st, "coffeemachine"))
        # Pour a held water cup at the machine.
        st.holding = "cup_1"
        st.objects["cup_1"].location = None
        st.objects["cup_1"].filled_with = "water"
        apply_planner_action(st, ("pour", ["cup_1", "water", "coffeemachine"]))
        self.assertIsNone(st.objects["cup_1"].filled_with)
        self.assertTrue(self._water_at(st, "coffeemachine"))

    def test_planner_problem_emits_is_jar_and_known_costs(self) -> None:
        st = RestaurantPlannerState.from_env(self.env)
        problem_text = build_restaurant_problem_text(self.env, st, self.env.task)
        self.assertIn("(is-jar jar_0)", problem_text)
        self.assertIn("(= (known-cost countertop coffeemachine)", problem_text)
        # No adjacency facts when movement_costs is present.
        self.assertNotIn("(adjacent", problem_text)


@unittest.skipUnless(PLANNER_PATH.exists(), "Fast Downward not available")
class ToyV3PlannerDomainEquivalence(unittest.TestCase):
    """FD-domain checks that the planner cannot plan what the env cannot execute."""

    def setUp(self) -> None:
        self.env = RestaurantSymbolicEnv(config_path=CONFIG)

    def _solve_make_coffee(self, mutate) -> list[str]:
        self.env.reset(seed=0)
        self.env.set_task("make_coffee", target_location="coffeemachine", task_source="iid")
        st = RestaurantPlannerState.from_env(self.env)
        mutate(st)
        res = solve_restaurant_task_with_fd(
            self.env,
            st,
            self.env.task,
            domain_path=DOMAIN_PATH,
            planner_path=PLANNER_PATH,
            search="astar(ff())",
            timeout_s=30.0,
        )
        self.assertTrue(res.success, f"planner failed: {res.error}")
        return res.plan_actions

    def test_dry_machine_brew_uses_real_water_not_poured_coffee(self) -> None:
        # Machine dry, no coffee anywhere, clean empty cups -> a genuine brew is forced.
        # The re-supply must fetch real water (fill) and any pour must carry water, never
        # coffee. This guards against make-coffee binding a poured `(is-at coffee loc)` as
        # its water source, which the executable env would refuse.
        def mutate(st: RestaurantPlannerState) -> None:
            st.objects["water_machine"].location = None
            st.objects["cup_0"].location = "coffeemachine"
            st.objects["cup_0"].dirty = False
            st.objects["cup_0"].filled_with = None
            st.objects["cup_1"].location = "countertop"
            st.objects["cup_1"].dirty = False
            st.objects["cup_1"].filled_with = None

        plan = self._solve_make_coffee(mutate)
        names = [a[0] for a in plan]
        self.assertIn("make-coffee", names)
        for name, args in plan:
            if name == "pour":
                self.assertEqual(args[1], "water", "pour must carry water, not coffee")

    def test_dry_machine_make_coffee_uses_refill_water_from_jar(self) -> None:
        # Put an empty clean cup with the jar at the shelf; the machine is dry.
        # The planner should refill from the jar, move to the machine, pour, and brew.
        self.env.reset(seed=0)
        self.env.set_task("make_coffee", target_location="coffeemachine", task_source="iid")
        s = self.env.state
        s.objects["water_machine"].location = None
        s.objects["cup_0"].location = "shelf"
        s.objects["cup_0"].dirty = False
        s.objects["cup_0"].filled_with = None
        s.objects["jar_0"].location = "shelf"
        s.objects["jar_0"].filled_with = "water"
        st = RestaurantPlannerState.from_env(self.env)
        res = solve_restaurant_task_with_fd(
            self.env,
            st,
            self.env.task,
            domain_path=DOMAIN_PATH,
            planner_path=PLANNER_PATH,
            search="astar(blind())",
            timeout_s=30.0,
        )
        self.assertTrue(res.success, f"planner failed: {res.error}")
        names = [a[0] for a in res.plan_actions]
        self.assertIn("refill_water", names)
        self.assertIn("make-coffee", names)
        for name, args in res.plan_actions:
            if name == "pour":
                self.assertEqual(args[1], "water", "pour must carry water, not coffee")


if __name__ == "__main__":
    unittest.main()
