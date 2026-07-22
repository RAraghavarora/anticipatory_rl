"""Restaurant PDDL builder + Fast Downward planner runner."""

from __future__ import annotations

import copy
import itertools
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from anticipatory_rl.envs.restaurant.env import (
    RestaurantObjectState,
    RestaurantState,
    RestaurantSymbolicEnv,
    RestaurantTask,
    consume_delivery,
)
from anticipatory_rl.envs.restaurant.fd_runner import run_planner


# ---------------------------------------------------------------------------
# Constants for the sequence domain
# ---------------------------------------------------------------------------

# The abstract liquid constants declared in the sequence domain. These are the
# only symbols permitted in liquid/source positions of the validator and the
# only ones the PDDL `(is-liquid ?obj)` precondition accepts.
_ABSTRACT_LIQUID_CONSTANTS: frozenset[str] = frozenset({"water", "coffee"})

# Concrete object kinds that may appear in liquid/source positions (for the
# `fill` source arg). The sequence init does NOT emit `(is-at water_fountain ?)`
# in sequence mode, so concrete water objects are NOT valid fill sources; we
# keep the constant for type-checking but it always rejects concrete water
# objects as fill sources.
_ABSTRACT_FILL_SOURCES: frozenset[str] = frozenset({"water"})

# The ten physical actions the sequence domain recognises; everything else is
# rejected. Completion actions are handled separately and always cost zero.
_PHYSICAL_ACTIONS: Tuple[str, ...] = (
    "move",
    "pick",
    "place",
    "wash",
    "fill",
    "drain",
    "pour",
    "make-coffee",
    "make-fruit-bowl",
    "refill_water",
)

# The six zero-cost `complete-*` actions emitted by the sequence domain.
_COMPLETION_ACTIONS: Tuple[str, ...] = (
    "complete-serve-water",
    "complete-make-coffee",
    "complete-make-fruit-bowl",
    "complete-clear-containers",
    "complete-wash-objects",
    "complete-pick-place",
)

# Mapping from a RestaurantTask.task_type to the matching completion action.
_TASK_TYPE_TO_COMPLETION: Dict[str, str] = {
    "serve_water": "complete-serve-water",
    "make_coffee": "complete-make-coffee",
    "make_fruit_bowl": "complete-make-fruit-bowl",
    "clear_containers": "complete-clear-containers",
    "wash_objects": "complete-wash-objects",
    "pick_place": "complete-pick-place",
}

# Object-kind names that are *not* valid sequence-domain `kind` constants
# because they collide with the abstract `(= (constants water coffee - object))`
# declarations in `pddl/toy_restaurant_sequence_domain.pddl`. PDDL types are
# disjoint, so a symbol may appear in only one type declaration; we keep
# `water`/`coffee` as object constants and reject them as kind values.
_SEQUENCE_KIND_BLACKLIST: frozenset[str] = frozenset({"water", "coffee"})

# Strict arity for each physical action. Malformed actions are rejected
# (they no longer silently get zero cost).
_PHYSICAL_ACTION_ARITY: Dict[str, int] = {
    "move": 2,
    "pick": 2,
    "place": 2,
    "wash": 2,
    "fill": 3,
    "drain": 2,
    "pour": 3,
    "make-coffee": 2,
    "make-fruit-bowl": 4,
    "refill_water": 3,
}

# Strict arity for each completion action. Completion markers carry task_id
# arguments plus task-specific object/location witnesses.
_COMPLETION_ACTION_ARITY: Dict[str, int] = {
    "complete-serve-water": 3,        # cur, nxt, cup
    "complete-make-coffee": 3,        # cur, nxt, cup
    "complete-make-fruit-bowl": 5,    # cur, nxt, apple, bowl, loc
    "complete-clear-containers": 2,    # cur, nxt
    "complete-wash-objects": 2,       # cur, nxt
    "complete-pick-place": 2,         # cur, nxt
}

# FD's `; cost = N (unit cost)` header. Captured best-effort; absent in some
# builds, which is fine because the canonical cost comes from the strict
# physical-cost recomputation.
_FD_COST_HEADER_RE = re.compile(r";\s*cost\s*=\s*(\d+)")


@dataclass
class PlannerResult:
    success: bool
    plan_actions: List[Tuple[str, List[str]]]
    plan_cost: float
    solve_time_s: float
    error: str | None = None


@dataclass
class SequenceTaskSegment:
    """One task's slice of a sequence plan.

    The completion action is always present and is the segment's last action.
    `auto_success` is true iff the physical prefix is empty.
    """

    task: RestaurantTask
    physical_actions: List[Tuple[str, List[str]]]
    completion_action: Tuple[str, List[str]]
    paper2_cost: float
    auto_success: bool


@dataclass
class SequencePlannerResult:
    success: bool
    plan_actions: List[Tuple[str, List[str]]]
    task_segments: List[SequenceTaskSegment]
    physical_actions: List[Tuple[str, List[str]]]
    physical_cost: float
    raw_fd_cost: Optional[float]
    solve_time_s: float
    completion_count: int
    error: str | None = None
    # ``selected_search`` records the alias that produced the satisficing plan.
    # ``None`` when planning failed or no successful search was recorded.
    selected_search: Optional[str] = None


@dataclass
class RestaurantPlannerState:
    agent_location: str
    holding: str | None
    objects: Dict[str, RestaurantObjectState]
    bread_spread: str | None = None

    @classmethod
    def from_env(cls, env: RestaurantSymbolicEnv) -> "RestaurantPlannerState":
        return cls(
            agent_location=str(env.state.agent_location),
            holding=None if env.state.holding is None else str(env.state.holding),
            objects={k: copy.deepcopy(v) for k, v in env.state.objects.items()},
            bread_spread=None if env.state.bread_spread is None else str(env.state.bread_spread),
        )

    def copy(self) -> "RestaurantPlannerState":
        return RestaurantPlannerState(
            agent_location=self.agent_location,
            holding=self.holding,
            objects={k: copy.deepcopy(v) for k, v in self.objects.items()},
            bread_spread=self.bread_spread,
        )


def parse_sas_plan(plan_text: str) -> List[Tuple[str, List[str]]]:
    actions: List[Tuple[str, List[str]]] = []
    for raw in plan_text.splitlines():
        line = raw.strip().lower()
        if not line or line.startswith(";"):
            continue
        if not line.startswith("("):
            continue
        toks = line.strip("()").split()
        if not toks:
            continue
        actions.append((toks[0], toks[1:]))
    return actions


_PDDL_FIXED_COST_KEY: Dict[str, str] = {
    "pick": "pick",
    "place": "place",
    "wash": "wash",
    "fill": "fill",
    "make-coffee": "make_coffee",
    "make-fruit-bowl": "make_fruit_bowl",
    "drain": "drain",
    "pour": "pour",
    "refill_water": "refill_water",
}


def _line_cost_from_action(action_name: str, args: Sequence[str], env: RestaurantSymbolicEnv) -> float:
    if action_name == "move":
        if len(args) < 2:
            raise ValueError(
                f"move requires 2 location arguments, got {len(args)}."
            )
        src, dst = args[0], args[1]
        return float(env.paper2_move_scale * env._dijkstra_distance(src, dst))
    if action_name not in _PDDL_FIXED_COST_KEY:
        raise ValueError(f"Unknown physical action {action_name!r}.")
    key = _PDDL_FIXED_COST_KEY[action_name]
    return float(env.paper2_fixed_costs.get(key, 0.0))


def planner_actions_paper2_cost(actions: Sequence[Tuple[str, List[str]]], env: RestaurantSymbolicEnv) -> float:
    total = 0.0
    for name, args in actions:
        total += _line_cost_from_action(name, args, env)
    return float(total)


def consume_delivery_from_state(state: RestaurantPlannerState, task_type: str, target_location: str | None) -> None:
    """Empty delivered artifact on a planner state."""
    consume_delivery(state.objects, task_type, target_location)


def _first_serve_water_cup(state: RestaurantPlannerState, target_location: str) -> Optional[str]:
    """First cup/mug at the target containing water (matches `consume_delivery`)."""
    for obj in state.objects.values():
        if obj.location == target_location and obj.kind in {"cup", "mug"} and obj.filled_with == "water":
            return obj.name
    return None


def _first_make_coffee_cup(state: RestaurantPlannerState, target_location: str) -> Optional[str]:
    """First cup/mug at the target containing coffee (matches `consume_delivery`)."""
    for obj in state.objects.values():
        if obj.location == target_location and obj.kind in {"cup", "mug"} and obj.filled_with == "coffee":
            return obj.name
    return None


def _first_make_fruit_bowl_witness(
    state: RestaurantPlannerState, target_location: str,
) -> Tuple[Optional[str], Optional[str]]:
    """First apple (in object order) whose containing bowl is at the target.

    Returns `(apple_name, bowl_name)` or `(None, None)` if no eligible
    apple exists. The bowl is the apple's actual containing bowl, not the
    first bowl at the target -- matches the env's `consume_delivery` scan
    over apples in insertion order.
    """
    bowls = {
        obj.name
        for obj in state.objects.values()
        if obj.kind == "bowl" and obj.location == target_location
    }
    if not bowls:
        return None, None
    for obj in state.objects.values():
        if obj.kind == "apple" and obj.contained_in in bowls:
            return obj.name, obj.contained_in
    return None, None


def _validate_completion_witness_against_state(
    name: str,
    args: Sequence[str],
    task: RestaurantTask,
    post_prefix_state: RestaurantPlannerState,
) -> None:
    """Strict state-based validation of a completion marker's witnesses.

    Compares the named witnesses against the simulated post-prefix state to
    catch known-but-ineligible plans (e.g. a cup witness that is at the
    wrong location, an apple that is not actually in the named bowl). The
    deterministic check matches what the env's `consume_delivery` would
    consume, so a Fast Downward plan that emits the deterministic witness
    is accepted -- we do not require the planner to non-deterministically
    pick a witness.

    Pre: `state.objects` retains insertion order and is the same dict the
    env uses, so iteration order matches the env's own scan.
    """
    target = task.target_location

    if name == "complete-serve-water":
        cup = args[2]
        cup_obj = post_prefix_state.objects.get(cup)
        if cup_obj is None or target is None:
            return  # Syntactic check already covers existence; nothing to verify.
        if cup_obj.location != target:
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} is at "
                f"{cup_obj.location!r}, expected the task's target {target!r}."
            )
        if cup_obj.filled_with != "water":
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} is filled_with "
                f"{cup_obj.filled_with!r}, expected 'water'."
            )
        expected = _first_serve_water_cup(post_prefix_state, target)
        if expected != cup:
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} does not match "
                f"the deterministic first-eligible cup {expected!r} at "
                f"target {target!r}."
            )
        return

    if name == "complete-make-coffee":
        cup = args[2]
        cup_obj = post_prefix_state.objects.get(cup)
        if cup_obj is None or target is None:
            return
        if cup_obj.location != target:
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} is at "
                f"{cup_obj.location!r}, expected the task's target {target!r}."
            )
        if cup_obj.filled_with != "coffee":
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} is filled_with "
                f"{cup_obj.filled_with!r}, expected 'coffee'."
            )
        expected = _first_make_coffee_cup(post_prefix_state, target)
        if expected != cup:
            raise ValueError(
                f"Completion {name!r} cup witness {cup!r} does not match "
                f"the deterministic first-eligible cup {expected!r} at "
                f"target {target!r}."
            )
        return

    if name == "complete-make-fruit-bowl":
        apple, bowl, loc = args[2], args[3], args[4]
        if target is None:
            return
        apple_obj = post_prefix_state.objects.get(apple)
        bowl_obj = post_prefix_state.objects.get(bowl)
        if apple_obj is None or bowl_obj is None:
            return
        if loc != target:
            raise ValueError(
                f"Completion {name!r} loc witness {loc!r} does not match "
                f"the task's target_location {target!r}."
            )
        if bowl_obj.location != target:
            raise ValueError(
                f"Completion {name!r} bowl witness {bowl!r} is at "
                f"{bowl_obj.location!r}, expected the task's target {target!r}."
            )
        if apple_obj.contained_in != bowl:
            raise ValueError(
                f"Completion {name!r} apple witness {apple!r} is contained in "
                f"{apple_obj.contained_in!r}, expected the named bowl {bowl!r}."
            )
        expected_apple, expected_bowl = _first_make_fruit_bowl_witness(
            post_prefix_state, target,
        )
        if (expected_apple, expected_bowl) != (apple, bowl):
            raise ValueError(
                f"Completion {name!r} apple/bowl witnesses "
                f"({apple!r}, {bowl!r}) do not match the deterministic "
                f"first-eligible pair ({expected_apple!r}, {expected_bowl!r}) "
                f"at target {target!r}."
            )
        return

    # clear_containers / wash_objects / pick_place carry no object witness;
    # the witness-free state-satisfaction check is in
    # `_current_task_satisfied`; arity/task-id checks happen before this
    # function is called.


def _current_task_satisfied(
    state: "RestaurantPlannerState",
    task: RestaurantTask,
    env: RestaurantSymbolicEnv,
) -> bool:
    """Mirrors the sequence-domain `current-task-satisfied` derived predicate.

    For each of the six task types, the check matches both the executable
    env's `_task_already_satisfied` (the ground truth) and the sequence
    PDDL's `current-task-satisfied` rule in
    `pddl/toy_restaurant_sequence_domain.pddl`. Two points require care:

    * `clear_containers`: the sequence init does not emit
      `(is-at water_machine ?loc)` in sequence mode (machine water is the
      abstract `(machine-water-available ?loc)` resource). The env's check
      iterates `state.objects`, which includes the concrete `water_machine`
      object, so "no object at target" naturally accounts for the machine
      water resource -- the sequence-domain `(not (machine-water-available
      ?loc))` and the env's "no object at target" agree at the state level.

    * `serve_water` / `make_coffee` / `make_fruit_bowl`: reuse the same
      deterministic-witness helpers as `_validate_completion_witness_against_state`
      so a task is satisfied iff a deterministic-witness-eligible artifact
      exists in the state. This matches the env's `consume_delivery` order
      and the PDDL's `selected-*` derived predicates.
    """
    target = task.target_location

    if task.task_type == "serve_water":
        return _first_serve_water_cup(state, target) is not None

    if task.task_type == "make_coffee":
        return _first_make_coffee_cup(state, target) is not None

    if task.task_type == "make_fruit_bowl":
        apple, _bowl = _first_make_fruit_bowl_witness(state, target)
        return apple is not None

    if task.task_type == "clear_containers":
        # Env check: no object at target. Concrete water objects (like
        # `water_machine`) are in `state.objects`, so this naturally
        # accounts for the machine-water resource the sequence domain
        # models separately as `(machine-water-available ?loc)`.
        return not any(
            obj.location == target for obj in state.objects.values()
        )

    if task.task_type == "wash_objects":
        kind = task.target_kind
        return any(
            obj.kind == kind
            and not obj.dirty
            and obj.filled_with is None
            and obj.location in env.wash_ready_locations
            and obj.contained_in is None
            for obj in state.objects.values()
        )

    if task.task_type == "pick_place":
        obj_name = task.object_name
        if obj_name is None or target is None:
            return False
        obj = state.objects.get(obj_name)
        return (
            obj is not None
            and obj.location == target
            and state.holding is None
        )

    raise ValueError(f"Unsupported task type: {task.task_type!r}")


def _known_cost_entries(env: RestaurantSymbolicEnv) -> Iterable[Tuple[str, str, int]]:
    """Emit (src, dst, cost) triples for the PDDL known-cost function.

    Uses env.movement_costs when present; otherwise falls back to a scaled
    grid Dijkstra distance so the planner still has a deterministic cost function.
    """
    for src, dst in itertools.product(env.locations, repeat=2):
        if env.movement_costs and src in env.movement_costs and dst in env.movement_costs[src]:
            dist = env.movement_costs[src][dst]
        else:
            dist = env._dijkstra_distance(src, dst)
        cost = int(round(env.paper2_move_scale * float(dist)))
        yield src, dst, cost


def _objects_of_kind(state: RestaurantPlannerState, kind: str) -> List[str]:
    return [name for name, obj in state.objects.items() if obj.kind == kind]


def task_goal_clauses(
    state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    service_locations: Sequence[str],
    wash_ready_locations: Sequence[str],
) -> List[str]:
    if task.task_type == "serve_water":
        assert task.target_location is not None
        candidates = [n for n, o in state.objects.items() if o.kind in {"cup", "mug"}]
        return [f"(or {' '.join([f'(and (is-at {o} {task.target_location}) (filled-with water {o}))' for o in candidates])})"]
    if task.task_type == "make_coffee":
        assert task.target_location is not None
        candidates = [n for n, o in state.objects.items() if o.kind in {"cup", "mug"}]
        return [f"(or {' '.join([f'(and (is-at {o} {task.target_location}) (filled-with coffee {o}))' for o in candidates])})"]
    if task.task_type == "make_fruit_bowl":
        assert task.target_location is not None
        bowls = _objects_of_kind(state, "bowl")
        apples = _objects_of_kind(state, "apple")
        disj_terms: List[str] = []
        for bowl in bowls:
            for apple in apples:
                disj_terms.append(f"(and (is-at {bowl} {task.target_location}) (is-in {apple} {bowl}))")
        return [f"(or {' '.join(disj_terms)})"] if disj_terms else ["(and)"]
    if task.task_type == "clear_containers":
        assert task.target_location is not None
        return [f"(not (is-at {o} {task.target_location}))" for o in state.objects.keys()]
    if task.task_type == "wash_objects":
        assert task.target_kind is not None
        candidates = _objects_of_kind(state, task.target_kind)
        disj_terms: List[str] = []
        for o in candidates:
            for wloc in wash_ready_locations:
                disj_terms.append(f"(and (is-at {o} {wloc}) (not (is-dirty {o})) (not (filled-with water {o})) (not (filled-with coffee {o})))")
        return [f"(or {' '.join(disj_terms)})"] if disj_terms else ["(and)"]
    if task.task_type == "pick_place":
        assert task.object_name is not None and task.target_location is not None
        return [f"(and (is-at {task.object_name} {task.target_location}) (hand-is-free))"]
    raise ValueError(f"Unsupported task type: {task.task_type}")


def _build_world_init_lines(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    *,
    for_sequence: bool = False,
) -> List[str]:
    """Build the world-state `:init` lines shared by every PDDL problem.

    Emits robot state, known movement costs, station predicates, the shared
    `water` / `coffee` liquid constants, the `(= (total-cost) 0)` seed, and one
    block of per-object facts. Order matches the original
    `build_restaurant_problem_text` so existing single-task problems stay
    byte-identical apart from the per-object predicate set.

    `for_sequence=True` switches on sequence-only semantics. The sequence
    domain's water model is:

    * `is-fountain ?loc` and `is-coffeemachine ?loc` are emitted for
      EVERY location recognized by `env._is_location(loc, role)` --
      both literal role names (e.g. "fountain", "coffeemachine") and
      configured role-declared locations (e.g. `servingtable` with
      `location_roles["servingtable"] = ("fountain",)`). The literal
      / role-declared fan-out is what lets a role-declared station
      drive the PDDL gate.
    * The abstract `(is-at water <loc>)` is bound to the actual
      `water_fountain` location (after the layout invariant confirms
      it is at a recognized fountain location). The concrete
      `water_fountain` is NOT emitted as a workaround.
    * The abstract `(machine-water-available <loc>)` is bound to the
      actual `water_machine` location (when water_machine is at a
      recognized coffeemachine location). The concrete `water_machine`
      is NEVER emitted.

    The `for_sequence=False` path preserves the historical
    `env.station_water` / `env.station_coffee` predicate emission and
    the per-water-object `(is-at water <loc>)` emission, so existing
    single-task problems stay byte-identical apart from the
    per-object predicate set.
    """
    init_lines: List[str] = [f"(rob-at {state.agent_location})"]
    if state.holding is None:
        init_lines.append("(hand-is-free)")
    else:
        init_lines.append(f"(is-holding {state.holding})")
    for src, dst, cost in _known_cost_entries(env):
        init_lines.append(f"(= (known-cost {src} {dst}) {cost})")
    if for_sequence:
        # Emit (is-fountain <loc>) for every recognized fountain
        # location: literal name "fountain" OR a role-declared
        # location (e.g. `servingtable` with the fountain role). The
        # PDDL `fill` action's `(is-fountain ?loc)` precondition then
        # matches whichever role-bearing location the action is at.
        for loc in env.locations:
            if env._is_location(loc, "fountain"):
                init_lines.append(f"(is-fountain {loc})")
        # Same fan-out for coffeemachine locations.
        for loc in env.locations:
            if env._is_location(loc, "coffeemachine"):
                init_lines.append(f"(is-coffeemachine {loc})")
    else:
        # Single-task legacy: emit canonical stations only.
        if env.station_water in env.location_index:
            init_lines.append(f"(is-fountain {env.station_water})")
        if env.station_coffee in env.location_index:
            init_lines.append(f"(is-coffeemachine {env.station_coffee})")
    # Role-based fan-out for `is-dishwasher` and `is-countertop` (Task 2
    # fix): the env's `_is_location(loc, role)` accepts BOTH the
    # canonical station name (e.g. `dishwasher`, `countertop`) and any
    # role-declared location (e.g. `servingtable` with
    # `location_roles["servingtable"] = ("dishwasher",)`). The
    # previous canonical-only emission let strict replay accept a
    # wash / make-fruit-bowl action at a role-declared location while
    # the PDDL silently rejected it. Emit the role predicates for
    # every recognized location so PDDL and strict replay agree.
    # This matches the existing fan-out for `is-fountain` and
    # `is-coffeemachine` above. Applied in both single-task and
    # sequence modes because `_build_world_init_lines` is shared.
    for loc in env.locations:
        if env._is_location(loc, "dishwasher"):
            init_lines.append(f"(is-dishwasher {loc})")
    for loc in env.locations:
        if env._is_location(loc, "countertop"):
            init_lines.append(f"(is-countertop {loc})")
    init_lines.append("(is-liquid water)")
    init_lines.append("(is-liquid coffee)")
    init_lines.append("(= (total-cost) 0)")

    if for_sequence:
        # Active toy_level_3 has a permanent fountain source and one
        # consumable machine-water resource.
        wf_loc = state.objects["water_fountain"].location
        init_lines.append(f"(is-at water {wf_loc})")
        wm = state.objects.get("water_machine")
        if wm is not None and wm.location is not None:
            init_lines.append(
                f"(machine-water-available {wm.location})"
            )

    for name, obj in state.objects.items():
        if obj.kind == "water":
            # Water is the abstract resource. The sequence domain
            # already emitted `(is-at water <wf_loc>)` and
            # `(machine-water-available <wm_loc>)` above, so we do
            # NOT emit concrete `(is-at water_fountain ...)` /
            # `(is-at water_machine ...)` facts here.
            if for_sequence:
                continue
            # Single-task domain: keep the historical single-(is-at water
            # ?loc) emission so the existing PDDL stays byte-identical
            # apart from the per-object predicate set.
            if obj.location is not None and obj.location != "__held__":
                init_lines.append(f"(is-at water {obj.location})")
            continue
        if obj.location is not None and obj.location != "__held__":
            init_lines.append(f"(is-at {name} {obj.location})")
        if obj.contained_in is not None:
            init_lines.append(f"(is-in {name} {obj.contained_in})")
        if obj.dirty:
            init_lines.append(f"(is-dirty {name})")
        if obj.kind in {"cup", "bowl", "mug", "jar"}:
            init_lines.append(f"(is-fillable {name})")
        if obj.kind == "bowl":
            init_lines.append(f"(is-container {name})")
        if obj.kind == "apple":
            init_lines.append(f"(is-slicable {name})")
        if obj.kind == "knife":
            init_lines.append(f"(is-knife {name})")
        if obj.kind == "jar":
            init_lines.append(f"(is-jar {name})")
        init_lines.append(f"(is-pickable {name})")
        if obj.filled_with == "water":
            init_lines.append(f"(filled-with water {name})")
        elif obj.filled_with == "coffee":
            init_lines.append(f"(filled-with coffee {name})")
    return init_lines


def build_restaurant_problem_text(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    extra_goal_clauses: Sequence[str] | None = None,
    problem_name: str = "restaurant-problem",
) -> str:
    locations = env.locations
    objects = list(state.objects.keys())
    obj_decl = " ".join(objects) + " - object"
    loc_decl = " ".join(locations) + " - location"
    object_block = f"    {obj_decl}\n    {loc_decl}\n"

    init_lines = _build_world_init_lines(env, state)

    goal_clauses = task_goal_clauses(
        state,
        task,
        service_locations=env.service_locations,
        wash_ready_locations=env.wash_ready_locations,
    ) + list(extra_goal_clauses or [])
    goal_text = "\n      ".join(goal_clauses) if goal_clauses else "(and)"
    init_text = "\n    ".join(init_lines)

    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain restaurant)\n"
        f"  (:objects\n{object_block}  )\n"
        f"  (:init\n    {init_text}\n  )\n"
        f"  (:goal\n    (and\n      {goal_text}\n    )\n  )\n"
        f"  (:metric minimize (total-cost))\n"
        f")\n"
    )


# ---------------------------------------------------------------------------
# Sequence-domain problem builder
# ---------------------------------------------------------------------------


def _validate_sequence_tasks(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    tasks: Sequence[RestaurantTask],
) -> None:
    """Reject an empty / malformed task sequence before invoking Fast Downward.

    Validation passes for the literal `tasks` list; the builder does not
    rewrite or substitute task parameters. Per task type:

    * `target_location` is required for serve_water, make_coffee,
      make_fruit_bowl, clear_containers, and pick_place, and forbidden
      otherwise (wash_objects is the only kind-only task).
    * `target_kind` is required for wash_objects and forbidden otherwise.
    * `object_name` is required for pick_place and forbidden otherwise.
    * All non-None fields must be present in the env/state lookups.
    """
    if not tasks:
        raise ValueError("Task sequence is empty.")
    requires_location = {
        "serve_water", "make_coffee", "make_fruit_bowl",
        "clear_containers", "pick_place",
    }
    sequence_kinds = _sequence_kind_constants(env)
    for idx, task in enumerate(tasks):
        if task.task_type not in env.task_type_index:
            raise ValueError(
                f"Task {idx} has unsupported type {task.task_type!r}; "
                f"supported: {sorted(env.task_type_index)}."
            )
        # target_location handling
        if task.task_type in requires_location:
            if task.target_location is None:
                raise ValueError(
                    f"Task {idx} ({task.task_type}) is missing target_location."
                )
            if task.target_location not in env.location_index:
                raise ValueError(
                    f"Task {idx} ({task.task_type}) references unknown location "
                    f"{task.target_location!r}; known: {sorted(env.location_index)}."
                )
        else:
            if task.target_location is not None:
                raise ValueError(
                    f"Task {idx} ({task.task_type}) must not set target_location "
                    f"(got {task.target_location!r}); target_location is forbidden for this task type."
                )
        # target_kind handling
        if task.task_type == "wash_objects":
            if task.target_kind is None:
                raise ValueError(
                    f"Task {idx} (wash_objects) is missing target_kind."
                )
            if task.target_kind not in env.object_kind_index:
                raise ValueError(
                    f"Task {idx} (wash_objects) references unknown kind "
                    f"{task.target_kind!r}; known: {sorted(env.object_kind_index)}."
                )
            if task.target_kind in _SEQUENCE_KIND_BLACKLIST:
                raise ValueError(
                    f"Task {idx} (wash_objects) target_kind {task.target_kind!r} "
                    f"is not a sequence-domain kind constant (collides with the "
                    f"abstract water/coffee object constant). Allowed kinds: "
                    f"{sequence_kinds}."
                )
            if task.target_kind not in sequence_kinds:
                raise ValueError(
                    f"Task {idx} (wash_objects) target_kind {task.target_kind!r} "
                    f"is not declared in the sequence kind constants "
                    f"({sequence_kinds})."
                )
        else:
            if task.target_kind is not None:
                raise ValueError(
                    f"Task {idx} ({task.task_type}) must not set target_kind "
                    f"(got {task.target_kind!r}); target_kind is only valid for wash_objects."
                )
        # object_name handling
        if task.task_type == "pick_place":
            if task.object_name is None:
                raise ValueError(
                    f"Task {idx} (pick_place) is missing object_name."
                )
            if task.object_name not in state.objects:
                raise ValueError(
                    f"Task {idx} (pick_place) references unknown object "
                    f"{task.object_name!r}; known: {sorted(state.objects)}."
                )
        else:
            if task.object_name is not None:
                raise ValueError(
                    f"Task {idx} ({task.task_type}) must not set object_name "
                    f"(got {task.object_name!r}); object_name is only valid for pick_place."
                )


def _sequence_kind_constants(env: RestaurantSymbolicEnv) -> List[str]:
    """Return the sorted list of sequence-domain `kind` constants.

    Filters out names that collide with the abstract `(water coffee - object)`
    constants in `pddl/toy_restaurant_sequence_domain.pddl`: PDDL types are
    disjoint, so a symbol can appear in only one type declaration. We keep
    `water`/`coffee` as object constants and exclude them from the `kind`
    type, which also forbids them as `wash_objects` targets.
    """
    return sorted(k for k in env.object_kinds if k not in _SEQUENCE_KIND_BLACKLIST)


def _task_facts_lines(task: RestaurantTask, task_id: str) -> List[str]:
    """Emit task-type tag and the task-parameter facts *relevant to the task type*.

    Only parameters that the sequence domain actually consults are emitted;
    irrelevant non-None fields are rejected by `_validate_sequence_tasks` before
    this runs, so the branch on task type is safe.
    """
    lines: List[str] = []
    tag_map = {
        "serve_water": "task-is-serve-water",
        "make_coffee": "task-is-make-coffee",
        "make_fruit_bowl": "task-is-make-fruit-bowl",
        "clear_containers": "task-is-clear-containers",
        "wash_objects": "task-is-wash-objects",
        "pick_place": "task-is-pick-place",
    }
    tag = tag_map.get(task.task_type)
    if tag is None:
        raise ValueError(f"Unsupported task type: {task.task_type}")
    lines.append(f"({tag} {task_id})")
    if task.task_type in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers", "pick_place"}:
        assert task.target_location is not None  # validated above
        lines.append(f"(task-target-location {task_id} {task.target_location})")
    if task.task_type == "wash_objects":
        assert task.target_kind is not None  # validated above
        lines.append(f"(task-target-kind {task_id} {task.target_kind})")
    if task.task_type == "pick_place":
        assert task.object_name is not None  # validated above
        lines.append(f"(task-object {task_id} {task.object_name})")
    return lines


def build_restaurant_sequence_problem_text(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    tasks: Sequence[RestaurantTask],
    *,
    problem_name: str = "restaurant-sequence",
) -> str:
    """Generate the PDDL problem text for the sequence domain.

    The problem represents a K-task chain (K = len(tasks)) starting at `t0` and
    ending at `t-end`. Reuses the world serialization of
    `build_restaurant_problem_text` and appends the state-machine facts.
    """
    _validate_sequence_tasks(env, state, tasks)
    locations = env.locations
    objects = list(state.objects.keys())
    kinds = _sequence_kind_constants(env)
    n = len(tasks)
    task_ids = [f"t{i}" for i in range(n)]  # t0, t1, ..., t<n-1>
    # The typed object declaration includes every task id AND the terminal
    # `t-end`. The `next-task` chain runs t0 -> t1 -> ... -> t<n-1> -> t-end,
    # so for a 1-task window the chain is just (next-task t0 t-end).
    task_ids_with_end = task_ids + ["t-end"]

    object_lines = [
        "    " + " ".join(objects) + " - object",
        "    " + " ".join(locations) + " - location",
        "    " + " ".join(kinds) + " - kind",
        "    " + " ".join(task_ids_with_end) + " - task_id",
    ]
    object_block = "\n".join(object_lines) + "\n"

    init_lines = _build_world_init_lines(env, state, for_sequence=True)

    # Sequence-domain-only classification facts.
    # The kind type cannot include `water`/`coffee` (they collide with the
    # abstract object constants), so we use the same blacklist when emitting
    # `(object-kind ?obj ?kind)` facts. The env's `wash_kind_distribution`
    # never targets these, so no task parameter loses its type binding.
    object_names_in_order = list(state.objects.keys())
    for name in object_names_in_order:
        obj = state.objects[name]
        if obj.kind in env.object_kind_index and obj.kind not in _SEQUENCE_KIND_BLACKLIST:
            init_lines.append(f"(object-kind {name} {obj.kind})")
        # The sequence domain's selection axioms (`selected-cup-for-serve-water`,
        # `selected-cup-for-make-coffee`) reference `is-drink-container` directly;
        # without these facts in the init, the witnesses are unsatisfiable and the
        # translator's relevance analysis drops every physical operator. The
        # single-task domain's predicates block does not declare this predicate,
        # so this fact is sequence-builder-only.
        if obj.kind in {"cup", "mug"}:
            init_lines.append(f"(is-drink-container {name})")
    # Complete pairwise object-precedes in state.objects insertion order.
    for i in range(len(object_names_in_order)):
        for j in range(i + 1, len(object_names_in_order)):
            init_lines.append(
                f"(object-precedes {object_names_in_order[i]} {object_names_in_order[j]})"
            )
    # Wash-ready locations.
    for loc in env.wash_ready_locations:
        if loc in env.location_index:
            init_lines.append(f"(is-wash-ready {loc})")

    # State machine: chain + first task current.
    init_lines.append(f"(is-current-task t0)")
    for i in range(n):
        successor = "t-end" if i == n - 1 else f"t{i + 1}"
        init_lines.append(f"(next-task t{i} {successor})")
    for i, task in enumerate(tasks):
        init_lines.extend(_task_facts_lines(task, f"t{i}"))

    init_text = "\n    ".join(init_lines)

    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain restaurant-sequence)\n"
        f"  (:objects\n{object_block}  )\n"
        f"  (:init\n    {init_text}\n  )\n"
        f"  (:goal\n    (and\n      (is-current-task t-end)\n    )\n  )\n"
        f"  (:metric minimize (total-cost))\n"
        f")\n"
    )


# ---------------------------------------------------------------------------
# Sequence solver
# ---------------------------------------------------------------------------


def _validate_physical_action_arity(name: str, args: Sequence[str]) -> None:
    expected = _PHYSICAL_ACTION_ARITY.get(name)
    if expected is None:
        raise ValueError(f"Unknown physical action {name!r}.")
    if len(args) != expected:
        raise ValueError(
            f"Physical action {name!r} has arity {len(args)}, expected {expected} "
            f"(args={list(args)})."
        )


def _validate_completion_action_arity(name: str, args: Sequence[str]) -> None:
    expected = _COMPLETION_ACTION_ARITY.get(name)
    if expected is None:
        raise ValueError(f"Unknown completion action {name!r}.")
    if len(args) != expected:
        raise ValueError(
            f"Completion action {name!r} has arity {len(args)}, expected "
            f"{expected} (args={list(args)})."
        )


def _validate_physical_action_args(
    name: str, args: Sequence[str], env: RestaurantSymbolicEnv, state: RestaurantPlannerState,
) -> None:
    """Validate that the physical action's arguments reference known env symbols.

    Argument-position rules (matches the sequence PDDL exactly):

    * `move(from, to)`: both must be known locations.
    * `pick`, `place`, `wash`, `make-coffee`, `drain`: the object arg must be
      a concrete state object. Abstract `water` / `coffee` constants are NOT
      valid objects for these actions -- they are liquid/source symbols only.
    * `fill(cnt, loc, src)`: `cnt` must be a concrete state object; `loc`
      must be a known location; `src` must be the abstract `water` constant
      (the only valid `fill` source per the sequence domain's init -- the
      concrete `water_fountain` / `water_machine` objects have no `(is-at
      water_* <loc>)` fact in sequence mode). `coffee` is not a valid
      fountain source.
    * `pour(cnt, liquid, loc)`: `cnt` must be a concrete state object;
      `liquid` must be the abstract `water` or `coffee` constant (the only
      `(is-liquid ?obj)` constants in the sequence domain); `loc` must be a
      known location.
    * `make-fruit-bowl(a, b, k, loc)`: `a` (apple), `b` (bowl), `k` (knife)
      must all be concrete state objects; `loc` must be a known location.
    * `refill_water(cnt, loc, jr)`: `cnt` and `jr` must be concrete state
      objects; `loc` must be a known location.

    Unknown concrete objects raise (no silent fall-through): if `state.objects`
    does not contain the name, the plan is malformed and must be rejected.
    """
    loc_set = set(env.locations)
    obj_set = set(state.objects.keys())

    def _check_concrete_obj(idx: int) -> None:
        if args[idx] not in obj_set:
            raise ValueError(
                f"{name!r} argument {idx} ({args[idx]!r}) is not a known "
                f"concrete state object; abstract liquid constants "
                f"({sorted(_ABSTRACT_LIQUID_CONSTANTS)}) are only valid in "
                f"the fill-source / pour-liquid positions."
            )

    def _check_loc(idx: int) -> None:
        if args[idx] not in loc_set:
            raise ValueError(
                f"{name!r} argument {idx} ({args[idx]!r}) is not a known location."
            )

    if name == "move":
        _check_loc(0)
        _check_loc(1)
    elif name in {"pick", "place", "wash", "make-coffee", "drain"}:
        _check_concrete_obj(0)
        _check_loc(1)
    elif name == "fill":
        _check_concrete_obj(0)
        _check_loc(1)
        src = args[2]
        if src not in _ABSTRACT_FILL_SOURCES:
            if src in _ABSTRACT_LIQUID_CONSTANTS:
                # e.g. `coffee` -- valid liquid for `pour`, invalid for `fill`.
                raise ValueError(
                    f"fill argument 2 ({src!r}) is not a valid fountain "
                    f"source; fill requires the abstract 'water' constant "
                    f"(allowed: {sorted(_ABSTRACT_FILL_SOURCES)})."
                )
            # Anything else (concrete objects, ghosts) is invalid here.
            raise ValueError(
                f"fill argument 2 ({src!r}) is not a valid fill source; "
                f"fill requires the abstract 'water' constant (allowed: "
                f"{sorted(_ABSTRACT_FILL_SOURCES)})."
            )
    elif name == "pour":
        _check_concrete_obj(0)
        liquid = args[1]
        if liquid not in _ABSTRACT_LIQUID_CONSTANTS:
            raise ValueError(
                f"pour argument 1 ({liquid!r}) is not a valid liquid; pour "
                f"requires the abstract 'water' or 'coffee' constant (allowed: "
                f"{sorted(_ABSTRACT_LIQUID_CONSTANTS)})."
            )
        _check_loc(2)
    elif name == "make-fruit-bowl":
        _check_concrete_obj(0)
        _check_concrete_obj(1)
        _check_concrete_obj(2)
        _check_loc(3)
    elif name == "refill_water":
        _check_concrete_obj(0)
        _check_loc(1)
        _check_concrete_obj(2)
    else:  # pragma: no cover - guarded by arity check above
        raise ValueError(f"Unhandled physical action {name!r}.")


def _validate_completion_action_args(
    name: str,
    args: Sequence[str],
    tasks: Sequence[RestaurantTask],
    task_index: int,
) -> None:
    """Validate completion task ID and successor sequencing."""
    valid_task_ids = {f"t{i}" for i in range(len(tasks))} | {"t-end"}
    cur, nxt = args[0], args[1]
    if cur not in valid_task_ids:
        raise ValueError(
            f"Completion {name!r} current task id {cur!r} is not a valid task id."
        )
    if nxt not in valid_task_ids:
        raise ValueError(
            f"Completion {name!r} successor task id {nxt!r} is not a valid task id."
        )
    expected_cur = f"t{task_index}"
    if cur != expected_cur:
        raise ValueError(
            f"Completion #{task_index} ({name!r}) targets task {cur!r}, "
            f"expected {expected_cur!r}."
        )
    expected_nxt = "t-end" if task_index == len(tasks) - 1 else f"t{task_index + 1}"
    if nxt != expected_nxt:
        raise ValueError(
            f"Completion #{task_index} ({name!r}) successor is {nxt!r}, "
            f"expected {expected_nxt!r}."
        )


def _split_sequence_plan(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    tasks: Sequence[RestaurantTask],
    actions: Sequence[Tuple[str, List[str]]],
) -> List[SequenceTaskSegment]:
    """Validate and split a parsed plan into one segment per task.

    Strictly rejects:
    * Unknown action names (anything outside the 10 physical + 6 completion set).
    * Wrong arity for any physical or completion action (e.g. `move` with no
      args, completion markers with arbitrary extra arguments).
    * Unknown object / location arguments.
    * Wrong completion action for the task at index k.
    * Wrong task-id / successor on a completion marker.
    * Wrong witness for the completion marker relative to the simulated
      post-prefix state (e.g. a cup at the wrong location, an apple not
      in the named bowl, or a non-deterministic witness choice).
    * Trailing actions after the last completion marker.
    * Wrong number / order of completion markers.

    The strict parser simulates each segment's physical prefix on a copy
    of the planner state, validates the completion marker against the
    post-prefix state, and then applies `consume_delivery` to advance the
    simulated state before the next segment is validated. The parser
    does NOT require Fast Downward to non-deterministically pick a
    completion witness: the deterministic first-eligible artifact
    (matching the env's `consume_delivery` scan order) is accepted.

    The strict parser never assigns zero cost to malformed actions.
    """
    if not actions:
        raise ValueError("Empty plan returned by Fast Downward.")

    known_actions = set(_PHYSICAL_ACTIONS) | set(_COMPLETION_ACTIONS)
    for name, args in actions:
        if name not in known_actions:
            raise ValueError(f"Plan contains unknown action {name!r}.")
        if name in _COMPLETION_ACTIONS:
            _validate_completion_action_arity(name, args)
        else:
            _validate_physical_action_arity(name, args)
            _validate_physical_action_args(name, args, env, state)

    n = len(tasks)
    expected_completion = [_TASK_TYPE_TO_COMPLETION[t.task_type] for t in tasks]
    completion_indices: List[int] = []
    for idx, (name, _args) in enumerate(actions):
        if name in _COMPLETION_ACTIONS:
            completion_indices.append(idx)
    if len(completion_indices) != n:
        raise ValueError(
            f"Plan has {len(completion_indices)} completion markers, expected {n}."
        )
    for k, idx in enumerate(completion_indices):
        expected = expected_completion[k]
        if actions[idx][0] != expected:
            raise ValueError(
                f"Completion #{k} is {actions[idx][0]!r}, expected {expected!r} "
                f"for task_type={tasks[k].task_type!r}."
            )
        _validate_completion_action_args(actions[idx][0], actions[idx][1], tasks, k)

    # Simulate the chain so the post-prefix state for each segment is the
    # env state at that point in the chain. The first segment starts from
    # the planner state passed in; each subsequent segment starts from the
    # state left behind by the previous segment's physical prefix plus its
    # `consume_delivery`. `post_prefix` is the state *after* the current
    # segment's physical actions; the chain state is then promoted to it
    # before the consumption effect runs.
    #
    # State-machine gate: the sequence PDDL requires
    #   `(not (current-task-satisfied))` for every physical action and
    #   `(current-task-satisfied)` for every completion action. The parser
    # mirrors this rule by:
    #   1. Checking `current-task-satisfied` is False BEFORE each physical
    #      action (gate on every prefix action, not just the first).
    #   2. Checking `current-task-satisfied` is True BEFORE the completion
    #      marker (catches the witness-free clear/wash/pick-place
    #      completions that would otherwise pass with auto_success=True
    #      even though the env would not auto-succeed).
    simulated_state = state.copy()
    segments: List[SequenceTaskSegment] = []
    cursor = 0
    for k, complete_idx in enumerate(completion_indices):
        physical = [a for a in actions[cursor:complete_idx]]
        completion = actions[complete_idx]
        task = tasks[k]
        # Run the physical prefix on a copy so we can validate the completion
        # witness against the post-prefix state without re-running the prefix
        # twice. The chain state is promoted to this copy at the end of the
        # iteration so the next segment sees the previous segment's effects.
        post_prefix = simulated_state.copy()
        for action in physical:
            # PDDL gate: no physical action may fire once the current task
            # is satisfied. Enforced at every step (not just the first), so
            # a "stray move after satisfaction" prefix is rejected.
            if _current_task_satisfied(post_prefix, task, env):
                raise ValueError(
                    f"Physical action {action[0]!r} fires after task "
                    f"{task.task_type!r} is already satisfied; the sequence "
                    f"PDDL disallows physical actions after "
                    f"current-task-satisfied becomes true. Only the matching "
                    f"complete-* action may run."
                )
            apply_planner_action(post_prefix, action)
        # PDDL gate: the completion action requires
        # `(current-task-satisfied)`. The post-prefix state must therefore
        # satisfy the task in the env's view (matches the sequence-domain
        # `current-task-satisfied` derived predicate, including the
        # witness-free clear_containers / wash_objects / pick_place arms).
        if not _current_task_satisfied(post_prefix, task, env):
            raise ValueError(
                f"Completion {completion[0]!r} for task {task.task_type!r} "
                f"fires before current-task-satisfied; the sequence PDDL "
                f"requires the task to be satisfied before completion. The "
                f"physical prefix left the task unsatisfied."
            )
        _validate_completion_witness_against_state(
            completion[0], completion[1], task, post_prefix,
        )
        seg_cost = float(planner_actions_paper2_cost(physical, env))
        segments.append(
            SequenceTaskSegment(
                task=task,
                physical_actions=physical,
                completion_action=completion,
                paper2_cost=seg_cost,
                auto_success=len(physical) == 0,
            )
        )
        # Advance the chain state: first carry the physical-prefix effects
        # (so the next segment's prefix starts from the new world), then
        # apply the same consumption the env would for this task.
        simulated_state = post_prefix
        consume_delivery_from_state(
            simulated_state, task.task_type, task.target_location,
        )
        cursor = complete_idx + 1
    if cursor != len(actions):
        raise ValueError(
            f"Plan has {len(actions) - cursor} trailing actions after the last "
            f"completion marker."
        )
    return segments


def _parse_raw_fd_cost(plan_text: str) -> Optional[float]:
    """Best-effort: parse `; cost = N` from an FD plan. Returns None if absent."""
    for raw in plan_text.splitlines():
        match = _FD_COST_HEADER_RE.search(raw)
        if match:
            return float(match.group(1))
    return None


def solve_restaurant_sequence_with_fd(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    tasks: Sequence[RestaurantTask],
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str = "seq-sat-lama-2011",
    timeout_s: float = 120.0,
) -> SequencePlannerResult:
    """Solve a K-task chain with the sequence domain and a satisficing planner.

    The returned physical cost is the selected plan's cost, not an optimum or
    lower bound.

    Returns a structured result rather than raising for normal
    planner failures, matching ``solve_restaurant_task_with_fd``.
    Validation errors (empty sequence, bad parameters) do raise --
    they are caller bugs.
    """
    _validate_sequence_tasks(env, state, tasks)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="restaurant_seq_fd_") as tmp:
        tmpdir = Path(tmp)
        problem_path = tmpdir / "problem.pddl"
        problem_path.write_text(
            build_restaurant_sequence_problem_text(
                env,
                state,
                tasks,
                problem_name="restaurant-sequence",
            ),
            encoding="utf-8",
        )
        try:
            plan_path = run_planner(
                planner_path,
                domain_path,
                problem_path,
                tmpdir,
                alias=alias,
                initial_search_time_limit=timeout_s,
                max_search_time_limit=timeout_s,
            )
            plan_text = plan_path.read_text(encoding="utf-8")
            actions = parse_sas_plan(plan_text)
            segments = _split_sequence_plan(env, state, tasks, actions)
            physical_actions: List[Tuple[str, List[str]]] = []
            for seg in segments:
                physical_actions.extend(seg.physical_actions)
            physical_cost = float(planner_actions_paper2_cost(physical_actions, env))
            return SequencePlannerResult(
                success=True,
                plan_actions=list(actions),
                task_segments=segments,
                physical_actions=physical_actions,
                physical_cost=physical_cost,
                raw_fd_cost=_parse_raw_fd_cost(plan_text),
                solve_time_s=float(time.perf_counter() - t0),
                completion_count=len(segments),
                selected_search=f"alias:{alias}",
            )
        except Exception as exc:
            elapsed = float(time.perf_counter() - t0)
            return SequencePlannerResult(
                success=False,
                plan_actions=[],
                task_segments=[],
                physical_actions=[],
                physical_cost=float("inf"),
                raw_fd_cost=None,
                solve_time_s=elapsed,
                completion_count=0,
                error=str(exc),
            )


def solve_restaurant_task_with_fd(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str = "seq-sat-lama-2011",
    extra_goal_clauses: Sequence[str] | None = None,
    timeout_s: float = 10.0,
) -> PlannerResult:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="restaurant_fd_") as tmp:
        tmpdir = Path(tmp)
        problem_path = tmpdir / "problem.pddl"
        problem_path.write_text(
            build_restaurant_problem_text(
                env,
                state,
                task,
                extra_goal_clauses=extra_goal_clauses,
                problem_name="restaurant-task",
            ),
            encoding="utf-8",
        )
        try:
            plan_path = run_planner(
                planner_path,
                domain_path,
                problem_path,
                tmpdir,
                alias=alias,
                initial_search_time_limit=timeout_s,
            )
            plan_text = plan_path.read_text(encoding="utf-8")
            actions = parse_sas_plan(plan_text)
            cost = planner_actions_paper2_cost(actions, env)
            return PlannerResult(
                success=True,
                plan_actions=actions,
                plan_cost=float(cost),
                solve_time_s=float(time.perf_counter() - t0),
            )
        except Exception as exc:
            elapsed = float(time.perf_counter() - t0)
            return PlannerResult(False, [], float("inf"), elapsed, error=str(exc))


def apply_planner_action(state: RestaurantPlannerState, action: Tuple[str, List[str]]) -> None:
    name, args = action
    if name == "move":
        if len(args) >= 2:
            state.agent_location = args[1]
        return
    if name == "pick":
        obj_name = args[0]
        state.holding = obj_name
        state.objects[obj_name].location = None
        state.objects[obj_name].contained_in = None
        return
    if name == "place":
        obj_name, loc = args[0], args[1]
        state.objects[obj_name].location = loc
        state.objects[obj_name].contained_in = None
        state.holding = None
        return
    if name == "wash":
        obj_name = args[0]
        obj = state.objects[obj_name]
        obj.dirty = False
        return
    if name == "fill":
        cnt = args[0]
        state.objects[cnt].filled_with = "water"
        return
    if name == "drain":
        cnt = args[0]
        state.objects[cnt].filled_with = None
        return
    if name == "pour":
        # container -> location: empty held container; restore machine water for `water`.
        cnt = args[0]
        liquid = state.objects[cnt].filled_with
        state.objects[cnt].filled_with = None
        loc = args[-1] if len(args) >= 2 else state.agent_location
        # Sequence-domain any-water rule aligned with the env's
        # `_restore_location_water`: pour of water restores the first
        # water-kind object with `location is None and contained_in is None`
        # to the agent's location (iteration follows `state.objects`
        # insertion order, matching the env's `dict.values()` scan).
        # Pour of any non-water liquid never restores a water-kind
        # object (matches the env's `if liquid == "water"` gate).
        if liquid == "water":
            for obj in state.objects.values():
                if (
                    obj.kind == "water"
                    and obj.location is None
                    and obj.contained_in is None
                ):
                    obj.location = loc
                    break
        return
    if name == "refill_water":
        cnt = args[0]
        state.objects[cnt].filled_with = "water"
        # Jar is not depleted.
        return
    if name == "make-coffee":
        obj = state.objects[args[0]]
        obj.filled_with = "coffee"
        obj.dirty = True
        # Sequence-domain any-water rule aligned with the env's
        # `_consume_machine_water`: make-coffee consumes the first
        # water-kind object at the coffeemachine (iteration follows
        # `state.objects` insertion order, matching the env's
        # `dict.values()` scan). Matches the sequence PDDL's
        # `(not (machine-water-available ?loc))` effect.
        loc = args[1] if len(args) >= 2 else state.agent_location
        for obj in state.objects.values():
            if obj.kind == "water" and obj.location == loc:
                obj.location = None
                break
        return
    if name == "make-fruit-bowl":
        apple_name, bowl_name, knife_name, loc = args[0], args[1], args[2], args[3]
        state.objects[apple_name].location = None
        state.objects[apple_name].contained_in = bowl_name
        state.objects[bowl_name].dirty = True
        state.objects[knife_name].dirty = True
        return


def apply_plan(state: RestaurantPlannerState, plan_actions: Sequence[Tuple[str, List[str]]]) -> RestaurantPlannerState:
    new_state = state.copy()
    for action in plan_actions:
        apply_planner_action(new_state, action)
    return new_state


def dump_planner_result_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
