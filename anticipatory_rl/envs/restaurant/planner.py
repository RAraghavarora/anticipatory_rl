"""Restaurant PDDL builder + Fast Downward planner runner."""

from __future__ import annotations

import copy
import itertools
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from anticipatory_rl.envs.restaurant.env import (
    RestaurantObjectState,
    RestaurantState,
    RestaurantSymbolicEnv,
    RestaurantTask,
    consume_delivery,
)
from anticipatory_rl.envs.restaurant.fd_runner import run_planner


@dataclass
class PlannerResult:
    success: bool
    plan_actions: List[Tuple[str, List[str]]]
    plan_cost: float
    solve_time_s: float
    error: str | None = None


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
            return 0.0
        src, dst = args[0], args[1]
        return float(env.paper2_move_scale * env._dijkstra_distance(src, dst))
    key = _PDDL_FIXED_COST_KEY.get(action_name)
    if key is not None:
        return float(env.paper2_fixed_costs.get(key, 0.0))
    return 0.0


def planner_actions_paper2_cost(actions: Sequence[Tuple[str, List[str]]], env: RestaurantSymbolicEnv) -> float:
    total = 0.0
    for name, args in actions:
        total += _line_cost_from_action(name, args, env)
    return float(total)


def consume_delivery_from_state(state: RestaurantPlannerState, task_type: str, target_location: str | None) -> None:
    """Empty delivered artifact on a planner state."""
    consume_delivery(state.objects, task_type, target_location)


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
    coords = env.location_coords
    extra_goal_clauses = list(extra_goal_clauses or [])

    obj_decl = " ".join(objects) + " - object"
    loc_decl = " ".join(locations) + " - location"
    object_block = f"    {obj_decl}\n    {loc_decl}\n"

    init_lines: List[str] = [f"(rob-at {state.agent_location})"]
    if state.holding is None:
        init_lines.append("(hand-is-free)")
    else:
        init_lines.append(f"(is-holding {state.holding})")
    for src, dst, cost in _known_cost_entries(env):
        init_lines.append(f"(= (known-cost {src} {dst}) {cost})")
    if env.station_water in env.location_index:
        init_lines.append(f"(is-fountain {env.station_water})")
    if env.station_coffee in env.location_index:
        init_lines.append(f"(is-coffeemachine {env.station_coffee})")
    if env.station_wash in env.location_index:
        init_lines.append(f"(is-dishwasher {env.station_wash})")
    if env.countertop_location in env.location_index:
        init_lines.append(f"(is-countertop {env.countertop_location})")
    init_lines.append("(is-liquid water)")
    init_lines.append("(is-liquid coffee)")
    init_lines.append("(= (total-cost) 0)")

    for name, obj in state.objects.items():
        if obj.kind == "water":
            # Water is modeled by the single `water` constant, not per-object instances.
            # A water object at a location contributes `(is-at water <loc>)`; it is never
            # picked/placed/washed. This keeps the water symbol identical to what fill/pour
            # produce, so make-coffee's `(is-at water ?loc)` cannot be satisfied by poured
            # coffee (which is a distinct constant), matching the executable env.
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

    goal_clauses = task_goal_clauses(
        state,
        task,
        service_locations=env.service_locations,
        wash_ready_locations=env.wash_ready_locations,
    ) + list(extra_goal_clauses)
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
        obj.filled_with = None
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
        if liquid == "water":
            for obj in state.objects.values():
                if obj.kind == "water" and obj.location is None and obj.contained_in is None:
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
        # Consume the water at the coffee machine (PDDL: not (is-at water coffeemachine)).
        loc = args[1] if len(args) >= 2 else state.agent_location
        for wobj in state.objects.values():
            if wobj.kind == "water" and wobj.location == loc:
                wobj.location = None
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
