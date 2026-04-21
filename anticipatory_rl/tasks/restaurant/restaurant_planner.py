"""Restaurant PDDL builder + Fast Downward planner runner."""

from __future__ import annotations

import copy
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from anticipatory_rl.envs.restaurant_symbolic_env import (
    RestaurantObjectState,
    RestaurantState,
    RestaurantSymbolicEnv,
    RestaurantTask,
)
from anticipatory_rl.tasks.planner_utils import run_planner


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

    @classmethod
    def from_env(cls, env: RestaurantSymbolicEnv) -> "RestaurantPlannerState":
        return cls(
            agent_location=str(env.state.agent_location),
            holding=None if env.state.holding is None else str(env.state.holding),
            objects={k: copy.deepcopy(v) for k, v in env.state.objects.items()},
        )

    def copy(self) -> "RestaurantPlannerState":
        return RestaurantPlannerState(
            agent_location=self.agent_location,
            holding=self.holding,
            objects={k: copy.deepcopy(v) for k, v in self.objects.items()},
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


def _line_cost_from_action(action_name: str, args: Sequence[str], env: RestaurantSymbolicEnv) -> float:
    fixed = env.paper2_fixed_costs
    if action_name == "move":
        if len(args) < 2:
            return 0.0
        src, dst = args[0], args[1]
        return float(env.paper2_move_scale * env._dijkstra_distance(src, dst))
    if action_name == "pick":
        return float(fixed.get("pick", 0.0))
    if action_name == "place":
        return float(fixed.get("place", 0.0))
    if action_name == "wash":
        return float(fixed.get("wash", 0.0))
    if action_name == "fill-water":
        return float(fixed.get("fill", 0.0))
    if action_name == "brew-coffee":
        return float(fixed.get("brew", 0.0))
    if action_name == "fill-fruit":
        return float(fixed.get("fruit", 0.0))
    return 0.0


def planner_actions_paper2_cost(actions: Sequence[Tuple[str, List[str]]], env: RestaurantSymbolicEnv) -> float:
    total = 0.0
    for name, args in actions:
        total += _line_cost_from_action(name, args, env)
    return float(total)


def _all_location_pairs(locations: Sequence[str], coords: Mapping[str, Tuple[int, int]]) -> Iterable[Tuple[str, str]]:
    for src in locations:
        sx, sy = coords[src]
        for dst in locations:
            if src == dst:
                continue
            dx, dy = coords[dst]
            if abs(dx - sx) + abs(dy - sy) == 1:
                yield src, dst


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
        candidates = [n for n, o in state.objects.items() if o.kind in {"mug", "glass"}]
        return [f"(or {' '.join([f'(and (at {o} {task.target_location}) (water {o}))' for o in candidates])})"]
    if task.task_type == "make_coffee":
        assert task.target_location is not None
        candidates = _objects_of_kind(state, "mug")
        return [f"(or {' '.join([f'(and (at {o} {task.target_location}) (coffee {o}))' for o in candidates])})"]
    if task.task_type == "serve_fruit_bowl":
        assert task.target_location is not None
        candidates = _objects_of_kind(state, "bowl")
        return [f"(or {' '.join([f'(and (at {o} {task.target_location}) (fruit {o}))' for o in candidates])})"]
    if task.task_type == "clear_containers":
        assert task.target_location is not None
        return [f"(not (at {o} {task.target_location}))" for o in state.objects.keys()]
    if task.task_type == "wash_objects":
        assert task.target_kind is not None
        candidates = _objects_of_kind(state, task.target_kind)
        disj_terms: List[str] = []
        for o in candidates:
            for wloc in wash_ready_locations:
                disj_terms.append(f"(and (at {o} {wloc}) (clean {o}) (empty {o}))")
        return [f"(or {' '.join(disj_terms)})"] if disj_terms else ["(and)"]
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

    init_lines: List[str] = [f"(agent-at {state.agent_location})"]
    if state.holding is None:
        init_lines.append("(handfree)")
    else:
        init_lines.append(f"(holding {state.holding})")
    for src, dst in _all_location_pairs(locations, coords):
        init_lines.append(f"(adjacent {src} {dst})")
    for loc in env.service_locations:
        init_lines.append(f"(service-loc {loc})")
    for loc in env.wash_ready_locations:
        init_lines.append(f"(wash-ready-loc {loc})")
    if env.station_wash in env.location_index:
        init_lines.append(f"(sink-loc {env.station_wash})")
    if env.station_water in env.location_index:
        init_lines.append(f"(water-loc {env.station_water})")
    if env.station_coffee in env.location_index:
        init_lines.append(f"(coffee-loc {env.station_coffee})")
    if env.station_fruit in env.location_index:
        init_lines.append(f"(fruit-loc {env.station_fruit})")

    for name, obj in state.objects.items():
        if obj.kind == "mug":
            init_lines.append(f"(mug-kind {name})")
        elif obj.kind == "glass":
            init_lines.append(f"(glass-kind {name})")
        elif obj.kind == "bowl":
            init_lines.append(f"(bowl-kind {name})")
        if obj.location != "__held__":
            init_lines.append(f"(at {name} {obj.location})")
        if not obj.dirty:
            init_lines.append(f"(clean {name})")
        if obj.contents == "empty":
            init_lines.append(f"(empty {name})")
        elif obj.contents == "water":
            init_lines.append(f"(water {name})")
        elif obj.contents == "coffee":
            init_lines.append(f"(coffee {name})")
        elif obj.contents == "fruit":
            init_lines.append(f"(fruit {name})")

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
        f"  (:domain restaurant_symbolic)\n"
        f"  (:objects\n{object_block}  )\n"
        f"  (:init\n    {init_text}\n  )\n"
        f"  (:goal\n    (and\n      {goal_text}\n    )\n  )\n"
        f")\n"
    )


def solve_restaurant_task_with_fd(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    planner_path: Path,
    domain_path: Path,
    search: str = "astar(lmcut())",
    extra_goal_clauses: Sequence[str] | None = None,
    timeout_s: float = 30.0,
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
            plan_path = run_planner(planner_path, domain_path, problem_path, search, tmpdir)
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
            if elapsed < timeout_s:
                return PlannerResult(False, [], float("inf"), elapsed, error=str(exc))
            return PlannerResult(False, [], float("inf"), elapsed, error=f"timeout: {exc}")


def apply_planner_action(state: RestaurantPlannerState, action: Tuple[str, List[str]]) -> None:
    name, args = action
    if name == "move":
        if len(args) >= 2:
            state.agent_location = args[1]
        return
    if name == "pick":
        obj_name = args[0]
        state.holding = obj_name
        state.objects[obj_name].location = "__held__"
        return
    if name == "place":
        obj_name, loc = args[0], args[1]
        obj = state.objects[obj_name]
        prev_contents = obj.contents
        obj.location = loc
        state.holding = None
        if loc in {"sink", "bus_tub"}:
            if prev_contents != "empty":
                obj.dirty = True
            obj.contents = "empty"
        elif loc in {"pass_counter", "table_left", "table_right"} and prev_contents != "empty":
            obj.dirty = True
        return
    if name == "wash":
        obj_name = args[0]
        obj = state.objects[obj_name]
        obj.dirty = False
        obj.contents = "empty"
        return
    if name == "fill-water":
        state.objects[args[0]].contents = "water"
        return
    if name == "brew-coffee":
        state.objects[args[0]].contents = "coffee"
        return
    if name == "fill-fruit":
        state.objects[args[0]].contents = "fruit"
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
