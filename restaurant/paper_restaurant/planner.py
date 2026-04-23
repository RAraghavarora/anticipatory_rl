from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .world import (
    CONTENTS,
    OBJECT_KINDS,
    SERVICE_LOCATIONS,
    WASH_READY_LOCATIONS,
    PaperRestaurantTask,
    RestaurantObjectState,
    RestaurantWorldConfig,
    RestaurantWorldGenerator,
    RestaurantWorldState,
)


@dataclass(frozen=True)
class GoalCandidate:
    bound_object: Optional[str] = None
    extra_placements: Tuple[Tuple[str, str], ...] = ()
    note: str = "baseline"

    def signature(self) -> Tuple[object, ...]:
        return (self.bound_object, self.extra_placements, self.note)


@dataclass
class PlanAction:
    name: str
    args: Tuple[str, ...]
    cost: float


@dataclass
class PlanResult:
    cost: float
    actions: List[PlanAction]
    final_state: RestaurantWorldState
    moved_objects: Tuple[str, ...]
    candidate: GoalCandidate


class FastDownwardRestaurantPlanner:
    """Paper-style symbolic planner with exact macro-action cost accounting.

    The restaurant PDDL/domain is not published in the paper, so this class uses
    an exact symbolic solver over macro actions while preserving the paper-style
    cost model: shortest-path movement through an occupancy map plus fixed action
    costs for manipulation, filling, clearing, and washing.
    """

    def __init__(self, config: RestaurantWorldConfig) -> None:
        self.config = config
        self.generator = RestaurantWorldGenerator(config)
        self._distance_cache: Dict[Tuple[str, str], int] = {}
        self._plan_cache: Dict[Tuple[Tuple[object, ...], str, Tuple[object, ...]], PlanResult] = {}

    def plan_for_task(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
    ) -> PlanResult:
        candidates = self.default_goal_candidates(state, task)
        results = [self.plan_to_candidate(state, task, candidate) for candidate in candidates]
        return min(results, key=lambda result: (result.cost, len(result.actions), result.final_state.signature()))

    def default_goal_candidates(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
    ) -> List[GoalCandidate]:
        if task.task_type in {"serve_water", "make_coffee", "make_fruit_bowl", "wash_objects"}:
            names = self.generator.candidate_object_names(state, task)
            return [GoalCandidate(bound_object=name, note=f"bind:{name}") for name in names]
        if task.task_type == "pick_place":
            return [GoalCandidate(bound_object=task.object_name, note="baseline")]
        if task.task_type == "clear_containers":
            return [GoalCandidate(note="baseline")]
        raise ValueError(f"Unsupported task type: {task.task_type}")

    def plan_to_candidate(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
        candidate: GoalCandidate,
    ) -> PlanResult:
        cache_key = (state.signature(), task.summary(), candidate.signature())
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return PlanResult(
                cost=cached.cost,
                actions=list(cached.actions),
                final_state=cached.final_state.clone(),
                moved_objects=tuple(cached.moved_objects),
                candidate=candidate,
            )

        working = state.clone()
        actions: List[PlanAction] = []
        moved: List[str] = []

        if self.generator.task_satisfied(working, task) and not candidate.extra_placements:
            result = PlanResult(0.0, [], working, tuple(), candidate)
            self._plan_cache[cache_key] = result
            return result

        if task.task_type == "serve_water":
            assert task.target_location is not None and candidate.bound_object is not None
            self._serve_contents(working, candidate.bound_object, "water", task.target_location, actions, moved)
        elif task.task_type == "make_coffee":
            assert task.target_location is not None and candidate.bound_object is not None
            self._serve_contents(working, candidate.bound_object, "coffee", task.target_location, actions, moved)
        elif task.task_type == "make_fruit_bowl":
            assert task.target_location is not None and candidate.bound_object is not None
            self._serve_contents(working, candidate.bound_object, "apple", task.target_location, actions, moved)
        elif task.task_type == "wash_objects":
            assert candidate.bound_object is not None
            self._wash_for_task(working, candidate.bound_object, actions, moved)
        elif task.task_type == "pick_place":
            assert task.object_name is not None and task.target_location is not None
            self._relocate_for_goal(
                working,
                task.object_name,
                task.target_location,
                actions,
                moved,
                process_for_destination=False,
            )
        elif task.task_type == "clear_containers":
            assert task.target_location is not None
            desired = dict(candidate.extra_placements)
            for object_name in list(working.objects_at(task.target_location)):
                destination = desired.get(object_name)
                if destination is None:
                    destination = self._best_default_clear_destination(working, task.target_location, object_name)
                self._relocate_for_goal(
                    working,
                    object_name,
                    destination,
                    actions,
                    moved,
                    process_for_destination=True,
                )
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")

        for object_name, destination in candidate.extra_placements:
            if task.task_type == "clear_containers" and working.objects[object_name].location == destination:
                continue
            if task.task_type == "pick_place" and object_name == task.object_name:
                continue
            self._relocate_for_goal(
                working,
                object_name,
                destination,
                actions,
                moved,
                process_for_destination=True,
            )

        result = PlanResult(
            cost=sum(action.cost for action in actions),
            actions=actions,
            final_state=working,
            moved_objects=tuple(dict.fromkeys(moved)),
            candidate=candidate,
        )
        self._plan_cache[cache_key] = PlanResult(
            cost=result.cost,
            actions=list(result.actions),
            final_state=result.final_state.clone(),
            moved_objects=result.moved_objects,
            candidate=candidate,
        )
        return result

    def _serve_contents(
        self,
        state: RestaurantWorldState,
        object_name: str,
        desired_contents: str,
        target_location: str,
        actions: List[PlanAction],
        moved: List[str],
    ) -> None:
        obj = state.objects[object_name]
        if obj.location == target_location and obj.contents == desired_contents:
            return
        self._pick_object(state, object_name, actions, moved)
        self._ensure_clean_and_empty(state, object_name, actions)
        if desired_contents == "water":
            self._fill_water_held(state, object_name, actions)
        elif desired_contents == "coffee":
            self._brew_coffee_held(state, object_name, actions)
        elif desired_contents == "apple":
            self._fill_fruit_held(state, object_name, actions)
        else:
            raise ValueError(f"Unsupported contents goal: {desired_contents}")
        self._place_held(state, target_location, actions)

    def _wash_for_task(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
        moved: List[str],
    ) -> None:
        obj = state.objects[object_name]
        if obj.contents == "empty" and not obj.dirty and obj.location in WASH_READY_LOCATIONS:
            return
        self._pick_object(state, object_name, actions, moved)
        self._ensure_clean_and_empty(state, object_name, actions)
        destination = self._best_wash_ready_destination(state, object_name)
        self._place_held(state, destination, actions)

    def _relocate_for_goal(
        self,
        state: RestaurantWorldState,
        object_name: str,
        destination: str,
        actions: List[PlanAction],
        moved: List[str],
        *,
        process_for_destination: bool,
    ) -> None:
        obj = state.objects[object_name]
        if obj.location == destination and (not process_for_destination or self._destination_prepared(obj, destination)):
            return
        self._pick_object(state, object_name, actions, moved)
        if process_for_destination and destination in WASH_READY_LOCATIONS:
            self._ensure_clean_and_empty(state, object_name, actions)
        elif process_for_destination and destination == "sink" and state.objects[object_name].contents != "empty":
            self._move_robot(state, "sink", actions)
            self._clear_held(state, object_name, actions)
        self._place_held(state, destination, actions)

    def _destination_prepared(self, obj: RestaurantObjectState, destination: str) -> bool:
        if destination in WASH_READY_LOCATIONS:
            return obj.contents == "empty" and not obj.dirty
        if destination == "sink":
            return obj.contents == "empty"
        return True

    def _ensure_clean_and_empty(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        held = state.holding
        if held != object_name:
            raise RuntimeError(f"Expected {object_name} to be held, found {held!r}")
        obj = state.objects[object_name]
        if obj.contents != "empty":
            self._move_robot(state, "sink", actions)
            self._clear_held(state, object_name, actions)
        if obj.dirty:
            self._move_robot(state, "sink", actions)
            self._wash_held(state, object_name, actions)

    def _pick_object(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
        moved: List[str],
    ) -> None:
        if state.holding == object_name:
            return
        if state.holding is not None:
            raise RuntimeError("Planner expects a hand-empty intermediate state.")
        obj = state.objects[object_name]
        self._move_robot(state, obj.location, actions)
        state.holding = object_name
        actions.append(PlanAction("pick", (object_name, obj.location), self.config.pick_cost))
        moved.append(object_name)

    def _place_held(
        self,
        state: RestaurantWorldState,
        destination: str,
        actions: List[PlanAction],
    ) -> None:
        if state.holding is None:
            raise RuntimeError("No held object to place.")
        object_name = state.holding
        self._move_robot(state, destination, actions)
        state.objects[object_name].location = destination
        state.holding = None
        actions.append(PlanAction("place", (object_name, destination), self.config.place_cost))

    def _clear_held(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        obj = state.objects[object_name]
        obj.contents = "empty"
        obj.dirty = True
        actions.append(PlanAction("clear", (object_name,), self.config.clear_cost))

    def _wash_held(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        obj = state.objects[object_name]
        obj.dirty = False
        obj.contents = "empty"
        actions.append(PlanAction("wash", (object_name,), self.config.wash_cost))

    def _fill_water_held(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        self._move_robot(state, "water_station", actions)
        obj = state.objects[object_name]
        obj.contents = "water"
        actions.append(PlanAction("fill_water", (object_name,), self.config.fill_cost))

    def _brew_coffee_held(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        self._move_robot(state, "coffee_machine", actions)
        obj = state.objects[object_name]
        obj.contents = "coffee"
        actions.append(PlanAction("make_coffee", (object_name,), self.config.brew_cost))

    def _fill_fruit_held(
        self,
        state: RestaurantWorldState,
        object_name: str,
        actions: List[PlanAction],
    ) -> None:
        self._move_robot(state, "fruit_station", actions)
        obj = state.objects[object_name]
        obj.contents = "apple"
        actions.append(PlanAction("fill_fruit", (object_name,), self.config.fruit_cost))

    def _move_robot(
        self,
        state: RestaurantWorldState,
        destination: str,
        actions: List[PlanAction],
    ) -> None:
        if state.robot_location == destination:
            return
        src = state.robot_location
        dist = self.shortest_distance(src, destination)
        state.robot_location = destination
        actions.append(
            PlanAction(
                "move",
                (src, destination),
                float(dist) * self.config.move_cost_per_step,
            )
        )

    def _best_wash_ready_destination(self, state: RestaurantWorldState, object_name: str) -> str:
        obj = state.objects[object_name]
        candidates = [
            location
            for location in self.generator.support_locations_for_kind(obj.kind)
            if location in WASH_READY_LOCATIONS
        ]
        if not candidates:
            candidates = list(WASH_READY_LOCATIONS)
        return min(candidates, key=lambda location: self.shortest_distance(state.robot_location, location))

    def _best_default_clear_destination(
        self,
        state: RestaurantWorldState,
        target_location: str,
        object_name: str,
    ) -> str:
        obj = state.objects[object_name]
        candidates = [
            location
            for location in self.generator.nearby_support_locations(state, target_location, kind=obj.kind)
            if location != target_location
        ]
        if not candidates:
            candidates = [loc for loc in self.config.containers if loc != target_location]
        best_score = None
        best_location = candidates[0]
        for location in candidates:
            current = state.robot_location
            score = self.shortest_distance(current, obj.location) + self.shortest_distance(obj.location, location)
            if location in WASH_READY_LOCATIONS and (obj.dirty or obj.contents != "empty"):
                score += self.shortest_distance(obj.location, "sink")
                score += 2
            if best_score is None or score < best_score:
                best_score = score
                best_location = location
        return best_location

    def shortest_distance(self, src_location: str, dst_location: str) -> int:
        key = (src_location, dst_location)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        src = self.config.container(src_location).coord
        dst = self.config.container(dst_location).coord
        door_y = self.config.container("pass_counter").coord[1]
        heap: List[Tuple[int, Tuple[int, int]]] = [(0, src)]
        seen: Dict[Tuple[int, int], int] = {src: 0}
        passable = set(self.config.passable_cells)
        while heap:
            cost, cell = heapq.heappop(heap)
            if cell == dst:
                self._distance_cache[key] = cost
                self._distance_cache[(dst_location, src_location)] = cost
                return cost
            if cost > seen.get(cell, cost):
                continue
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                nxt = (nx, ny)
                if nxt not in passable:
                    continue
                if {x, nx} == {7, 8} and y != door_y and ny != door_y:
                    continue
                new_cost = cost + 1
                if new_cost < seen.get(nxt, 10**9):
                    seen[nxt] = new_cost
                    heapq.heappush(heap, (new_cost, nxt))
        raise RuntimeError(f"No path from {src_location} to {dst_location}")
