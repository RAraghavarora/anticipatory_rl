from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .planner import GoalCandidate, PlanResult
from .world import (
    WASH_READY_LOCATIONS,
    PaperRestaurantTask,
    RestaurantWorldGenerator,
    RestaurantWorldState,
)


def expand_goal_candidates(
    generator: RestaurantWorldGenerator,
    state: RestaurantWorldState,
    task: PaperRestaurantTask,
    base_candidates: Sequence[GoalCandidate],
    *,
    candidate_goal_limit: int = 24,
) -> List[GoalCandidate]:
    expanded: List[GoalCandidate] = []
    seen: Set[Tuple[object, ...]] = set()

    def add(candidate: GoalCandidate) -> None:
        sig = candidate.signature()
        if sig in seen:
            return
        seen.add(sig)
        expanded.append(candidate)

    for candidate in base_candidates:
        add(candidate)
        if len(expanded) >= candidate_goal_limit:
            return expanded[:candidate_goal_limit]

        if task.task_type == "clear_containers" and task.target_location is not None:
            object_names = [name for name in state.objects_at(task.target_location)]
            if not object_names:
                continue
            object_names = object_names[:3]
            destination_lists: List[List[str]] = []
            for object_name in object_names:
                obj = state.objects[object_name]
                options = generator.nearby_support_locations(
                    state,
                    task.target_location,
                    kind=obj.kind,
                )
                if obj.dirty or obj.contents != "empty":
                    for location in WASH_READY_LOCATIONS:
                        if location not in options and location != task.target_location:
                            options.append(location)
                deduped: List[str] = []
                seen_locations: Set[str] = set()
                for location in options:
                    if location == task.target_location or location in seen_locations:
                        continue
                    seen_locations.add(location)
                    deduped.append(location)
                destination_lists.append(deduped[:4] if deduped else [])
            for assignment in product(*destination_lists):
                pairs = tuple(sorted(zip(object_names, assignment)))
                add(
                    GoalCandidate(
                        bound_object=candidate.bound_object,
                        extra_placements=pairs,
                        note="clear_augment",
                    )
                )
                if len(expanded) >= candidate_goal_limit:
                    return expanded[:candidate_goal_limit]
            continue

        if task.task_type == "pick_place" and task.target_location is not None:
            anchor = task.target_location
            nearby = _related_objects(generator, state, anchor, exclude={task.object_name})
            for object_name in nearby[:2]:
                obj = state.objects[object_name]
                for destination in generator.nearby_support_locations(state, anchor, kind=obj.kind)[:2]:
                    add(
                        GoalCandidate(
                            bound_object=candidate.bound_object,
                            extra_placements=((object_name, destination),),
                            note="pick_place_augment",
                        )
                    )
                    if len(expanded) >= candidate_goal_limit:
                        return expanded[:candidate_goal_limit]

    return expanded[:candidate_goal_limit]


def _related_objects(
    generator: RestaurantWorldGenerator,
    state: RestaurantWorldState,
    anchor_location: str,
    *,
    exclude: Set[str],
) -> List[str]:
    anchor_room = generator.config.location_room(anchor_location)
    ranked = [
        (
            generator.location_distance(anchor_location, obj.location),
            name,
        )
        for name, obj in state.objects.items()
        if name not in exclude and generator.config.location_room(obj.location) == anchor_room
    ]
    ranked.sort()
    return [name for _, name in ranked]
