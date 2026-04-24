from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from anticipatory_rl.envs.restaurant.pddl_domain import (
    PDDL_ACTION_COSTS,
    get_pddl_cost,
)


ROOMS: Tuple[str, ...] = ("kitchen", "serving_room")
CONTENTS: Tuple[str, ...] = ("empty", "water", "coffee", "spread", "apple")
TASK_TYPES: Tuple[str, ...] = (
    "serve_water",
    "make_coffee",
    "make_fruit_bowl",
    "clear_containers",
    "wash_objects",
    "pick_place",
)
OBJECT_KINDS: Tuple[str, ...] = (
    "cup",
    "mug",
    "jar",
    "coffeegrinds",
    "water",
    "bread",
    "knife",
    "plate",
    "bowl",
    "spread",
    "apple",
)

WASH_READY_LOCATIONS: Tuple[str, ...] = ("dish_rack", "prep_counter", "kitchen_counter")
SERVICE_LOCATIONS: Tuple[str, ...] = ("pass_counter", "table_left", "table_center", "table_right")
PICK_PLACE_TARGETS: Tuple[str, ...] = (
    "kitchen_counter",
    "prep_counter",
    "pantry_shelf",
    "service_shelf",
    "pass_counter",
    "table_left",
    "table_center",
    "table_right",
)


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    category: str
    room: str
    coord: Tuple[int, int]
    wash_source: bool = False
    water_source: bool = False
    coffee_source: bool = False
    fruit_source: bool = False


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str
    category: str
    preferred_service_locations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperRestaurantTask:
    task_type: str
    target_location: Optional[str] = None
    target_kind: Optional[str] = None
    object_name: Optional[str] = None
    weight: float = 1.0

    def summary(self) -> str:
        if self.task_type == "wash_objects":
            return f"{self.task_type}:{self.target_kind}"
        if self.task_type == "pick_place":
            return f"{self.task_type}:{self.object_name}->{self.target_location}"
        return f"{self.task_type}:{self.target_location}"


@dataclass
class RestaurantObjectState:
    name: str
    kind: str
    category: str
    location: str
    dirty: bool = False
    contents: str = "empty"

    def clone(self) -> "RestaurantObjectState":
        return RestaurantObjectState(
            name=self.name,
            kind=self.kind,
            category=self.category,
            location=self.location,
            dirty=bool(self.dirty),
            contents=self.contents,
        )


@dataclass
class RestaurantWorldState:
    robot_location: str
    holding: Optional[str]
    objects: Dict[str, RestaurantObjectState]

    def clone(self) -> "RestaurantWorldState":
        return RestaurantWorldState(
            robot_location=self.robot_location,
            holding=self.holding,
            objects={name: obj.clone() for name, obj in self.objects.items()},
        )

    def signature(self) -> Tuple[object, ...]:
        return (
            self.robot_location,
            self.holding,
            tuple(
                (
                    name,
                    obj.kind,
                    obj.location,
                    int(obj.dirty),
                    obj.contents,
                )
                for name, obj in sorted(self.objects.items())
            ),
        )

    def objects_at(self, location: str) -> List[str]:
        return [name for name, obj in self.objects.items() if obj.location == location]

    def object_state(self, name: str) -> RestaurantObjectState:
        return self.objects[name]


@dataclass
class RestaurantTaskLibrary:
    tasks: List[PaperRestaurantTask]
    weights: Dict[str, float]

    def normalized_weights(self) -> List[float]:
        vals = [max(float(self.weights.get(task.summary(), task.weight)), 0.0) for task in self.tasks]
        total = sum(vals)
        if total <= 0.0:
            return [1.0 / max(1, len(self.tasks)) for _ in self.tasks]
        return [value / total for value in vals]

    def sample_task(self, rng: random.Random) -> PaperRestaurantTask:
        probs = self.normalized_weights()
        return rng.choices(self.tasks, weights=probs, k=1)[0]


@dataclass(frozen=True)
class RestaurantWorldConfig:
    width: int
    height: int
    kitchen_cells: Tuple[Tuple[int, int], ...]
    serving_cells: Tuple[Tuple[int, int], ...]
    passable_cells: Tuple[Tuple[int, int], ...]
    containers: Dict[str, ContainerSpec]
    object_specs: Dict[str, ObjectSpec]
    move_cost_per_step: float = 10.0
    pick_cost: float = 40.0
    place_cost: float = 40.0
    clear_cost: float = 35.0
    wash_cost: float = 45.0
    fill_cost: float = 35.0
    brew_cost: float = 50.0
    fruit_cost: float = 35.0
    task_count_range: Tuple[int, int] = (50, 100)
    service_task_bias: Dict[str, float] = field(default_factory=dict)

    def container(self, name: str) -> ContainerSpec:
        return self.containers[name]

    def location_room(self, location: str) -> str:
        return self.containers[location].room

    def nearest_service_locations(self, room: str) -> Tuple[str, ...]:
        if room == "kitchen":
            return ("pass_counter",)
        return tuple(loc for loc in SERVICE_LOCATIONS if self.containers[loc].room == room)

    @staticmethod
    def sample(rng: random.Random) -> "RestaurantWorldConfig":
        width = 16
        height = 8
        kitchen_cells = tuple((x, y) for x in range(0, 8) for y in range(height))
        serving_cells = tuple((x, y) for x in range(8, width) for y in range(height))
        door_y = rng.choice((2, 3, 4, 5))
        passable = tuple((x, y) for x in range(width) for y in range(height))

        containers = {
            "kitchen_counter": ContainerSpec("kitchen_counter", "counter", "kitchen", (1, 1)),
            "prep_counter": ContainerSpec("prep_counter", "prep_counter", "kitchen", (2, 2)),
            "sink": ContainerSpec("sink", "sink", "kitchen", (1, 5), wash_source=True),
            "dish_rack": ContainerSpec("dish_rack", "dish_rack", "kitchen", (3, 5), wash_source=True),
            "water_station": ContainerSpec("water_station", "water_station", "kitchen", (4, 1), water_source=True),
            "coffee_machine": ContainerSpec("coffee_machine", "coffee_machine", "kitchen", (5, 1), coffee_source=True),
            "fruit_station": ContainerSpec("fruit_station", "fruit_station", "kitchen", (6, 1), fruit_source=True),
            "pantry_shelf": ContainerSpec("pantry_shelf", "pantry_shelf", "kitchen", (6, 5)),
            "pass_counter": ContainerSpec("pass_counter", "pass_counter", "serving_room", (8, door_y)),
            "service_shelf": ContainerSpec("service_shelf", "service_shelf", "serving_room", (10, 1)),
            "host_stand": ContainerSpec("host_stand", "host_stand", "serving_room", (10, 6)),
            "bus_tub": ContainerSpec("bus_tub", "bus_tub", "serving_room", (9, 5)),
            "table_left": ContainerSpec("table_left", "table", "serving_room", (13, 1)),
            "table_center": ContainerSpec("table_center", "table", "serving_room", (13, 3)),
            "table_right": ContainerSpec("table_right", "table", "serving_room", (13, 5)),
        }

        object_specs = {
            "cup_small": ObjectSpec("cup_small", "cup", "cup", ("table_left", "table_center")),
            "cup_large": ObjectSpec("cup_large", "cup", "cup", ("table_center", "table_right")),
            "mug_red": ObjectSpec("mug_red", "mug", "mug", ("table_left", "table_center")),
            "mug_blue": ObjectSpec("mug_blue", "mug", "mug", ("table_center", "table_right")),
            "jar_sugar": ObjectSpec("jar_sugar", "jar", "jar", ("coffee_machine",)),
            "jar_coffee": ObjectSpec("jar_coffee", "jar", "jar", ("pantry_shelf",)),
            "coffeegrinds": ObjectSpec("coffeegrinds", "coffeegrinds", "coffeegrinds", ("coffee_machine",)),
            "water_pitcher": ObjectSpec("water_pitcher", "water", "water", ("kitchen_counter",)),
            "bread_loaf": ObjectSpec("bread_loaf", "bread", "bread", ("pantry_shelf",)),
            "bread_slice": ObjectSpec("bread_slice", "bread", "bread", ("pantry_shelf",)),
            "knife_chef": ObjectSpec("knife_chef", "knife", "knife", ("kitchen_counter",)),
            "knife_butter": ObjectSpec("knife_butter", "knife", "knife", ("kitchen_counter",)),
            "plate_dinner": ObjectSpec("plate_dinner", "plate", "plate", ("table_left",)),
            "plate_side": ObjectSpec("plate_side", "plate", "plate", ("table_right",)),
            "bowl_small": ObjectSpec("bowl_small", "bowl", "bowl", ("table_center", "table_right")),
            "bowl_large": ObjectSpec("bowl_large", "bowl", "bowl", ("table_left", "table_center")),
            "spread_butter": ObjectSpec("spread_butter", "spread", "spread", ("pantry_shelf",)),
            "spread_jam": ObjectSpec("spread_jam", "spread", "spread", ("pantry_shelf",)),
            "apple_red": ObjectSpec("apple_red", "apple", "apple", ("fruit_station",)),
            "apple_green": ObjectSpec("apple_green", "apple", "apple", ("pantry_shelf",)),
        }

        service_bias = {
            "serve_water": 0.24,
            "make_coffee": 0.20,
            "make_fruit_bowl": 0.14,
            "clear_containers": 0.16,
            "wash_objects": 0.10,
            "pick_place": 0.16,
        }

        # Always use PDDL costs per author clarification
        return RestaurantWorldConfig(
            width=width,
            height=height,
            kitchen_cells=kitchen_cells,
            serving_cells=serving_cells,
            passable_cells=passable,
            containers=containers,
            object_specs=object_specs,
            move_cost_per_step=10.0,
            pick_cost=get_pddl_cost("pick"),
            place_cost=get_pddl_cost("place"),
            clear_cost=get_pddl_cost("clear"),
            wash_cost=get_pddl_cost("wash"),
            fill_cost=get_pddl_cost("fill"),
            brew_cost=get_pddl_cost("make-coffee"),
            fruit_cost=get_pddl_cost("make-fruit-bowl"),
            task_count_range=(50, 100),
            service_task_bias=service_bias,
        )


class RestaurantWorldGenerator:
    def __init__(self, config: RestaurantWorldConfig) -> None:
        self.config = config

    def sample_initial_state(self, rng: random.Random) -> RestaurantWorldState:
        placements: Dict[str, RestaurantObjectState] = {}
        for name, spec in self.config.object_specs.items():
            location = self._sample_initial_location(spec, rng)
            dirty, contents = self._sample_initial_status(spec, location, rng)
            placements[name] = RestaurantObjectState(
                name=name,
                kind=spec.kind,
                category=spec.category,
                location=location,
                dirty=dirty,
                contents=contents,
            )
        return RestaurantWorldState(
            robot_location="kitchen_counter",
            holding=None,
            objects=placements,
        )

    def sample_task_library(
        self,
        rng: random.Random,
        *,
        count: Optional[int] = None,
    ) -> RestaurantTaskLibrary:
        feasible = self._enumerate_feasible_tasks()
        desired = count or rng.randint(
            self.config.task_count_range[0],
            self.config.task_count_range[1],
        )
        rng.shuffle(feasible)
        tasks = feasible[: min(desired, len(feasible))]
        weights = self._sample_task_weights(tasks, rng)
        return RestaurantTaskLibrary(tasks=tasks, weights=weights)

    def sample_task_sequence(
        self,
        rng: random.Random,
        task_library: RestaurantTaskLibrary,
        length: int,
    ) -> List[PaperRestaurantTask]:
        return [task_library.sample_task(rng) for _ in range(max(0, length))]

    def task_satisfied(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
    ) -> bool:
        if task.task_type == "serve_water":
            assert task.target_location is not None
            return any(
                obj.location == task.target_location
                and obj.kind in {"cup", "mug"}
                and obj.contents == "water"
                for obj in state.objects.values()
            )
        if task.task_type == "make_coffee":
            assert task.target_location is not None
            return any(
                obj.location == task.target_location
                and obj.kind in {"cup", "mug"}
                and obj.contents == "coffee"
                for obj in state.objects.values()
            )
        if task.task_type == "make_fruit_bowl":
            assert task.target_location is not None
            return any(
                obj.location == task.target_location
                and obj.kind == "bowl"
                and obj.contents == "apple"
                for obj in state.objects.values()
            )
        if task.task_type == "clear_containers":
            assert task.target_location is not None
            return all(obj.location != task.target_location for obj in state.objects.values())
        if task.task_type == "wash_objects":
            assert task.target_kind is not None
            return any(
                obj.kind == task.target_kind
                and not obj.dirty
                and obj.contents == "empty"
                and obj.location in WASH_READY_LOCATIONS
                for obj in state.objects.values()
            )
        if task.task_type == "pick_place":
            assert task.object_name is not None and task.target_location is not None
            obj = state.objects[task.object_name]
            return obj.location == task.target_location and state.holding is None
        raise ValueError(f"Unsupported task type: {task.task_type}")

    def candidate_object_names(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
    ) -> List[str]:
        if task.task_type == "serve_water":
            return [name for name, obj in state.objects.items() if obj.kind in {"cup", "mug"}]
        if task.task_type == "make_coffee":
            return [name for name, obj in state.objects.items() if obj.kind == "mug"]
        if task.task_type == "make_fruit_bowl":
            return [name for name, obj in state.objects.items() if obj.kind == "bowl"]
        if task.task_type == "wash_objects":
            assert task.target_kind is not None
            return [name for name, obj in state.objects.items() if obj.kind == task.target_kind]
        if task.task_type == "pick_place":
            assert task.object_name is not None
            return [task.object_name]
        if task.task_type == "clear_containers":
            assert task.target_location is not None
            return [name for name, obj in state.objects.items() if obj.location == task.target_location]
        raise ValueError(f"Unsupported task type: {task.task_type}")

    def room_of_object(self, state: RestaurantWorldState, object_name: str) -> str:
        return self.config.location_room(state.objects[object_name].location)

    def support_locations_for_kind(self, kind: str) -> Tuple[str, ...]:
        if kind == "mug":
            return ("dish_rack", "prep_counter", "pass_counter", "kitchen_counter")
        elif kind == "cup":
            return ("dish_rack", "pass_counter", "service_shelf", "kitchen_counter")
        if kind == "bowl":
            return ("dish_rack", "fruit_station", "pass_counter", "service_shelf")
        if kind in {"plate", "spoon", "fork", "knife"}:
            return ("dish_rack", "service_shelf", "kitchen_counter", "pass_counter")
        return ("kitchen_counter", "prep_counter", "service_shelf", "pantry_shelf", "pass_counter")

    def nearby_support_locations(
        self,
        state: RestaurantWorldState,
        anchor_location: str,
        *,
        kind: Optional[str] = None,
    ) -> List[str]:
        base = list(self.support_locations_for_kind(kind or "plate"))
        room = self.config.location_room(anchor_location)
        same_room = [loc for loc in base if self.config.location_room(loc) == room and loc != anchor_location]
        other_room = [loc for loc in base if self.config.location_room(loc) != room and loc != anchor_location]
        ranked = same_room + other_room
        ranked.sort(key=lambda loc: self.location_distance(anchor_location, loc))
        seen: set[str] = set()
        unique: List[str] = []
        for loc in ranked:
            if loc not in seen:
                seen.add(loc)
                unique.append(loc)
        return unique

    def location_distance(self, src: str, dst: str) -> int:
        return abs(self.config.container(src).coord[0] - self.config.container(dst).coord[0]) + abs(
            self.config.container(src).coord[1] - self.config.container(dst).coord[1]
        )

    def _sample_initial_location(self, spec: ObjectSpec, rng: random.Random) -> str:
        if spec.kind == "mug":
            candidates = ("dish_rack", "kitchen_counter", "prep_counter", "table_left", "table_center")
        elif spec.kind == "cup":
            candidates = ("dish_rack", "service_shelf", "table_left", "table_center", "table_right")
        elif spec.kind == "bowl":
            candidates = ("dish_rack", "fruit_station", "table_center", "table_right")
        elif spec.kind in {"plate", "spoon", "fork"}:
            candidates = ("service_shelf", "table_left", "table_center", "table_right", "dish_rack")
        else:
            candidates = ("kitchen_counter", "prep_counter", "service_shelf", "pantry_shelf", "host_stand")
        return rng.choice(candidates)

    def _sample_initial_status(
        self,
        spec: ObjectSpec,
        location: str,
        rng: random.Random,
    ) -> Tuple[bool, str]:
        if location in {"dish_rack", "prep_counter", "kitchen_counter", "pantry_shelf"}:
            return False, "empty"
        if location == "sink":
            return True, "empty"
        if location == "bus_tub":
            return True, "empty"
        if spec.kind == "mug":
            contents = rng.choices(["empty", "water", "coffee"], weights=[0.55, 0.20, 0.25], k=1)[0]
        elif spec.kind == "cup":
            contents = rng.choices(["empty", "water"], weights=[0.75, 0.25], k=1)[0]
        elif spec.kind == "bowl":
            contents = rng.choices(["empty", "apple"], weights=[0.70, 0.30], k=1)[0]
        else:
            contents = "empty"
        dirty = contents != "empty" or rng.random() < 0.15
        return dirty, contents

    def _enumerate_feasible_tasks(self) -> List[PaperRestaurantTask]:
        tasks: List[PaperRestaurantTask] = []
        for location in SERVICE_LOCATIONS:
            tasks.append(PaperRestaurantTask("serve_water", target_location=location))
            tasks.append(PaperRestaurantTask("make_coffee", target_location=location))
            tasks.append(PaperRestaurantTask("make_fruit_bowl", target_location=location))
            tasks.append(PaperRestaurantTask("clear_containers", target_location=location))
        washable = sorted({spec.kind for spec in self.config.object_specs.values() if spec.kind in {"mug", "cup", "bowl", "plate"}})
        for kind in washable:
            tasks.append(PaperRestaurantTask("wash_objects", target_kind=kind))
        for name, spec in self.config.object_specs.items():
            candidate_targets = [
                loc
                for loc in PICK_PLACE_TARGETS
                if loc not in spec.preferred_service_locations
            ]
            for location in candidate_targets[:6]:
                tasks.append(PaperRestaurantTask("pick_place", object_name=name, target_location=location))
        return tasks

    def _sample_task_weights(
        self,
        tasks: Sequence[PaperRestaurantTask],
        rng: random.Random,
    ) -> Dict[str, float]:
        # Always use uniform weights per author clarification
        return {task.summary(): 1.0 for task in tasks}
