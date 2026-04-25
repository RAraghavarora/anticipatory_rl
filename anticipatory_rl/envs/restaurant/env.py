"""Configurable symbolic restaurant environment for continual-task RL."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from gymnasium import Env, spaces

from .pddl_domain import get_pddl_cost


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "restaurant" / "restaurant_symbolic.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


DEFAULT_LOCATIONS: Tuple[str, ...] = (
    "kitchen_counter",
    "coffee_machine",
    "water_station",
    "fruit_station",
    "dish_rack",
    "sink",
    "pass_counter",
    "table_left",
    "bus_tub",
    "table_right",
)
DEFAULT_LOCATION_COORDS: Dict[str, Tuple[int, int]] = {
    "kitchen_counter": (0, 0),
    "coffee_machine": (1, 0),
    "water_station": (1, 1),
    "fruit_station": (1, 2),
    "dish_rack": (0, 1),
    "sink": (0, 2),
    "pass_counter": (2, 1),
    "table_left": (3, 0),
    "bus_tub": (3, 1),
    "table_right": (3, 2),
}
DEFAULT_SERVICE_LOCATIONS: Tuple[str, ...] = ("pass_counter", "table_left", "table_right")
DEFAULT_WASH_READY_LOCATIONS: Tuple[str, ...] = ("dish_rack", "kitchen_counter")
DEFAULT_OBJECT_KINDS: Tuple[str, ...] = ("cup", "mug", "jar", "coffeegrinds", "water", "bread", "knife", "plate", "bowl", "spread", "apple")
DEFAULT_CONTENTS: Tuple[str, ...] = ("empty", "water", "coffee")
DEFAULT_TASK_TYPES: Tuple[str, ...] = (
    "serve_water",
    "make_coffee",
    "make_fruit_bowl",
    "clear_containers",
    "wash_objects",
    "pick_place",
)
DEFAULT_OBJECT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("cup_small", "cup"),
    ("cup_large", "cup"),
    ("mug_red", "mug"),
    ("mug_blue", "mug"),
    ("jar_sugar", "jar"),
    ("jar_coffee", "jar"),
    ("coffeegrinds", "coffeegrinds"),
    ("water_pitcher", "water"),
    ("bread_loaf", "bread"),
    ("bread_slice", "bread"),
    ("knife_chef", "knife"),
    ("knife_butter", "knife"),
    ("plate_dinner", "plate"),
    ("plate_side", "plate"),
    ("bowl_small", "bowl"),
    ("bowl_large", "bowl"),
    ("spread_butter", "spread"),
    ("spread_jam", "spread"),
    ("apple_red", "apple"),
    ("apple_green", "apple"),
)

ACTION_TYPES: Tuple[str, ...] = (
    "move",
    "pick",
    "place",
    "wash",
    "fill",
    "make_coffee",
    "make_fruit_bowl",
    "apply_spread",
    "pour",
    "refill_water",
    "drain",
)
ACTION_TYPE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(ACTION_TYPES)}
ACTION_HEADS: Dict[str, Tuple[str, ...]] = {
    "move": ("location",),
    "pick": ("object1",),
    "place": ("location",),
    "wash": ("object1",),
    "fill": ("object1",),
    "make_coffee": ("object1",),
    "make_fruit_bowl": ("object1", "object2"),
    "apply_spread": ("object1",),
    "pour": ("object1", "object2"),
    "refill_water": ("object1", "object2"),
    "drain": ("object1",),
}

# Exported for compatibility with existing imports.
TASK_TYPES: Tuple[str, ...] = DEFAULT_TASK_TYPES


@dataclass(frozen=True)
class RestaurantTask:
    task_type: str
    target_location: str | None = None
    target_kind: str | None = None
    object_name: str | None = None


@dataclass
class RestaurantObjectState:
    name: str
    kind: str
    location: str | None
    dirty: bool = False
    filled_with: str | None = None
    contained_in: str | None = None


@dataclass
class RestaurantState:
    agent_location: str
    holding: str | None
    objects: Dict[str, RestaurantObjectState]
    bread_spread: str | None = None


class RestaurantSymbolicEnv(Env):
    """Symbolic continuing-task restaurant benchmark with factorized PDDL-style actions."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        max_steps_per_task: int = 100,
        success_reward: float = 15.0,
        invalid_action_penalty: float = 6.0,
        travel_cost_scale: float = 25.0,
        pick_cost: float = 25.0,
        place_cost: float = 25.0,
        wash_cost: float = 25.0,
        fill_cost: float = 25.0,
        brew_cost: float = 25.0,
        fruit_cost: float = 25.0,
        pour_cost: float = 25.0,
        refill_cost: float = 25.0,
        drain_cost: float = 25.0,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._base_config = _load_config(Path(config_path)) if config_path is not None else _load_config(CONFIG_PATH)
        self._rng = np.random.default_rng(rng_seed)

        self.max_steps_per_task = max(1, int(max_steps_per_task))
        self.success_reward = float(success_reward)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.travel_cost_scale = float(travel_cost_scale)
        self.pick_cost = float(pick_cost)
        self.place_cost = float(place_cost)
        self.wash_cost = float(wash_cost)
        self.fill_cost = float(fill_cost)
        self.brew_cost = float(brew_cost)
        self.fruit_cost = float(fruit_cost)
        self.spread_cost = float(fruit_cost)
        self.pour_cost = float(pour_cost)
        self.refill_cost = float(refill_cost)
        self.drain_cost = float(drain_cost)

        self._task_library: List[RestaurantTask] = []
        self._task_library_index = 0
        self._task_source = "iid"
        self._pending_auto_success = False
        self._task_steps = 0
        self._active_layout_id: str | None = None
        self._action_mask_cache_key: tuple[object, ...] | None = None
        self._action_mask_cache: Dict[str, np.ndarray] | None = None

        self._apply_schema(self._base_config)
        self._configure_paper2_cost(self._base_config.get("paper2_cost", {}))

        self.state = RestaurantState(
            agent_location=self._default_agent_location(),
            holding=None,
            objects={},
            bread_spread=None,
        )
        self.task = RestaurantTask(
            task_type=self.task_types[0],
            target_location=self.service_locations[0] if self.service_locations else None,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        layout = options.get("layout")
        if isinstance(layout, Mapping):
            self._apply_schema(dict(layout), merge_with_base=False)
            self._configure_paper2_cost(dict(layout).get("paper2_cost", self._base_config.get("paper2_cost", {})))
            self._active_layout_id = str(dict(layout).get("layout_id", "external_layout"))
        elif bool(options.get("reload_base_schema", False)):
            self._apply_schema(self._base_config)
            self._configure_paper2_cost(self._base_config.get("paper2_cost", {}))
            self._active_layout_id = None

        task_library_cfg = options.get("task_library")
        if isinstance(task_library_cfg, Sequence) and not isinstance(task_library_cfg, (str, bytes)):
            self._task_library = self._parse_task_library(task_library_cfg)
            self._task_library_index = 0
        elif bool(options.get("clear_task_library", False)):
            self._task_library = []
            self._task_library_index = 0

        task_distribution_override = options.get("task_distribution")
        if isinstance(task_distribution_override, Mapping):
            self.task_distribution = {
                name: float(task_distribution_override.get(name, 0.0))
                for name in self.task_types
            }

        agent_location = str(options.get("agent_location", self._default_agent_location()))
        if agent_location not in self.location_index:
            agent_location = self._default_agent_location()
        objects = self._sample_object_layout()
        self.state = RestaurantState(
            agent_location=agent_location,
            holding=None,
            objects=objects,
            bread_spread=None,
        )
        self._task_steps = 0
        self._task_source = "iid"
        self._paper2_total_cost = 0.0
        self._paper2_task_cost = 0.0
        self._paper2_last_step_cost = 0.0
        self._paper2_last_step_breakdown = {"move_cost": 0.0, "fixed_cost": 0.0, "action_type": "none", "valid": True}
        self._resample_task()
        return self._obs(), self._info(success=False)

    def step(self, action: Mapping[str, int] | Sequence[int]):
        if self._pending_auto_success:
            completed_task = self.task
            self._pending_auto_success = False
            reward = self.success_reward
            success = True
            truncated = False
            self._advance_after_task_success(completed_task)
            self._task_steps = 0
            self._paper2_task_cost = 0.0
            return self._obs(), reward, success, truncated, self._info(success=success)

        parsed = self._normalize_action(action)
        self._task_steps += 1
        reward = 0.0
        success = False
        truncated = False
        src_location = self.state.agent_location

        action_reward, valid = self._execute_action(parsed)
        reward += action_reward
        if not valid:
            reward -= self.invalid_action_penalty
        self._update_paper2_cost(action_spec=parsed, src_location=src_location, valid=valid)

        if self._task_already_satisfied():
            reward += self.success_reward
            success = True
            completed_task = self.task
            self._advance_after_task_success(completed_task)
            self._task_steps = 0
            self._paper2_task_cost = 0.0
        elif self._task_steps >= self.max_steps_per_task:
            truncated = True
            self._task_steps = 0
            self._paper2_task_cost = 0.0
            self._resample_task()

        return self._obs(), reward, success, truncated, self._info(success=success)

    def set_task(
        self,
        task_type: str,
        *,
        target_location: str | None = None,
        target_kind: str | None = None,
        object_name: str | None = None,
        task_source: str = "external",
    ) -> None:
        if task_type not in self.task_type_index:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if target_location is not None and target_location not in self.location_index:
            raise ValueError(f"Unknown target location: {target_location}")
        if target_kind is not None and target_kind not in self.object_kind_index:
            raise ValueError(f"Unknown target kind: {target_kind}")
        if object_name is not None and object_name not in self.object_name_index:
            raise ValueError(f"Unknown object name: {object_name}")
        self.task = RestaurantTask(
            task_type=task_type,
            target_location=target_location,
            target_kind=target_kind,
            object_name=object_name,
        )
        self._task_source = task_source
        self._update_pending_auto_success()

    def get_action_meanings(self) -> List[str]:
        return [
            "move(location)",
            "pick(object)",
            "place(location)",
            "wash(object)",
            "fill(object)",
            "make_coffee(object)",
            "make_fruit_bowl(apple, bowl)",
            "apply_spread(spread)",
            "pour(src, dst)",
            "refill_water(container, jar)",
            "drain(object)",
        ]

    def _apply_schema(self, config: Mapping[str, Any], *, merge_with_base: bool = True) -> None:
        schema: Dict[str, Any] = {}
        if merge_with_base:
            schema.update(self._base_config)
            schema.update(config)
        else:
            schema.update(config)

        self.locations, self.location_coords, self.location_roles = self._parse_locations(schema.get("locations"))
        self.location_index = {name: idx for idx, name in enumerate(self.locations)}
        self.num_locations = len(self.locations)

        self.object_kinds = tuple(str(x) for x in schema.get("object_kinds", DEFAULT_OBJECT_KINDS))
        self.contents = tuple(str(x) for x in schema.get("contents", DEFAULT_CONTENTS))
        self.task_types = tuple(str(x) for x in schema.get("task_types", DEFAULT_TASK_TYPES))
        self.object_kind_index = {name: idx for idx, name in enumerate(self.object_kinds)}
        self.content_index = {name: idx for idx, name in enumerate(self.contents)}
        self.task_type_index = {name: idx for idx, name in enumerate(self.task_types)}
        self.action_type_index = dict(ACTION_TYPE_TO_INDEX)

        service_default = tuple(loc for loc in DEFAULT_SERVICE_LOCATIONS if loc in self.location_index)
        wash_ready_default = tuple(loc for loc in DEFAULT_WASH_READY_LOCATIONS if loc in self.location_index)
        self.service_locations = tuple(
            str(x)
            for x in schema.get("service_locations", service_default)
            if str(x) in self.location_index
        )
        self.wash_ready_locations = tuple(
            str(x)
            for x in schema.get("wash_ready_locations", wash_ready_default)
            if str(x) in self.location_index
        )
        self.dirty_drop_locations = set(
            str(x)
            for x in schema.get("dirty_drop_locations", ["sink", "bus_tub"])
            if str(x) in self.location_index
        )
        self.restricted_locations = set(
            str(x)
            for x in schema.get("restricted_locations", [])
            if str(x) in self.location_index
        )

        stations = schema.get("stations", {})
        self.station_water = str(stations.get("water", "water_station"))
        self.station_coffee = str(stations.get("coffee", "coffee_machine"))
        self.station_fruit = str(stations.get("fruit", "fruit_station"))
        self.station_wash = str(stations.get("wash", "sink"))
        self.countertop_location = str(stations.get("countertop", "prep_counter" if "prep_counter" in self.location_index else self._default_agent_location()))
        for attr in ("station_water", "station_coffee", "station_fruit", "station_wash", "countertop_location"):
            value = getattr(self, attr)
            if value not in self.location_index:
                setattr(self, attr, self._default_agent_location())

        object_specs_cfg = schema.get("object_specs", DEFAULT_OBJECT_SPECS)
        self.object_specs = self._parse_object_specs(object_specs_cfg)
        self.object_names = tuple(name for name, _ in self.object_specs)
        self.object_name_index = {name: idx for idx, name in enumerate(self.object_names)}
        self.num_objects = len(self.object_names)
        self.none_object_index = self.num_objects
        self.none_location_index = self.num_locations

        self.task_distribution = {
            name: float(schema.get("task_distribution", {}).get(name, 0.0))
            for name in self.task_types
        }
        self.service_location_distribution = {
            name: float(schema.get("service_location_distribution", {}).get(name, 0.0))
            for name in self.service_locations
        }
        self.wash_kind_distribution = {
            name: float(schema.get("wash_kind_distribution", {}).get(name, 0.0))
            for name in self.object_kinds
        }
        reset_loc_cfg = schema.get("reset_location_distribution", {})
        self.reset_location_distribution = {
            kind: {loc: float(reset_loc_cfg.get(kind, {}).get(loc, 0.0)) for loc in self.locations}
            for kind in self.object_kinds
        }
        self._task_library = self._parse_task_library(schema.get("task_library", []))
        self._task_library_index = 0

        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ACTION_TYPES)),
                "object1": spaces.Discrete(self.num_objects + 1),
                "location": spaces.Discrete(self.num_locations + 1),
                "object2": spaces.Discrete(self.num_objects + 1),
            }
        )

        obs_dim = (
            self.num_locations
            + 1
            + (self.num_objects + 1)
            + self.num_objects * (
                (self.num_locations + 1)
                + 1
                + (len(self.contents) + 1)
                + (self.num_objects + 1)
                + len(self.object_kinds)
            )
            + len(self.object_names)
            + len(self.task_types)
            + (self.num_locations + 1)
            + (len(self.object_kinds) + 1)
            + (self.num_objects + 1)
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def _parse_locations(
        self,
        location_defs: Any,
    ) -> Tuple[Tuple[str, ...], Dict[str, Tuple[int, int]], Dict[str, Tuple[str, ...]]]:
        if not isinstance(location_defs, Sequence) or isinstance(location_defs, (str, bytes)):
            roles = {name: self._infer_location_roles(name) for name in DEFAULT_LOCATIONS}
            return DEFAULT_LOCATIONS, dict(DEFAULT_LOCATION_COORDS), roles
        names: List[str] = []
        coords: Dict[str, Tuple[int, int]] = {}
        roles: Dict[str, Tuple[str, ...]] = {}
        for idx, entry in enumerate(location_defs):
            if isinstance(entry, Mapping):
                name = str(entry.get("name", f"loc_{idx}"))
                coord_raw = entry.get("coord", [idx, 0])
                raw_roles = entry.get("roles", None)
            else:
                name = str(entry)
                coord_raw = [idx, 0]
                raw_roles = None
            if name in coords:
                continue
            if isinstance(coord_raw, Sequence) and len(coord_raw) >= 2:
                cx = int(coord_raw[0])
                cy = int(coord_raw[1])
            else:
                cx, cy = idx, 0
            names.append(name)
            coords[name] = (cx, cy)
            if isinstance(raw_roles, Sequence) and not isinstance(raw_roles, (str, bytes)):
                roles[name] = tuple(str(x) for x in raw_roles)
            else:
                roles[name] = self._infer_location_roles(name)
        if not names:
            roles = {name: self._infer_location_roles(name) for name in DEFAULT_LOCATIONS}
            return DEFAULT_LOCATIONS, dict(DEFAULT_LOCATION_COORDS), roles
        return tuple(names), coords, roles

    def _infer_location_roles(self, name: str) -> Tuple[str, ...]:
        roles: List[str] = []
        if "counter" in name:
            roles.append("countertop")
        if "water" in name:
            roles.append("fountain")
        if "coffee" in name:
            roles.append("coffeemachine")
        if "sink" in name or "dish" in name:
            roles.append("dishwasher")
        if "table" in name or "pass" in name:
            roles.append("servingtable")
        if "shelf" in name or "stand" in name:
            roles.append("shelf")
        if not roles:
            roles.append("location")
        return tuple(roles)

    def _parse_object_specs(self, object_specs_cfg: Any) -> Tuple[Tuple[str, str], ...]:
        parsed: List[Tuple[str, str]] = []
        if isinstance(object_specs_cfg, Sequence) and not isinstance(object_specs_cfg, (str, bytes)):
            for idx, item in enumerate(object_specs_cfg):
                if isinstance(item, Mapping):
                    name = str(item.get("name", f"obj_{idx}"))
                    kind = str(item.get("kind", self.object_kinds[0] if self.object_kinds else "obj"))
                elif isinstance(item, Sequence) and len(item) >= 2:
                    name = str(item[0])
                    kind = str(item[1])
                else:
                    continue
                if kind not in self.object_kind_index:
                    continue
                parsed.append((name, kind))
        if not parsed:
            parsed = list(DEFAULT_OBJECT_SPECS)
        return tuple(parsed)

    def _parse_task_library(self, task_library_cfg: Any) -> List[RestaurantTask]:
        parsed: List[RestaurantTask] = []
        if not isinstance(task_library_cfg, Sequence) or isinstance(task_library_cfg, (str, bytes)):
            return parsed
        for item in task_library_cfg:
            if not isinstance(item, Mapping):
                continue
            task_type = str(item.get("task_type", ""))
            if task_type not in self.task_type_index:
                continue
            target_location = item.get("target_location")
            target_kind = item.get("target_kind")
            object_name = item.get("object_name")
            if target_location is not None and str(target_location) not in self.location_index:
                continue
            if target_kind is not None and str(target_kind) not in self.object_kind_index:
                continue
            if object_name is not None and str(object_name) not in self.object_name_index:
                continue
            parsed.append(
                RestaurantTask(
                    task_type=task_type,
                    target_location=None if target_location is None else str(target_location),
                    target_kind=None if target_kind is None else str(target_kind),
                    object_name=None if object_name is None else str(object_name),
                )
            )
        return parsed

    def _default_agent_location(self) -> str:
        if hasattr(self, "location_index") and "kitchen_counter" in self.location_index:
            return "kitchen_counter"
        if hasattr(self, "locations") and self.locations:
            return self.locations[0]
        return DEFAULT_LOCATIONS[0]

    def _advance_after_task_success(self, completed_task: RestaurantTask) -> None:
        del completed_task
        self._resample_task()

    def _resample_task(self) -> None:
        if self._task_library:
            task = self._task_library[self._task_library_index % len(self._task_library)]
            self._task_library_index += 1
            self.set_task(
                task.task_type,
                target_location=task.target_location,
                target_kind=task.target_kind,
                object_name=task.object_name,
                task_source="library",
            )
            return

        task_type = str(self._rng.choice(self.task_types))
        if task_type in {"serve_water", "make_coffee", "make_fruit_bowl", "clear_containers"}:
            target_location = str(self._rng.choice(self.service_locations))
            self.set_task(task_type, target_location=target_location, task_source="iid")
            return
        if task_type == "pick_place":
            object_name = str(self._rng.choice(self.object_names))
            target_location = str(self._rng.choice(self.locations))
            self.set_task(task_type, target_location=target_location, object_name=object_name, task_source="iid")
            return
        target_kind = str(self._rng.choice(self.object_kinds))
        self.set_task(task_type, target_kind=target_kind, task_source="iid")

    def _task_already_satisfied(self) -> bool:
        if self.task.task_type == "serve_water":
            assert self.task.target_location is not None
            return any(
                obj.location == self.task.target_location
                and obj.kind in {"cup", "mug"}
                and obj.filled_with == "water"
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "make_coffee":
            assert self.task.target_location is not None
            return any(
                obj.location == self.task.target_location
                and obj.kind in {"cup", "mug"}
                and obj.filled_with == "coffee"
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "make_fruit_bowl":
            assert self.task.target_location is not None
            bowls = [
                obj.name
                for obj in self.state.objects.values()
                if obj.kind == "bowl" and obj.location == self.task.target_location
            ]
            if not bowls:
                return False
            return any(
                apple.kind == "apple" and apple.contained_in in bowls
                for apple in self.state.objects.values()
            )
        if self.task.task_type == "clear_containers":
            assert self.task.target_location is not None
            return not any(obj.location == self.task.target_location for obj in self.state.objects.values())
        if self.task.task_type == "wash_objects":
            assert self.task.target_kind is not None
            return any(
                obj.kind == self.task.target_kind
                and not obj.dirty
                and obj.filled_with is None
                and obj.location in self.wash_ready_locations
                and obj.contained_in is None
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "pick_place":
            assert self.task.object_name is not None and self.task.target_location is not None
            obj = self.state.objects.get(self.task.object_name)
            return obj is not None and obj.location == self.task.target_location and self.state.holding is None
        raise ValueError(f"Unsupported task type: {self.task.task_type}")

    def _update_pending_auto_success(self) -> None:
        self._pending_auto_success = self._task_already_satisfied()

    def _normalize_action(self, action: Mapping[str, int] | Sequence[int]) -> Dict[str, int | str]:
        if isinstance(action, Mapping):
            action_type_idx = int(action.get("action_type", 0))
            object1_idx = int(action.get("object1", self.none_object_index))
            location_idx = int(action.get("location", self.none_location_index))
            object2_idx = int(action.get("object2", self.none_object_index))
        elif isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            values = list(action)
            if len(values) != 4:
                raise ValueError("Structured restaurant action sequence must have length 4.")
            action_type_idx = int(values[0])
            object1_idx = int(values[1])
            location_idx = int(values[2])
            object2_idx = int(values[3])
        else:
            raise TypeError("RestaurantSymbolicEnv.step expects a mapping or length-4 sequence action.")

        if action_type_idx < 0 or action_type_idx >= len(ACTION_TYPES):
            raise ValueError(f"Invalid action_type index {action_type_idx}.")
        if object1_idx < 0 or object1_idx > self.none_object_index:
            raise ValueError(f"Invalid object1 index {object1_idx}.")
        if location_idx < 0 or location_idx > self.none_location_index:
            raise ValueError(f"Invalid location index {location_idx}.")
        if object2_idx < 0 or object2_idx > self.none_object_index:
            raise ValueError(f"Invalid object2 index {object2_idx}.")

        return {
            "action_type_idx": action_type_idx,
            "action_type": ACTION_TYPES[action_type_idx],
            "object1_idx": object1_idx,
            "location_idx": location_idx,
            "object2_idx": object2_idx,
            "object1_name": None if object1_idx == self.none_object_index else self.object_names[object1_idx],
            "location_name": None if location_idx == self.none_location_index else self.locations[location_idx],
            "object2_name": None if object2_idx == self.none_object_index else self.object_names[object2_idx],
        }

    def _execute_action(self, action_spec: Mapping[str, Any]) -> Tuple[float, bool]:
        action_type = str(action_spec["action_type"])
        if not self._is_action_valid(action_spec):
            return 0.0, False
        if action_type == "move":
            return self._move_to(str(action_spec["location_name"])), True
        if action_type == "pick":
            return self._pick_object(str(action_spec["object1_name"])), True
        if action_type == "place":
            return self._place_object(str(action_spec["location_name"])), True
        if action_type == "wash":
            return self._wash_object(str(action_spec["object1_name"])), True
        if action_type == "fill":
            return self._fill_container(str(action_spec["object1_name"])), True
        if action_type == "make_coffee":
            return self._make_coffee(str(action_spec["object1_name"])), True
        if action_type == "make_fruit_bowl":
            return self._make_fruit_bowl(str(action_spec["object1_name"]), str(action_spec["object2_name"])), True
        if action_type == "apply_spread":
            return self._apply_spread(str(action_spec["object1_name"])), True
        if action_type == "pour":
            return self._pour(str(action_spec["object1_name"]), str(action_spec["object2_name"])), True
        if action_type == "refill_water":
            return self._refill_water(str(action_spec["object1_name"]), str(action_spec["object2_name"])), True
        if action_type == "drain":
            return self._drain(str(action_spec["object1_name"])), True
        raise ValueError(f"Unsupported action type: {action_type}")

    def _move_to(self, location: str) -> float:
        travel = self._travel_cost(self.state.agent_location, location)
        self.state.agent_location = location
        return -travel

    def _pick_object(self, obj_name: str) -> float:
        obj = self.state.objects[obj_name]
        obj.location = None
        obj.contained_in = None
        self.state.holding = obj_name
        return -self.pick_cost

    def _place_object(self, location: str) -> float:
        assert self.state.holding is not None
        obj = self.state.objects[self.state.holding]
        self.state.holding = None
        obj.location = location
        obj.contained_in = None
        return -self.place_cost

    def _wash_object(self, obj_name: str) -> float:
        obj = self.state.objects[obj_name]
        obj.dirty = False
        return -self.wash_cost

    def _fill_container(self, obj_name: str) -> float:
        obj = self.state.objects[obj_name]
        obj.filled_with = "water"
        return -self.fill_cost

    def _make_coffee(self, obj_name: str) -> float:
        obj = self.state.objects[obj_name]
        obj.filled_with = "coffee"
        obj.dirty = True
        return -self.brew_cost

    def _make_fruit_bowl(self, apple_name: str, bowl_name: str) -> float:
        knife_name = self.state.holding
        assert knife_name is not None
        apple = self.state.objects[apple_name]
        bowl = self.state.objects[bowl_name]
        knife = self.state.objects[knife_name]
        apple.location = None
        apple.contained_in = bowl_name
        bowl.dirty = True
        knife.dirty = True
        return -self.fruit_cost

    def _apply_spread(self, spread_name: str) -> float:
        knife_name = self.state.holding
        assert knife_name is not None
        knife = self.state.objects[knife_name]
        self.state.bread_spread = spread_name
        knife.dirty = True
        return -self.spread_cost

    def _pour(self, src_name: str, dst_name: str) -> float:
        src = self.state.objects[src_name]
        dst = self.state.objects[dst_name]
        dst.filled_with = src.filled_with
        src.filled_with = None
        return -self.pour_cost

    def _refill_water(self, container_name: str, jar_name: str) -> float:
        del jar_name
        container = self.state.objects[container_name]
        container.filled_with = "water"
        return -self.refill_cost

    def _drain(self, container_name: str) -> float:
        container = self.state.objects[container_name]
        container.filled_with = None
        return -self.drain_cost

    def _has_role(self, location: str, role: str) -> bool:
        return role in self.location_roles.get(location, ())

    def _object_at_robot(self, obj_name: str) -> bool:
        obj = self.state.objects[obj_name]
        return obj.location == self.state.agent_location and obj.contained_in is None

    def _held_clean_knife(self) -> bool:
        if self.state.holding is None:
            return False
        held = self.state.objects[self.state.holding]
        return held.kind == "knife" and not held.dirty

    def _is_fillable_kind(self, kind: str) -> bool:
        return kind in {"cup", "mug", "jar", "bowl"}

    def _is_container_kind(self, kind: str) -> bool:
        return kind in {"cup", "mug", "jar", "plate", "bowl"}

    def _water_available_at(self, location: str) -> bool:
        return any(obj.kind == "water" and obj.location == location for obj in self.state.objects.values())

    def _coffeegrinds_available_at(self, location: str) -> bool:
        return any(obj.kind == "coffeegrinds" and obj.location == location for obj in self.state.objects.values())

    def _bread_available_at(self, location: str) -> bool:
        return any(obj.kind == "bread" and obj.location == location for obj in self.state.objects.values())

    def _is_action_valid(self, action_spec: Mapping[str, Any]) -> bool:
        action_type = str(action_spec["action_type"])
        object1 = action_spec.get("object1_name")
        location = action_spec.get("location_name")
        object2 = action_spec.get("object2_name")
        held_name = self.state.holding

        if action_type == "move":
            return location is not None and location != self.state.agent_location
        if action_type == "pick":
            if object1 is None or held_name is not None:
                return False
            obj = self.state.objects[object1]
            return obj.location == self.state.agent_location and self.state.agent_location not in self.restricted_locations
        if action_type == "place":
            return held_name is not None and location is not None and location == self.state.agent_location and location not in self.restricted_locations
        if action_type == "wash":
            if object1 is None:
                return False
            obj = self.state.objects[object1]
            return obj.dirty and obj.location == self.state.agent_location and self._has_role(self.state.agent_location, "dishwasher")
        if action_type == "fill":
            if object1 is None or held_name != object1:
                return False
            obj = self.state.objects[object1]
            return (
                self._has_role(self.state.agent_location, "fountain")
                and self._water_available_at(self.state.agent_location)
                and self._is_fillable_kind(obj.kind)
                and not obj.dirty
                and obj.filled_with is None
            )
        if action_type == "make_coffee":
            if object1 is None:
                return False
            obj = self.state.objects[object1]
            return (
                self._has_role(self.state.agent_location, "coffeemachine")
                and obj.location == self.state.agent_location
                and obj.kind in {"cup", "mug"}
                and not obj.dirty
                and obj.filled_with is None
                and self._water_available_at(self.state.agent_location)
                and self._coffeegrinds_available_at(self.state.agent_location)
            )
        if action_type == "make_fruit_bowl":
            if object1 is None or object2 is None:
                return False
            apple = self.state.objects[object1]
            bowl = self.state.objects[object2]
            return (
                self._has_role(self.state.agent_location, "countertop")
                and apple.kind == "apple"
                and apple.location == self.state.agent_location
                and apple.contained_in is None
                and bowl.kind == "bowl"
                and bowl.location == self.state.agent_location
                and bowl.contained_in is None
                and not bowl.dirty
                and bowl.filled_with is None
                and self._held_clean_knife()
            )
        if action_type == "apply_spread":
            if object1 is None:
                return False
            spread = self.state.objects[object1]
            return (
                self._has_role(self.state.agent_location, "countertop")
                and spread.kind == "spread"
                and spread.location == self.state.agent_location
                and self._bread_available_at(self.state.agent_location)
                and self._held_clean_knife()
                and self.state.bread_spread != object1
            )
        if action_type == "pour":
            if object1 is None or object2 is None or object1 == object2:
                return False
            src = self.state.objects[object1]
            dst = self.state.objects[object2]
            return (
                held_name == object1
                and src.filled_with is not None
                and dst.location == self.state.agent_location
                and dst.contained_in is None
                and self._is_fillable_kind(dst.kind)
                and dst.filled_with is None
            )
        if action_type == "refill_water":
            if object1 is None or object2 is None or held_name != object1:
                return False
            container = self.state.objects[object1]
            jar = self.state.objects[object2]
            return (
                jar.kind == "jar"
                and jar.location == self.state.agent_location
                and jar.filled_with == "water"
                and self._is_fillable_kind(container.kind)
                and not container.dirty
                and container.filled_with is None
            )
        if action_type == "drain":
            if object1 is None or held_name != object1:
                return False
            container = self.state.objects[object1]
            return self._has_role(self.state.agent_location, "fountain") and container.filled_with == "water"
        return False

    def _configure_paper2_cost(self, cfg: Mapping[str, Any]) -> None:
        self.paper2_enabled = bool(cfg.get("enabled", False))
        self.paper2_fixed_costs = {
            "move": 0.0,
            "pick": get_pddl_cost("pick"),
            "place": get_pddl_cost("place"),
            "wash": get_pddl_cost("wash"),
            "fill": get_pddl_cost("fill"),
            "make_coffee": get_pddl_cost("make-coffee"),
            "make_fruit_bowl": get_pddl_cost("make-fruit-bowl"),
            "apply_spread": get_pddl_cost("apply-spread"),
            "pour": get_pddl_cost("pour"),
            "refill_water": get_pddl_cost("refill_water"),
            "drain": get_pddl_cost("drain"),
        }

        move_cfg = cfg.get("move", {})
        self.paper2_move_scale = float(move_cfg.get("scale", 1.0))
        grid_cfg = move_cfg.get("grid", {})
        self.paper2_grid_width = int(grid_cfg.get("width", 10))
        self.paper2_grid_height = int(grid_cfg.get("height", 10))
        blocked = grid_cfg.get("blocked_cells", [])
        self.paper2_blocked_cells = {
            (int(cell[0]), int(cell[1]))
            for cell in blocked
            if isinstance(cell, Sequence) and len(cell) >= 2
        }
        loc_cells_cfg = move_cfg.get("location_cells", {})
        self.paper2_location_cells = {}
        for loc in self.locations:
            raw = loc_cells_cfg.get(loc)
            if isinstance(raw, Sequence) and len(raw) >= 2:
                self.paper2_location_cells[loc] = (int(raw[0]), int(raw[1]))
            else:
                sx, sy = self.location_coords[loc]
                self.paper2_location_cells[loc] = (int(sx), int(sy))
        self._paper2_total_cost = 0.0
        self._paper2_task_cost = 0.0
        self._paper2_last_step_cost = 0.0
        self._paper2_last_step_breakdown = {"move_cost": 0.0, "fixed_cost": 0.0, "action_type": "none", "valid": True}

    def _update_paper2_cost(self, *, action_spec: Mapping[str, Any], src_location: str, valid: bool) -> None:
        action_type = str(action_spec.get("action_type", "unknown"))
        if not self.paper2_enabled or not valid:
            self._paper2_last_step_cost = 0.0
            self._paper2_last_step_breakdown = {
                "move_cost": 0.0,
                "fixed_cost": 0.0,
                "action_type": action_type,
                "valid": bool(valid),
            }
            return
        dst_location = self.state.agent_location
        move_dist = float(self._dijkstra_distance(src_location, dst_location))
        move_cost = self.paper2_move_scale * move_dist
        fixed_cost = float(self.paper2_fixed_costs.get(action_type, 0.0))
        step_cost = move_cost + fixed_cost
        self._paper2_last_step_cost = step_cost
        self._paper2_last_step_breakdown = {
            "move_cost": move_cost,
            "fixed_cost": fixed_cost,
            "action_type": action_type,
            "valid": bool(valid),
        }
        self._paper2_task_cost += step_cost
        self._paper2_total_cost += step_cost

    def _dijkstra_distance(self, src_location: str, dst_location: str) -> float:
        src = self.paper2_location_cells.get(src_location)
        dst = self.paper2_location_cells.get(dst_location)
        if src is None or dst is None:
            return float(
                abs(self.location_coords[src_location][0] - self.location_coords[dst_location][0])
                + abs(self.location_coords[src_location][1] - self.location_coords[dst_location][1])
            )
        if src == dst:
            return 0.0
        visited: set[Tuple[int, int]] = set()
        heap: List[Tuple[int, Tuple[int, int]]] = [(0, src)]
        while heap:
            dist, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == dst:
                return float(dist)
            x, y = node
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or ny < 0 or nx >= self.paper2_grid_width or ny >= self.paper2_grid_height:
                    continue
                nxt = (nx, ny)
                if nxt in self.paper2_blocked_cells or nxt in visited:
                    continue
                heapq.heappush(heap, (dist + 1, nxt))
        return float(abs(src[0] - dst[0]) + abs(src[1] - dst[1]))

    def _sample_object_layout(self) -> Dict[str, RestaurantObjectState]:
        objects: Dict[str, RestaurantObjectState] = {}
        for name, kind in self.object_specs:
            location = self._weighted_choice(self.reset_location_distribution.get(kind, {}), self.locations)
            dirty, filled_with = self._sample_object_status(kind, location)
            objects[name] = RestaurantObjectState(
                name=name,
                kind=kind,
                location=location,
                dirty=dirty,
                filled_with=filled_with,
                contained_in=None,
            )
        return objects

    def _sample_object_status(self, kind: str, location: str) -> Tuple[bool, str | None]:
        prep_like = {self._default_agent_location(), self.station_coffee, self.station_water, self.station_fruit}
        prep_like.update(self.wash_ready_locations)
        if kind in {"water", "coffeegrinds"}:
            return False, None
        if location in prep_like:
            return False, None
        if location in self.dirty_drop_locations:
            return True, None
        if location in self.service_locations:
            if kind in {"cup", "mug"}:
                filled = str(self._rng.choice(["water", "coffee", "empty"], p=[0.35, 0.35, 0.30]))
                return filled != "empty", None if filled == "empty" else filled
            return False, None
        return False, None

    def _obs(self) -> np.ndarray:
        pieces: List[np.ndarray] = []

        agent_one_hot = np.zeros((self.num_locations,), dtype=np.float32)
        agent_one_hot[self.location_index[self.state.agent_location]] = 1.0
        pieces.append(agent_one_hot)
        pieces.append(np.array([1.0 if self.state.holding is None else 0.0], dtype=np.float32))

        held_vec = np.zeros((self.num_objects + 1,), dtype=np.float32)
        if self.state.holding is None:
            held_vec[self.none_object_index] = 1.0
        else:
            held_vec[self.object_name_index[self.state.holding]] = 1.0
        pieces.append(held_vec)

        for name in self.object_names:
            obj = self.state.objects[name]
            loc_vec = np.zeros((self.num_locations + 1,), dtype=np.float32)
            if obj.location is None:
                loc_vec[self.none_location_index] = 1.0
            else:
                loc_vec[self.location_index[obj.location]] = 1.0
            dirty_vec = np.array([1.0 if obj.dirty else 0.0], dtype=np.float32)
            fill_vec = np.zeros((len(self.contents) + 1,), dtype=np.float32)
            if obj.filled_with is None:
                fill_vec[-1] = 1.0
            else:
                fill_vec[self.content_index[obj.filled_with]] = 1.0
            contained_vec = np.zeros((self.num_objects + 1,), dtype=np.float32)
            if obj.contained_in is None:
                contained_vec[self.none_object_index] = 1.0
            else:
                contained_vec[self.object_name_index[obj.contained_in]] = 1.0
            kind_vec = np.zeros((len(self.object_kinds),), dtype=np.float32)
            kind_vec[self.object_kind_index[obj.kind]] = 1.0
            pieces.extend([loc_vec, dirty_vec, fill_vec, contained_vec, kind_vec])

        bread_spread_vec = np.zeros((self.num_objects,), dtype=np.float32)
        if self.state.bread_spread is not None:
            bread_spread_vec[self.object_name_index[self.state.bread_spread]] = 1.0
        pieces.append(bread_spread_vec)

        task_type_vec = np.zeros((len(self.task_types),), dtype=np.float32)
        task_type_vec[self.task_type_index[self.task.task_type]] = 1.0
        target_location_vec = np.zeros((self.num_locations + 1,), dtype=np.float32)
        if self.task.target_location is None:
            target_location_vec[self.none_location_index] = 1.0
        else:
            target_location_vec[self.location_index[self.task.target_location]] = 1.0
        target_kind_vec = np.zeros((len(self.object_kinds) + 1,), dtype=np.float32)
        if self.task.target_kind is None:
            target_kind_vec[-1] = 1.0
        else:
            target_kind_vec[self.object_kind_index[self.task.target_kind]] = 1.0
        target_object_vec = np.zeros((self.num_objects + 1,), dtype=np.float32)
        if self.task.object_name is None:
            target_object_vec[self.none_object_index] = 1.0
        else:
            target_object_vec[self.object_name_index[self.task.object_name]] = 1.0
        pieces.extend([task_type_vec, target_location_vec, target_kind_vec, target_object_vec])
        return np.concatenate(pieces, axis=0)

    def _action_mask_state_key(self) -> tuple[object, ...]:
        object_state = []
        for name in self.object_names:
            obj = self.state.objects[name]
            object_state.append((obj.location, obj.dirty, obj.filled_with, obj.contained_in))
        return (
            self.state.agent_location,
            self.state.holding,
            self.state.bread_spread,
            tuple(object_state),
        )

    def _empty_action_masks(self) -> Dict[str, np.ndarray]:
        return {
            "valid_action_type_mask": np.zeros((len(ACTION_TYPES),), dtype=np.float32),
            "valid_object1_mask": np.zeros((len(ACTION_TYPES), self.num_objects + 1), dtype=np.float32),
            "valid_location_mask": np.zeros((len(ACTION_TYPES), self.num_locations + 1), dtype=np.float32),
            "valid_object2_mask": np.zeros((len(ACTION_TYPES), self.num_objects + 1, self.num_objects + 1), dtype=np.float32),
        }

    def _finalize_action_masks(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for action_type_idx, action_type in enumerate(ACTION_TYPES):
            required = ACTION_HEADS[action_type]
            if "object1" not in required:
                masks["valid_object1_mask"][action_type_idx, self.none_object_index] = 1.0
            if "location" not in required:
                masks["valid_location_mask"][action_type_idx, self.none_location_index] = 1.0
            if "object2" not in required:
                masks["valid_object2_mask"][action_type_idx, :, self.none_object_index] = 1.0
        masks["valid_action_mask"] = masks["valid_action_type_mask"].copy()
        return masks

    def _set_action_type_valid(self, masks: Dict[str, np.ndarray], action_type: str) -> int:
        idx = self.action_type_index[action_type]
        masks["valid_action_type_mask"][idx] = 1.0
        return idx

    def _set_none_defaults_for_action(self, masks: Dict[str, np.ndarray], action_type_idx: int) -> None:
        masks["valid_object1_mask"][action_type_idx, self.none_object_index] = 1.0
        masks["valid_location_mask"][action_type_idx, self.none_location_index] = 1.0
        masks["valid_object2_mask"][action_type_idx, self.none_object_index, self.none_object_index] = 1.0

    def _build_action_masks(self) -> Dict[str, np.ndarray]:
        masks = self._empty_action_masks()
        held_name = self.state.holding
        held_idx = self.none_object_index if held_name is None else self.object_name_index[held_name]

        move_idx = self.action_type_index["move"]
        if self.num_locations > 1:
            masks["valid_action_type_mask"][move_idx] = 1.0
            masks["valid_object1_mask"][move_idx, self.none_object_index] = 1.0
            for location in self.locations:
                if location != self.state.agent_location:
                    masks["valid_location_mask"][move_idx, self.location_index[location]] = 1.0
            masks["valid_object2_mask"][move_idx, self.none_object_index, self.none_object_index] = 1.0

        if held_name is None and self.state.agent_location not in self.restricted_locations:
            valid_pick_objects = [
                self.object_name_index[name]
                for name in self.object_names
                if self.state.objects[name].location == self.state.agent_location
            ]
            if valid_pick_objects:
                pick_idx = self._set_action_type_valid(masks, "pick")
                for obj_idx in valid_pick_objects:
                    masks["valid_object1_mask"][pick_idx, obj_idx] = 1.0
                    masks["valid_object2_mask"][pick_idx, obj_idx, self.none_object_index] = 1.0
                masks["valid_location_mask"][pick_idx, self.none_location_index] = 1.0

        if held_name is not None and self.state.agent_location not in self.restricted_locations:
            place_idx = self._set_action_type_valid(masks, "place")
            masks["valid_object1_mask"][place_idx, self.none_object_index] = 1.0
            loc_idx = self.location_index[self.state.agent_location]
            masks["valid_location_mask"][place_idx, loc_idx] = 1.0
            masks["valid_object2_mask"][place_idx, self.none_object_index, self.none_object_index] = 1.0

        if self._has_role(self.state.agent_location, "dishwasher"):
            washable = [
                self.object_name_index[name]
                for name in self.object_names
                if self.state.objects[name].dirty and self.state.objects[name].location == self.state.agent_location
            ]
            if washable:
                wash_idx = self._set_action_type_valid(masks, "wash")
                for obj_idx in washable:
                    masks["valid_object1_mask"][wash_idx, obj_idx] = 1.0
                    masks["valid_object2_mask"][wash_idx, obj_idx, self.none_object_index] = 1.0
                masks["valid_location_mask"][wash_idx, self.none_location_index] = 1.0

        if held_name is not None:
            held_obj = self.state.objects[held_name]
            held_obj_idx = self.object_name_index[held_name]

            if (
                self._has_role(self.state.agent_location, "fountain")
                and self._water_available_at(self.state.agent_location)
                and self._is_fillable_kind(held_obj.kind)
                and not held_obj.dirty
                and held_obj.filled_with is None
            ):
                fill_idx = self._set_action_type_valid(masks, "fill")
                masks["valid_object1_mask"][fill_idx, held_obj_idx] = 1.0
                masks["valid_location_mask"][fill_idx, self.none_location_index] = 1.0
                masks["valid_object2_mask"][fill_idx, held_obj_idx, self.none_object_index] = 1.0

            if self._held_clean_knife() and self._has_role(self.state.agent_location, "countertop"):
                apple_indices = [
                    self.object_name_index[name]
                    for name in self.object_names
                    if self.state.objects[name].kind == "apple"
                    and self.state.objects[name].location == self.state.agent_location
                    and self.state.objects[name].contained_in is None
                ]
                bowl_indices = [
                    self.object_name_index[name]
                    for name in self.object_names
                    if self.state.objects[name].kind == "bowl"
                    and self.state.objects[name].location == self.state.agent_location
                    and self.state.objects[name].contained_in is None
                    and not self.state.objects[name].dirty
                    and self.state.objects[name].filled_with is None
                ]
                if apple_indices and bowl_indices:
                    fruit_idx = self._set_action_type_valid(masks, "make_fruit_bowl")
                    masks["valid_location_mask"][fruit_idx, self.none_location_index] = 1.0
                    for apple_idx in apple_indices:
                        masks["valid_object1_mask"][fruit_idx, apple_idx] = 1.0
                        for bowl_idx in bowl_indices:
                            masks["valid_object2_mask"][fruit_idx, apple_idx, bowl_idx] = 1.0

                spread_indices = [
                    self.object_name_index[name]
                    for name in self.object_names
                    if self.state.objects[name].kind == "spread"
                    and self.state.objects[name].location == self.state.agent_location
                    and self.state.bread_spread != name
                ]
                if spread_indices and self._bread_available_at(self.state.agent_location):
                    spread_idx = self._set_action_type_valid(masks, "apply_spread")
                    masks["valid_location_mask"][spread_idx, self.none_location_index] = 1.0
                    for obj_idx in spread_indices:
                        masks["valid_object1_mask"][spread_idx, obj_idx] = 1.0
                        masks["valid_object2_mask"][spread_idx, obj_idx, self.none_object_index] = 1.0

            if held_obj.filled_with is not None:
                pour_targets = [
                    self.object_name_index[name]
                    for name in self.object_names
                    if name != held_name
                    and self.state.objects[name].location == self.state.agent_location
                    and self.state.objects[name].contained_in is None
                    and self._is_fillable_kind(self.state.objects[name].kind)
                    and self.state.objects[name].filled_with is None
                ]
                if pour_targets:
                    pour_idx = self._set_action_type_valid(masks, "pour")
                    masks["valid_object1_mask"][pour_idx, held_obj_idx] = 1.0
                    masks["valid_location_mask"][pour_idx, self.none_location_index] = 1.0
                    for target_idx in pour_targets:
                        masks["valid_object2_mask"][pour_idx, held_obj_idx, target_idx] = 1.0

            if held_obj.filled_with is None and not held_obj.dirty and self._is_fillable_kind(held_obj.kind):
                refill_sources = [
                    self.object_name_index[name]
                    for name in self.object_names
                    if self.state.objects[name].kind == "jar"
                    and self.state.objects[name].location == self.state.agent_location
                    and self.state.objects[name].filled_with == "water"
                ]
                if refill_sources:
                    refill_idx = self._set_action_type_valid(masks, "refill_water")
                    masks["valid_object1_mask"][refill_idx, held_obj_idx] = 1.0
                    masks["valid_location_mask"][refill_idx, self.none_location_index] = 1.0
                    for jar_idx in refill_sources:
                        masks["valid_object2_mask"][refill_idx, held_obj_idx, jar_idx] = 1.0

            if self._has_role(self.state.agent_location, "fountain") and held_obj.filled_with == "water":
                drain_idx = self._set_action_type_valid(masks, "drain")
                masks["valid_object1_mask"][drain_idx, held_obj_idx] = 1.0
                masks["valid_location_mask"][drain_idx, self.none_location_index] = 1.0
                masks["valid_object2_mask"][drain_idx, held_obj_idx, self.none_object_index] = 1.0

        coffee_candidates = [
            self.object_name_index[name]
            for name in self.object_names
            if self.state.objects[name].location == self.state.agent_location
            and self.state.objects[name].kind in {"cup", "mug"}
            and not self.state.objects[name].dirty
            and self.state.objects[name].filled_with is None
        ]
        if (
            self._has_role(self.state.agent_location, "coffeemachine")
            and coffee_candidates
            and self._water_available_at(self.state.agent_location)
            and self._coffeegrinds_available_at(self.state.agent_location)
        ):
            coffee_idx = self._set_action_type_valid(masks, "make_coffee")
            masks["valid_location_mask"][coffee_idx, self.none_location_index] = 1.0
            for obj_idx in coffee_candidates:
                masks["valid_object1_mask"][coffee_idx, obj_idx] = 1.0
                masks["valid_object2_mask"][coffee_idx, obj_idx, self.none_object_index] = 1.0

        return self._finalize_action_masks(masks)

    def _compute_action_masks(self) -> Dict[str, np.ndarray]:
        cache_key = self._action_mask_state_key()
        if self._action_mask_cache_key == cache_key and self._action_mask_cache is not None:
            return {key: value.copy() for key, value in self._action_mask_cache.items()}
        masks = self._build_action_masks()
        self._action_mask_cache_key = cache_key
        self._action_mask_cache = {key: value.copy() for key, value in masks.items()}
        return masks

    def _info(self, *, success: bool) -> Dict[str, Any]:
        masks = self._compute_action_masks()
        return {
            "agent_location": self.state.agent_location,
            "holding": self.state.holding,
            "objects": {
                name: {
                    "kind": obj.kind,
                    "location": obj.location,
                    "dirty": bool(obj.dirty),
                    "filled_with": obj.filled_with,
                    "contained_in": obj.contained_in,
                }
                for name, obj in self.state.objects.items()
            },
            "bread_spread": self.state.bread_spread,
            "task": {
                "task_type": self.task.task_type,
                "target_location": self.task.target_location,
                "target_kind": self.task.target_kind,
                "object_name": self.task.object_name,
            },
            "success": bool(success),
            "task_source": self._task_source,
            "layout_id": self._active_layout_id,
            "next_auto_satisfied": bool(self._pending_auto_success),
            **masks,
            "paper2_cost_step": float(self._paper2_last_step_cost),
            "paper2_cost_task": float(self._paper2_task_cost),
            "paper2_cost_total": float(self._paper2_total_cost),
            "paper2_cost_breakdown": dict(self._paper2_last_step_breakdown),
        }

    def _travel_cost(self, src: str, dst: str) -> float:
        sx, sy = self.location_coords[src]
        dx, dy = self.location_coords[dst]
        return self.travel_cost_scale * float(abs(sx - dx) + abs(sy - dy))

    def _weighted_choice(self, distribution: Mapping[str, float], candidates: Sequence[str]) -> str:
        if not candidates:
            raise ValueError("No candidates available for weighted choice.")
        weights = np.array([max(float(distribution.get(name, 0.0)), 0.0) for name in candidates], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return str(self._rng.choice(candidates))
        probs = weights / total
        return str(self._rng.choice(candidates, p=probs))
