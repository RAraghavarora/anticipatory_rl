"""Configurable symbolic restaurant environment for continual-task RL."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from gymnasium import Env, spaces


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "restaurant_symbolic.yaml"


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
DEFAULT_OBJECT_KINDS: Tuple[str, ...] = ("mug", "glass", "bowl")
DEFAULT_CONTENTS: Tuple[str, ...] = ("empty", "water", "coffee", "fruit")
DEFAULT_TASK_TYPES: Tuple[str, ...] = (
    "serve_water",
    "make_coffee",
    "serve_fruit_bowl",
    "clear_containers",
    "wash_objects",
)
DEFAULT_OBJECT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("mug_red", "mug"),
    ("glass_tall", "glass"),
    ("glass_short", "glass"),
    ("bowl_small", "bowl"),
    ("bowl_large", "bowl"),
)

# Exported for compatibility with existing imports.
TASK_TYPES: Tuple[str, ...] = DEFAULT_TASK_TYPES


@dataclass(frozen=True)
class RestaurantTask:
    task_type: str
    target_location: str | None = None
    target_kind: str | None = None


@dataclass
class RestaurantObjectState:
    name: str
    kind: str
    location: str
    dirty: bool = False
    contents: str = "empty"


@dataclass
class RestaurantState:
    agent_location: str
    holding: str | None
    objects: Dict[str, RestaurantObjectState]


class RestaurantSymbolicEnv(Env):
    """Symbolic continuing-task restaurant benchmark with macro actions."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        max_task_steps: int = 24,
        success_reward: float = 15.0,
        invalid_action_penalty: float = 6.0,
        travel_cost_scale: float = 25.0,
        pick_cost: float = 25.0,
        place_cost: float = 25.0,
        wash_cost: float = 25.0,
        fill_cost: float = 25.0,
        brew_cost: float = 25.0,
        fruit_cost: float = 25.0,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._base_config = _load_config(Path(config_path)) if config_path is not None else _load_config(CONFIG_PATH)
        self._rng = np.random.default_rng(rng_seed)

        self.max_task_steps = max(1, int(max_task_steps))
        self.success_reward = float(success_reward)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.travel_cost_scale = float(travel_cost_scale)
        self.pick_cost = float(pick_cost)
        self.place_cost = float(place_cost)
        self.wash_cost = float(wash_cost)
        self.fill_cost = float(fill_cost)
        self.brew_cost = float(brew_cost)
        self.fruit_cost = float(fruit_cost)

        self._task_library: List[RestaurantTask] = []
        self._task_library_index = 0
        self._task_source = "iid"
        self._pending_auto_success = False
        self._task_steps = 0
        self._active_layout_id: str | None = None

        self._apply_schema(self._base_config)
        self._configure_paper2_cost(self._base_config.get("paper2_cost", {}))

        self.state = RestaurantState(agent_location=self._default_agent_location(), holding=None, objects={})
        self.task = RestaurantTask(task_type=self.task_types[0], target_location=self.service_locations[0] if self.service_locations else None)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
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
        self.state = RestaurantState(agent_location=agent_location, holding=None, objects=objects)
        self._task_steps = 0
        self._task_source = "iid"
        self._paper2_total_cost = 0.0
        self._paper2_task_cost = 0.0
        self._paper2_last_step_cost = 0.0
        self._paper2_last_step_breakdown = {"move_cost": 0.0, "fixed_cost": 0.0, "action_type": "none", "valid": True}
        self._resample_task()
        return self._obs(), self._info(success=False)

    def step(self, action: int):
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

        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action index {action}.")

        self._task_steps += 1
        reward = 0.0
        success = False
        truncated = False
        src_location = self.state.agent_location
        action_spec = self._decode_action(int(action))

        action_reward, valid = self._execute_action(int(action))
        reward += action_reward
        if not valid:
            reward -= self.invalid_action_penalty
        self._update_paper2_cost(action_spec=action_spec, src_location=src_location, valid=valid)

        if self._task_already_satisfied():
            reward += self.success_reward
            success = True
            completed_task = self.task
            self._advance_after_task_success(completed_task)
            self._task_steps = 0
            self._paper2_task_cost = 0.0

        if not success and self._task_steps >= self.max_task_steps:
            truncated = True
            self._paper2_task_cost = 0.0

        return self._obs(), reward, success, truncated, self._info(success=success)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_task(
        self,
        task_type: str,
        *,
        target_location: str | None = None,
        target_kind: str | None = None,
        task_source: str = "external",
    ) -> None:
        if task_type not in self.task_type_index:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if target_location is not None and target_location not in self.location_index:
            raise ValueError(f"Unknown target location: {target_location}")
        if target_kind is not None and target_kind not in self.object_kind_index:
            raise ValueError(f"Unknown target kind: {target_kind}")
        self.task = RestaurantTask(task_type=task_type, target_location=target_location, target_kind=target_kind)
        self._task_source = task_source
        self._update_pending_auto_success()

    def get_action_meanings(self) -> List[str]:
        labels = [f"pick:{name}" for name in self.object_names]
        labels.extend(f"place:{name}" for name in self.locations)
        labels.extend(["wash_held", "fill_water_held", "make_coffee_held", "fill_fruit_held"])
        return labels

    # ------------------------------------------------------------------
    # Configuration / schema helpers
    # ------------------------------------------------------------------
    def _apply_schema(self, config: Mapping[str, Any], *, merge_with_base: bool = True) -> None:
        schema: Dict[str, Any] = {}
        if merge_with_base:
            schema.update(self._base_config)
            schema.update(config)
        else:
            schema.update(config)

        location_defs = schema.get("locations")
        self.locations, self.location_coords = self._parse_locations(location_defs)
        self.location_index = {name: idx for idx, name in enumerate(self.locations)}

        self.object_kinds = tuple(str(x) for x in schema.get("object_kinds", DEFAULT_OBJECT_KINDS))
        self.contents = tuple(str(x) for x in schema.get("contents", DEFAULT_CONTENTS))
        self.task_types = tuple(str(x) for x in schema.get("task_types", DEFAULT_TASK_TYPES))
        self.object_kind_index = {name: idx for idx, name in enumerate(self.object_kinds)}
        self.content_index = {name: idx for idx, name in enumerate(self.contents)}
        self.task_type_index = {name: idx for idx, name in enumerate(self.task_types)}

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
        stations = schema.get("stations", {})
        self.station_water = str(stations.get("water", "water_station"))
        self.station_coffee = str(stations.get("coffee", "coffee_machine"))
        self.station_fruit = str(stations.get("fruit", "fruit_station"))
        self.station_wash = str(stations.get("wash", "sink"))
        if self.station_water not in self.location_index and self.locations:
            self.station_water = self.locations[0]
        if self.station_coffee not in self.location_index and self.locations:
            self.station_coffee = self.locations[0]
        if self.station_fruit not in self.location_index and self.locations:
            self.station_fruit = self.locations[0]
        if self.station_wash not in self.location_index and self.locations:
            self.station_wash = self.locations[0]

        object_specs_cfg = schema.get("object_specs", DEFAULT_OBJECT_SPECS)
        self.object_specs = self._parse_object_specs(object_specs_cfg)
        self.object_names = tuple(name for name, _ in self.object_specs)
        self.object_name_index = {name: idx for idx, name in enumerate(self.object_names)}
        self.num_objects = len(self.object_names)
        self.num_locations = len(self.locations)
        self.container_capacity = schema.get("container_capacity", None)

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

        self._pick_offset = 0
        self._place_offset = self.num_objects
        self._wash_action = self._place_offset + self.num_locations
        self._fill_action = self._wash_action + 1
        self._brew_action = self._wash_action + 2
        self._fruit_action = self._wash_action + 3
        self.action_space = spaces.Discrete(self._fruit_action + 1)

        self._held_slot = self.num_locations
        self._target_location_slot = self.num_locations
        self._target_kind_slot = len(self.object_kinds)
        obs_dim = (
            self.num_locations
            + (self.num_objects + 1)
            + self.num_objects * ((self.num_locations + 1) + 1 + len(self.contents) + len(self.object_kinds))
            + len(self.task_types)
            + (self.num_locations + 1)
            + (len(self.object_kinds) + 1)
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def _parse_locations(
        self,
        location_defs: Any,
    ) -> Tuple[Tuple[str, ...], Dict[str, Tuple[int, int]]]:
        if not isinstance(location_defs, Sequence) or isinstance(location_defs, (str, bytes)):
            return DEFAULT_LOCATIONS, dict(DEFAULT_LOCATION_COORDS)
        names: List[str] = []
        coords: Dict[str, Tuple[int, int]] = {}
        for idx, entry in enumerate(location_defs):
            if isinstance(entry, Mapping):
                name = str(entry.get("name", f"loc_{idx}"))
                coord_raw = entry.get("coord", [idx, 0])
            else:
                name = str(entry)
                coord_raw = [idx, 0]
            if name in coords:
                continue
            if isinstance(coord_raw, Sequence) and len(coord_raw) >= 2:
                cx = int(coord_raw[0])
                cy = int(coord_raw[1])
            else:
                cx, cy = idx, 0
            names.append(name)
            coords[name] = (cx, cy)
        if not names:
            return DEFAULT_LOCATIONS, dict(DEFAULT_LOCATION_COORDS)
        return tuple(names), coords

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
            if target_location is not None and str(target_location) not in self.location_index:
                continue
            if target_kind is not None and str(target_kind) not in self.object_kind_index:
                continue
            parsed.append(
                RestaurantTask(
                    task_type=task_type,
                    target_location=None if target_location is None else str(target_location),
                    target_kind=None if target_kind is None else str(target_kind),
                )
            )
        return parsed

    def _default_agent_location(self) -> str:
        if "kitchen_counter" in self.location_index:
            return "kitchen_counter"
        return self.locations[0]

    # ------------------------------------------------------------------
    # Internal task process
    # ------------------------------------------------------------------
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
                task_source="library",
            )
            return
        task_type = self._weighted_choice(self.task_distribution, self.task_types)
        if task_type in {"serve_water", "make_coffee", "serve_fruit_bowl", "clear_containers"}:
            target_location = self._weighted_choice(self.service_location_distribution, self.service_locations)
            self.set_task(task_type, target_location=target_location, target_kind=None, task_source="iid")
            return
        target_kind = self._weighted_choice(self.wash_kind_distribution, self.object_kinds)
        self.set_task(task_type, target_location=None, target_kind=target_kind, task_source="iid")

    def _task_already_satisfied(self) -> bool:
        if self.task.task_type == "serve_water":
            assert self.task.target_location is not None
            return any(
                obj.location == self.task.target_location
                and obj.kind in {"mug", "glass"}
                and obj.contents == "water"
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "make_coffee":
            assert self.task.target_location is not None
            return any(
                obj.location == self.task.target_location
                and obj.kind == "mug"
                and obj.contents == "coffee"
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "serve_fruit_bowl":
            assert self.task.target_location is not None
            return any(
                obj.location == self.task.target_location
                and obj.kind == "bowl"
                and obj.contents == "fruit"
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "clear_containers":
            assert self.task.target_location is not None
            return not any(obj.location == self.task.target_location for obj in self.state.objects.values())
        if self.task.task_type == "wash_objects":
            assert self.task.target_kind is not None
            return any(
                obj.kind == self.task.target_kind
                and not obj.dirty
                and obj.contents == "empty"
                and obj.location in self.wash_ready_locations
                for obj in self.state.objects.values()
            )
        raise ValueError(f"Unsupported task type: {self.task.task_type}")

    def _update_pending_auto_success(self) -> None:
        self._pending_auto_success = self._task_already_satisfied()

    # ------------------------------------------------------------------
    # Internal dynamics
    # ------------------------------------------------------------------
    def _decode_action(self, action: int) -> Dict[str, Any]:
        if action < self._place_offset:
            obj_name = self.object_names[action - self._pick_offset]
            return {"action_type": "pick", "obj_name": obj_name, "dst_location": None}
        if action < self._wash_action:
            location = self.locations[action - self._place_offset]
            return {"action_type": "place", "obj_name": None, "dst_location": location}
        if action == self._wash_action:
            return {"action_type": "wash", "obj_name": None, "dst_location": self.station_wash}
        if action == self._fill_action:
            return {"action_type": "fill", "obj_name": None, "dst_location": self.station_water}
        if action == self._brew_action:
            return {"action_type": "brew", "obj_name": None, "dst_location": self.station_coffee}
        if action == self._fruit_action:
            return {"action_type": "fruit", "obj_name": None, "dst_location": self.station_fruit}
        return {"action_type": "unknown", "obj_name": None, "dst_location": None}

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        if action < self._place_offset:
            obj_name = self.object_names[action - self._pick_offset]
            return self._pick_object(obj_name)
        if action < self._wash_action:
            location = self.locations[action - self._place_offset]
            return self._place_object(location)
        if action == self._wash_action:
            return self._wash_held()
        if action == self._fill_action:
            return self._fill_water_held()
        if action == self._brew_action:
            return self._make_coffee_held()
        if action == self._fruit_action:
            return self._fill_fruit_held()
        raise ValueError(f"Unsupported action index {action}.")

    def _pick_object(self, obj_name: str) -> Tuple[float, bool]:
        if self.state.holding is not None:
            return 0.0, False
        obj = self.state.objects[obj_name]
        travel = self._travel_cost(self.state.agent_location, obj.location)
        self.state.agent_location = obj.location
        self.state.holding = obj_name
        obj.location = "__held__"
        return -(travel + self.pick_cost), True

    def _place_object(self, location: str) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        travel = self._travel_cost(self.state.agent_location, location)
        self.state.agent_location = location
        self.state.holding = None
        previous_contents = obj.contents
        obj.location = location
        if location in self.dirty_drop_locations:
            if previous_contents != "empty":
                obj.dirty = True
            obj.contents = "empty"
        elif location in self.service_locations and previous_contents != "empty":
            obj.dirty = True
        return -(travel + self.place_cost), True

    def _wash_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if not obj.dirty:
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, self.station_wash)
        self.state.agent_location = self.station_wash
        obj.dirty = False
        obj.contents = "empty"
        return -(travel + self.wash_cost), True

    def _fill_water_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind not in {"mug", "glass"} or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, self.station_water)
        self.state.agent_location = self.station_water
        obj.contents = "water"
        return -(travel + self.fill_cost), True

    def _make_coffee_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind != "mug" or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, self.station_coffee)
        self.state.agent_location = self.station_coffee
        obj.contents = "coffee"
        return -(travel + self.brew_cost), True

    def _fill_fruit_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind != "bowl" or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, self.station_fruit)
        self.state.agent_location = self.station_fruit
        obj.contents = "fruit"
        return -(travel + self.fruit_cost), True

    # ------------------------------------------------------------------
    # Paper2 planner-cost bridge
    # ------------------------------------------------------------------
    def _configure_paper2_cost(self, cfg: Mapping[str, Any]) -> None:
        self.paper2_enabled = bool(cfg.get("enabled", False))
        fixed_cfg = cfg.get("fixed_costs", {})
        self.paper2_fixed_costs = {
            "pick": float(fixed_cfg.get("pick", 100.0)),
            "place": float(fixed_cfg.get("place", 100.0)),
            "wash": float(fixed_cfg.get("wash", 100.0)),
            "fill": float(fixed_cfg.get("fill", 100.0)),
            "brew": float(fixed_cfg.get("brew", 100.0)),
            "fruit": float(fixed_cfg.get("fruit", 100.0)),
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
        if not self.paper2_enabled or not valid:
            self._paper2_last_step_cost = 0.0
            self._paper2_last_step_breakdown = {
                "move_cost": 0.0,
                "fixed_cost": 0.0,
                "action_type": str(action_spec.get("action_type", "unknown")),
                "valid": bool(valid),
            }
            return
        dst_location = self.state.agent_location
        action_type = str(action_spec.get("action_type", "unknown"))
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
            return float(abs(self.location_coords[src_location][0] - self.location_coords[dst_location][0]) + abs(self.location_coords[src_location][1] - self.location_coords[dst_location][1]))
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
        # Fallback if disconnected.
        return float(abs(src[0] - dst[0]) + abs(src[1] - dst[1]))

    # ------------------------------------------------------------------
    # State sampling and encoding
    # ------------------------------------------------------------------
    def _sample_object_layout(self) -> Dict[str, RestaurantObjectState]:
        objects: Dict[str, RestaurantObjectState] = {}
        for name, kind in self.object_specs:
            location = self._weighted_choice(self.reset_location_distribution.get(kind, {}), self.locations)
            dirty, contents = self._sample_object_status(kind, location)
            objects[name] = RestaurantObjectState(name=name, kind=kind, location=location, dirty=dirty, contents=contents)
        return objects

    def _sample_object_status(self, kind: str, location: str) -> Tuple[bool, str]:
        prep_like = {self._default_agent_location(), self.station_coffee, self.station_water, self.station_fruit}
        prep_like.update(self.wash_ready_locations)
        if location in prep_like:
            return False, "empty"
        if location in self.dirty_drop_locations:
            return True, "empty"
        if location in self.service_locations:
            if kind == "mug":
                contents = str(self._rng.choice(["water", "coffee", "empty"], p=[0.35, 0.35, 0.30]))
            elif kind == "glass":
                contents = str(self._rng.choice(["water", "empty"], p=[0.7, 0.3]))
            else:
                contents = str(self._rng.choice(["fruit", "empty"], p=[0.7, 0.3]))
            dirty = contents != "empty" or bool(self._rng.random() < 0.4)
            return dirty, contents
        return False, "empty"

    def _obs(self) -> np.ndarray:
        pieces: List[np.ndarray] = []
        agent_one_hot = np.zeros((self.num_locations,), dtype=np.float32)
        agent_one_hot[self.location_index[self.state.agent_location]] = 1.0
        pieces.append(agent_one_hot)

        held_vec = np.zeros((self.num_objects + 1,), dtype=np.float32)
        if self.state.holding is None:
            held_vec[-1] = 1.0
        else:
            held_vec[self.object_name_index[self.state.holding]] = 1.0
        pieces.append(held_vec)

        for name in self.object_names:
            obj = self.state.objects[name]
            loc_vec = np.zeros((self.num_locations + 1,), dtype=np.float32)
            if obj.location == "__held__":
                loc_vec[self._held_slot] = 1.0
            else:
                loc_vec[self.location_index[obj.location]] = 1.0
            dirty_vec = np.array([1.0 if obj.dirty else 0.0], dtype=np.float32)
            contents_vec = np.zeros((len(self.contents),), dtype=np.float32)
            contents_vec[self.content_index[obj.contents]] = 1.0
            kind_vec = np.zeros((len(self.object_kinds),), dtype=np.float32)
            kind_vec[self.object_kind_index[obj.kind]] = 1.0
            pieces.extend([loc_vec, dirty_vec, contents_vec, kind_vec])

        task_type_vec = np.zeros((len(self.task_types),), dtype=np.float32)
        task_type_vec[self.task_type_index[self.task.task_type]] = 1.0
        target_location_vec = np.zeros((self.num_locations + 1,), dtype=np.float32)
        if self.task.target_location is None:
            target_location_vec[self._target_location_slot] = 1.0
        else:
            target_location_vec[self.location_index[self.task.target_location]] = 1.0
        target_kind_vec = np.zeros((len(self.object_kinds) + 1,), dtype=np.float32)
        if self.task.target_kind is None:
            target_kind_vec[self._target_kind_slot] = 1.0
        else:
            target_kind_vec[self.object_kind_index[self.task.target_kind]] = 1.0
        pieces.extend([task_type_vec, target_location_vec, target_kind_vec])
        return np.concatenate(pieces, axis=0)

    def _info(self, *, success: bool) -> Dict[str, Any]:
        return {
            "agent_location": self.state.agent_location,
            "holding": self.state.holding,
            "objects": {
                name: {
                    "kind": obj.kind,
                    "location": obj.location,
                    "dirty": bool(obj.dirty),
                    "contents": obj.contents,
                }
                for name, obj in self.state.objects.items()
            },
            "task": {
                "task_type": self.task.task_type,
                "target_location": self.task.target_location,
                "target_kind": self.task.target_kind,
            },
            "success": bool(success),
            "task_source": self._task_source,
            "layout_id": self._active_layout_id,
            "next_auto_satisfied": bool(self._pending_auto_success),
            "valid_action_mask": self._valid_action_mask(),
            "paper2_cost_step": float(self._paper2_last_step_cost),
            "paper2_cost_task": float(self._paper2_task_cost),
            "paper2_cost_total": float(self._paper2_total_cost),
            "paper2_cost_breakdown": dict(self._paper2_last_step_breakdown),
        }

    def _valid_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_space.n,), dtype=np.float32)
        for idx in range(self.action_space.n):
            mask[idx] = 1.0 if self._is_action_valid(idx) else 0.0
        return mask

    def _is_action_valid(self, action: int) -> bool:
        held = self.state.holding
        if action < self._place_offset:
            return held is None
        if action < self._wash_action:
            return held is not None
        if action == self._wash_action:
            return held is not None and self.state.objects[held].dirty
        if action == self._fill_action:
            if held is None:
                return False
            obj = self.state.objects[held]
            return obj.kind in {"mug", "glass"} and (not obj.dirty) and obj.contents == "empty"
        if action == self._brew_action:
            if held is None:
                return False
            obj = self.state.objects[held]
            return obj.kind == "mug" and (not obj.dirty) and obj.contents == "empty"
        if action == self._fruit_action:
            if held is None:
                return False
            obj = self.state.objects[held]
            return obj.kind == "bowl" and (not obj.dirty) and obj.contents == "empty"
        return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
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
