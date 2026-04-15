"""Compact restaurant-style persistent-task environment.

This is a symbolic Gymnasium environment inspired by the restaurant domain in
Dhakal et al. It models reusable tableware, preparation stations, and
continuing task streams where the way an agent completes the current task can
change the cost of future tasks.
"""

from __future__ import annotations

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
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected config file path, got directory: {path}. "
            f"Pass a YAML file such as {CONFIG_PATH}."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


DEFAULT_CONFIG = _load_config(CONFIG_PATH)

LOCATIONS: Tuple[str, ...] = (
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
LOCATION_INDEX = {name: idx for idx, name in enumerate(LOCATIONS)}
LOCATION_COORDS: Dict[str, Tuple[int, int]] = {
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
SERVICE_LOCATIONS: Tuple[str, ...] = ("pass_counter", "table_left", "table_right")
WASH_READY_LOCATIONS: Tuple[str, ...] = ("dish_rack", "kitchen_counter")

OBJECT_KINDS: Tuple[str, ...] = ("mug", "glass", "bowl")
OBJECT_KIND_INDEX = {name: idx for idx, name in enumerate(OBJECT_KINDS)}
CONTENTS: Tuple[str, ...] = ("empty", "water", "coffee", "fruit")
CONTENT_INDEX = {name: idx for idx, name in enumerate(CONTENTS)}
TASK_TYPES: Tuple[str, ...] = (
    "serve_water",
    "make_coffee",
    "serve_fruit_bowl",
    "clear_containers",
    "wash_objects",
)
TASK_TYPE_INDEX = {name: idx for idx, name in enumerate(TASK_TYPES)}


def _default_objects() -> Tuple[Tuple[str, str], ...]:
    return (
        ("mug_red", "mug"),
        ("mug_blue", "mug"),
        ("glass_tall", "glass"),
        ("glass_short", "glass"),
        ("bowl_small", "bowl"),
        ("bowl_large", "bowl"),
    )


OBJECT_SPECS: Tuple[Tuple[str, str], ...] = _default_objects()
OBJECT_NAMES: Tuple[str, ...] = tuple(name for name, _ in OBJECT_SPECS)
OBJECT_NAME_INDEX = {name: idx for idx, name in enumerate(OBJECT_NAMES)}


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

    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        max_task_steps: int = 24,
        success_reward: float = 15.0,
        invalid_action_penalty: float = 6.0,
        travel_cost_scale: float = 1.0,
        pick_cost: float = 1.0,
        place_cost: float = 1.0,
        wash_cost: float = 2.0,
        fill_cost: float = 1.0,
        brew_cost: float = 2.0,
        fruit_cost: float = 2.0,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()
        loaded_config = DEFAULT_CONFIG
        if config_path is not None:
            loaded_config = _load_config(Path(config_path))
        self.object_specs = self._normalize_object_specs(loaded_config.get("object_specs"))
        self.task_distribution: Dict[str, float] = {
            name: float(loaded_config.get("task_distribution", {}).get(name, 0.0))
            for name in TASK_TYPES
        }
        self.service_location_distribution: Dict[str, float] = {
            name: float(loaded_config.get("service_location_distribution", {}).get(name, 0.0))
            for name in SERVICE_LOCATIONS
        }
        self.wash_kind_distribution: Dict[str, float] = {
            name: float(loaded_config.get("wash_kind_distribution", {}).get(name, 0.0))
            for name in OBJECT_KINDS
        }
        transition_cfg = loaded_config.get("task_transition_distribution", {})
        self.task_transition_distribution: Dict[str, Dict[str, float]] = {
            task_type: {
                name: float(transition_cfg.get(task_type, {}).get(name, 0.0))
                for name in TASK_TYPES
            }
            for task_type in TASK_TYPES
        }
        kind_transition_cfg = loaded_config.get("wash_followup_task_distribution", {})
        self.wash_followup_task_distribution: Dict[str, Dict[str, float]] = {
            kind: {
                name: float(kind_transition_cfg.get(kind, {}).get(name, 0.0))
                for name in TASK_TYPES
            }
            for kind in OBJECT_KINDS
        }
        same_loc_cfg = loaded_config.get("same_location_followup_prob", {})
        self.same_location_followup_prob: Dict[str, float] = {
            task_type: float(same_loc_cfg.get(task_type, 0.0))
            for task_type in TASK_TYPES
        }
        self.active_service_location_stickiness = float(
            loaded_config.get("active_service_location_stickiness", 0.0)
        )
        self.followup_wash_from_cleared_prob = float(
            loaded_config.get("followup_wash_from_cleared_prob", 0.0)
        )
        consume_cfg = loaded_config.get("service_consumption_prob", {})
        self.service_consumption_prob: Dict[str, float] = {
            task_type: float(consume_cfg.get(task_type, 0.0))
            for task_type in TASK_TYPES
        }
        reset_loc_cfg = loaded_config.get("reset_location_distribution", {})
        self.reset_location_distribution: Dict[str, Dict[str, float]] = {
            kind: {
                loc: float(reset_loc_cfg.get(kind, {}).get(loc, 0.0))
                for loc in LOCATIONS
            }
            for kind in OBJECT_KINDS
        }
        service_content_cfg = loaded_config.get("service_contents_distribution", {})
        self.service_contents_distribution: Dict[str, Dict[str, float]] = {
            kind: {
                content: float(service_content_cfg.get(kind, {}).get(content, 0.0))
                for content in CONTENTS
            }
            for kind in OBJECT_KINDS
        }
        dirty_cfg = loaded_config.get("service_empty_dirty_prob", {})
        self.service_empty_dirty_prob: Dict[str, float] = {
            kind: float(dirty_cfg.get(kind, 0.0))
            for kind in OBJECT_KINDS
        }
        capacity_cfg = loaded_config.get("location_capacity", {})
        self.location_capacity: Dict[str, int] = {
            name: max(1, int(capacity_cfg.get(name, 99)))
            for name in LOCATIONS
        }
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
        self._rng = np.random.default_rng(rng_seed)
        if render_mode not in {None, "rgb_array", "ansi"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        self.object_names = tuple(name for name, _ in self.object_specs)
        self.object_name_index = {name: idx for idx, name in enumerate(self.object_names)}
        self.num_objects = len(self.object_names)
        self.num_locations = len(LOCATIONS)

        self._pick_offset = 0
        self._place_offset = self.num_objects
        self._wash_action = self._place_offset + self.num_locations
        self._fill_action = self._wash_action + 1
        self._brew_action = self._wash_action + 2
        self._fruit_action = self._wash_action + 3
        self.action_space = spaces.Discrete(self._fruit_action + 1)

        self._held_slot = self.num_locations
        self._target_location_slot = len(LOCATIONS)
        self._target_kind_slot = len(OBJECT_KINDS)

        obs_dim = (
            self.num_locations  # agent location
            + (self.num_objects + 1)  # held object or empty
            + self.num_objects * (
                (self.num_locations + 1) + 1 + len(CONTENTS) + len(OBJECT_KINDS)
            )
            + len(TASK_TYPES)
            + (len(LOCATIONS) + 1)
            + (len(OBJECT_KINDS) + 1)
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.state = RestaurantState(agent_location="kitchen_counter", holding=None, objects={})
        self.task = RestaurantTask(task_type="serve_water", target_location="pass_counter")
        self._task_steps = 0
        self._pending_auto_success = False
        self._task_source = "iid"
        self._active_service_location: Optional[str] = None
        self._task_context: Dict[str, Any] = {}
        self._last_picked_object: Optional[str] = None
        self._last_placed_object: Optional[str] = None

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
        agent_location = str(options.get("agent_location", "kitchen_counter"))
        objects = self._sample_object_layout()
        self.state = RestaurantState(
            agent_location=agent_location,
            holding=None,
            objects=objects,
        )
        self._task_steps = 0
        self._task_source = "iid"
        self._active_service_location = self._weighted_choice(
            self.service_location_distribution,
            SERVICE_LOCATIONS,
        )
        self._last_picked_object = None
        self._last_placed_object = None
        self._resample_task()
        return self._obs(), self._info(success=False)

    def step(self, action: int):
        if self._pending_auto_success:
            completed_task = self.task
            self._pending_auto_success = False
            reward = self.success_reward
            success = True
            horizon = False
            self._advance_after_task_success(completed_task)
            self._task_steps = 0
            return self._obs(), reward, success, horizon, self._info(success=success)

        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action index {action}.")

        self._task_steps += 1
        reward = 0.0
        success = False
        horizon = False

        action_reward, valid = self._execute_action(int(action))
        reward += action_reward
        if not valid:
            reward -= self.invalid_action_penalty

        if self._task_already_satisfied():
            reward += self.success_reward
            success = True
            completed_task = self.task
            self._advance_after_task_success(completed_task)
            self._task_steps = 0

        if not success and self._task_steps >= self.max_task_steps:
            horizon = True

        return self._obs(), reward, success, horizon, self._info(success=success)

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
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if target_location is not None and target_location not in LOCATIONS:
            raise ValueError(f"Unknown target location: {target_location}")
        if target_kind is not None and target_kind not in OBJECT_KINDS:
            raise ValueError(f"Unknown target kind: {target_kind}")
        self.task = RestaurantTask(
            task_type=task_type,
            target_location=target_location,
            target_kind=target_kind,
        )
        self._task_source = task_source
        self._start_task_context()
        self._update_pending_auto_success()

    def get_action_meanings(self) -> List[str]:
        labels = [f"pick:{name}" for name in self.object_names]
        labels.extend(f"place:{name}" for name in LOCATIONS)
        labels.extend(["wash_held", "fill_water_held", "make_coffee_held", "fill_fruit_held"])
        return labels

    def render(self):
        mode = self.render_mode or "rgb_array"
        if mode == "ansi":
            return self._render_ansi()
        if mode == "rgb_array":
            return self._render_rgb_array()
        raise ValueError(f"Unsupported render mode: {mode}")

    # ------------------------------------------------------------------
    # Internal task process
    # ------------------------------------------------------------------
    def _advance_after_task_success(self, completed_task: RestaurantTask) -> None:
        self._apply_between_task_transition(completed_task)
        if completed_task.target_location in SERVICE_LOCATIONS:
            self._active_service_location = completed_task.target_location
        next_task, task_source = self._sample_next_task(completed_task)
        self.set_task(
            next_task.task_type,
            target_location=next_task.target_location,
            target_kind=next_task.target_kind,
            task_source=task_source,
        )

    def _resample_task(self) -> None:
        next_task, task_source = self._sample_iid_task()
        self.set_task(
            next_task.task_type,
            target_location=next_task.target_location,
            target_kind=next_task.target_kind,
            task_source=task_source,
        )

    def _sample_iid_task(self) -> Tuple[RestaurantTask, str]:
        task_type = self._weighted_choice(self.task_distribution, TASK_TYPES)
        if task_type in {"serve_water", "make_coffee", "serve_fruit_bowl", "clear_containers"}:
            target_location = self._sample_service_location(None)
            return (
                RestaurantTask(
                    task_type=task_type,
                    target_location=target_location,
                    target_kind=None,
                ),
                "iid",
            )
        target_kind = self._weighted_choice(self.wash_kind_distribution, OBJECT_KINDS)
        return (
            RestaurantTask(
                task_type=task_type,
                target_location=None,
                target_kind=target_kind,
            ),
            "iid",
        )

    def _sample_next_task(self, completed_task: RestaurantTask) -> Tuple[RestaurantTask, str]:
        source_distribution = self._transition_distribution_for(completed_task)
        if source_distribution is None:
            return self._sample_iid_task()

        next_task_type = self._weighted_choice(source_distribution, TASK_TYPES)
        if next_task_type == "wash_objects":
            target_kind = self._sample_followup_wash_kind(completed_task)
            return (
                RestaurantTask(task_type="wash_objects", target_location=None, target_kind=target_kind),
                f"transition:{completed_task.task_type}->wash_objects",
            )

        target_location = self._sample_service_location(completed_task)
        return (
            RestaurantTask(task_type=next_task_type, target_location=target_location, target_kind=None),
            f"transition:{completed_task.task_type}->{next_task_type}",
        )

    def _transition_distribution_for(
        self,
        completed_task: RestaurantTask,
    ) -> Optional[Mapping[str, float]]:
        if completed_task.task_type == "wash_objects" and completed_task.target_kind is not None:
            by_kind = self.wash_followup_task_distribution.get(completed_task.target_kind, {})
            if any(weight > 0.0 for weight in by_kind.values()):
                return by_kind
        by_task = self.task_transition_distribution.get(completed_task.task_type, {})
        if any(weight > 0.0 for weight in by_task.values()):
            return by_task
        return None

    def _sample_service_location(self, completed_task: Optional[RestaurantTask]) -> str:
        if (
            completed_task is not None
            and completed_task.target_location in SERVICE_LOCATIONS
            and self._rng.random()
            < self.same_location_followup_prob.get(completed_task.task_type, 0.0)
        ):
            return str(completed_task.target_location)
        if (
            self._active_service_location in SERVICE_LOCATIONS
            and self._rng.random() < self.active_service_location_stickiness
        ):
            return str(self._active_service_location)
        return self._weighted_choice(self.service_location_distribution, SERVICE_LOCATIONS)

    def _sample_followup_wash_kind(self, completed_task: RestaurantTask) -> str:
        if completed_task.task_type == "clear_containers":
            washable_kinds = list(self._task_context.get("washable_kinds", []))
            if washable_kinds and self._rng.random() < self.followup_wash_from_cleared_prob:
                weighted_kinds = {kind: float(washable_kinds.count(kind)) for kind in set(washable_kinds)}
                return self._weighted_choice(weighted_kinds, OBJECT_KINDS)
        return self._weighted_choice(self.wash_kind_distribution, OBJECT_KINDS)

    def _start_task_context(self) -> None:
        context: Dict[str, Any] = {}
        if self.task.task_type == "clear_containers" and self.task.target_location is not None:
            target_objects = [
                obj for obj in self.state.objects.values() if obj.location == self.task.target_location
            ]
            context["initial_target_objects"] = [obj.name for obj in target_objects]
            context["washable_kinds"] = [
                obj.kind for obj in target_objects if obj.dirty or obj.contents != "empty"
            ]
        self._task_context = context

    def _apply_between_task_transition(self, completed_task: RestaurantTask) -> None:
        if completed_task.task_type not in {"serve_water", "make_coffee", "serve_fruit_bowl"}:
            return
        probability = self.service_consumption_prob.get(completed_task.task_type, 0.0)
        if probability <= 0.0 or self._rng.random() >= probability:
            return
        served_name = self._select_served_object(completed_task)
        if served_name is None:
            return
        served_obj = self.state.objects[served_name]
        if completed_task.target_location is None or served_obj.location != completed_task.target_location:
            return
        served_obj.contents = "empty"
        served_obj.dirty = True

    def _select_served_object(self, completed_task: RestaurantTask) -> Optional[str]:
        if completed_task.target_location is None:
            return None
        candidates: List[str] = []
        for name, obj in self.state.objects.items():
            if obj.location != completed_task.target_location:
                continue
            if completed_task.task_type == "serve_water":
                if obj.kind in {"mug", "glass"} and obj.contents == "water":
                    candidates.append(name)
            elif completed_task.task_type == "make_coffee":
                if obj.kind == "mug" and obj.contents == "coffee":
                    candidates.append(name)
            elif completed_task.task_type == "serve_fruit_bowl":
                if obj.kind == "bowl" and obj.contents == "fruit":
                    candidates.append(name)
        if not candidates:
            return None
        if self._last_placed_object in candidates:
            return self._last_placed_object
        if self._last_picked_object in candidates:
            return self._last_picked_object
        return str(self._rng.choice(candidates))

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
            return not any(
                obj.location == self.task.target_location
                for obj in self.state.objects.values()
            )
        if self.task.task_type == "wash_objects":
            assert self.task.target_kind is not None
            return any(
                obj.kind == self.task.target_kind
                and not obj.dirty
                and obj.contents == "empty"
                and obj.location in WASH_READY_LOCATIONS
                for obj in self.state.objects.values()
            )
        raise ValueError(f"Unsupported task type: {self.task.task_type}")

    def _update_pending_auto_success(self) -> None:
        self._pending_auto_success = self._task_already_satisfied()

    # ------------------------------------------------------------------
    # Internal dynamics
    # ------------------------------------------------------------------
    def _execute_action(self, action: int) -> Tuple[float, bool]:
        if action < self._place_offset:
            obj_name = self.object_names[action - self._pick_offset]
            return self._pick_object(obj_name)
        if action < self._wash_action:
            location = LOCATIONS[action - self._place_offset]
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
        self._last_picked_object = obj_name
        return -(travel + self.pick_cost), True

    def _place_object(self, location: str) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        if not self._location_has_space(location):
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        travel = self._travel_cost(self.state.agent_location, location)
        self.state.agent_location = location
        self.state.holding = None
        previous_contents = obj.contents
        obj.location = location
        if location == "sink":
            if previous_contents != "empty":
                obj.dirty = True
            obj.contents = "empty"
        elif location == "bus_tub":
            obj.dirty = True
        elif location in SERVICE_LOCATIONS and previous_contents != "empty":
            obj.dirty = True
        self._last_placed_object = obj.name
        return -(travel + self.place_cost), True

    def _wash_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if not obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, "sink")
        self.state.agent_location = "sink"
        obj.dirty = False
        obj.contents = "empty"
        return -(travel + self.wash_cost), True

    def _fill_water_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind not in {"mug", "glass"} or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, "water_station")
        self.state.agent_location = "water_station"
        obj.contents = "water"
        return -(travel + self.fill_cost), True

    def _make_coffee_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind != "mug" or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, "coffee_machine")
        self.state.agent_location = "coffee_machine"
        obj.contents = "coffee"
        return -(travel + self.brew_cost), True

    def _fill_fruit_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if obj.kind != "bowl" or obj.dirty or obj.contents != "empty":
            return 0.0, False
        travel = self._travel_cost(self.state.agent_location, "fruit_station")
        self.state.agent_location = "fruit_station"
        obj.contents = "fruit"
        return -(travel + self.fruit_cost), True

    # ------------------------------------------------------------------
    # State sampling and encoding
    # ------------------------------------------------------------------
    def _sample_object_layout(self) -> Dict[str, RestaurantObjectState]:
        objects: Dict[str, RestaurantObjectState] = {}
        for name, kind in self.object_specs:
            location = self._sample_reset_location(kind, objects)
            dirty, contents = self._sample_object_status(kind, location)
            objects[name] = RestaurantObjectState(
                name=name,
                kind=kind,
                location=location,
                dirty=dirty,
                contents=contents,
            )
        return objects

    def _sample_object_status(self, kind: str, location: str) -> Tuple[bool, str]:
        if location in {"dish_rack", "kitchen_counter", "coffee_machine", "water_station", "fruit_station"}:
            return False, "empty"
        if location == "sink":
            return True, "empty"
        if location == "bus_tub":
            return True, "empty"
        if location in SERVICE_LOCATIONS:
            configured = self.service_contents_distribution.get(kind, {})
            if any(weight > 0.0 for weight in configured.values()):
                contents = self._weighted_choice(configured, CONTENTS)
            elif kind == "mug":
                contents = str(self._rng.choice(["water", "coffee", "empty"], p=[0.35, 0.35, 0.30]))
            elif kind == "glass":
                contents = str(self._rng.choice(["water", "empty"], p=[0.7, 0.3]))
            else:
                contents = str(self._rng.choice(["fruit", "empty"], p=[0.7, 0.3]))
            dirty = contents != "empty" or bool(
                self._rng.random() < self.service_empty_dirty_prob.get(kind, 0.4)
            )
            return dirty, contents
        return False, "empty"

    def _obs(self) -> np.ndarray:
        pieces: List[np.ndarray] = []

        agent_one_hot = np.zeros((self.num_locations,), dtype=np.float32)
        agent_one_hot[LOCATION_INDEX[self.state.agent_location]] = 1.0
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
                loc_vec[LOCATION_INDEX[obj.location]] = 1.0
            dirty_vec = np.array([1.0 if obj.dirty else 0.0], dtype=np.float32)
            contents_vec = np.zeros((len(CONTENTS),), dtype=np.float32)
            contents_vec[CONTENT_INDEX[obj.contents]] = 1.0
            kind_vec = np.zeros((len(OBJECT_KINDS),), dtype=np.float32)
            kind_vec[OBJECT_KIND_INDEX[obj.kind]] = 1.0
            pieces.extend([loc_vec, dirty_vec, contents_vec, kind_vec])

        task_type_vec = np.zeros((len(TASK_TYPES),), dtype=np.float32)
        task_type_vec[TASK_TYPE_INDEX[self.task.task_type]] = 1.0
        target_location_vec = np.zeros((len(LOCATIONS) + 1,), dtype=np.float32)
        if self.task.target_location is None:
            target_location_vec[self._target_location_slot] = 1.0
        else:
            target_location_vec[LOCATION_INDEX[self.task.target_location]] = 1.0
        target_kind_vec = np.zeros((len(OBJECT_KINDS) + 1,), dtype=np.float32)
        if self.task.target_kind is None:
            target_kind_vec[self._target_kind_slot] = 1.0
        else:
            target_kind_vec[OBJECT_KIND_INDEX[self.task.target_kind]] = 1.0
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
            "active_service_location": self._active_service_location,
            "next_auto_satisfied": bool(self._pending_auto_success),
            "valid_action_mask": self._valid_action_mask(),
        }

    def _valid_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_space.n,), dtype=np.float32)
        for idx in range(self.action_space.n):
            mask[idx] = 1.0 if self._is_action_valid(idx) else 0.0
        return mask

    def _render_ansi(self) -> str:
        lines = [
            f"Task: {self.task.task_type}"
            + (
                f" @ {self.task.target_location}"
                if self.task.target_location is not None
                else f" [{self.task.target_kind}]"
            ),
            f"Agent: {self.state.agent_location}",
            f"Holding: {self.state.holding or 'none'}",
            f"Task source: {self._task_source}",
            f"Active service location: {self._active_service_location or 'none'}",
            "Objects:",
        ]
        for location in LOCATIONS:
            objs = [
                self._format_object_label(obj)
                for obj in self.state.objects.values()
                if obj.location == location
            ]
            marker = " <agent>" if self.state.agent_location == location else ""
            lines.append(f"  - {location}:{marker} {', '.join(objs) if objs else '(empty)'}")
        if self.state.holding is not None:
            lines.append(
                "  - __held__: "
                + self._format_object_label(self.state.objects[self.state.holding])
            )
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

        fig = Figure(figsize=(12.5, 7.5), constrained_layout=True)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        side = fig.add_subplot(gs[0, 1])

        ax.set_facecolor("#f7f6f3")
        side.set_facecolor("#fffdf8")

        min_x = min(coord[0] for coord in LOCATION_COORDS.values())
        max_x = max(coord[0] for coord in LOCATION_COORDS.values())
        min_y = min(coord[1] for coord in LOCATION_COORDS.values())
        max_y = max(coord[1] for coord in LOCATION_COORDS.values())

        for location, (gx, gy) in LOCATION_COORDS.items():
            x = float(gx)
            y = float(max_y - gy)
            is_target = self.task.target_location == location
            is_agent = self.state.agent_location == location
            face = self._location_face_color(location)
            edge = "#d97706" if is_target else "#50525b"
            rect = FancyBboxPatch(
                (x, y),
                0.95,
                0.95,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face,
                edgecolor=edge,
                linewidth=2.8 if is_target else 1.6,
            )
            ax.add_patch(rect)
            ax.text(
                x + 0.475,
                y + 0.77,
                location.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="600",
                color="#262626",
            )
            cap = self.location_capacity.get(location, 99)
            ax.text(
                x + 0.86,
                y + 0.12,
                f"cap {cap}",
                ha="right",
                va="bottom",
                fontsize=8,
                color="#5c6066",
            )
            if is_agent:
                agent = Circle(
                    (x + 0.12, y + 0.84),
                    0.08,
                    facecolor="#b91c1c",
                    edgecolor="#450a0a",
                    linewidth=1.2,
                )
                ax.add_patch(agent)

            objects_here = [
                obj for obj in self.state.objects.values() if obj.location == location
            ]
            slot_count = max(cap, max(1, len(objects_here)))
            for idx, obj in enumerate(objects_here):
                chip_h = 0.16
                top_margin = 0.18
                usable_h = 0.68
                if slot_count <= 1:
                    cy = y + 0.42
                else:
                    cy = y + top_margin + idx * min(chip_h + 0.04, usable_h / slot_count)
                chip = FancyBboxPatch(
                    (x + 0.12, cy),
                    0.72,
                    chip_h,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    facecolor=self._object_fill_color(obj.kind),
                    edgecolor="#111827" if obj.dirty else "#374151",
                    linewidth=2.0 if obj.dirty else 1.2,
                    linestyle="--" if obj.dirty else "-",
                )
                ax.add_patch(chip)
                ax.text(
                    x + 0.16,
                    cy + chip_h / 2.0,
                    self._object_short_label(obj),
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    fontweight="600",
                    color="#111827",
                )

        ax.set_xlim(min_x - 0.15, max_x + 1.15)
        ax.set_ylim(min_y - 0.15, max_y + 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Restaurant State", fontsize=16, fontweight="700", color="#1f2937")

        side.axis("off")
        y = 0.96
        side.text(0.03, y, "Task", fontsize=16, fontweight="700", color="#1f2937", transform=side.transAxes)
        y -= 0.07
        side.text(
            0.03,
            y,
            self._task_summary_line(),
            fontsize=12,
            color="#374151",
            transform=side.transAxes,
        )
        y -= 0.06
        side.text(
            0.03,
            y,
            f"Task source: {self._task_source}",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )
        y -= 0.05
        side.text(
            0.03,
            y,
            f"Auto-satisfied next: {'yes' if self._pending_auto_success else 'no'}",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )
        y -= 0.05
        side.text(
            0.03,
            y,
            f"Active table: {self._active_service_location or 'none'}",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )

        y -= 0.10
        side.text(0.03, y, "Agent", fontsize=16, fontweight="700", color="#1f2937", transform=side.transAxes)
        y -= 0.07
        side.text(
            0.03,
            y,
            f"Location: {self.state.agent_location}",
            fontsize=11,
            color="#374151",
            transform=side.transAxes,
        )
        y -= 0.05
        side.text(
            0.03,
            y,
            f"Holding: {self._format_holding_label()}",
            fontsize=11,
            color="#374151",
            transform=side.transAxes,
        )

        y -= 0.10
        side.text(0.03, y, "Legend", fontsize=16, fontweight="700", color="#1f2937", transform=side.transAxes)
        y -= 0.07
        legend_rows = [
            ("Mug", self._object_fill_color("mug")),
            ("Glass", self._object_fill_color("glass")),
            ("Bowl", self._object_fill_color("bowl")),
        ]
        for label, color in legend_rows:
            chip = Rectangle((0.03, y - 0.015), 0.05, 0.03, transform=side.transAxes, facecolor=color, edgecolor="#374151")
            side.add_patch(chip)
            side.text(0.10, y, label, fontsize=10.5, color="#374151", va="center", transform=side.transAxes)
            y -= 0.055
        side.text(
            0.03,
            y,
            "Dashed border = dirty",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )
        y -= 0.05
        side.text(
            0.03,
            y,
            "Chip suffix: [E/W/C/F]",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )
        y -= 0.045
        side.text(
            0.03,
            y,
            "Target location = amber border",
            fontsize=10.5,
            color="#4b5563",
            transform=side.transAxes,
        )

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        return image

    def _is_action_valid(self, action: int) -> bool:
        held = self.state.holding
        if action < self._place_offset:
            return held is None
        if action < self._wash_action:
            if held is None:
                return False
            location = LOCATIONS[action - self._place_offset]
            return self._location_has_space(location)
        if action == self._wash_action:
            return (
                held is not None
                and self.state.objects[held].dirty
                and self.state.objects[held].contents == "empty"
            )
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
        sx, sy = LOCATION_COORDS[src]
        dx, dy = LOCATION_COORDS[dst]
        return self.travel_cost_scale * float(abs(sx - dx) + abs(sy - dy))

    def _location_face_color(self, location: str) -> str:
        palette = {
            "kitchen_counter": "#e7dcc6",
            "coffee_machine": "#d7d2cb",
            "water_station": "#d9ecff",
            "fruit_station": "#fce7b2",
            "dish_rack": "#dbe7f1",
            "sink": "#cfe5ef",
            "pass_counter": "#f0dbc7",
            "table_left": "#edd7cf",
            "bus_tub": "#d7d4eb",
            "table_right": "#edd7cf",
        }
        return palette.get(location, "#ececec")

    def _object_fill_color(self, kind: str) -> str:
        return {
            "mug": "#f59e0b",
            "glass": "#60a5fa",
            "bowl": "#34d399",
        }.get(kind, "#d1d5db")

    def _content_short_code(self, contents: str) -> str:
        return {
            "empty": "E",
            "water": "W",
            "coffee": "C",
            "fruit": "F",
        }.get(contents, "?")

    def _object_short_label(self, obj: RestaurantObjectState) -> str:
        base = obj.name.replace("_", " ")
        return f"{base} [{self._content_short_code(obj.contents)}]"

    def _format_object_label(self, obj: RestaurantObjectState) -> str:
        dirty = ", dirty" if obj.dirty else ", clean"
        return f"{obj.name}<{obj.kind}, {obj.contents}{dirty}>"

    def _task_summary_line(self) -> str:
        if self.task.target_location is not None:
            return f"{self.task.task_type} @ {self.task.target_location}"
        if self.task.target_kind is not None:
            return f"{self.task.task_type} [{self.task.target_kind}]"
        return self.task.task_type

    def _format_holding_label(self) -> str:
        if self.state.holding is None:
            return "none"
        return self._format_object_label(self.state.objects[self.state.holding])

    def _weighted_choice(
        self,
        distribution: Mapping[str, float],
        candidates: Sequence[str],
    ) -> str:
        weights = np.array([max(float(distribution.get(name, 0.0)), 0.0) for name in candidates], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return str(self._rng.choice(candidates))
        probs = weights / total
        return str(self._rng.choice(candidates, p=probs))

    def _sample_reset_location(
        self,
        kind: str,
        placed_objects: Mapping[str, RestaurantObjectState],
    ) -> str:
        feasible = [
            location
            for location in LOCATIONS
            if self._location_occupancy(location, placed_objects) < self.location_capacity.get(location, 99)
        ]
        if not feasible:
            feasible = list(LOCATIONS)
        distribution = self.reset_location_distribution.get(kind, {})
        return self._weighted_choice(distribution, feasible)

    def _location_occupancy(
        self,
        location: str,
        objects: Optional[Mapping[str, RestaurantObjectState]] = None,
    ) -> int:
        source = self.state.objects if objects is None else objects
        return sum(1 for obj in source.values() if obj.location == location)

    def _location_has_space(self, location: str) -> bool:
        return self._location_occupancy(location) < self.location_capacity.get(location, 99)

    def _normalize_object_specs(
        self,
        object_specs: Any,
    ) -> Tuple[Tuple[str, str], ...]:
        if not object_specs:
            return OBJECT_SPECS
        normalized: List[Tuple[str, str]] = []
        for item in object_specs:
            if isinstance(item, Mapping):
                name = str(item["name"])
                kind = str(item["kind"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                name = str(item[0])
                kind = str(item[1])
            else:
                raise ValueError(f"Invalid object spec entry: {item!r}")
            if kind not in OBJECT_KINDS:
                raise ValueError(f"Unsupported object kind in object_specs: {kind}")
            normalized.append((name, kind))
        names = [name for name, _ in normalized]
        if len(set(names)) != len(names):
            raise ValueError("object_specs must use unique object names.")
        return tuple(normalized)
