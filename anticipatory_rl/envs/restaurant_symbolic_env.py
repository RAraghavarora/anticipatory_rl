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

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
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
        reset_loc_cfg = loaded_config.get("reset_location_distribution", {})
        self.reset_location_distribution: Dict[str, Dict[str, float]] = {
            kind: {
                loc: float(reset_loc_cfg.get(kind, {}).get(loc, 0.0))
                for loc in LOCATIONS
            }
            for kind in OBJECT_KINDS
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

        self.object_specs = OBJECT_SPECS
        self.object_names = OBJECT_NAMES
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
        self._update_pending_auto_success()

    def get_action_meanings(self) -> List[str]:
        labels = [f"pick:{name}" for name in self.object_names]
        labels.extend(f"place:{name}" for name in LOCATIONS)
        labels.extend(["wash_held", "fill_water_held", "make_coffee_held", "fill_fruit_held"])
        return labels

    # ------------------------------------------------------------------
    # Internal task process
    # ------------------------------------------------------------------
    def _advance_after_task_success(self, completed_task: RestaurantTask) -> None:
        del completed_task
        self._resample_task()

    def _resample_task(self) -> None:
        task_type = self._weighted_choice(self.task_distribution, TASK_TYPES)
        if task_type in {"serve_water", "make_coffee", "serve_fruit_bowl", "clear_containers"}:
            target_location = self._weighted_choice(
                self.service_location_distribution,
                SERVICE_LOCATIONS,
            )
            self.set_task(
                task_type,
                target_location=target_location,
                target_kind=None,
                task_source="iid",
            )
            return
        target_kind = self._weighted_choice(self.wash_kind_distribution, OBJECT_KINDS)
        self.set_task(
            task_type,
            target_location=None,
            target_kind=target_kind,
            task_source="iid",
        )

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
        if location in {"sink", "bus_tub"}:
            if previous_contents != "empty":
                obj.dirty = True
            obj.contents = "empty"
        elif location in SERVICE_LOCATIONS and previous_contents != "empty":
            obj.dirty = True
        return -(travel + self.place_cost), True

    def _wash_held(self) -> Tuple[float, bool]:
        if self.state.holding is None:
            return 0.0, False
        obj = self.state.objects[self.state.holding]
        if not obj.dirty:
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
            location = self._weighted_choice(
                self.reset_location_distribution.get(kind, {}),
                LOCATIONS,
            )
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
        agent_one_hot[LOCATION_INDEX[self.state.agent_location]] = 1.0
        pieces.append(agent_one_hot)

        held_vec = np.zeros((self.num_objects + 1,), dtype=np.float32)
        if self.state.holding is None:
            held_vec[-1] = 1.0
        else:
            held_vec[OBJECT_NAME_INDEX[self.state.holding]] = 1.0
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
            "next_auto_satisfied": bool(self._pending_auto_success),
            "valid_action_mask": self._valid_action_mask(),
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
        sx, sy = LOCATION_COORDS[src]
        dx, dy = LOCATION_COORDS[dst]
        return self.travel_cost_scale * float(abs(sx - dx) + abs(sy - dy))

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
