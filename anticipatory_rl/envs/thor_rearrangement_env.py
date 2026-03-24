"""Gymnasium-style wrapper around AI2-THOR/ProcTHOR scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from gymnasium import Env, spaces

try:
    from ai2thor.controller import Controller  # type: ignore
except ImportError:  # pragma: no cover - optional extra
    Controller = None  # type: ignore


DEFAULT_SCENES: tuple[str, ...] = (
    tuple(f"FloorPlan{i}" for i in range(1, 31))
    + tuple(f"FloorPlan{i}" for i in range(201, 231))
    + tuple(f"FloorPlan{i}" for i in range(301, 331))
    + tuple(f"FloorPlan{i}" for i in range(401, 431))
)
DEFAULT_OBJECT_TYPES = ("Apple", "Bottle", "Vase", "Laptop", "Book")
DEFAULT_RECEPTACLE_TYPES = ("CounterTop", "DiningTable", "CoffeeTable", "Desk", "Sofa")


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _vectorize(position: Mapping[str, float]) -> np.ndarray:
    return np.array([position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)], dtype=np.float32)


def _transpose_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("AI2-THOR frames are expected to be HxWxC arrays.")
    return np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))


@dataclass
class ThorTask:
    object_type: str
    receptacle_type: str
    object_id: str
    receptacle_id: str


class ThorRearrangementEnv(Env):
    """Single-task rearrangement environment backed by AI2-THOR/ProcTHOR scenes."""

    MOVE_AHEAD = 0
    MOVE_BACK = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    LOOK_UP = 6
    LOOK_DOWN = 7
    PICKUP = 8
    PLACE = 9

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        controller: Optional[Controller] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        scene_pool: Sequence[str] | None = None,
        procthor_house_paths: Optional[Sequence[str | Path]] = None,
        procthor_house_sampler: Optional[Callable[[np.random.Generator], Mapping[str, Any]]] = None,
        frame_width: int = 400,
        frame_height: int = 300,
        move_magnitude: float = 0.25,
        rotate_degrees: float = 15.0,
        look_degrees: float = 10.0,
        max_task_steps: int = 200,
        success_reward: float = 50.0,
        step_penalty: float = -0.01,
        pickup_reward: float = 1.0,
        place_reward: float = 2.0,
        failure_penalty: float = -5.0,
        distance_reward: bool = True,
        distance_scale: float = 1.0,
        randomize_on_reset: bool = True,
        scene_retry_limit: int = 5,
        object_types: Sequence[str] | None = None,
        receptacle_types: Sequence[str] | None = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if Controller is None and controller is None:
            raise ImportError("ai2thor is not installed; install ai2thor>=4.0.0 to use ThorRearrangementEnv.")
        self.scene_pool: tuple[str, ...] = tuple(scene_pool) if scene_pool else DEFAULT_SCENES
        self.object_types = tuple(object_types) if object_types else DEFAULT_OBJECT_TYPES
        self.receptacle_types = tuple(receptacle_types) if receptacle_types else DEFAULT_RECEPTACLE_TYPES
        self.max_task_steps = max(1, max_task_steps)
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.pickup_reward = pickup_reward
        self.place_reward = place_reward
        self.failure_penalty = failure_penalty
        self.distance_reward = distance_reward
        self.distance_scale = distance_scale
        self.randomize_on_reset = randomize_on_reset
        self._scene_retry_limit = max(1, scene_retry_limit)
        self.move_magnitude = move_magnitude
        self.rotate_degrees = rotate_degrees
        self.look_degrees = look_degrees
        self._rng = _rng(rng_seed)
        self._task: Optional[ThorTask] = None
        self._step_count = 0
        self._prev_goal_distance = 0.0
        self._house_paths = tuple(Path(p) for p in procthor_house_paths or [])
        self._house_sampler = procthor_house_sampler
        self._reset_kwargs = dict(reset_kwargs or {})
        base_kwargs = dict(controller_kwargs or {})
        base_kwargs.setdefault("width", frame_width)
        base_kwargs.setdefault("height", frame_height)
        self._owns_controller = controller is None
        self.controller: Controller = controller or Controller(**base_kwargs)  # type: ignore[assignment]
        self._last_event = None
        self._current_scene_id: Optional[str] = None
        self._held_object_id: Optional[str] = None
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, frame_height, frame_width),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            self._rng = _rng(seed)
        options = options or {}
        attempts = 0
        while True:
            event = self._reset_scene(options)
            if self.randomize_on_reset:
                event = self._randomize_scene(seed)
            if self._assign_task(event.metadata):
                break
            attempts += 1
            if attempts >= self._scene_retry_limit:
                raise RuntimeError("Unable to find target object/receptacle pairing in sampled scenes.")
        self._held_object_id = self._current_inventory(event)
        self._prev_goal_distance = self._goal_distance(event.metadata)
        self._step_count = 0
        self._last_event = event
        obs = _transpose_frame(event.frame)
        return obs, self._info(event, last_action=None)

    def step(self, action: int):  # type: ignore[override]
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action index {action}.")
        event = self._execute_action(action)
        self._step_count += 1
        success = self._is_success(event)
        horizon = self._step_count >= self.max_task_steps
        reward = self._compute_reward(action, event, success, horizon)
        obs = _transpose_frame(event.frame)
        info = self._info(event, last_action=action)
        self._last_event = event
        return obs, reward, success, horizon, info

    def close(self):  # type: ignore[override]
        if self._owns_controller and self.controller is not None:
            self.controller.stop()
        super().close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_action(self, action: int):
        action_map = {
            self.MOVE_AHEAD: ("MoveAhead", {"moveMagnitude": self.move_magnitude}),
            self.MOVE_BACK: ("MoveBack", {"moveMagnitude": self.move_magnitude}),
            self.MOVE_LEFT: ("MoveLeft", {"moveMagnitude": self.move_magnitude}),
            self.MOVE_RIGHT: ("MoveRight", {"moveMagnitude": self.move_magnitude}),
            self.ROTATE_LEFT: ("RotateLeft", {"degrees": self.rotate_degrees}),
            self.ROTATE_RIGHT: ("RotateRight", {"degrees": self.rotate_degrees}),
            self.LOOK_UP: ("LookUp", {"degrees": self.look_degrees}),
            self.LOOK_DOWN: ("LookDown", {"degrees": self.look_degrees}),
        }
        if action in action_map:
            name, params = action_map[action]
            event = self.controller.step(action=name, **params)
        elif action == self.PICKUP:
            event = self._pickup_target()
        elif action == self.PLACE:
            event = self._place_target()
        else:  # pragma: no cover - sanity guard
            raise ValueError(f"Unsupported action index {action}.")
        self._held_object_id = self._current_inventory(event)
        return event

    def _pickup_target(self):
        if not self._task:
            return self.controller.step(action="Pass")
        return self.controller.step(
            action="PickupObject",
            objectId=self._task.object_id,
            forceAction=False,
        )

    def _place_target(self):
        if not self._task or self._held_object_id != self._task.object_id:
            return self.controller.step(action="Pass")
        return self.controller.step(
            action="PutObject",
            objectId=self._task.receptacle_id,
        )

    def _reset_scene(self, options: Mapping[str, Any]):
        manual_scene = options.get("scene")
        manual_house = options.get("house")
        if manual_house is not None:
            event = self._reset_with_house(manual_house)
        elif manual_scene is not None:
            event = self.controller.reset(scene=str(manual_scene), **self._reset_kwargs)
            self._current_scene_id = str(manual_scene)
        elif self._house_sampler is not None:
            house_spec = self._house_sampler(self._rng)
            event = self._reset_with_house(house_spec)
        elif self._house_paths:
            path = self._house_paths[int(self._rng.integers(len(self._house_paths)))]
            event = self._reset_with_house(path)
        else:
            scene = self.scene_pool[int(self._rng.integers(len(self.scene_pool)))]
            event = self.controller.reset(scene=scene, **self._reset_kwargs)
            self._current_scene_id = scene
        return event

    def _reset_with_house(self, spec: Any):
        house_data = self._normalize_house_spec(spec)
        event = self.controller.reset(scene="Procedural", house=house_data, **self._reset_kwargs)
        name = house_data.get("metadata", {}).get("id") if isinstance(house_data, Mapping) else None
        self._current_scene_id = name or "Procedural"
        return event

    def _normalize_house_spec(self, spec: Any) -> MutableMapping[str, Any]:
        if isinstance(spec, Mapping):
            return dict(spec)
        if isinstance(spec, Path):
            return json.loads(spec.read_text())
        if isinstance(spec, str):
            path = Path(spec)
            if path.exists():
                return json.loads(path.read_text())
            return json.loads(spec)
        if hasattr(spec, "to_dict"):
            return dict(spec.to_dict())  # type: ignore[call-arg]
        if hasattr(spec, "to_json"):
            data = spec.to_json()  # type: ignore[attr-defined]
            return json.loads(data) if isinstance(data, str) else dict(data)
        raise TypeError("Unsupported house specification type for ProcTHOR scenes.")

    def _randomize_scene(self, seed: Optional[int]):
        random_seed = int(seed) if seed is not None else int(self._rng.integers(np.iinfo(np.int32).max))
        event = self.controller.step(
            action="InitialRandomSpawn",
            randomSeed=random_seed,
            forceVisible=False,
            placeStationary=True,
        )
        return event

    def _assign_task(self, metadata: Mapping[str, Any]) -> bool:
        objects = list(metadata.get("objects", []))
        pickup_candidates = [
            obj
            for obj in objects
            if obj.get("pickupable", False)
            and obj.get("objectType") in self.object_types
            and not obj.get("isPickedUp", False)
        ]
        if not pickup_candidates:
            return False
        receptacle_candidates = [
            obj
            for obj in objects
            if obj.get("receptacle", False)
            and obj.get("objectType") in self.receptacle_types
        ]
        if not receptacle_candidates:
            return False
        pick_idx = int(self._rng.integers(len(pickup_candidates)))
        rec_idx = int(self._rng.integers(len(receptacle_candidates)))
        target_obj = pickup_candidates[pick_idx]
        target_rec = receptacle_candidates[rec_idx]
        self._task = ThorTask(
            object_type=str(target_obj.get("objectType")),
            receptacle_type=str(target_rec.get("objectType")),
            object_id=str(target_obj.get("objectId")),
            receptacle_id=str(target_rec.get("objectId")),
        )
        return True

    def _current_inventory(self, event: Any) -> Optional[str]:
        inventory = event.metadata.get("inventoryObjects", [])
        if inventory:
            return str(inventory[0].get("objectId"))
        return None

    def _goal_distance(self, metadata: Mapping[str, Any]) -> float:
        if not self._task:
            return 0.0
        agent = metadata.get("agent", {})
        agent_pos = _vectorize(agent.get("position", {}))
        target_id = self._task.receptacle_id if self._held_object_id == self._task.object_id else self._task.object_id
        target = self._find_object(metadata.get("objects", []), target_id)
        if target is None:
            return 0.0
        target_pos = _vectorize(target.get("position", {}))
        return float(np.linalg.norm(agent_pos - target_pos))

    def _find_object(self, objects: Sequence[Mapping[str, Any]], object_id: str) -> Optional[Mapping[str, Any]]:
        for obj in objects:
            if obj.get("objectId") == object_id:
                return obj
        return None

    def _is_success(self, event: Any) -> bool:
        if not self._task:
            return False
        metadata = event.metadata
        target = self._find_object(metadata.get("objects", []), self._task.object_id)
        if not target:
            return False
        parents = target.get("parentReceptacles") or []
        return self._task.receptacle_id in parents

    def _compute_reward(self, action: int, event: Any, success: bool, horizon: bool) -> float:
        reward = self.step_penalty
        if self.distance_reward:
            current_dist = self._goal_distance(event.metadata)
            reward += self.distance_scale * (self._prev_goal_distance - current_dist)
            self._prev_goal_distance = current_dist
        last_success = bool(event.metadata.get("lastActionSuccess", False))
        if action == self.PICKUP and last_success:
            reward += self.pickup_reward
        if action == self.PLACE and last_success:
            reward += self.place_reward
        if success:
            reward += self.success_reward
        elif horizon:
            reward += self.failure_penalty
        return float(reward)

    def _info(self, event: Any, last_action: Optional[int]) -> Dict[str, Any]:
        info = {
            "scene": self._current_scene_id,
            "task": self._task,
            "step": self._step_count,
            "held_object_id": self._held_object_id,
            "last_action": last_action,
            "last_action_success": bool(event.metadata.get("lastActionSuccess", False)),
            "error": event.metadata.get("errorMessage"),
        }
        return info
