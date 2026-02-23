"""Minimal 3x3 gridworld with a single object and target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import Env, spaces

Coord = Tuple[int, int]


OBJECT_NAMES = ["obj_a", "obj_b"]
RECEPTACLE_LIST = ["rec_a", "rec_b"]


@dataclass
class SimpleGridState:
    agent: Coord
    objects: Dict[str, Coord]
    carrying: str | None


class SimpleGridEnv(Env):
    """
    Deterministic NxN grid:
    - Agent moves with four cardinal actions.
    - A single object can be picked up when the agent stands on it and dropped elsewhere.
    - Goal is to place the object on the fixed target coordinate.
    """

    metadata = {"render.modes": []}

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICK = 4
    PLACE = 5

    def __init__(
        self,
        grid_size: int = 4,
        max_task_steps: int = 200,
        success_reward: float = 50.0,
        num_objects: int = len(OBJECT_NAMES),
        correct_pick_bonus: float = 1.0,
        distance_reward: bool = False,
        distance_reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_task_steps = max_task_steps
        self.success_reward = success_reward
        self.correct_pick_bonus = correct_pick_bonus
        self.distance_reward = distance_reward
        self.distance_reward_scale = distance_reward_scale
        self.target_object: str = OBJECT_NAMES[0]
        self.target_receptacle: str = RECEPTACLE_LIST[0]
        self._last_target_receptacle: str | None = None
        self.action_space = spaces.Discrete(6)
        self.max_objects = len(OBJECT_NAMES)
        self.active_count = max(1, min(num_objects, self.max_objects))
        self.active_objects = OBJECT_NAMES[: self.active_count]
        self.receptacles = {
            "rec_a": (0, 0),
            "rec_b": (self.grid_size - 1, self.grid_size - 1),
        }
        obs_dim = self._feature_length()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.state = SimpleGridState(agent=(0, 0), objects={}, carrying=None)
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        options = options or {}

        agent_override = options.get("agent_pos")
        if agent_override is not None:
            agent = self._validate_coord(agent_override)
        else:
            agent = self._sample_coord()

        objects: Dict[str, Coord] = {}
        occupied = {agent}
        for name in self.active_objects:
            coord = self._sample_coord(exclude=occupied)
            objects[name] = coord
            occupied.add(coord)

        object_under_agent = options.get("object_under_agent")
        if object_under_agent:
            obj_name = (
                object_under_agent
                if isinstance(object_under_agent, str)
                else self.active_objects[0]
            )
            if obj_name not in self.active_objects:
                raise ValueError(f"Object '{obj_name}' is not active.")
            objects[obj_name] = agent

        self.state = SimpleGridState(agent=agent, objects=objects, carrying=None)
        self._last_target_receptacle = None
        self._resample_task()
        self._task_steps = 0
        return self._obs(), self._info()

    def step(self, action: int):
        self._task_steps += 1
        reward = -1.0
        success = False
        horizon = False

        prev_obj_dist = self._distance_to_target_object()
        prev_target_dist = self._distance_to_target_receptacle()

        if action == self.MOVE_UP:
            self._move_agent(0, -1)
        elif action == self.MOVE_DOWN:
            self._move_agent(0, 1)
        elif action == self.MOVE_LEFT:
            self._move_agent(-1, 0)
        elif action == self.MOVE_RIGHT:
            self._move_agent(1, 0)
        elif action == self.PICK:
            picked = self._handle_pick()
            if picked == self.target_object:
                reward += self.correct_pick_bonus
        elif action == self.PLACE:
            self._handle_place()

        if self.distance_reward:
            if self.state.carrying == self.target_object:
                new_dist = self._distance_to_target_receptacle()
                if prev_target_dist is None:
                    prev_target_dist = new_dist
                reward += self.distance_reward_scale * (prev_target_dist - new_dist)
            else:
                new_dist = self._distance_to_target_object()
                if prev_obj_dist is None:
                    prev_obj_dist = new_dist
                reward += self.distance_reward_scale * (prev_obj_dist - new_dist)

        target_coord = self.receptacles[self.target_receptacle]
        obj_pos = self._object_position(self.target_object)
        if obj_pos == target_coord:
            reward = self.success_reward
            success = True
            self._resample_task()
            self._task_steps = 0
        elif self._task_steps >= self.max_task_steps:
            horizon = True
            self._resample_task()
            self._task_steps = 0
        return self._obs(), reward, success, horizon, self._info(success=success)

    # ------------------------------------------------------------------ Helpers
    def _move_agent(self, dx: int, dy: int) -> None:
        ax, ay = self.state.agent
        nx = np.clip(ax + dx, 0, self.grid_size - 1)
        ny = np.clip(ay + dy, 0, self.grid_size - 1)
        self.state.agent = (nx, ny)
        if self.state.carrying is not None:
            self.state.objects[self.state.carrying] = (nx, ny)

    def _handle_pick(self) -> Optional[str]:
        if self.state.carrying is not None:
            return None
        for name, coord in self.state.objects.items():
            if coord == self.state.agent and name in self.active_objects:
                self.state.carrying = name
                return name
        return None

    def _handle_place(self) -> None:
        if self.state.carrying is None:
            return
        self.state.objects[self.state.carrying] = self.state.agent
        self.state.carrying = None

    def _validate_coord(self, coord: Coord) -> Coord:
        x, y = coord
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Coordinate {coord} is outside the grid.")
        return int(x), int(y)

    def _obs(self) -> np.ndarray:
        return self._encode_state()

    def _info(self, success: bool | None = None) -> Dict[str, object]:
        return {
            "agent": self.state.agent,
            "objects": dict(self.state.objects),
            "carrying": self.state.carrying,
            "target_object": self.target_object,
            "target_receptacle": self.target_receptacle,
            "success": bool(success),
        }

    # ------------------------------------------------------------------ Task helpers
    def _resample_task(self) -> None:
        obj = self._rng.choice(self.active_objects)
        rec_choices = RECEPTACLE_LIST
        if self._last_target_receptacle is not None and len(RECEPTACLE_LIST) > 1:
            rec_choices = [r for r in RECEPTACLE_LIST if r != self._last_target_receptacle]
            if not rec_choices:
                rec_choices = RECEPTACLE_LIST
        rec = self._rng.choice(rec_choices)
        target_coord = self.receptacles[rec]
        attempts = 0
        while self.state.objects.get(obj) == target_coord and attempts < 10:
            obj = self._rng.choice(self.active_objects)
            rec = self._rng.choice(rec_choices)
            target_coord = self.receptacles[rec]
            attempts += 1
        self.target_object = obj
        self.target_receptacle = rec
        self._last_target_receptacle = rec

    def _object_position(self, name: str) -> Coord:
        if self.state.carrying == name:
            return self.state.agent
        return self.state.objects.get(name, self.state.agent)

    def _encode_state(self) -> np.ndarray:
        grid = self.grid_size
        features: List[float] = []
        ax, ay = self.state.agent
        features.append(ax / (grid - 1))
        features.append(ay / (grid - 1))

        holding = np.zeros(len(OBJECT_NAMES) + 1, dtype=np.float32)
        if self.state.carrying is None:
            holding[-1] = 1.0
        else:
            holding[OBJECT_NAMES.index(self.state.carrying)] = 1.0
        features.extend(holding.tolist())

        tile_dim = grid * grid
        for name in OBJECT_NAMES:
            vec = np.zeros(tile_dim + 1, dtype=np.float32)
            coord = self.state.objects.get(name)
            if self.state.carrying == name:
                vec[-1] = 1.0
            elif coord is not None:
                idx = coord[1] * grid + coord[0]
                vec[idx] = 1.0
            features.extend(vec.tolist())

        target_obj_vec = np.zeros(len(OBJECT_NAMES), dtype=np.float32)
        target_obj_vec[OBJECT_NAMES.index(self.target_object)] = 1.0
        features.extend(target_obj_vec.tolist())

        target_rec_vec = np.zeros(len(RECEPTACLE_LIST), dtype=np.float32)
        target_rec_vec[RECEPTACLE_LIST.index(self.target_receptacle)] = 1.0
        features.extend(target_rec_vec.tolist())
        return np.asarray(features, dtype=np.float32)

    def _distance(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _distance_to_target_object(self) -> int | None:
        obj_pos = self._object_position(self.target_object)
        if obj_pos is None:
            return None
        return self._distance(self.state.agent, obj_pos)

    def _distance_to_target_receptacle(self) -> int | None:
        target_coord = self.receptacles[self.target_receptacle]
        return self._distance(self.state.agent, target_coord)

    def _feature_length(self) -> int:
        grid = self.grid_size
        tile_dim = grid * grid
        return (
            2
            + (len(OBJECT_NAMES) + 1)
            + len(OBJECT_NAMES) * (tile_dim + 1)
            + len(OBJECT_NAMES)
            + len(RECEPTACLE_LIST)
        )

    def _sample_coord(self, exclude: Set[Coord] | None = None) -> Coord:
        exclude = exclude or set()
        while True:
            coord = (
                int(self._rng.integers(0, self.grid_size)),
                int(self._rng.integers(0, self.grid_size)),
            )
            if coord not in exclude:
                return coord

    def set_active_objects(self, count: int) -> None:
        count = max(1, min(count, self.max_objects))
        if count == self.active_count:
            return
        self.active_count = count
        self.active_objects = OBJECT_NAMES[:count]
        occupied = set(self.state.objects.values())
        occupied.add(self.state.agent)
        for name in OBJECT_NAMES:
            if name not in self.active_objects and name in self.state.objects:
                if self.state.carrying == name:
                    self.state.carrying = None
                occupied.discard(self.state.objects[name])
                del self.state.objects[name]
        for name in self.active_objects:
            if name not in self.state.objects:
                coord = self._sample_coord(exclude=occupied)
                self.state.objects[name] = coord
                occupied.add(coord)
        self._resample_task()
