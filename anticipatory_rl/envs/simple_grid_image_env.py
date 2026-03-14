"""SimpleGrid variant with MiniWorld-inspired image observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml
from gymnasium import Env, spaces
from PIL import Image, ImageDraw

Coord = Tuple[int, int]


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_config(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


CONFIG = _load_config(CONFIG_PATH)

DEFAULT_OBJECT_NAMES = ["obj_a", "obj_b", "obj_c", "obj_d", "obj_e"]
DEFAULT_RECEPTACLE_NAMES = ["rec_a", "rec_b", "rec_c", "rec_d", "rec_e"]

OBJECT_DISTRIBUTION: Dict[str, float] = CONFIG.get("object_distribution", {})
SURFACE_DISTRIBUTION: Dict[str, float] = CONFIG.get("surface_distribution", {})
OBJECT_SOURCE_DISTRIBUTION: Dict[str, Dict[str, float]] = CONFIG.get(
    "object_source_distribution", {}
)
TASK_DISTRIBUTION: Dict[str, float] = CONFIG.get("task_distribution", {})

OBJECT_NAMES = list(OBJECT_DISTRIBUTION.keys()) or DEFAULT_OBJECT_NAMES
RECEPTACLE_LIST = list(SURFACE_DISTRIBUTION.keys()) or DEFAULT_RECEPTACLE_NAMES
DEFAULT_CLEAR_TASK_PROB = float(TASK_DISTRIBUTION.get("clear_receptacle", 0.0))

OBJECT_COLORS: Dict[str, np.ndarray] = {
    "water_bottle": np.array([0.1, 0.6, 0.95], dtype=np.float32),
    "tiffin_box": np.array([0.9, 0.3, 0.25], dtype=np.float32),
    "apple": np.array([0.3, 0.8, 0.2], dtype=np.float32),
    "soda_can": np.array([0.95, 0.7, 0.2], dtype=np.float32),
    "drinking_glass": np.array([0.7, 0.4, 0.95], dtype=np.float32),
}
RECEPTACLE_COLORS: Dict[str, np.ndarray] = {
    "kitchen_table": np.array([0.9, 0.35, 0.35], dtype=np.float32),
    "kitchen_counter": np.array([0.95, 0.7, 0.2], dtype=np.float32),
    "dining_table": np.array([0.3, 0.8, 0.6], dtype=np.float32),
    "study_table": np.array([0.35, 0.35, 0.9], dtype=np.float32),
    "shelf": np.array([0.8, 0.6, 0.2], dtype=np.float32),
}
DEFAULT_OBJECT_COLOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)
DEFAULT_RECEPTACLE_COLOR = np.array([0.8, 0.4, 0.4], dtype=np.float32)
AGENT_COLOR = np.array([0.2, 0.3, 0.95], dtype=np.float32)
AGENT_WITH_OBJECT_COLOR = np.array([0.95, 0.9, 0.1], dtype=np.float32)

BACKGROUND_RGB = (110, 207, 246)
ROOM_RGB = (128, 128, 128)
AGENT_TRIANGLE_RGB = (180, 30, 30)
AGENT_TRIANGLE_OUTLINE = (60, 0, 0)
OBJECT_OUTLINE_RGB = (10, 10, 10)


@dataclass
class SimpleGridState:
    agent: Coord
    objects: Dict[str, Coord]
    carrying: str | None = None


class SimpleGridImageEnv(Env):
    """
    Deterministic NxN grid:
    - Agent moves with four cardinal actions.
    - Objects sit on receptacle tiles; the agent can carry at most one object at a time.
    - Placement is restricted to receptacle tiles and yields success when the target object reaches its target surface.
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
        grid_size: int = 10,
        max_task_steps: int = 200,
        success_reward: float = 50.0,
        num_objects: int = len(OBJECT_NAMES),
        correct_pick_bonus: float = 1.0,
        distance_reward: bool = False,
        distance_reward_scale: float = 1.0,
        clear_receptacle_shaping_scale: float = 2.0,
        render_tile_px: int = 24,
        render_margin_px: Optional[int] = None,
        clear_task_prob: Optional[float] = None,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        loaded_config = CONFIG
        if config_path is not None:
            loaded_config = _load_config(Path(config_path))
        self.object_distribution: Dict[str, float] = loaded_config.get(
            "object_distribution", {}
        )
        self.surface_distribution: Dict[str, float] = loaded_config.get(
            "surface_distribution", {}
        )
        self.object_source_distribution: Dict[str, Dict[str, float]] = loaded_config.get(
            "object_source_distribution", {}
        )
        task_distribution: Dict[str, float] = loaded_config.get("task_distribution", {})
        self.grid_size = grid_size
        self.max_task_steps = max_task_steps
        self.success_reward = success_reward
        self.correct_pick_bonus = correct_pick_bonus
        self.distance_reward = distance_reward
        self.distance_reward_scale = distance_reward_scale
        self.clear_receptacle_shaping_scale = clear_receptacle_shaping_scale
        self.object_names = list(self.object_distribution.keys()) or list(DEFAULT_OBJECT_NAMES)
        self.receptacle_names = list(self.surface_distribution.keys()) or list(DEFAULT_RECEPTACLE_NAMES)
        self.target_object: str | None = self.object_names[0]
        self.target_receptacle: str = self.receptacle_names[0]
        self.task_type: str = "move"
        self._last_target_receptacle: str | None = None
        self.action_space = spaces.Discrete(6)
        self.max_objects = len(self.object_names)
        self.active_count = max(1, min(num_objects, self.max_objects))
        self.active_objects = self.object_names[: self.active_count]
        self.receptacles: Dict[str, List[Coord]] = {name: [] for name in self.receptacle_names}
        self._receptacle_tiles: Set[Coord] = set()
        self.tile_px = max(4, int(render_tile_px))
        self.margin_px = (
            max(2, int(render_margin_px)) if render_margin_px is not None else self.tile_px
        )
        self.render_size = self.grid_size * self.tile_px + 2 * self.margin_px
        channels = self._channel_count()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, self.render_size, self.render_size),
            dtype=np.float32,
        )
        self.state = SimpleGridState(agent=(0, 0), objects={})
        self._rng = np.random.default_rng()
        default_prob = float(np.clip(task_distribution.get("clear_receptacle", 0.0), 0.0, 1.0))
        prob = default_prob if clear_task_prob is None else clear_task_prob
        self.clear_task_prob = float(np.clip(prob, 0.0, 1.0))
        self._pending_auto_success = False

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        options = options or {}
        self._generate_receptacles()

        agent_override = options.get("agent_pos")
        if agent_override is not None:
            agent = self._validate_coord(agent_override)
        else:
            agent = self._sample_coord()

        objects: Dict[str, Coord] = {}
        occupied: Set[Coord] = {agent}
        for name in self.active_objects:
            coord = self._sample_receptacle_coord(exclude=occupied)
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

        self.state = SimpleGridState(agent=agent, objects=objects)
        self._last_target_receptacle = None
        self._resample_task()
        self._task_steps = 0
        return self._obs(), self._info()

    def step(self, action: int):
        if self._pending_auto_success:
            self._pending_auto_success = False
            reward = self.success_reward
            success = True
            horizon = False
            self._resample_task()
            self._task_steps = 0
            return self._obs(), reward, success, horizon, self._info(success=success)

        self._task_steps += 1
        reward = -1.0
        success = False
        horizon = False

        n_clear_before = (
            len(self._objects_on_receptacle(self.target_receptacle))
            if self.task_type == "clear"
            else 0
        )
        prev_obj_dist = self._distance_to_target_object()
        prev_target_dist = self._distance_to_target_receptacle()

        if action == self.MOVE_UP:
            reward += self._move_agent(0, -1)
        elif action == self.MOVE_DOWN:
            reward += self._move_agent(0, 1)
        elif action == self.MOVE_LEFT:
            reward += self._move_agent(-1, 0)
        elif action == self.MOVE_RIGHT:
            reward += self._move_agent(1, 0)
        elif action == self.PICK:
            picked = self._handle_pick()
            if picked is None:
                reward -= 5.0
            elif self.task_type == "move" and picked == self.target_object:
                reward += self.correct_pick_bonus
        elif action == self.PLACE:
            if not self._handle_place():
                reward -= 5.0

        if self.distance_reward:
            reward += self._progress_shaping(prev_obj_dist, prev_target_dist)

        target_tiles = self.receptacles[self.target_receptacle]
        carrying_empty = self.state.carrying is None
        if self.task_type == "move":
            obj_pos = self._object_position(self.target_object)
            if obj_pos in target_tiles and carrying_empty:
                reward = self.success_reward
                success = True
                self._resample_task()
                self._task_steps = 0
        else:
            if not self._objects_on_receptacle(self.target_receptacle) and carrying_empty:
                reward = self.success_reward
                success = True
                self._resample_task()
                self._task_steps = 0

        if self.task_type == "clear" and not success:
            n_clear_after = len(self._objects_on_receptacle(self.target_receptacle))
            reward += self.clear_receptacle_shaping_scale * (n_clear_before - n_clear_after)

        if not success and self._task_steps >= self.max_task_steps:
            horizon = True
            self._resample_task()
            self._task_steps = 0
        return self._obs(), reward, success, horizon, self._info(success=success)

    # ------------------------------------------------------------------ Helpers
    def _move_agent(self, dx: int, dy: int) -> float:
        ax, ay = self.state.agent
        nx = np.clip(ax + dx, 0, self.grid_size - 1)
        ny = np.clip(ay + dy, 0, self.grid_size - 1)
        penalty = -5.0 if (nx == ax and ny == ay) else 0.0
        self.state.agent = (nx, ny)
        if self.state.carrying is not None:
            self.state.objects[self.state.carrying] = (nx, ny)
        return penalty

    def _handle_pick(self) -> Optional[str]:
        if self.state.carrying is not None:
            return None
        for name, coord in self.state.objects.items():
            if (
                coord == self.state.agent
                and name in self.active_objects
            ):
                self.state.carrying = name
                self.state.objects[name] = self.state.agent
                return name
        return None

    def _handle_place(self) -> bool:
        if self.state.carrying is None:
            return False
        if not self._coord_on_receptacle(self.state.agent):
            return False
        obj_name = self.state.carrying
        self.state.carrying = None
        self.state.objects[obj_name] = self.state.agent
        return True

    def _validate_coord(self, coord: Coord) -> Coord:
        x, y = coord
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Coordinate {coord} is outside the grid.")
        return int(x), int(y)

    def _obs(self) -> np.ndarray:
        return self._encode_grid()

    def _info(self, success: bool | None = None) -> Dict[str, object]:
        can_pick = any(
            coord == self.state.agent
            and name in self.active_objects
            for name, coord in self.state.objects.items()
        ) and self.state.carrying is None
        can_place = (
            self.state.carrying is not None
            and self._coord_on_receptacle(self.state.agent)
        )
        return {
            "agent": self.state.agent,
            "objects": dict(self.state.objects),
            "carrying": self.state.carrying,
            "target_object": self.target_object,
            "target_receptacle": self.target_receptacle,
            "task_type": self.task_type,
            "success": bool(success),
            "on_receptacle": self._coord_on_receptacle(self.state.agent),
            "can_pick": can_pick,
            "can_place": can_place,
            "next_auto_satisfied": self._pending_auto_success,
        }

    # ------------------------------------------------------------------ Task helpers
    def _resample_task(self) -> None:
        if (
            self.clear_task_prob > 0.0
            and self._rng.random() < self.clear_task_prob
        ):
            if self._resample_clear_task():
                return
        self._resample_move_task()

    def _resample_move_task(self) -> None:
        rec_choices: List[str] = list(self.receptacle_names)
        if self._last_target_receptacle is not None and len(self.receptacle_names) > 1:
            rec_choices = [r for r in self.receptacle_names if r != self._last_target_receptacle]
            if not rec_choices:
                rec_choices = list(self.receptacle_names)
        obj = self._weighted_choice(self.object_distribution, self.active_objects)
        source_dist = self.object_source_distribution.get(obj, self.surface_distribution)
        rec = self._weighted_choice(source_dist, rec_choices)
        self.task_type = "move"
        self.target_object = obj
        self.target_receptacle = rec
        self._last_target_receptacle = rec
        self._update_pending_auto_success()

    def _resample_clear_task(self) -> bool:
        rec_choices: List[str] = list(self.receptacle_names)
        rec = None
        for _ in range(30):
            candidate = self._weighted_choice(self.surface_distribution, rec_choices)
            if self._objects_on_receptacle(candidate):
                rec = candidate
                break
        if rec is None:
            if not rec_choices:
                return False
            rec = self._weighted_choice(self.surface_distribution, rec_choices)
        self.task_type = "clear"
        self.target_object = None
        self.target_receptacle = rec
        self._last_target_receptacle = rec
        self._update_pending_auto_success()
        return True

    def _object_position(self, name: str | None) -> Coord | None:
        if name is None:
            return None
        if name == self.state.carrying:
            return self.state.agent
        return self.state.objects.get(name, self.state.agent)

    def _encode_grid(self) -> np.ndarray:
        frame = self._render_top_view()
        height, width, _ = frame.shape
        grid = np.zeros((self._channel_count(), height, width), dtype=np.float32)
        grid[:3] = frame.transpose(2, 0, 1)
        grid[3] = self._object_mask(self.target_object, height, width)
        grid[4] = self._receptacle_mask(self.target_receptacle, height, width)
        grid[5].fill(1.0 if self.task_type == "clear" else 0.0)
        return grid

    def _distance(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _distance_to_target_object(self) -> int | None:
        if self.target_object is None:
            return None
        obj_pos = self._object_position(self.target_object)
        if obj_pos is None:
            return None
        return self._distance(self.state.agent, obj_pos)

    def _distance_to_target_receptacle(self) -> int | None:
        tiles = self.receptacles[self.target_receptacle]
        return min(self._distance(self.state.agent, tile) for tile in tiles)

    def _progress_shaping(self, prev_obj_dist: int | None, prev_target_dist: int | None) -> float:
        if self.task_type == "clear":
            return 0.0
        if self.target_object == self.state.carrying:
            new_dist = self._distance_to_target_receptacle()
            if prev_target_dist is None or new_dist is None:
                return 0.0
            return self.distance_reward_scale * (prev_target_dist - new_dist)
        new_dist = self._distance_to_target_object()
        if prev_obj_dist is None or new_dist is None:
            return 0.0
        return self.distance_reward_scale * (prev_obj_dist - new_dist)

    def _channel_count(self) -> int:
        return 6

    def _sample_coord(self, exclude: Set[Coord] | None = None) -> Coord:
        exclude = exclude or set()
        while True:
            coord = (
                int(self._rng.integers(0, self.grid_size)),
                int(self._rng.integers(0, self.grid_size)),
            )
            if coord not in exclude:
                return coord

    def _sample_receptacle_coord(self, exclude: Set[Coord] | None = None) -> Coord:
        tiles = list(self._receptacle_tiles)
        if not tiles:
            return self._sample_coord(exclude)
        exclude = exclude or set()
        max_attempts = max(10, len(tiles) * 2)
        for _ in range(max_attempts):
            coord = tiles[int(self._rng.integers(len(tiles)))]
            if coord not in exclude:
                return coord
        return tiles[int(self._rng.integers(len(tiles)))]

    def _coord_on_receptacle(self, coord: Coord) -> bool:
        return coord in self._receptacle_tiles

    def _objects_on_receptacle(self, receptacle_name: str) -> List[str]:
        tiles = self.receptacles[receptacle_name]
        return [
            name
            for name, coord in self.state.objects.items()
            if name in self.active_objects
            and name != self.state.carrying
            and coord in tiles
        ]

    def _task_already_satisfied(self) -> bool:
        if self.state.carrying is not None:
            return False
        if self.task_type == "move":
            if self.target_object is None:
                return False
            obj_pos = self._object_position(self.target_object)
            if obj_pos is None:
                return False
            return obj_pos in self.receptacles[self.target_receptacle]
        return len(self._objects_on_receptacle(self.target_receptacle)) == 0

    def _update_pending_auto_success(self) -> None:
        self._pending_auto_success = self._task_already_satisfied()

    def _place_receptacle_tiles(
        self,
        width: int,
        height: int,
        anchor: Coord,
        used_tiles: Set[Coord],
    ) -> List[Coord] | None:
        max_x = self.grid_size - width
        max_y = self.grid_size - height
        if max_x < 0 or max_y < 0:
            return None
        anchor_x = max(0, min(anchor[0], max_x))
        anchor_y = max(0, min(anchor[1], max_y))
        candidates: List[Tuple[int, int]] = [
            (x0, y0) for x0 in range(max_x + 1) for y0 in range(max_y + 1)
        ]
        candidates.sort(key=lambda pos: abs(pos[0] - anchor_x) + abs(pos[1] - anchor_y))
        for x0, y0 in candidates:
            tiles = [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]
            if all(tile not in used_tiles for tile in tiles):
                return tiles
        return None

    def _generate_receptacles(self) -> None:
        shapes = [(1, 1), (2, 1), (1, 2), (2, 2)]
        receptacles: Dict[str, List[Coord]] = {}
        used_tiles: Set[Coord] = set()
        for name in self.receptacle_names:
            anchor = self._anchor_for_receptacle(name)
            placed = False
            for _ in range(25):
                width, height = shapes[int(self._rng.integers(len(shapes)))]
                tiles = self._place_receptacle_tiles(width, height, anchor, used_tiles)
                if tiles is not None:
                    receptacles[name] = tiles
                    used_tiles.update(tiles)
                    placed = True
                    break
            if not placed:
                tiles = self._place_receptacle_tiles(1, 1, anchor, used_tiles)
                if tiles is None:
                    raise RuntimeError("Unable to place non-overlapping receptacles on the grid.")
                receptacles[name] = tiles
                used_tiles.update(tiles)
        self.receptacles = receptacles
        self._receptacle_tiles = set(used_tiles)

    def _anchor_for_receptacle(self, name: str) -> Coord:
        size = self.grid_size
        if name == "kitchen_table":
            return (0, 0)
        if name == "kitchen_counter":
            return (size - 2, 0)
        if name == "dining_table":
            return (size - 2, size - 2)
        if name == "study_table":
            return (0, size - 2)
        if name == "shelf":
            return (size // 2 - 1, size // 2 - 1)
        return (0, 0)

    def _weighted_choice(self, distribution: Dict[str, float], candidates: Sequence[str]) -> str:
        if not candidates:
            raise ValueError("No candidates available for sampling.")
        weights = np.array([max(distribution.get(name, 0.0), 0.0) for name in candidates], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            return str(self._rng.choice(candidates))
        probs = weights / total
        return str(self._rng.choice(candidates, p=probs))

    def _tile_bounds(self, coord: Coord) -> Tuple[int, int, int, int]:
        x, y = coord
        x0 = self.margin_px + x * self.tile_px
        y0 = self.margin_px + y * self.tile_px
        x1 = x0 + self.tile_px
        y1 = y0 + self.tile_px
        return x0, y0, x1, y1

    def _render_top_view(self) -> np.ndarray:
        size = self.render_size
        tile_px = self.tile_px
        margin = self.margin_px
        img = Image.new("RGB", (size, size), BACKGROUND_RGB)
        draw = ImageDraw.Draw(img)

        draw.rectangle(
            [margin, margin, size - margin - 1, size - margin - 1],
            fill=ROOM_RGB,
        )

        for rec_name, tiles in self.receptacles.items():
            color = self._color_bytes(RECEPTACLE_COLORS.get(rec_name, DEFAULT_RECEPTACLE_COLOR))
            for tile in tiles:
                x0, y0, x1, y1 = self._tile_bounds(tile)
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)

        obj_size = int(tile_px * 0.55)
        obj_offset = max(1, (tile_px - obj_size) // 2)
        for name in self.active_objects:
            if name == self.state.carrying:
                continue
            coord = self.state.objects.get(name)
            if coord is None:
                continue
            color = self._color_bytes(OBJECT_COLORS.get(name, DEFAULT_OBJECT_COLOR))
            x0, y0, _, _ = self._tile_bounds(coord)
            x0 += obj_offset
            y0 += obj_offset
            x1 = x0 + obj_size
            y1 = y0 + obj_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=OBJECT_OUTLINE_RGB, width=2)

        ax, ay = self.state.agent
        center_x = margin + ax * tile_px + tile_px // 2
        center_y = margin + ay * tile_px + tile_px // 2
        half = tile_px // 2
        triangle = [
            (center_x - half // 2, center_y - half // 2),
            (center_x - half // 2, center_y + half // 2),
            (center_x + half // 2, center_y),
        ]
        draw.polygon(triangle, fill=AGENT_TRIANGLE_RGB, outline=AGENT_TRIANGLE_OUTLINE)

        if self.state.carrying is not None:
            last_carried = self.state.carrying
            carry_color = self._color_bytes(OBJECT_COLORS.get(last_carried, DEFAULT_OBJECT_COLOR))
            size_c = max(2, tile_px // 3)
            draw.rectangle(
                [
                    center_x - size_c // 2,
                    center_y - size_c // 2,
                    center_x + size_c // 2,
                    center_y + size_c // 2,
                ],
                outline=carry_color,
                width=3,
            )

        frame = np.asarray(img, dtype=np.float32) / 255.0
        return frame

    def _object_mask(self, obj_name: str | None, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)
        if obj_name is None:
            return mask
        coord = self._object_position(obj_name)
        if coord is None:
            return mask
        x0, y0, x1, y1 = self._tile_bounds(coord)
        mask[y0:y1, x0:x1] = 1.0
        return mask

    def _receptacle_mask(self, rec_name: str, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)
        for coord in self.receptacles[rec_name]:
            x0, y0, x1, y1 = self._tile_bounds(coord)
            mask[y0:y1, x0:x1] = 1.0
        return mask

    def _color_bytes(self, color: np.ndarray) -> Tuple[int, int, int]:
        arr = np.asarray(color, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return tuple(int(round(val * 255)) for val in arr[:3])

    def set_active_objects(self, count: int) -> None:
        count = max(1, min(count, self.max_objects))
        if count == self.active_count:
            return
        self.active_count = count
        self.active_objects = self.object_names[:count]
        occupied = set(self.state.objects.values())
        occupied.add(self.state.agent)
        for name in self.object_names:
            if name not in self.active_objects and name in self.state.objects:
                if name == self.state.carrying:
                    self.state.carrying = None
                occupied.discard(self.state.objects[name])
                del self.state.objects[name]
        for name in self.active_objects:
            if name not in self.state.objects:
                coord = self._sample_receptacle_coord(exclude=occupied)
                self.state.objects[name] = coord
                occupied.add(coord)
        self._resample_task()
