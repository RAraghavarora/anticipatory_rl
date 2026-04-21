"""Image-based RL environment for the reproduced paper1 blockworld."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces


Coord = Tuple[int, int]

COLORED_REGIONS: Tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "orange",
    "teal",
    # "purple",
    # "yellow",
)
WHITE_REGIONS: Tuple[str, ...] = (
    "white_1",
    "white_2",
    # "white_3",
)
BLOCK_NAMES: Tuple[str, ...] = (
    "a", 
    "b", 
    "c", 
    "d", 
    "e", 
    # "f", 
    # "g", 
    # "h"
)
COLORED_BLOCK_COLORS: Tuple[str, ...] = (
    "red",
    "blue",
    "green",
    # "orange",
    # "teal",
)

BACKGROUND_RGB = (236, 236, 236)
GRID_RGB = (215, 215, 215)
ROBOT_RGB = (180, 30, 30)
ROBOT_OUTLINE_RGB = (70, 0, 0)
BLOCK_OUTLINE_RGB = (20, 20, 20)

COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 70, 70),
    "blue": (70, 110, 220),
    "green": (80, 180, 90),
    "orange": (235, 145, 60),
    "teal": (45, 160, 175),
    "purple": (155, 90, 210),
    "yellow": (235, 200, 70),
    "cyan": (80, 200, 205),
    "pink": (225, 80, 195),
    "brown": (160, 95, 55),
    "white": (250, 250, 250),
}

DEFAULT_REGION_ANCHORS: Tuple[Coord, ...] = (
    (0, 0),
    (2, 0),
    (4, 0),
    (6, 0),
    (8, 0),
    (0, 2),
    (2, 2),
    (4, 2),
    (6, 2),
    (8, 2),
)


def region_color(region: str) -> str:
    if region.startswith("white"):
        return "white"
    return region


@dataclass(frozen=True)
class Task:
    assignments: Tuple[Tuple[str, str], ...]

    @property
    def blocks(self) -> Tuple[str, ...]:
        return tuple(block for block, _ in self.assignments)

    @property
    def target_regions(self) -> Tuple[str, ...]:
        return tuple(region for _, region in self.assignments)


@dataclass(frozen=True)
class WorldConfig:
    width: int = 10
    height: int = 10
    region_size: int = 2
    move_cost: int = 25
    pick_cost: int = 100
    place_cost: int = 100
    region_layout: Tuple[Tuple[str, Coord], ...] | None = None
    block_colors: Tuple[Tuple[str, str], ...] | None = None

    @property
    def robot_start(self) -> Coord:
        return (self.width // 2, self.height // 2)

    @property
    def all_regions(self) -> Tuple[str, ...]:
        if self.region_layout is not None:
            return tuple(region for region, _ in self.region_layout)
        return COLORED_REGIONS + WHITE_REGIONS

    @property
    def nonwhite_regions(self) -> Tuple[str, ...]:
        if self.region_layout is not None:
            return tuple(region for region in self.all_regions if not region.startswith("white"))
        return COLORED_REGIONS

    @property
    def white_regions(self) -> Tuple[str, ...]:
        if self.region_layout is not None:
            return tuple(region for region in self.all_regions if region.startswith("white"))
        return WHITE_REGIONS

    @property
    def all_blocks(self) -> Tuple[str, ...]:
        return BLOCK_NAMES

    @property
    def block_color_map(self) -> Dict[str, str]:
        if self.block_colors is not None:
            return dict(self.block_colors)
        mapping: Dict[str, str] = {}
        for idx, block in enumerate(BLOCK_NAMES):
            if idx < len(COLORED_BLOCK_COLORS):
                mapping[block] = COLORED_BLOCK_COLORS[idx]
            else:
                mapping[block] = "white"
        return mapping

    @property
    def nonwhite_blocks(self) -> Tuple[str, ...]:
        return tuple(
            block
            for block in BLOCK_NAMES
            if self.block_color_map[block] != "white"
        )

    @property
    def white_blocks(self) -> Tuple[str, ...]:
        return tuple(
            block
            for block in BLOCK_NAMES
            if self.block_color_map[block] == "white"
        )

    @property
    def region_anchors(self) -> Dict[str, Coord]:
        if self.region_layout is not None:
            return dict(self.region_layout)
        return dict(zip(self.all_regions, DEFAULT_REGION_ANCHORS))

    @property
    def region_coords(self) -> Dict[str, Coord]:
        return self.region_anchors

    @property
    def region_tiles(self) -> Dict[str, Tuple[Coord, ...]]:
        tiles: Dict[str, Tuple[Coord, ...]] = {}
        for region, anchor in self.region_anchors.items():
            ax, ay = anchor
            coords = tuple(
                (ax + dx, ay + dy)
                for dx in range(self.region_size)
                for dy in range(self.region_size)
            )
            tiles[region] = coords
        return tiles

    @property
    def region_cells(self) -> Tuple[Coord, ...]:
        return tuple(coord for tiles in self.region_tiles.values() for coord in tiles)

    @property
    def all_cells(self) -> Tuple[Coord, ...]:
        return tuple(
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
        )

    @property
    def manipulation_cells(self) -> Tuple[Coord, ...]:
        cells = set()
        for tile in self.region_cells:
            for neighbor in self.neighbors(tile):
                cells.add(neighbor)
        return tuple(sorted(cells))

    def region_for_coord(self, coord: Coord) -> str | None:
        for region, coords in self.region_tiles.items():
            if coord in coords:
                return region
        return None

    def neighbors(self, coord: Coord) -> Tuple[Coord, ...]:
        x, y = coord
        neighbors: List[Coord] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return tuple(neighbors)

    def block_color(self, block: str) -> str:
        return self.block_color_map[block]

    @classmethod
    def sample(cls, rng: random.Random) -> "WorldConfig":
        anchors = _sample_region_anchors(
            rng,
            width=cls.width,
            height=cls.height,
            region_size=cls.region_size,
            count=len(COLORED_REGIONS) + len(WHITE_REGIONS),
        )
        rng.shuffle(anchors)
        region_layout = tuple(zip(COLORED_REGIONS + WHITE_REGIONS, anchors))

        colored_blocks = set(rng.sample(BLOCK_NAMES, k=len(COLORED_BLOCK_COLORS)))
        color_pool = list(COLORED_BLOCK_COLORS)
        rng.shuffle(color_pool)
        block_colors: List[Tuple[str, str]] = []
        for block in BLOCK_NAMES:
            if block in colored_blocks:
                block_colors.append((block, color_pool.pop()))
            else:
                block_colors.append((block, "white"))
        return cls(
            region_layout=region_layout,
            block_colors=tuple(block_colors),
        )


@dataclass
class WorldState:
    robot: Coord
    placements: Dict[str, Coord]
    holding: str | None = None

    def clone(self) -> "WorldState":
        return WorldState(
            robot=self.robot,
            placements=dict(self.placements),
            holding=self.holding,
        )

    def block_at(self, coord: Coord) -> str | None:
        for block, placement in self.placements.items():
            if placement == coord:
                return block
        return None

    def is_task_satisfied(self, task: Task, config: WorldConfig) -> bool:
        for block, region in task.assignments:
            if self.placements.get(block) not in config.region_tiles[region]:
                return False
        return self.holding is None


class WorldGenerator:
    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()

    def sample_initial_state(self, rng: random.Random) -> WorldState:
        regions = list(self.config.all_regions)
        rng.shuffle(regions)
        placements: Dict[str, Coord] = {}
        for block, region in zip(self.config.all_blocks, regions):
            coord = rng.choice(list(self.config.region_tiles[region]))
            placements[block] = coord
        free_cells = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in placements.values()
        ]
        return WorldState(
            robot=rng.choice(free_cells),
            placements=placements,
            holding=None,
        )

    def sample_task_library(
        self,
        rng: random.Random,
        count: int = 20,
    ) -> List[Task]:
        chosen: List[Task] = []
        seen = set()
        max_attempts = max(200, count * 20)
        attempts = 0
        while len(chosen) < count and attempts < max_attempts:
            attempts += 1
            task_size = rng.choice((1, 2))
            blocks = tuple(rng.sample(self.config.nonwhite_blocks, k=task_size))
            regions = tuple(rng.sample(self.config.nonwhite_regions, k=task_size))
            task = Task(tuple(zip(blocks, regions)))
            frozen = tuple(sorted(task.assignments))
            if frozen in seen:
                continue
            seen.add(frozen)
            chosen.append(task)
        if len(chosen) != count:
            raise RuntimeError(
                f"Unable to sample {count} unique tasks from the current environment."
            )
        return chosen

    def sample_task_sequence(
        self,
        rng: random.Random,
        task_library: Sequence[Task],
        length: int,
    ) -> List[Task]:
        return [rng.choice(task_library) for _ in range(length)]


def _sample_region_anchors(
    rng: random.Random,
    *,
    width: int,
    height: int,
    region_size: int,
    count: int,
) -> List[Coord]:
    anchors = [
        (x, y)
        for y in range(height - region_size + 1)
        for x in range(width - region_size + 1)
    ]
    for _ in range(2000):
        rng.shuffle(anchors)
        chosen: List[Coord] = []
        occupied: set[Coord] = set()
        for anchor in anchors:
            tiles = set(_tiles_for_anchor(anchor, region_size))
            if occupied & tiles:
                continue
            chosen.append(anchor)
            occupied |= tiles
            if len(chosen) == count:
                return chosen
    raise RuntimeError("Failed to sample non-overlapping region anchors.")


def _tiles_for_anchor(anchor: Coord, region_size: int) -> Iterable[Coord]:
    ax, ay = anchor
    for dx in range(region_size):
        for dy in range(region_size):
            yield (ax + dx, ay + dy)


@dataclass
class BlockworldInfo:
    robot: Coord
    placements: Dict[str, Coord]
    holding: str | None
    task_assignments: Tuple[Tuple[str, str], ...]
    task_size: int
    success: bool
    can_pick: bool
    can_place: bool
    next_auto_satisfied: bool


class Paper1BlockworldImageEnv(Env):
    """Primitive-control blockworld with image observations and task channels."""

    metadata = {"render.modes": []}

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICK = 4
    PLACE = 5

    def __init__(
        self,
        *,
        base_config: WorldConfig | None = None,
        task_library_size: int = 20,
        max_task_steps: int = 64,
        success_reward: float = 12.0,
        step_penalty: float = 1.0,
        invalid_action_penalty: float = 5.0,
        correct_pick_bonus: float = 1.0,
        render_tile_px: int = 24,
        render_margin_px: Optional[int] = None,
        procedural_layout: bool = True,
    ) -> None:
        super().__init__()
        self.base_config = base_config or WorldConfig()
        self.task_library_size = max(1, int(task_library_size))
        self.max_task_steps = max(1, int(max_task_steps))
        self.success_reward = float(success_reward)
        self.step_penalty = float(step_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.correct_pick_bonus = float(correct_pick_bonus)
        self.procedural_layout = bool(procedural_layout)

        self.tile_px = max(4, int(render_tile_px))
        self.margin_px = (
            max(2, int(render_margin_px)) if render_margin_px is not None else self.tile_px
        )
        self.render_size = self.base_config.width * self.tile_px + 2 * self.margin_px

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8, self.render_size, self.render_size),
            dtype=np.float32,
        )

        self.config = self.base_config
        self.generator = WorldGenerator(self.config)
        self.state = WorldState(robot=self.config.robot_start, placements={}, holding=None)
        self.task_library: List[Task] = []
        self.current_task = Task((("a", "red"),))
        self._pending_auto_success = False
        self._task_steps = 0
        self._py_rng = random.Random()
        self._rgb_canvas = np.zeros((3, self.render_size, self.render_size), dtype=np.float32)
        self._obs_canvas = np.zeros((8, self.render_size, self.render_size), dtype=np.float32)
        self._rebuild_static_canvas()

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._py_rng = random.Random(seed)
        options = options or {}

        self.config = self._select_config(options)
        self.generator = WorldGenerator(self.config)
        self.state = self._select_state(options)
        self.task_library = self._select_task_library(options)
        self.current_task = self._select_current_task(options)
        self._task_steps = 0
        self._rebuild_static_canvas()
        self._update_pending_auto_success()
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
        reward = -self.step_penalty
        success = False
        horizon = False

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
                reward -= self.invalid_action_penalty
            elif picked in self.current_task.blocks:
                reward += self.correct_pick_bonus
        elif action == self.PLACE:
            if not self._handle_place():
                reward -= self.invalid_action_penalty
        else:
            raise ValueError(f"Unsupported action: {action}")

        if self._task_already_satisfied():
            reward = self.success_reward
            success = True
            self._resample_task()
            self._task_steps = 0
        elif self._task_steps >= self.max_task_steps:
            horizon = True
            self._resample_task()
            self._task_steps = 0

        return self._obs(), reward, success, horizon, self._info(success=success)

    def _select_config(self, options: Mapping[str, object]) -> WorldConfig:
        override = options.get("world_config")
        coerced = self._coerce_world_config(override)
        if coerced is not None:
            return coerced
        if self.procedural_layout:
            return WorldConfig.sample(self._py_rng)
        return self.base_config

    def _select_state(self, options: Mapping[str, object]) -> WorldState:
        state_override = options.get("state")
        state = self._coerce_world_state(state_override)
        if state is None:
            state = self.generator.sample_initial_state(self._py_rng)
        placements_override = options.get("placements")
        if placements_override is not None:
            placements: Dict[str, Coord] = {}
            for block, coord in dict(placements_override).items():
                coord_tuple = self._validate_coord(tuple(coord))
                placements[str(block)] = coord_tuple
            self._validate_region_capacity(placements)
            state.placements = placements
        robot_override = options.get("robot_pos")
        if robot_override is not None:
            state.robot = self._validate_coord(tuple(robot_override))
        holding_override = options.get("holding", state.holding)
        state.holding = None if holding_override is None else str(holding_override)
        return state

    def _coerce_world_config(self, raw_config: object) -> WorldConfig | None:
        if raw_config is None:
            return None
        if isinstance(raw_config, WorldConfig):
            return raw_config
        required = {
            "width",
            "height",
            "move_cost",
            "pick_cost",
            "place_cost",
        }
        if not all(hasattr(raw_config, name) for name in required):
            return None
        region_layout = getattr(raw_config, "region_layout", None)
        if region_layout is None and hasattr(raw_config, "region_coords"):
            region_layout = tuple(getattr(raw_config, "region_coords").items())
        block_colors = getattr(raw_config, "block_colors", None)
        if block_colors is None and hasattr(raw_config, "block_color_map"):
            block_colors = tuple(getattr(raw_config, "block_color_map").items())
        region_size = getattr(raw_config, "region_size", 1)
        return WorldConfig(
            width=int(getattr(raw_config, "width")),
            height=int(getattr(raw_config, "height")),
            region_size=int(region_size),
            move_cost=int(getattr(raw_config, "move_cost")),
            pick_cost=int(getattr(raw_config, "pick_cost")),
            place_cost=int(getattr(raw_config, "place_cost")),
            region_layout=tuple(region_layout) if region_layout is not None else None,
            block_colors=tuple(block_colors) if block_colors is not None else None,
        )

    def _coerce_world_state(self, raw_state: object) -> WorldState | None:
        if raw_state is None:
            return None
        if isinstance(raw_state, WorldState):
            return raw_state.clone()
        if not all(hasattr(raw_state, name) for name in ("robot", "placements", "holding")):
            return None
        placements = dict(getattr(raw_state, "placements"))
        robot = tuple(getattr(raw_state, "robot"))
        holding = getattr(raw_state, "holding")
        return WorldState(
            robot=self._validate_coord(robot),
            placements={str(block): tuple(coord) for block, coord in placements.items()},
            holding=None if holding is None else str(holding),
        )

    def _select_task_library(self, options: Mapping[str, object]) -> List[Task]:
        library_override = options.get("task_library")
        if library_override is None:
            return self.generator.sample_task_library(
                self._py_rng,
                count=self.task_library_size,
            )
        library = [self._coerce_task(task) for task in list(library_override)]
        if not library:
            raise ValueError("task_library override must contain at least one task.")
        return library

    def _select_current_task(self, options: Mapping[str, object]) -> Task:
        task_override = options.get("task")
        if task_override is not None:
            return self._coerce_task(task_override)
        return self._py_rng.choice(self.task_library)

    def _coerce_task(self, raw_task: object) -> Task:
        if isinstance(raw_task, Task):
            return raw_task
        assignments = self._extract_assignments(raw_task)
        if assignments is not None:
            return Task(tuple(assignments))
        raise TypeError(f"Unsupported task representation: {type(raw_task)!r}")

    def _extract_assignments(self, raw_task: object) -> List[Tuple[str, str]] | None:
        if hasattr(raw_task, "assignments"):
            raw_task = getattr(raw_task, "assignments")
        if not isinstance(raw_task, Sequence):
            return None
        assignments: List[Tuple[str, str]] = []
        for item in raw_task:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError(f"Invalid task assignment: {item!r}")
            block, region = item
            assignments.append((str(block), str(region)))
        if not assignments:
            raise ValueError("Task override must contain at least one assignment.")
        return assignments

    def _validate_coord(self, coord: Sequence[int]) -> Coord:
        x, y = int(coord[0]), int(coord[1])
        if not (0 <= x < self.config.width and 0 <= y < self.config.height):
            raise ValueError(f"Coordinate {coord} is outside the grid.")
        return x, y

    def _move_agent(self, dx: int, dy: int) -> float:
        ax, ay = self.state.robot
        nx = int(np.clip(ax + dx, 0, self.config.width - 1))
        ny = int(np.clip(ay + dy, 0, self.config.height - 1))
        blocked = nx == ax and ny == ay
        self.state.robot = (nx, ny)
        return -self.invalid_action_penalty if blocked else 0.0

    def _handle_pick(self) -> str | None:
        if self.state.holding is not None:
            return None
        block = self.state.block_at(self.state.robot)
        if block is None:
            return None
        del self.state.placements[block]
        self.state.holding = block
        return block

    def _handle_place(self) -> bool:
        if self.state.holding is None:
            return False
        coord = self._current_place_coord()
        if coord is None:
            return False
        block = self.state.holding
        self.state.placements[block] = coord
        self.state.holding = None
        return True

    def _coord_is_region(self, coord: Coord) -> bool:
        return self.config.region_for_coord(coord) is not None

    def _task_already_satisfied(self) -> bool:
        return self.state.is_task_satisfied(self.current_task, self.config)

    def _resample_task(self) -> None:
        self.current_task = self._py_rng.choice(self.task_library)
        self._update_pending_auto_success()

    def _update_pending_auto_success(self) -> None:
        self._pending_auto_success = self._task_already_satisfied()

    def _obs(self) -> np.ndarray:
        height = self.render_size
        width = self.render_size
        obs = self._obs_canvas
        obs.fill(0.0)
        obs[:3] = self._rgb_canvas

        block_size = int(self.tile_px * 0.6)
        block_offset = max(1, (self.tile_px - block_size) // 2)
        for block, coord in sorted(self.state.placements.items()):
            x0, y0, _, _ = self._tile_bounds(coord)
            bx0 = x0 + block_offset
            by0 = y0 + block_offset
            bx1 = bx0 + block_size
            by1 = by0 + block_size
            rgb = self._rgb_triplet(self.config.block_color(block))
            obs[0, by0:by1, bx0:bx1] = rgb[0]
            obs[1, by0:by1, bx0:bx1] = rgb[1]
            obs[2, by0:by1, bx0:bx1] = rgb[2]
            self._draw_outline(obs[:3], bx0, by0, bx1, by1, self._rgb_triplet(BLOCK_OUTLINE_RGB))

        ax, ay = self.state.robot
        x0, y0, x1, y1 = self._tile_bounds((ax, ay))
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        robot_half = max(2, self.tile_px // 4)
        rx0 = max(0, cx - robot_half)
        ry0 = max(0, cy - robot_half)
        rx1 = min(width, cx + robot_half)
        ry1 = min(height, cy + robot_half)
        robot_rgb = self._rgb_triplet(ROBOT_RGB)
        height_px = max(1, ry1 - ry0)
        for y in range(ry0, ry1):
            frac = (y - ry0 + 1) / height_px
            half_width = max(1, int(frac * (rx1 - rx0) / 2))
            x0 = max(0, cx - half_width)
            x1 = min(width, cx + half_width)
            if x1 <= x0:
                continue
            obs[0, y, x0:x1] = robot_rgb[0]
            obs[1, y, x0:x1] = robot_rgb[1]
            obs[2, y, x0:x1] = robot_rgb[2]

        if self.state.holding is not None:
            held_rgb = self._rgb_triplet(self.config.block_color(self.state.holding))
            held_half = max(1, self.tile_px // 8)
            hx0 = max(0, cx - held_half)
            hy0 = max(0, cy - held_half)
            hx1 = min(width, cx + held_half)
            hy1 = min(height, cy + held_half)
            obs[0, hy0:hy1, hx0:hx1] = held_rgb[0]
            obs[1, hy0:hy1, hx0:hx1] = held_rgb[1]
            obs[2, hy0:hy1, hx0:hx1] = held_rgb[2]

        assignments = self.current_task.assignments
        self._fill_target_block_mask(obs[3], assignments[0][0])
        self._fill_target_region_mask(obs[4], assignments[0][1])
        if len(assignments) > 1:
            self._fill_target_block_mask(obs[5], assignments[1][0])
            self._fill_target_region_mask(obs[6], assignments[1][1])
            obs[7].fill(1.0)
        return obs.copy()

    def _info(self, success: bool | None = None) -> Dict[str, object]:
        can_pick = self.state.holding is None and self.state.block_at(self.state.robot) is not None
        can_place = self._current_place_coord() is not None
        info = BlockworldInfo(
            robot=self.state.robot,
            placements=dict(self.state.placements),
            holding=self.state.holding,
            task_assignments=self.current_task.assignments,
            task_size=len(self.current_task.assignments),
            success=bool(success),
            can_pick=can_pick,
            can_place=can_place,
            next_auto_satisfied=self._pending_auto_success,
        )
        return info.__dict__

    def _adjacent_coords(self, coord: Coord | None = None) -> List[Coord]:
        x, y = self.state.robot if coord is None else coord
        coords: List[Coord] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                coords.append((nx, ny))
        return coords

    def _adjacent_blocks(self) -> List[str]:
        blocks: List[str] = []
        for coord in self._adjacent_coords():
            block = self.state.block_at(coord)
            if block is not None:
                blocks.append(block)
        return blocks

    def _adjacent_region_tiles(self) -> List[Coord]:
        coords: List[Coord] = []
        for coord in self._adjacent_coords():
            if not self._coord_is_region(coord):
                continue
            if self.state.block_at(coord) is not None:
                continue
            coords.append(coord)
        return coords

    def _adjacent_place_coord(self) -> Coord | None:
        if self.state.holding is None:
            return None
        tiles = self._adjacent_region_tiles()
        if not tiles:
            return None
        regions = {self.config.region_for_coord(coord) for coord in tiles}
        regions.discard(None)
        if len(regions) != 1:
            return None
        region = next(iter(regions))
        if self._region_occupied(region):
            return None
        return min(tiles)

    def _current_place_coord(self) -> Coord | None:
        if self.state.holding is None:
            return None
        coord = self.state.robot
        if self.state.block_at(coord) is not None:
            return None
        region = self.config.region_for_coord(coord)
        if region is None:
            return None
        if self._region_occupied(region):
            return None
        return coord

    def _region_occupied(self, region: str, *, placements: Mapping[str, Coord] | None = None) -> bool:
        placements = placements or self.state.placements
        region_tiles = set(self.config.region_tiles[region])
        return any(coord in region_tiles for coord in placements.values())

    def _validate_region_capacity(self, placements: Mapping[str, Coord]) -> None:
        seen: Dict[str, str] = {}
        for block, coord in placements.items():
            region = self.config.region_for_coord(coord)
            if region is None:
                continue
            if region in seen:
                raise ValueError(
                    f"Region {region} already occupied by {seen[region]} when placing {block}."
                )
            seen[region] = block

    def _tile_bounds(self, coord: Coord) -> Tuple[int, int, int, int]:
        x, y = coord
        x0 = self.margin_px + x * self.tile_px
        y0 = self.margin_px + y * self.tile_px
        x1 = x0 + self.tile_px
        y1 = y0 + self.tile_px
        return x0, y0, x1, y1

    def _fill_target_block_mask(self, mask: np.ndarray, block: str) -> None:
        if self.state.holding == block:
            coord = self.state.robot
        else:
            coord = self.state.placements.get(block)
        if coord is None:
            return
        x0, y0, x1, y1 = self._tile_bounds(coord)
        mask[y0:y1, x0:x1] = 1.0

    def _fill_target_region_mask(self, mask: np.ndarray, region: str) -> None:
        for coord in self.config.region_tiles[region]:
            x0, y0, x1, y1 = self._tile_bounds(coord)
            mask[y0:y1, x0:x1] = 1.0

    def _rebuild_static_canvas(self) -> None:
        rgb = self._rgb_canvas
        rgb[0].fill(BACKGROUND_RGB[0] / 255.0)
        rgb[1].fill(BACKGROUND_RGB[1] / 255.0)
        rgb[2].fill(BACKGROUND_RGB[2] / 255.0)

        margin = self.margin_px
        x0 = margin
        y0 = margin
        x1 = self.render_size - margin
        y1 = self.render_size - margin
        grid_rgb = self._rgb_triplet(GRID_RGB)
        rgb[0, y0:y1, x0:x1] = grid_rgb[0]
        rgb[1, y0:y1, x0:x1] = grid_rgb[1]
        rgb[2, y0:y1, x0:x1] = grid_rgb[2]

        for region_name, coords in self.config.region_tiles.items():
            region_rgb = self._rgb_triplet(COLOR_RGB[region_color(region_name)])
            for coord in coords:
                rx0, ry0, rx1, ry1 = self._tile_bounds(coord)
                rgb[0, ry0:ry1, rx0:rx1] = region_rgb[0]
                rgb[1, ry0:ry1, rx0:rx1] = region_rgb[1]
                rgb[2, ry0:ry1, rx0:rx1] = region_rgb[2]
            for coord in coords:
                rx0, ry0, rx1, ry1 = self._tile_bounds(coord)
                self._draw_outline(rgb, rx0, ry0, rx1, ry1, self._rgb_triplet(BLOCK_OUTLINE_RGB))

    @staticmethod
    def _rgb_triplet(rgb: Tuple[int, int, int] | str) -> Tuple[float, float, float]:
        if isinstance(rgb, str):
            rgb = COLOR_RGB[rgb]
        return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    @staticmethod
    def _draw_outline(
        rgb: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Tuple[float, float, float],
    ) -> None:
        if x1 <= x0 or y1 <= y0:
            return
        rgb[0, y0:y1, x0] = color[0]
        rgb[1, y0:y1, x0] = color[1]
        rgb[2, y0:y1, x0] = color[2]
        rgb[0, y0:y1, x1 - 1] = color[0]
        rgb[1, y0:y1, x1 - 1] = color[1]
        rgb[2, y0:y1, x1 - 1] = color[2]
        rgb[0, y0, x0:x1] = color[0]
        rgb[1, y0, x0:x1] = color[1]
        rgb[2, y0, x0:x1] = color[2]
        rgb[0, y1 - 1, x0:x1] = color[0]
        rgb[1, y1 - 1, x0:x1] = color[1]
        rgb[2, y1 - 1, x0:x1] = color[2]
