"""Image-based RL environment for the reproduced paper1 blockworld."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces

from paper1_blockworld.world import (
    Coord,
    Task,
    WorldConfig,
    WorldGenerator,
    WorldState,
    region_color,
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
    "cyan": (80, 200, 205),
    "pink": (225, 80, 195),
    "orange": (235, 145, 60),
    "brown": (160, 95, 55),
    "white": (250, 250, 250),
}


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
        if isinstance(override, WorldConfig):
            return override
        if self.procedural_layout:
            return WorldConfig.sample(self._py_rng)
        return self.base_config

    def _select_state(self, options: Mapping[str, object]) -> WorldState:
        state_override = options.get("state")
        if isinstance(state_override, WorldState):
            state = state_override.clone()
        else:
            state = self.generator.sample_initial_state(self._py_rng)
        placements_override = options.get("placements")
        if placements_override is not None:
            state.placements = {
                str(block): self._validate_coord(tuple(coord))
                for block, coord in dict(placements_override).items()
            }
        robot_override = options.get("robot_pos")
        if robot_override is not None:
            state.robot = self._validate_coord(tuple(robot_override))
        holding_override = options.get("holding", state.holding)
        state.holding = None if holding_override is None else str(holding_override)
        return state

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
        if isinstance(raw_task, Sequence):
            assignments: List[Tuple[str, str]] = []
            for item in raw_task:
                if not isinstance(item, Sequence) or len(item) != 2:
                    raise ValueError(f"Invalid task assignment: {item!r}")
                block, region = item
                assignments.append((str(block), str(region)))
            if not assignments:
                raise ValueError("Task override must contain at least one assignment.")
            return Task(tuple(assignments))
        raise TypeError(f"Unsupported task representation: {type(raw_task)!r}")

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
        if not self._coord_is_region(self.state.robot):
            return False
        if self.state.block_at(self.state.robot) is not None:
            return False
        block = self.state.holding
        self.state.placements[block] = self.state.robot
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
        obs[0, ry0:ry1, rx0:rx1] = robot_rgb[0]
        obs[1, ry0:ry1, rx0:rx1] = robot_rgb[1]
        obs[2, ry0:ry1, rx0:rx1] = robot_rgb[2]
        self._draw_outline(obs[:3], rx0, ry0, rx1, ry1, self._rgb_triplet(ROBOT_OUTLINE_RGB))

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
        can_place = (
            self.state.holding is not None
            and self._coord_is_region(self.state.robot)
            and self.state.block_at(self.state.robot) is None
        )
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
        coord = self.config.region_coords[region]
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

        for region_name, coord in self.config.region_coords.items():
            rx0, ry0, rx1, ry1 = self._tile_bounds(coord)
            region_rgb = self._rgb_triplet(COLOR_RGB[region_color(region_name)])
            rgb[0, ry0:ry1, rx0:rx1] = region_rgb[0]
            rgb[1, ry0:ry1, rx0:rx1] = region_rgb[1]
            rgb[2, ry0:ry1, rx0:rx1] = region_rgb[2]
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
