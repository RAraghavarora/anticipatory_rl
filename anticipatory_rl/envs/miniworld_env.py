"""
MiniWorld recreation of the earlier gridworld rearrangement setting.

The environment keeps the same discrete action semantics
(`move_{up,down,left,right}`, `pick`, `place`) while leveraging MiniWorld's
rendering and Gymnasium integration. The world is a 6x6 grid with labeled
receptacles (kitchen table, kitchen counter, dining table, study table, shelf)
and the same starter objects (water bottle, tiffin box, apple, soda can,
drinking glass).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces, utils
from PIL import Image, ImageDraw

from miniworld.entity import Box, COLORS
from miniworld.miniworld import MiniWorldEnv

Coord = Tuple[int, int]


@dataclass(frozen=True)
class ReceptacleSpec:
    name: str
    tiles: List[Coord]
    color: str


CUSTOM_COLORS = {
    "orange": np.array([1.0, 0.6, 0.0]),
    "teal": np.array([0.0, 0.65, 0.65]),
    "pink": np.array([1.0, 0.45, 0.7]),
    "navy": np.array([0.1, 0.15, 0.45]),
}

for name, rgb in CUSTOM_COLORS.items():
    if name not in COLORS:
        COLORS[name] = rgb

RECEPTACLES: List[ReceptacleSpec] = [
    ReceptacleSpec("kitchen_table", [(0, 0), (1, 0)], "orange"),
    ReceptacleSpec("kitchen_counter", [(1, 1), (2, 1), (1, 2), (2, 2)], "teal"),
    ReceptacleSpec("dining_table", [(4, 1), (4, 2), (5, 1), (5, 2)], "pink"),
    ReceptacleSpec("study_table", [(3, 3), (4, 3), (3, 4), (4, 4)], "navy"),
    ReceptacleSpec("shelf", [(0, 5), (1, 5)], "grey"),
]
RECEPTACLE_TILE_MAP: Dict[str, List[Coord]] = {spec.name: spec.tiles for spec in RECEPTACLES}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    color: str
    start_tile: Coord
    home_region: str
    stack_index: int = 0


OBJECT_SPECS: List[ObjectSpec] = [
    ObjectSpec("water_bottle", "blue", (1, 1), "kitchen_counter", stack_index=0),
    ObjectSpec("tiffin_box", "red", (1, 1), "kitchen_counter", stack_index=1),
    ObjectSpec("apple", "green", (4, 2), "dining_table", stack_index=0),
    ObjectSpec("soda_can", "yellow", (5, 2), "dining_table", stack_index=0),
    ObjectSpec("drinking_glass", "purple", (3, 4), "study_table", stack_index=0),
]
OBJECT_NAMES = [spec.name for spec in OBJECT_SPECS]


class SixAction(IntEnum):
    move_up = 0
    move_down = 1
    move_left = 2
    move_right = 3
    pick = 4
    place = 5


class MiniWorldGridRearrange(MiniWorldEnv, utils.EzPickle):
    """
    MiniWorld environment with discrete grid actions and pick/place semantics.
    """

    def __init__(
        self,
        grid_size: int = 6,
        tile_size: float = 1.0,
        max_episode_steps: int = 1500,
        object_specs: Optional[Sequence[ObjectSpec]] = None,
        receptacles: Optional[Sequence[ReceptacleSpec]] = None,
        **kwargs,
    ) -> None:
        self.grid_size = grid_size
        self.tile_size = tile_size

        self.tile_contents: Dict[Coord, List[str]] = {}
        self.object_registry: Dict[str, Dict[str, object]] = {}
        self.tile_to_receptacle: Dict[Coord, str] = {}
        self.receptacle_entities: Dict[str, Box] = {}
        self.carrying: Optional[str] = None
        self.agent_grid: Coord = (0, 0)

        self.object_specs: List[ObjectSpec] = (
            list(object_specs) if object_specs is not None else list(OBJECT_SPECS)
        )
        if not self.object_specs:
            raise ValueError("MiniWorldGridRearrange requires at least one object spec.")
        self.object_names: List[str] = [spec.name for spec in self.object_specs]
        self.object_color_map: Dict[str, str] = {
            spec.name: spec.color for spec in self.object_specs
        }

        self.receptacles: List[ReceptacleSpec] = (
            list(receptacles) if receptacles is not None else list(RECEPTACLES)
        )
        if not self.receptacles:
            raise ValueError("MiniWorldGridRearrange requires at least one receptacle.")
        self.receptacle_tile_map: Dict[str, List[Coord]] = {
            spec.name: list(spec.tiles) for spec in self.receptacles
        }

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=max_episode_steps,
            view="top",
            **kwargs,
        )
        utils.EzPickle.__init__(self, grid_size, tile_size, max_episode_steps, **kwargs)

        # Override action space with custom six-action enumeration
        self.actions = SixAction
        self.action_space = spaces.Discrete(len(self.actions))

    # ------------------------------------------------------------------ World setup
    def _gen_world(self) -> None:
        self.tile_contents = {}
        self.object_registry = {}
        self.tile_to_receptacle = {}
        self.receptacle_entities = {}
        self.carrying = None

        # Build a single rectangular room matching the discrete grid extents.
        self.add_rect_room(
            min_x=0,
            max_x=self.grid_size,
            min_z=0,
            max_z=self.grid_size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            ceil_tex="ceiling_tiles",
            no_ceiling=True,
        )

        self._place_receptacle_surfaces()
        self._spawn_objects()
        self._place_agent()

    def _place_receptacle_surfaces(self) -> None:
        for spec in self.receptacles:
            for tile in spec.tiles:
                self.tile_to_receptacle[tile] = spec.name

            xs = [tile[0] for tile in spec.tiles]
            zs = [tile[1] for tile in spec.tiles]
            min_x = min(xs)
            max_x = max(xs) + 1
            min_z = min(zs)
            max_z = max(zs) + 1

            width = (max_x - min_x) * self.tile_size
            depth = (max_z - min_z) * self.tile_size
            center = np.array(
                [
                    (min_x + max_x) * 0.5 * self.tile_size,
                    0.0,
                    (min_z + max_z) * 0.5 * self.tile_size,
                ]
            )

            surface = Box(color=spec.color, size=[width, 0.05, depth])
            self.place_entity(surface, pos=center, dir=0.0)
            self.receptacle_entities[spec.name] = surface

    def _spawn_objects(self) -> None:
        for spec in sorted(self.object_specs, key=lambda s: (s.start_tile, s.stack_index)):
            entity = Box(color=spec.color, size=0.4)
            self.place_entity(entity, pos=self._grid_to_world(spec.start_tile))
            entity.dir = 0.0

            self.object_registry[spec.name] = {
                "entity": entity,
                "tile": spec.start_tile,
                "region": spec.home_region,
            }
            stack = self.tile_contents.setdefault(spec.start_tile, [])
            stack.append(spec.name)
            self._update_stack_poses(spec.start_tile)

    def _place_agent(self) -> None:
        self.agent_grid = (0, 0)
        start_pos = self._grid_to_world(self.agent_grid)
        self.place_agent(pos=start_pos, dir=0.0)
        self.agent.carrying = None
        self.carrying = None

    # ------------------------------------------------------------------ Helpers
    def _grid_to_world(self, tile: Coord, height_offset: float = 0.0) -> np.ndarray:
        x, y = tile
        return np.array(
            [(x + 0.5) * self.tile_size, height_offset, (y + 0.5) * self.tile_size]
        )

    def _region_for_tile(self, tile: Coord) -> Optional[str]:
        return self.tile_to_receptacle.get(tile)

    def _update_stack_poses(self, tile: Coord) -> None:
        stack = self.tile_contents.get(tile)
        if not stack:
            self.tile_contents.pop(tile, None)
            return

        for level, name in enumerate(stack):
            entity = self.object_registry[name]["entity"]
            height = level * 0.45
            entity.pos = self._grid_to_world(tile, height_offset=height)

    def _move_agent(self, dx: int, dy: int) -> bool:
        nx = self.agent_grid[0] + dx
        ny = self.agent_grid[1] + dy
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return False

        self.agent_grid = (nx, ny)
        self.agent.pos = self._grid_to_world(self.agent_grid)
        self.agent.dir = 0.0

        if self.carrying:
            carried_entity = self.object_registry[self.carrying]["entity"]
            carried_entity.pos = self._get_carry_pos(self.agent.pos, carried_entity)

        return True

    def _top_object_at_agent(self) -> Optional[str]:
        stack = self.tile_contents.get(self.agent_grid, [])
        return stack[-1] if stack else None

    # ------------------------------------------------------------------ State/query
    def get_state(self) -> Dict[str, object]:
        return {
            "agent": {
                "grid_pos": self.agent_grid,
                "carrying": self.carrying,
            },
            "objects": {
                name: {
                    "tile": data["tile"],
                    "region": data["region"],
                }
                for name, data in self.object_registry.items()
            },
        }

    # ------------------------------------------------------------------ Actions
    def step(self, action: int):
        try:
            action_enum = self.actions(action)
        except ValueError as exc:
            raise ValueError(
                f"Invalid action {action}; expected one of {list(self.actions)}"
            ) from exc

        self.step_count += 1
        if action_enum == self.actions.move_up:
            self._move_agent(0, -1)
        elif action_enum == self.actions.move_down:
            self._move_agent(0, 1)
        elif action_enum == self.actions.move_left:
            self._move_agent(-1, 0)
        elif action_enum == self.actions.move_right:
            self._move_agent(1, 0)
        elif action_enum == self.actions.pick:
            self._handle_pick()
        elif action_enum == self.actions.place:
            self._handle_place()

        obs = self.render_obs()
        truncation = self.step_count >= self.max_episode_steps
        info = {"state": self.get_state()}
        return obs, 0.0, False, truncation, info

    def act(self, action_name: str):
        """
        Convenience helper: apply an action by its string name.
        """
        if not hasattr(self.actions, action_name):
            raise KeyError(f"Unknown action '{action_name}'")
        action = getattr(self.actions, action_name)
        return self.step(int(action))

    def _handle_pick(self) -> bool:
        if self.carrying:
            return False

        top_object = self._top_object_at_agent()
        if top_object is None:
            return False

        stack = self.tile_contents[self.agent_grid]
        stack.pop()
        self._update_stack_poses(self.agent_grid)

        self.carrying = top_object
        obj_data = self.object_registry[top_object]
        obj_data["tile"] = None
        obj_data["region"] = None
        self.agent.carrying = obj_data["entity"]
        obj_data["entity"].pos = self._get_carry_pos(
            self.agent.pos, obj_data["entity"]
        )
        return True

    def _handle_place(self) -> bool:
        if not self.carrying:
            return False

        stack = self.tile_contents.setdefault(self.agent_grid, [])
        name = self.carrying
        stack.append(name)
        self._update_stack_poses(self.agent_grid)

        obj_data = self.object_registry[name]
        obj_data["tile"] = self.agent_grid
        obj_data["region"] = self._region_for_tile(self.agent_grid)
        self.agent.carrying = None
        self.carrying = None
        return True

    # ------------------------------------------------------------------ Debug utilities
    def apply_object_placements(self, placements: Dict[str, str]) -> None:
        """Force objects onto specific receptacles (for visualization/testing)."""
        for spec in self.object_specs:
            if spec.name not in placements:
                raise KeyError(f"Missing placement for object '{spec.name}'")

        self.tile_contents.clear()
        self.carrying = None
        self.agent.carrying = None

        counters: Dict[str, int] = {name: 0 for name in self.receptacle_tile_map.keys()}

        for obj_name in self.object_names:
            region = placements[obj_name]
            if region not in self.receptacle_tile_map:
                raise KeyError(f"Unknown receptacle '{region}' in placements")
            tiles = self.receptacle_tile_map[region]
            idx = counters[region] % len(tiles)
            counters[region] += 1
            tile = tiles[idx]

            entity = self.object_registry[obj_name]["entity"]
            entity.pos = self._grid_to_world(tile)
            entity.dir = 0.0

            self.object_registry[obj_name]["tile"] = tile
            self.object_registry[obj_name]["region"] = region
            self.tile_contents.setdefault(tile, []).append(obj_name)

        for tile in list(self.tile_contents.keys()):
            self._update_stack_poses(tile)

    # ------------------------------------------------------------------ Rendering stub (top-down)
    def render_top_view(self, frame_buffer=None):
        tile_px = 64
        margin = tile_px
        size = self.grid_size * tile_px + 2 * margin
        background = (110, 207, 246)
        room_color = (128, 128, 128)

        img = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [margin, margin, size - margin - 1, size - margin - 1],
            fill=room_color,
        )

        for spec in self.receptacles:
            tile_color = self._color_bytes(spec.color)
            for tile in spec.tiles:
                x0 = margin + tile[0] * tile_px
                y0 = margin + tile[1] * tile_px
                x1 = x0 + tile_px - 1
                y1 = y0 + tile_px - 1
                draw.rectangle([x0, y0, x1, y1], fill=tile_color)

        obj_size = int(tile_px * 0.55)
        obj_offset = (tile_px - obj_size) // 2
        for spec in self.object_specs:
            data = self.object_registry.get(spec.name)
            if not data:
                continue
            tile = data.get("tile")
            if tile is None:
                continue
            color = self._color_bytes(spec.color)
            x0 = margin + tile[0] * tile_px + obj_offset
            y0 = margin + tile[1] * tile_px + obj_offset
            x1 = x0 + obj_size - 1
            y1 = y0 + obj_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=color)

        # Agent triangle approximating the MiniWorld avatar.
        ax = margin + self.agent_grid[0] * tile_px + tile_px // 2
        ay = margin + self.agent_grid[1] * tile_px + tile_px // 2
        half = tile_px // 2
        triangle = [
            (ax - half // 2, ay - half // 2),
            (ax - half // 2, ay + half // 2),
            (ax + half // 2, ay),
        ]
        draw.polygon(triangle, fill=(180, 30, 30), outline=(60, 0, 0))

        if self.carrying:
            carry_color = self._color_bytes(self.object_color_map[self.carrying])
            size_c = tile_px // 3
            x0 = ax - size_c // 2
            y0 = ay - size_c // 2
            draw.rectangle(
                [x0, y0, x0 + size_c, y0 + size_c],
                outline=carry_color,
                width=3,
            )

        frame = np.array(img, dtype=np.uint8)
        if frame_buffer is not None and getattr(frame_buffer, "shape", None) == frame.shape:
            frame_buffer[...] = frame
            return frame_buffer
        return frame

    def _color_bytes(self, name: str) -> Tuple[int, int, int]:
        rgb = COLORS.get(name)
        if rgb is None:
            return (255, 255, 255)
        arr = np.asarray(rgb, dtype=float)
        arr = np.clip(arr, 0.0, 1.0)
        return tuple(int(round(val * 255)) for val in arr[:3])


def demo_episode(steps: int = 5) -> None:
    """
    Simple manual rollout for quick verification without a training loop.
    """

    env = MiniWorldGridRearrange()
    obs, _ = env.reset()
    print(f"Initial state: {env.get_state()}")
    for action_name in ["move_right", "move_down", "move_down", "pick"]:
        try:
            env.act(action_name)
        except RuntimeError as err:
            print(f"Action {action_name} failed: {err}")
    print(f"Final state: {env.get_state()}")


if __name__ == "__main__":
    demo_episode()
