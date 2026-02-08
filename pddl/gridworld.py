"""Lightweight gridworld with agent, objects, regions, and pick/place tasks.

The module focuses on data structures that make it easy to express tasks where
an agent must pick up one or multiple objects and move them into target
regions. Objects can be obstructed by other objects, so higher-level tasks can
capture ordering constraints (e.g., remove the top crate before accessing the
bottom crate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

Coord = Tuple[int, int]

ACTION_SPACE = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "pick",
    "place",
)

ACTION_SPACE = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "pick",
    "place",
)


@dataclass
class Agent:
    """Simple agent that can carry a single object at a time."""

    name: str
    position: Coord
    carrying: Optional[str] = None

    def move(self, dx: int, dy: int) -> None:
        self.position = (self.position[0] + dx, self.position[1] + dy)


@dataclass
class GridObject:
    """Object in the grid that may be obstructed by other objects."""

    name: str
    position: Optional[Coord]
    region: str
    obstructed_by: List[str] = field(default_factory=list)
    carried_by: Optional[str] = None

    def is_available(self) -> bool:
        return self.carried_by is None and self.position is not None


@dataclass
class Region:
    name: str
    tiles: List[Coord]

    def contains(self, coord: Coord) -> bool:
        return coord in self.tiles


class GridWorld:
    """Grid that tracks the agent, objects, regions, and basic interactions."""

    def __init__(
        self,
        width: int,
        height: int,
        agent: Agent,
        objects: Iterable[GridObject],
        regions: Iterable[Region],
    ) -> None:
        self.width = width
        self.height = height
        self.agent = agent
        self.objects: Dict[str, GridObject] = {obj.name: obj for obj in objects}
        self.regions: Dict[str, Region] = {region.name: region for region in regions}
        self._validate()

    def _validate(self) -> None:
        if not (0 <= self.agent.position[0] < self.width and 0 <= self.agent.position[1] < self.height):
            raise ValueError("Agent position must be within bounds")
        for obj in self.objects.values():
            if obj.position is None:
                continue
            if not self.is_within_bounds(obj.position):
                raise ValueError(f"Object {obj.name} out of bounds: {obj.position}")
            for blocker in obj.obstructed_by:
                if blocker not in self.objects:
                    raise ValueError(f"Unknown blocker '{blocker}' for object '{obj.name}'")
        for region in self.regions.values():
            for tile in region.tiles:
                if not self.is_within_bounds(tile):
                    raise ValueError(f"Region '{region.name}' includes out-of-bounds tile {tile}")

    def is_within_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    # ------------------------------------------------------------------
    # Interaction helpers
    def object_at(self, coord: Coord) -> Optional[GridObject]:
        for obj in self.objects.values():
            if obj.position == coord:
                return obj
        return None

    def is_object_obstructed(self, object_name: str) -> bool:
        obj = self.objects[object_name]
        for blocker_name in obj.obstructed_by:
            blocker = self.objects[blocker_name]
            # A blocker only matters when it shares the tile and is still present.
            if blocker.position == obj.position and blocker.carried_by is None:
                return True
        return False

    def pick(self, object_name: str) -> None:
        if self.agent.carrying is not None:
            raise RuntimeError("Agent is already carrying an object")
        obj = self.objects[object_name]
        if obj.position != self.agent.position:
            raise RuntimeError(f"Agent is not at {object_name}'s tile")
        if self.is_object_obstructed(object_name):
            raise RuntimeError(f"{object_name} is obstructed")
        obj.position = None
        obj.carried_by = self.agent.name
        self.agent.carrying = object_name

    def place(self, region_name: str, tile: Optional[Coord] = None) -> None:
        if self.agent.carrying is None:
            raise RuntimeError("Agent is not carrying anything")
        if region_name not in self.regions:
            raise KeyError(f"Unknown region '{region_name}'")
        region = self.regions[region_name]
        coord = tile or self.agent.position
        if not region.contains(coord):
            raise RuntimeError(f"Tile {coord} is not part of region '{region_name}'")
        obj = self.objects[self.agent.carrying]
        obj.position = coord
        obj.region = region_name
        obj.carried_by = None
        self.agent.carrying = None

    def drop(self) -> None:
        """Place the carried object on the agent's current tile."""
        if self.agent.carrying is None:
            raise RuntimeError("Agent is not carrying anything")
        if self.object_at(self.agent.position):
            raise RuntimeError("Cannot place object on an occupied tile")
        coord = self.agent.position
        region_name = self.region_at(coord)
        obj = self.objects[self.agent.carrying]
        obj.position = coord
        obj.region = region_name or obj.region
        obj.carried_by = None
        self.agent.carrying = None

    # ------------------------------------------------------------------
    # Visualization helpers
    def render(self) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for region in self.regions.values():
            for x, y in region.tiles:
                grid[y][x] = grid[y][x].lower()
        for obj in self.objects.values():
            if obj.position is None:
                continue
            x, y = obj.position
            grid[y][x] = obj.name[0].upper()
        ax, ay = self.agent.position
        grid[ay][ax] = "A"
        rows = [" ".join(row) for row in grid]
        legend = "Legend: uppercase letters are objects, 'A' is the agent, lowercase tiles show regions."
        return "\n".join(rows + [legend])

    def get_state(self) -> Dict[str, object]:
        """Return a snapshot of the world state suitable for an RL agent."""
        return {
            "agent": {
                "position": self.agent.position,
                "carrying": self.agent.carrying,
            },
            "objects": {
                name: {
                    "position": obj.position,
                    "region": obj.region,
                    "carried_by": obj.carried_by,
                    "obstructed_by": list(obj.obstructed_by),
                }
                for name, obj in self.objects.items()
            },
        }

    def region_at(self, coord: Coord) -> Optional[str]:
        for name, region in self.regions.items():
            if region.contains(coord):
                return name
        return None

    def step(self, action: str) -> Dict[str, object]:
        """Apply an action from the fixed action space and return the new state."""
        if action not in ACTION_SPACE:
            raise ValueError(f"Invalid action '{action}'. Valid actions: {ACTION_SPACE}")

        dxdy = {
            "move_up": (0, -1),
            "move_down": (0, 1),
            "move_left": (-1, 0),
            "move_right": (1, 0),
        }

        if action in dxdy:
            dx, dy = dxdy[action]
            nx = self.agent.position[0] + dx
            ny = self.agent.position[1] + dy
            if not self.is_within_bounds((nx, ny)):
                raise RuntimeError("Move would leave the grid")
            self.agent.move(dx, dy)
            return self.get_state()

        if action == "pick":
            obj = self.object_at(self.agent.position)
            if obj is None:
                raise RuntimeError("No object to pick up at the current tile")
            self.pick(obj.name)
            return self.get_state()

        if action == "place":
            self.drop()
            return self.get_state()

        raise RuntimeError(f"Unhandled action '{action}'")


def build_sample_world() -> GridWorld:
    regions = [
        Region("charging_pad", [(0, 0), (1, 0)]),
        Region("storage_a", [(1, 1), (2, 1), (1, 2), (2, 2)]),
        Region("storage_b", [(4, 1), (4, 2), (5, 1), (5, 2)]),
        Region("assembly_zone", [(3, 3), (4, 3), (3, 4), (4, 4)]),
        Region("shipping", [(0, 5), (1, 5)]),
    ]

    objects = [
        GridObject("red_crate", position=(1, 1), region="storage_a"),
        GridObject("blue_crate", position=(1, 1), region="storage_a", obstructed_by=["red_crate"]),
        GridObject("sensor_pack", position=(4, 2), region="storage_b"),
        GridObject("toolkit", position=(3, 4), region="assembly_zone"),
        GridObject("shipping_label", position=(5, 2), region="storage_b", obstructed_by=["sensor_pack"]),
    ]

    agent = Agent("mobile_bot", position=(0, 0))
    return GridWorld(width=6, height=6, agent=agent, objects=objects, regions=regions)


if __name__ == "__main__":
    world = build_sample_world()
    print(world.render())
