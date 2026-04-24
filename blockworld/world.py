from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Coord = Tuple[int, int]
Point = Tuple[float, float]


COLORED_REGIONS: Tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "cyan",
    "pink",
    "orange",
    "brown",
)
WHITE_REGIONS: Tuple[str, ...] = (
    "white_1",
    "white_2",
    "white_3",
)
BLOCK_NAMES: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h")
DEFAULT_REGION_ANCHORS: Tuple[Coord, ...] = (
    (0, 0),
    (3, 0),
    (6, 0),
    (8, 1),
    (0, 3),
    (3, 4),
    (7, 4),
    (0, 7),
    (4, 8),
    (8, 7),
)


def _tiles_for_anchor(anchor: Coord, region_size: int) -> Tuple[Coord, ...]:
    ax, ay = anchor
    return tuple(
        (ax + dx, ay + dy)
        for dy in range(region_size)
        for dx in range(region_size)
    )


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
    for _ in range(4000):
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


@dataclass(frozen=True)
class Task:
    assignments: Tuple[Tuple[str, str], ...]

    @property
    def blocks(self) -> Tuple[str, ...]:
        return tuple(block for block, _ in self.assignments)

    @property
    def target_regions(self) -> Tuple[str, ...]:
        return tuple(region for _, region in self.assignments)

    def goal_regions(self) -> Dict[str, str]:
        return dict(self.assignments)

    def goal_tiles(self, config: "WorldConfig") -> Dict[str, Tuple[Coord, ...]]:
        return {
            block: config.region_tiles[region]
            for block, region in self.assignments
        }

    def describe(self) -> str:
        return ", ".join(f"{block}->{region}" for block, region in self.assignments)


@dataclass(frozen=True)
class WorldConfig:
    """
    Planner/training-side 2D Blockworld with geometric regions.
    """

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
        return tuple(region for region in self.all_regions if not region.startswith("white"))

    @property
    def white_regions(self) -> Tuple[str, ...]:
        return tuple(region for region in self.all_regions if region.startswith("white"))

    @property
    def all_blocks(self) -> Tuple[str, ...]:
        return BLOCK_NAMES

    @property
    def block_color_map(self) -> Dict[str, str]:
        if self.block_colors is not None:
            return dict(self.block_colors)
        mapping: Dict[str, str] = {}
        for idx, block in enumerate(BLOCK_NAMES):
            mapping[block] = COLORED_REGIONS[idx] if idx < 5 else "white"
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
        return {
            region: _tiles_for_anchor(anchor, self.region_size)
            for region, anchor in self.region_anchors.items()
        }

    @property
    def region_cells(self) -> Tuple[Coord, ...]:
        return tuple(coord for tiles in self.region_tiles.values() for coord in tiles)

    @property
    def floor_cells(self) -> Tuple[Coord, ...]:
        region_cells = set(self.region_cells)
        return tuple(
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in region_cells
        )

    @property
    def all_cells(self) -> Tuple[Coord, ...]:
        return tuple(
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
        )

    @property
    def manipulation_cells(self) -> Tuple[Coord, ...]:
        return tuple(sorted(self.region_cells))

    def is_floor_connected(self) -> bool:
        cells = set(self.all_cells)
        if not cells:
            return False
        frontier = [next(iter(cells))]
        visited: set[Coord] = set()
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.neighbors(current):
                if neighbor in cells and neighbor not in visited:
                    frontier.append(neighbor)
        return visited == cells

    @classmethod
    def sample(cls, rng: random.Random) -> "WorldConfig":
        for _ in range(1000):
            anchors = _sample_region_anchors(
                rng,
                width=cls.width,
                height=cls.height,
                region_size=cls.region_size,
                count=len(COLORED_REGIONS) + len(WHITE_REGIONS),
            )
            rng.shuffle(anchors)
            region_layout = tuple(zip(COLORED_REGIONS + WHITE_REGIONS, anchors))

            colored_blocks = set(rng.sample(BLOCK_NAMES, k=5))
            color_pool = list(COLORED_REGIONS)
            rng.shuffle(color_pool)
            block_colors: List[Tuple[str, str]] = []
            for block in BLOCK_NAMES:
                if block in colored_blocks:
                    block_colors.append((block, color_pool.pop()))
                else:
                    block_colors.append((block, "white"))
            config = cls(
                region_layout=region_layout,
                block_colors=tuple(block_colors),
            )
            if (
                config.is_floor_connected()
                and all(config.placeable_tiles_for_region(region) for region in config.all_regions)
            ):
                return config
        raise RuntimeError("Failed to sample a region layout with accessible tiles.")

    def region_for_coord(self, coord: Coord) -> str | None:
        for region, coords in self.region_tiles.items():
            if coord in coords:
                return region
        return None

    def region_centroid(self, region: str) -> Point:
        ax, ay = self.region_anchors[region]
        offset = (self.region_size - 1) / 2.0
        return (ax + offset, ay + offset)

    def tiles_for_region(self, region: str) -> Tuple[Coord, ...]:
        return self.region_tiles[region]

    def access_cells_for_tile(self, coord: Coord) -> Tuple[Coord, ...]:
        return tuple(
            neighbor
            for neighbor in self.neighbors(coord)
        )

    def access_cells_for_region(self, region: str) -> Tuple[Coord, ...]:
        cells = set()
        for tile in self.region_tiles[region]:
            cells.update(self.access_cells_for_tile(tile))
        return tuple(sorted(cells))

    def placeable_tiles_for_region(self, region: str) -> Tuple[Coord, ...]:
        return tuple(self.region_tiles[region])

    def neighbors(self, coord: Coord) -> Tuple[Coord, ...]:
        x, y = coord
        neighbors: List[Coord] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return tuple(neighbors)

    def location_name(self, coord: Coord) -> str:
        return f"loc_{coord[0]}_{coord[1]}"

    def block_color(self, block: str) -> str:
        return self.block_color_map[block]


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

    def signature(self) -> Tuple[Coord, str | None, Tuple[Tuple[str, Coord], ...]]:
        return (
            self.robot,
            self.holding,
            tuple(sorted(self.placements.items())),
        )

    def block_at(self, coord: Coord) -> str | None:
        for block, placement in self.placements.items():
            if placement == coord:
                return block
        return None

    def region_of_block(self, block: str, config: WorldConfig) -> str | None:
        placement = self.placements.get(block)
        if placement is None:
            return None
        return config.region_for_coord(placement)

    def occupied_regions(self, config: WorldConfig) -> Dict[str, str]:
        occupied: Dict[str, str] = {}
        for block, coord in self.placements.items():
            region = config.region_for_coord(coord)
            if region is None:
                continue
            occupied[region] = block
        return occupied

    def is_task_satisfied(self, task: Task, config: WorldConfig) -> bool:
        for block, region in task.assignments:
            if self.placements.get(block) not in config.region_tiles[region]:
                return False
        return self.holding is None

    def render(self, config: WorldConfig) -> str:
        grid = [["." for _ in range(config.width)] for _ in range(config.height)]
        for region, tiles in config.region_tiles.items():
            token = region[0].lower() if not region.startswith("white") else "w"
            for x, y in tiles:
                grid[y][x] = token
        for block, coord in self.placements.items():
            x, y = coord
            grid[y][x] = block.upper()
        rx, ry = self.robot
        grid[ry][rx] = "@"
        lines = [" ".join(row) for row in grid]
        if self.holding is not None:
            lines.append(f"Holding: {self.holding}")
        return "\n".join(lines)


class WorldGenerator:
    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()

    def sample_initial_state(self, rng: random.Random) -> WorldState:
        regions = list(self.config.all_regions)
        rng.shuffle(regions)
        placements: Dict[str, Coord] = {}
        for block, region in zip(self.config.all_blocks, regions):
            placeable_tiles = self.config.placeable_tiles_for_region(region)
            if not placeable_tiles:
                raise RuntimeError(f"Region {region} has no manipulable placement tile.")
            placements[block] = rng.choice(placeable_tiles)
        free_floor = [
            coord
            for coord in self.config.floor_cells
            if coord not in placements.values()
        ]
        if not free_floor:
            raise RuntimeError("No free floor cell available for robot start.")
        return WorldState(
            robot=rng.choice(free_floor),
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

    def candidate_parking_cells(
        self,
        state: WorldState,
        task: Task,
    ) -> List[Coord]:
        occupied_regions = state.occupied_regions(self.config)
        candidate_regions: List[str] = []
        for region in self.config.white_regions:
            occupant = occupied_regions.get(region)
            if occupant is None or occupant in task.blocks:
                candidate_regions.append(region)
        for block in task.blocks:
            region = state.region_of_block(block, self.config)
            if region is not None:
                candidate_regions.append(region)

        blocked_goal_regions = set(task.target_regions)
        cells: List[Coord] = []
        seen: set[Coord] = set()
        for region in candidate_regions:
            if region in blocked_goal_regions:
                continue
            for coord in self.config.placeable_tiles_for_region(region):
                if coord in seen:
                    continue
                seen.add(coord)
                cells.append(coord)
        return cells


def block_color(block: str, config: WorldConfig) -> str:
    return config.block_color(block)


def region_color(region: str) -> str:
    if region.startswith("white"):
        return "white"
    return region
