from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import itertools
import random


Coord = Tuple[int, int]


NONWHITE_REGIONS: Tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "violet",
    "cyan",
)
WHITE_REGIONS: Tuple[str, ...] = (
    "white_north",
    "white_south",
    "white_west",
    "white_east",
)
NONWHITE_BLOCKS: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
WHITE_BLOCKS: Tuple[str, ...] = ("w1", "w2", "w3", "w4")
BLOCK_COLOR_MAP: Dict[str, str] = {
    "a": "red",
    "b": "blue",
    "c": "green",
    "d": "yellow",
    "e": "orange",
    "f": "purple",
    "g": "violet",
    "w1": "white",
    "w2": "white",
    "w3": "white",
    "w4": "white",
}


@dataclass(frozen=True)
class Task:
    assignments: Tuple[Tuple[str, str], ...]

    @property
    def blocks(self) -> Tuple[str, ...]:
        return tuple(block for block, _ in self.assignments)

    @property
    def target_regions(self) -> Tuple[str, ...]:
        return tuple(region for _, region in self.assignments)

    def goal_positions(self, config: "WorldConfig") -> Dict[str, Coord]:
        return {
            block: config.region_coords[region]
            for block, region in self.assignments
        }

    def describe(self) -> str:
        parts = [f"{block}->{region}" for block, region in self.assignments]
        return ", ".join(parts)


@dataclass(frozen=True)
class WorldConfig:
    width: int = 7
    height: int = 7
    move_cost: int = 25
    pick_cost: int = 100
    place_cost: int = 100
    region_layout: Tuple[Tuple[str, Coord], ...] | None = None

    @property
    def robot_start(self) -> Coord:
        return (3, 3)

    @property
    def region_coords(self) -> Dict[str, Coord]:
        if self.region_layout is not None:
            return dict(self.region_layout)
        return {
            "red": (1, 1),
            "blue": (3, 1),
            "green": (5, 1),
            "yellow": (1, 3),
            "orange": (5, 3),
            "purple": (1, 5),
            "violet": (3, 5),
            "cyan": (5, 5),
            "white_north": (3, 0),
            "white_south": (3, 6),
            "white_west": (0, 3),
            "white_east": (6, 3),
        }

    @classmethod
    def sample(cls, rng: random.Random) -> "WorldConfig":
        slots = [
            (1, 1),
            (3, 1),
            (5, 1),
            (1, 3),
            (5, 3),
            (1, 5),
            (3, 5),
            (5, 5),
            (3, 0),
            (3, 6),
            (0, 3),
            (6, 3),
            (0, 1),
            (6, 1),
            (0, 5),
            (6, 5),
        ]
        chosen = rng.sample(slots, k=len(NONWHITE_REGIONS) + len(WHITE_REGIONS))
        names = list(NONWHITE_REGIONS + WHITE_REGIONS)
        rng.shuffle(names)
        layout = tuple(sorted(zip(names, chosen)))
        return cls(region_layout=layout)

    @property
    def all_blocks(self) -> Tuple[str, ...]:
        return NONWHITE_BLOCKS + WHITE_BLOCKS

    @property
    def all_regions(self) -> Tuple[str, ...]:
        return NONWHITE_REGIONS + WHITE_REGIONS

    @property
    def region_cells(self) -> Tuple[Coord, ...]:
        return tuple(self.region_coords.values())

    def region_for_coord(self, coord: Coord) -> str | None:
        for name, region_coord in self.region_coords.items():
            if region_coord == coord:
                return name
        return None

    def location_name(self, coord: Coord) -> str:
        return f"loc_{coord[0]}_{coord[1]}"

    def region_location(self, region: str) -> str:
        return self.location_name(self.region_coords[region])


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

    def is_task_satisfied(self, task: Task, config: WorldConfig) -> bool:
        for block, region in task.assignments:
            if self.placements.get(block) != config.region_coords[region]:
                return False
        return self.holding is None

    def render(self, config: WorldConfig) -> str:
        grid = [["." for _ in range(config.width)] for _ in range(config.height)]
        for region, coord in config.region_coords.items():
            x, y = coord
            token = region[0].upper() if not region.startswith("white") else "W"
            grid[y][x] = token.lower()
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
        region_cells = list(self.config.region_cells)
        rng.shuffle(region_cells)
        placements = {
            block: coord
            for block, coord in zip(self.config.all_blocks, region_cells)
        }
        return WorldState(
            robot=self.config.robot_start,
            placements=placements,
            holding=None,
        )

    def sample_task_library(
        self,
        rng: random.Random,
        count: int = 24,
    ) -> List[Task]:
        single_tasks = [
            Task(((block, region),))
            for block in NONWHITE_BLOCKS
            for region in NONWHITE_REGIONS
        ]
        pair_tasks = [
            Task(tuple(zip(blocks, regions)))
            for blocks in itertools.combinations(NONWHITE_BLOCKS, 2)
            for regions in itertools.permutations(NONWHITE_REGIONS, 2)
        ]
        candidates = single_tasks + pair_tasks
        rng.shuffle(candidates)
        chosen: List[Task] = []
        seen = set()
        for task in candidates:
            if task.assignments in seen:
                continue
            seen.add(task.assignments)
            chosen.append(task)
            if len(chosen) >= count:
                break
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
        cells: List[Coord] = []
        for name in WHITE_REGIONS:
            coord = self.config.region_coords[name]
            occupant = state.block_at(coord)
            if occupant is None or occupant in task.blocks:
                cells.append(coord)
        for block in task.blocks:
            cells.append(state.placements[block])
        deduped: List[Coord] = []
        seen = set()
        task_targets = set(task.goal_positions(self.config).values())
        for cell in cells:
            if cell in task_targets:
                continue
            if cell in seen:
                continue
            seen.add(cell)
            deduped.append(cell)
        return deduped


def block_color(block: str) -> str:
    return BLOCK_COLOR_MAP[block]


def region_color(region: str) -> str:
    if region.startswith("white"):
        return "white"
    return region
