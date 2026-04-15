from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random


Coord = Tuple[int, int]


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
    '''
    2D Blocksworld configuration.
    '''
    
    width: int = 7
    height: int = 7
    move_cost: int = 25
    pick_cost: int = 100
    place_cost: int = 100
    region_layout: Tuple[Tuple[str, Coord], ...] | None = None
    block_colors: Tuple[Tuple[str, str], ...] | None = None

    @property
    def robot_start(self) -> Coord:
        return (self.width // 2, self.height // 2)

    @property
    def region_coords(self) -> Dict[str, Coord]:
        if self.region_layout is not None:
            return dict(self.region_layout)
        return {
            "red": (1, 1),
            "blue": (3, 1),
            "green": (5, 1),
            "cyan": (1, 3),
            "pink": (3, 3),
            "orange": (5, 3),
            "brown": (1, 5),
            "white_1": (3, 5),
            "white_2": (5, 5),
            "white_3": (5, 6),
        }

    @classmethod
    def sample(cls, rng: random.Random) -> "WorldConfig":
        slots = [(x, y) for y in range(cls.height) for x in range(cls.width)]
        chosen = rng.sample(slots, k=len(COLORED_REGIONS) + len(WHITE_REGIONS))
        region_names = list(COLORED_REGIONS + WHITE_REGIONS)
        rng.shuffle(region_names)
        layout = tuple(sorted(zip(region_names, chosen)))

        colored_blocks = set(rng.sample(BLOCK_NAMES, k=5))
        color_pool = rng.sample(COLORED_REGIONS, k=len(colored_blocks))
        block_colors = []
        for block in BLOCK_NAMES:
            if block in colored_blocks:
                color = color_pool.pop()
            else:
                color = "white"
            block_colors.append((block, color))
        return cls(
            region_layout=layout,
            block_colors=tuple(block_colors),
        )

    @property
    def block_color_map(self) -> Dict[str, str]:
        if self.block_colors is not None:
            return dict(self.block_colors)
        return {
            "a": "red",
            "b": "blue",
            "c": "green",
            "d": "cyan",
            "e": "pink",
            "f": "white",
            "g": "white",
            "h": "white",
        }

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
    def nonwhite_regions(self) -> Tuple[str, ...]:
        return COLORED_REGIONS

    @property
    def white_regions(self) -> Tuple[str, ...]:
        return WHITE_REGIONS

    @property
    def all_blocks(self) -> Tuple[str, ...]:
        return BLOCK_NAMES

    @property
    def all_regions(self) -> Tuple[str, ...]:
        return COLORED_REGIONS + WHITE_REGIONS

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
    """
    Given a config with regions and a task, generate a 2D blockworld state.
    """
    
    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()

    def sample_initial_state(self, rng: random.Random) -> WorldState:
        region_cells = list(self.config.region_cells)
        rng.shuffle(region_cells)
        placements = {
            block: coord
            for block, coord in zip(self.config.all_blocks, region_cells)
        }
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

    def candidate_parking_cells(
        self,
        state: WorldState,
        task: Task,
    ) -> List[Coord]:
        cells: List[Coord] = []
        for name in self.config.white_regions:
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

def block_color(block: str, config: WorldConfig) -> str:
    return config.block_color(block)


def region_color(region: str) -> str:
    if region.startswith("white"):
        return "white"
    return region
