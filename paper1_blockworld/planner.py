from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Mapping, Sequence, Tuple

from .motion import LazyPRMMotionPlanner
from .world import Coord, Task, WorldConfig, WorldState


PLAN_LINE = re.compile(r"^\((?P<name>[a-z0-9_-]+)\s+(?P<args>[^)]*)\)$")


@dataclass
class PlanResult:
    cost: int
    actions: List[Tuple[str, Tuple[str, ...]]]
    final_state: WorldState
    moved_blocks: Tuple[str, ...]


@dataclass(frozen=True)
class ProblemContext:
    move_costs: Dict[Tuple[Coord, Coord], int]


class FastDownwardBlockworldPlanner:
    def __init__(
        self,
        config: WorldConfig,
        *,
        fast_downward_path: str | Path | None = None,
        search: str = "astar(hmax())",
    ) -> None:
        self.config = config
        self.search = search
        self.fast_downward_path = (
            Path(fast_downward_path)
            if fast_downward_path is not None
            else Path(__file__).resolve().parents[1] / "downward" / "fast-downward.py"
        )
        self.domain_path = Path(__file__).resolve().parent / "pddl" / "blockworld_domain.pddl"
        self._cache: Dict[Tuple[Tuple[object, ...], Tuple[object, ...]], PlanResult] = {}

    def plan_for_task(
        self,
        state: WorldState,
        task: Task,
    ) -> PlanResult:
        cache_key = (state.signature(), ("task", tuple(sorted(task.assignments))))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        problem_text, problem_context = self._build_problem_text(
            state,
            task=task,
        )
        actions = self._run_fast_downward(problem_text)
        result = self._simulate_plan(state, actions, problem_context)
        self._cache[cache_key] = result
        return result

    def plan_to_placements(
        self,
        state: WorldState,
        goal_placements: Mapping[str, Coord],
    ) -> PlanResult:
        self._validate_goal_placements(goal_placements)
        frozen_goals = tuple(sorted(goal_placements.items()))
        cache_key = (state.signature(), ("placements", frozen_goals))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        problem_text, problem_context = self._build_problem_text(
            state,
            goal_placements=goal_placements,
        )
        actions = self._run_fast_downward(problem_text)
        result = self._simulate_plan(state, actions, problem_context)
        self._cache[cache_key] = result
        return result

    def _run_fast_downward(
        self,
        problem_text: str,
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            problem_path = tmpdir_path / "problem.pddl"
            plan_path = tmpdir_path / "sas_plan"
            problem_path.write_text(problem_text, encoding="utf-8")
            cmd = [
                sys.executable,
                str(self.fast_downward_path),
                "--plan-file",
                str(plan_path),
                str(self.domain_path),
                str(problem_path),
                "--search",
                self.search,
            ]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Fast Downward failed.\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )
            if not plan_path.exists():
                raise FileNotFoundError(
                    "Fast Downward completed but did not write a plan file."
                )
            return self._parse_plan(plan_path.read_text(encoding="utf-8"))

    def _build_problem_text(
        self,
        state: WorldState,
        *,
        task: Task | None = None,
        goal_placements: Mapping[str, Coord] | None = None,
    ) -> Tuple[str, ProblemContext]:
        if (task is None) == (goal_placements is None):
            raise ValueError("Specify exactly one of task or goal_placements.")

        motion_planner = LazyPRMMotionPlanner(self.config, state)
        interest_poses = motion_planner.interest_poses()
        pairwise_paths = motion_planner.pairwise_paths(interest_poses)
        move_costs = {
            (src, dst): path.cost
            for (src, dst), path in pairwise_paths.items()
            if src != dst
        }
        location_coords = sorted(set(interest_poses) | set(self.config.region_cells))
        location_names = [self.config.location_name(coord) for coord in location_coords]

        init_lines: List[str] = []
        init_lines.append(f"(at bot {self.config.location_name(state.robot)})")
        if state.holding is None:
            init_lines.append("(handempty bot)")
        else:
            init_lines.append(f"(holding bot {state.holding})")

        for (src, dst), cost in sorted(move_costs.items()):
            init_lines.append(
                f"(move-edge {self.config.location_name(src)} {self.config.location_name(dst)})"
            )
            init_lines.append(
                f"(= (move-cost {self.config.location_name(src)} {self.config.location_name(dst)}) {cost})"
            )

        interest_set = set(interest_poses)
        for region, tiles in self.config.region_tiles.items():
            for tile in tiles:
                tile_name = self.config.location_name(tile)
                init_lines.append(f"(belongs {tile_name} {region})")
                for neighbor in self.config.neighbors(tile):
                    if neighbor in interest_set:
                        init_lines.append(
                            f"(adjacent {self.config.location_name(neighbor)} {tile_name})"
                        )

        occupied_regions = state.occupied_regions(self.config)
        occupied_tiles = set(state.placements.values())
        for region in self.config.all_regions:
            if region not in occupied_regions:
                init_lines.append(f"(region-empty {region})")

        for tile in self.config.region_cells:
            if tile not in occupied_tiles:
                init_lines.append(f"(clear {self.config.location_name(tile)})")

        for block, coord in sorted(state.placements.items()):
            region = self.config.region_for_coord(coord)
            if region is None:
                raise ValueError(f"Block {block} is not in a region tile: {coord}")
            loc_name = self.config.location_name(coord)
            init_lines.append(f"(on {block} {loc_name})")
            init_lines.append(f"(in-region {block} {region})")

        goal_lines = ["(handempty bot)"]
        if task is not None:
            for block, region in sorted(task.assignments):
                goal_lines.append(f"(in-region {block} {region})")
        else:
            assert goal_placements is not None
            for block, coord in sorted(goal_placements.items()):
                goal_lines.append(f"(on {block} {self.config.location_name(coord)})")

        problem_text = "\n".join(
            [
                "(define (problem paper1-blockworld-problem)",
                "  (:domain paper1-blockworld)",
                "  (:objects",
                "    bot - robot",
                f"    {' '.join(location_names)} - location",
                f"    {' '.join(self.config.all_blocks)} - block",
                f"    {' '.join(self.config.all_regions)} - region",
                "  )",
                "  (:init",
                "    (= (total-cost) 0)",
                *[f"    {line}" for line in init_lines],
                "  )",
                "  (:goal",
                "    (and",
                *[f"      {line}" for line in goal_lines],
                "    )",
                "  )",
                "  (:metric minimize (total-cost))",
                ")",
            ]
        )
        return problem_text, ProblemContext(move_costs=move_costs)

    def _parse_plan(self, text: str) -> List[Tuple[str, Tuple[str, ...]]]:
        actions: List[Tuple[str, Tuple[str, ...]]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lower()
            if not line or line.startswith(";"):
                continue
            match = PLAN_LINE.match(line)
            if match is None:
                continue
            actions.append((match.group("name"), tuple(match.group("args").split())))
        return actions

    def _simulate_plan(
        self,
        initial_state: WorldState,
        actions: Sequence[Tuple[str, Tuple[str, ...]]],
        context: ProblemContext,
    ) -> PlanResult:
        state = initial_state.clone()
        moved_blocks: List[str] = []
        total_cost = 0

        for name, args in actions:
            if name == "move":
                _, src, dst = args
                src_coord = self._coord_from_loc(src)
                dst_coord = self._coord_from_loc(dst)
                if state.robot != src_coord:
                    raise RuntimeError(f"Robot not at expected source {src}.")
                edge_cost = context.move_costs.get((src_coord, dst_coord))
                if edge_cost is None:
                    raise RuntimeError(f"No PRM move edge from {src} to {dst}.")
                state.robot = dst_coord
                total_cost += edge_cost
                continue

            if name == "pick":
                _, robot_loc, block, block_loc, region = args
                robot_coord = self._coord_from_loc(robot_loc)
                coord = self._coord_from_loc(block_loc)
                if state.robot != robot_coord:
                    raise RuntimeError(f"Robot not at expected pick source {robot_loc}.")
                if not self._is_adjacent(robot_coord, coord):
                    raise RuntimeError(
                        f"Robot at {robot_loc} is not adjacent to pick location {block_loc}."
                    )
                if state.holding is not None:
                    raise RuntimeError("Cannot pick while already holding a block.")
                if state.placements.get(block) != coord:
                    raise RuntimeError(f"Block {block} not at pick location {block_loc}.")
                if self.config.region_for_coord(coord) != region:
                    raise RuntimeError(f"Pick region mismatch for {block} at {block_loc}.")
                del state.placements[block]
                state.holding = block
                moved_blocks.append(block)
                total_cost += self.config.pick_cost
                continue

            if name == "place":
                _, robot_loc, block, loc, region = args
                robot_coord = self._coord_from_loc(robot_loc)
                coord = self._coord_from_loc(loc)
                if state.robot != robot_coord:
                    raise RuntimeError(f"Robot not at expected place source {robot_loc}.")
                if not self._is_adjacent(robot_coord, coord):
                    raise RuntimeError(
                        f"Robot at {robot_loc} is not adjacent to place location {loc}."
                    )
                if state.holding != block:
                    raise RuntimeError(f"Robot is not holding {block}.")
                if self.config.region_for_coord(coord) != region:
                    raise RuntimeError(f"Place region mismatch for {block} at {loc}.")
                if state.block_at(coord) is not None:
                    raise RuntimeError(f"Location {loc} is already occupied.")
                occupied_regions = state.occupied_regions(self.config)
                if region in occupied_regions:
                    raise RuntimeError(f"Region {region} is already occupied.")
                state.placements[block] = coord
                state.holding = None
                total_cost += self.config.place_cost
                continue

            raise RuntimeError(f"Unsupported planner action: {name}")

        return PlanResult(
            cost=total_cost,
            actions=list(actions),
            final_state=state,
            moved_blocks=tuple(dict.fromkeys(moved_blocks)),
        )

    def _validate_goal_placements(self, goal_placements: Mapping[str, Coord]) -> None:
        occupied_regions: Dict[str, str] = {}
        for block, coord in goal_placements.items():
            region = self.config.region_for_coord(coord)
            if region is None:
                raise ValueError(f"Goal placement for {block} is not inside a region: {coord}")
            previous = occupied_regions.get(region)
            if previous is not None and previous != block:
                raise ValueError(
                    f"Goal placements assign multiple blocks to region {region}: {previous}, {block}"
                )
            occupied_regions[region] = block

    @staticmethod
    def _coord_from_loc(loc: str) -> Coord:
        _, x, y = loc.split("_")
        return int(x), int(y)

    @staticmethod
    def _is_adjacent(src: Coord, dst: Coord) -> bool:
        return abs(src[0] - dst[0]) + abs(src[1] - dst[1]) == 1
