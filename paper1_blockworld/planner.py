from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .world import Coord, Task, WorldConfig, WorldState


PLAN_LINE = re.compile(r"^\((?P<name>[a-z0-9_-]+)\s+(?P<args>[^)]*)\)$")


@dataclass
class PlanResult:
    cost: int
    actions: List[Tuple[str, Tuple[str, ...]]]
    final_state: WorldState
    moved_blocks: Tuple[str, ...]


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
        self._cache: Dict[Tuple[Tuple[object, ...], Tuple[Tuple[str, Coord], ...]], PlanResult] = {}

    def plan_for_task(
        self,
        state: WorldState,
        task: Task,
    ) -> PlanResult:
        return self.plan_to_placements(state, task.goal_positions(self.config))

    def plan_to_placements(
        self,
        state: WorldState,
        goal_placements: Mapping[str, Coord],
    ) -> PlanResult:
        cache_key = (state.signature(), tuple(sorted(goal_placements.items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        problem_text = self._build_problem_text(state, goal_placements)
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
            actions = self._parse_plan(plan_path.read_text(encoding="utf-8"))
        result = self._simulate_plan(state, actions)
        self._cache[cache_key] = result
        return result

    def _build_problem_text(
        self,
        state: WorldState,
        goal_placements: Mapping[str, Coord],
    ) -> str:
        location_names = [
            self.config.location_name((x, y))
            for y in range(self.config.height)
            for x in range(self.config.width)
        ]
        block_names = " ".join(self.config.all_blocks)
        region_names = " ".join(self.config.all_regions)
        init_lines: List[str] = []
        init_lines.append(f"(at bot {self.config.location_name(state.robot)})")
        if state.holding is None:
            init_lines.append("(handempty bot)")
        else:
            init_lines.append(f"(holding bot {state.holding})")

        occupied = set(state.placements.values())
        for y in range(self.config.height):
            for x in range(self.config.width):
                loc = self.config.location_name((x, y))
                if (x, y) not in occupied:
                    init_lines.append(f"(clear {loc})")
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                        init_lines.append(
                            f"(adjacent {loc} {self.config.location_name((nx, ny))})"
                        )

        for region, coord in self.config.region_coords.items():
            init_lines.append(
                f"(belongs {self.config.location_name(coord)} {region})"
            )

        for block, coord in sorted(state.placements.items()):
            init_lines.append(f"(on {block} {self.config.location_name(coord)})")

        goal_lines = ["(handempty bot)"]
        for block, coord in sorted(goal_placements.items()):
            goal_lines.append(f"(on {block} {self.config.location_name(coord)})")

        return "\n".join(
            [
                "(define (problem paper1-blockworld-problem)",
                "  (:domain paper1-blockworld)",
                "  (:objects",
                "    bot - robot",
                f"    {' '.join(location_names)} - location",
                f"    {block_names} - block",
                f"    {region_names} - region",
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

    def _parse_plan(self, text: str) -> List[Tuple[str, Tuple[str, ...]]]:
        actions: List[Tuple[str, Tuple[str, ...]]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lower()
            if not line or line.startswith(";"):
                continue
            match = PLAN_LINE.match(line)
            if match is None:
                continue
            name = match.group("name")
            args = tuple(match.group("args").split())
            actions.append((name, args))
        return actions

    def _simulate_plan(
        self,
        initial_state: WorldState,
        actions: Sequence[Tuple[str, Tuple[str, ...]]],
    ) -> PlanResult:
        state = initial_state.clone()
        moved_blocks: List[str] = []
        total_cost = 0
        for name, args in actions:
            if name == "move":
                _, src, dst = args
                if state.robot != self._coord_from_loc(src):
                    raise RuntimeError(f"Robot not at expected source {src}.")
                state.robot = self._coord_from_loc(dst)
                total_cost += self.config.move_cost
                continue
            if name == "pick":
                _, block, loc = args
                coord = self._coord_from_loc(loc)
                if state.robot != coord:
                    raise RuntimeError(f"Robot not at pick location {loc}.")
                if state.holding is not None:
                    raise RuntimeError("Cannot pick while already holding a block.")
                if state.placements.get(block) != coord:
                    raise RuntimeError(f"Block {block} not at pick location {loc}.")
                del state.placements[block]
                state.holding = block
                moved_blocks.append(block)
                total_cost += self.config.pick_cost
                continue
            if name == "place":
                _, block, loc, _region = args
                coord = self._coord_from_loc(loc)
                if state.robot != coord:
                    raise RuntimeError(f"Robot not at place location {loc}.")
                if state.holding != block:
                    raise RuntimeError(f"Robot is not holding {block}.")
                if state.block_at(coord) is not None:
                    raise RuntimeError(f"Location {loc} is already occupied.")
                state.placements[block] = coord
                state.holding = None
                total_cost += self.config.place_cost
                continue
            raise RuntimeError(f"Unsupported planner action: {name}")

        moved_tuple = tuple(dict.fromkeys(moved_blocks))
        return PlanResult(
            cost=total_cost,
            actions=list(actions),
            final_state=state,
            moved_blocks=moved_tuple,
        )

    @staticmethod
    def _coord_from_loc(loc: str) -> Coord:
        _, x, y = loc.split("_")
        return int(x), int(y)
