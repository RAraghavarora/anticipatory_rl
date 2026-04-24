from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .world import Coord, Point, WorldConfig, WorldState


@dataclass(frozen=True)
class MotionPath:
    length: float
    waypoints: Tuple[Coord, ...]

    @property
    def cost(self) -> int:
        return int(round(self.length * 25.0))


class LazyPRMMotionPlanner:
    """
    Small in-repo Lazy PRM over cell centers.

    Region tiles remain traversable even when occupied because manipulation
    semantics require the robot to stand on the object/destination tile.
    Interest poses are the current robot cell plus region tiles.
    """

    def __init__(
        self,
        config: WorldConfig,
        state: WorldState,
        *,
        k_neighbors: int = 10,
    ) -> None:
        self.config = config
        self.state = state
        self.k_neighbors = max(4, k_neighbors)
        self.sample_nodes = tuple(sorted(self._build_sample_nodes()))
        self._candidate_edges = self._build_candidate_edges()
        self._edge_validity: Dict[Tuple[Coord, Coord], bool] = {}

    def interest_poses(self) -> Tuple[Coord, ...]:
        poses = set(self.config.manipulation_cells)
        poses.add(self.state.robot)
        traversable = set(self.sample_nodes)
        return tuple(sorted(pose for pose in poses if pose in traversable))

    def pairwise_paths(
        self,
        poses: Sequence[Coord],
    ) -> Dict[Tuple[Coord, Coord], MotionPath]:
        paths: Dict[Tuple[Coord, Coord], MotionPath] = {}
        ordered = tuple(dict.fromkeys(poses))
        for idx, start in enumerate(ordered):
            for goal in ordered[idx + 1:]:
                path = self.shortest_path(start, goal)
                if path is None:
                    continue
                paths[(start, goal)] = path
                paths[(goal, start)] = MotionPath(
                    length=path.length,
                    waypoints=tuple(reversed(path.waypoints)),
                )
        return paths

    def shortest_path(self, start: Coord, goal: Coord) -> MotionPath | None:
        if start == goal:
            return MotionPath(length=0.0, waypoints=(start,))
        if start not in self.sample_nodes or goal not in self.sample_nodes:
            return None

        frontier: List[Tuple[float, Coord]] = [(0.0, start)]
        best_distance: Dict[Coord, float] = {start: 0.0}
        parent: Dict[Coord, Coord] = {}

        while frontier:
            current_distance, current = heapq.heappop(frontier)
            if current_distance > best_distance.get(current, math.inf):
                continue
            if current == goal:
                break
            for neighbor in self._candidate_edges.get(current, ()):
                edge = _canonical_edge(current, neighbor)
                valid = self._edge_validity.get(edge)
                if valid is None:
                    valid = self._edge_is_valid(current, neighbor)
                    self._edge_validity[edge] = valid
                if not valid:
                    continue
                step_cost = _euclidean(self._center(current), self._center(neighbor))
                candidate_distance = current_distance + step_cost
                if candidate_distance + 1e-9 >= best_distance.get(neighbor, math.inf):
                    continue
                best_distance[neighbor] = candidate_distance
                parent[neighbor] = current
                heapq.heappush(frontier, (candidate_distance, neighbor))

        if goal not in parent and goal != start:
            return None

        waypoints = [goal]
        cursor = goal
        while cursor != start:
            cursor = parent[cursor]
            waypoints.append(cursor)
        waypoints.reverse()
        return MotionPath(length=best_distance[goal], waypoints=tuple(waypoints))

    def _build_sample_nodes(self) -> Iterable[Coord]:
        traversable = set(self.config.all_cells)
        traversable.discard(self.state.robot)  # added back explicitly to preserve ordering
        yield self.state.robot
        for coord in sorted(traversable):
            yield coord

    def _build_candidate_edges(self) -> Dict[Coord, Tuple[Coord, ...]]:
        nodes = list(self.sample_nodes)
        centers = {coord: self._center(coord) for coord in nodes}
        adjacency: Dict[Coord, set[Coord]] = {coord: set() for coord in nodes}

        for coord in nodes:
            for neighbor in self.config.neighbors(coord):
                if neighbor in adjacency:
                    adjacency[coord].add(neighbor)
                    adjacency[neighbor].add(coord)

        for coord in nodes:
            ranked = sorted(
                (other for other in nodes if other != coord),
                key=lambda other: (
                    _euclidean(centers[coord], centers[other]),
                    other[1],
                    other[0],
                ),
            )
            for other in ranked[: self.k_neighbors]:
                adjacency[coord].add(other)
                adjacency[other].add(coord)

        return {
            coord: tuple(sorted(neighbors))
            for coord, neighbors in adjacency.items()
        }

    def _edge_is_valid(self, src: Coord, dst: Coord) -> bool:
        src_center = self._center(src)
        dst_center = self._center(dst)
        distance = _euclidean(src_center, dst_center)
        steps = max(2, int(math.ceil(distance * 40.0)))
        for step in range(steps + 1):
            t = step / steps
            x = src_center[0] + t * (dst_center[0] - src_center[0])
            y = src_center[1] + t * (dst_center[1] - src_center[1])
            if self._point_hits_obstacle((x, y)):
                return False
        return True

    def _point_hits_obstacle(self, point: Point) -> bool:
        return False

    @staticmethod
    def _center(coord: Coord) -> Point:
        return (coord[0] + 0.5, coord[1] + 0.5)


def _canonical_edge(a: Coord, b: Coord) -> Tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
