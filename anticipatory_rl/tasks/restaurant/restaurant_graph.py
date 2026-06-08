"""Restaurant state -> graph features for anticipatory-cost learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from anticipatory_rl.tasks.restaurant.restaurant_planner import RestaurantPlannerState


@dataclass
class RestaurantGraph:
    node_features: np.ndarray  # [N, F]
    edge_index: np.ndarray  # [2, E]


def _location_feature(
    loc: str,
    *,
    coords: Dict[str, Tuple[int, int]],
    max_x: float,
    max_y: float,
    is_agent: bool,
) -> np.ndarray:
    x, y = coords[loc]
    return np.array(
        [
            1.0,  # node is location
            0.0,  # node is object
            1.0 if is_agent else 0.0,
            float(x) / max_x,
            float(y) / max_y,
            0.0,  # dirty
            0.0,  # empty
            0.0,  # water
            0.0,  # coffee
            0.0,  # apple
            0.0,  # mug
            0.0,  # cup
            0.0,  # bowl
        ],
        dtype=np.float32,
    )


def _object_feature(
    name: str,
    state: RestaurantPlannerState,
    *,
    coords: Dict[str, Tuple[int, int]],
    max_x: float,
    max_y: float,
) -> np.ndarray:
    obj = state.objects[name]
    if state.holding == name:
        loc = state.agent_location
    elif obj.contained_in is not None and obj.contained_in in state.objects:
        loc = state.objects[obj.contained_in].location
        if loc is None:
            loc = state.agent_location
    elif obj.location is not None:
        loc = obj.location
    else:
        loc = state.agent_location
    x, y = coords[loc]
    return np.array(
        [
            0.0,  # node is location
            1.0,  # node is object
            1.0 if state.holding == name else 0.0,
            float(x) / max_x,
            float(y) / max_y,
            1.0 if obj.dirty else 0.0,
            1.0 if obj.filled_with is None and obj.contained_in is None else 0.0,  # empty
            1.0 if obj.filled_with == "water" else 0.0,  # water
            1.0 if obj.filled_with == "coffee" else 0.0,  # coffee
            1.0 if obj.contained_in is not None else 0.0,  # apple (contained)
            1.0 if obj.kind == "mug" else 0.0,  # mug
            1.0 if obj.kind == "cup" else 0.0,  # cup
            1.0 if obj.kind == "bowl" else 0.0,  # bowl
        ],
        dtype=np.float32,
    )


def build_restaurant_graph(
    state: RestaurantPlannerState,
    *,
    locations: Sequence[str],
    location_coords: Dict[str, Tuple[int, int]],
) -> RestaurantGraph:
    max_x = float(max(v[0] for v in location_coords.values()) + 1)
    max_y = float(max(v[1] for v in location_coords.values()) + 1)
    loc_nodes = list(locations)
    obj_nodes = sorted(state.objects.keys())
    nodes = loc_nodes + obj_nodes
    idx = {name: i for i, name in enumerate(nodes)}

    features: List[np.ndarray] = []
    for loc in loc_nodes:
        features.append(
            _location_feature(
                loc,
                coords=location_coords,
                max_x=max_x,
                max_y=max_y,
                is_agent=(state.agent_location == loc),
            )
        )
    for obj_name in obj_nodes:
        features.append(
            _object_feature(
                obj_name,
                state,
                coords=location_coords,
                max_x=max_x,
                max_y=max_y,
            )
        )

    edge_src: List[int] = []
    edge_dst: List[int] = []
    # Location adjacency (4-neighborhood by Manhattan distance 1).
    for i, a in enumerate(loc_nodes):
        ax, ay = location_coords[a]
        for j, b in enumerate(loc_nodes):
            if i == j:
                continue
            bx, by = location_coords[b]
            if abs(ax - bx) + abs(ay - by) == 1:
                edge_src.append(idx[a])
                edge_dst.append(idx[b])
    # Object <-> location containment edges.
    for obj_name in obj_nodes:
        obj = state.objects[obj_name]
        if state.holding == obj_name:
            loc = state.agent_location
        elif obj.contained_in is not None and obj.contained_in in state.objects:
            container_loc = state.objects[obj.contained_in].location
            loc = container_loc if container_loc is not None else state.agent_location
        elif obj.location is not None:
            loc = obj.location
        else:
            loc = state.agent_location
        edge_src.append(idx[obj_name])
        edge_dst.append(idx[loc])
        edge_src.append(idx[loc])
        edge_dst.append(idx[obj_name])

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    node_features = np.stack(features, axis=0).astype(np.float32)
    return RestaurantGraph(node_features=node_features, edge_index=edge_index)
