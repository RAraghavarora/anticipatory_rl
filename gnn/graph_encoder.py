from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

from anticipatory_rl.envs.restaurant.planner import RestaurantPlannerState

NODE_TYPES = ("agent", "room", "location", "object")
ROOM_ASSIGNMENTS = {
    "Kitchen": ("countertop", "coffeemachine", "dishwasher", "shelf"),
    "Serving": ("servingtable", "fountain"),
    "Storage": ("pantry",),
}
SBERT_DIM = 384

BINARY_ATTRS = (
    "is_dirty",
    "filled_water",
    "filled_coffee",
    "is_empty",
    "is_held",
    "is_liquid_source",
    "is_container",
    "hand_free",
    "bread_spread",
)


@lru_cache(maxsize=1)
def _get_sbert() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def _sbert_encode(names: tuple[str, ...]) -> torch.Tensor:
    model = _get_sbert()
    return torch.tensor(model.encode(list(names)), dtype=torch.float32)


def _type_onehot(type_idx: int) -> torch.Tensor:
    v = torch.zeros(len(NODE_TYPES))
    v[type_idx] = 1.0
    return v


def _binary_attrs_for_object(obj_state, env) -> torch.Tensor:
    attrs = torch.zeros(len(BINARY_ATTRS))
    attrs[0] = float(obj_state.dirty)
    attrs[1] = float(obj_state.filled_with == "water")
    attrs[2] = float(obj_state.filled_with == "coffee")
    attrs[3] = float(obj_state.filled_with is None and not obj_state.dirty)
    attrs[4] = 0.0
    kind = obj_state.kind if hasattr(obj_state, "kind") else ""
    attrs[5] = float(kind in ("water_fountain", "water_machine") or "water" in obj_state.name)
    attrs[6] = float(kind in ("cup", "mug", "bowl", "jar"))
    return attrs


def state_to_graph(
    state: RestaurantPlannerState,
    env,
) -> Data:
    room_assignments = {
        room: tuple(loc for loc in locations if loc in env.location_coords)
        for room, locations in ROOM_ASSIGNMENTS.items()
    }
    room_assignments = {room: locations for room, locations in room_assignments.items() if locations}

    names: list[str] = []
    types: list[int] = []
    positions: list[tuple[float, float]] = []
    obj_indices: dict[str, int] = {}

    names.append("robot")
    types.append(0)
    agent_loc = state.agent_location
    agent_pos = env.location_coords.get(agent_loc, (0, 0))
    positions.append((float(agent_pos[0]), float(agent_pos[1])))

    for room_name in room_assignments:
        names.append(room_name)
        types.append(1)
        locs_in_room = room_assignments[room_name]
        coords = [env.location_coords.get(loc, (0, 0)) for loc in locs_in_room if loc in env.location_coords]
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
        else:
            cx, cy = 0.0, 0.0
        positions.append((cx, cy))

    room_name_to_idx = {rn: i for i, rn in enumerate(room_assignments)}
    loc_name_to_idx: dict[str, int] = {}
    for loc_name in env.locations:
        names.append(loc_name)
        types.append(2)
        pos = env.location_coords.get(loc_name, (0, 0))
        positions.append((float(pos[0]), float(pos[1])))
        loc_name_to_idx[loc_name] = len(names) - 1

    for obj_name, obj_state in state.objects.items():
        obj_indices[obj_name] = len(names)
        names.append(obj_name)
        types.append(3)
        if obj_state.location and obj_state.location in env.location_coords:
            pos = env.location_coords[obj_state.location]
        elif state.holding == obj_name:
            pos = agent_pos
        else:
            pos = (0, 0)
        positions.append((float(pos[0]), float(pos[1])))

    n = len(names)
    sbert_feats = _sbert_encode(tuple(names))
    features = []
    for i in range(n):
        type_oh = _type_onehot(types[i])
        px, py = positions[i]
        pos_t = torch.tensor([px, py], dtype=torch.float32)
        if types[i] == 3:
            obj_name = names[i]
            obj_state = state.objects[obj_name]
            bin_attrs = _binary_attrs_for_object(obj_state, env)
        elif types[i] == 0:
            bin_attrs = torch.zeros(len(BINARY_ATTRS))
            bin_attrs[7] = float(state.holding is None)
            bin_attrs[8] = float(state.bread_spread is not None)
        else:
            bin_attrs = torch.zeros(len(BINARY_ATTRS))
        feat = torch.cat([sbert_feats[i], type_oh, pos_t, bin_attrs])
        features.append(feat)

    x = torch.stack(features)

    edge_src: list[int] = []
    edge_dst: list[int] = []

    agent_idx = 0
    if state.agent_location in loc_name_to_idx:
        loc_idx = loc_name_to_idx[state.agent_location]
        edge_src.extend([agent_idx, loc_idx])
        edge_dst.extend([loc_idx, agent_idx])

    for obj_name, obj_node_idx in obj_indices.items():
        obj_state = state.objects[obj_name]
        if state.holding == obj_name:
            edge_src.extend([obj_node_idx, agent_idx])
            edge_dst.extend([agent_idx, obj_node_idx])
        elif obj_state.location and obj_state.location in loc_name_to_idx:
            loc_idx = loc_name_to_idx[obj_state.location]
            edge_src.extend([obj_node_idx, loc_idx])
            edge_dst.extend([loc_idx, obj_node_idx])
        if obj_state.contained_in and obj_state.contained_in in obj_indices:
            container_idx = obj_indices[obj_state.contained_in]
            edge_src.extend([obj_node_idx, container_idx])
            edge_dst.extend([container_idx, obj_node_idx])

    for room_name, locs in room_assignments.items():
        room_idx = 1 + room_name_to_idx[room_name]
        for loc in locs:
            if loc in loc_name_to_idx:
                loc_idx = loc_name_to_idx[loc]
                edge_src.extend([loc_idx, room_idx])
                edge_dst.extend([room_idx, loc_idx])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=n)
