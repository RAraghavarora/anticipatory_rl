"""Relational-graph observation variant of :class:`SimpleGridImageEnv`."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from .simple_grid_image_env import (
    OBJECT_NAMES,
    RECEPTACLE_LIST,
    SimpleGridImageEnv,
)


Coord = Tuple[int, int]

NODE_TYPES: Tuple[str, ...] = ("agent", "object", "receptacle")
EDGE_TYPES: Tuple[str, ...] = (
    "agent_object",
    "object_agent",
    "object_receptacle",
    "receptacle_object",
    "agent_receptacle",
    "receptacle_agent",
    "object_object",
)
EDGE_TYPE_TO_IDX = {name: idx for idx, name in enumerate(EDGE_TYPES)}


class SimpleGridEnv(SimpleGridImageEnv):
    """State-sharing grid environment that emits relational graph observations.

    The core dynamics (task sampling, transitions, rewards, config handling, etc.)
    come directly from :class:`SimpleGridImageEnv`. This subclass changes only the
    observation head so that downstream agents can consume a structured graph with
    per-entity features and typed edges describing spatial relationships.
    """

    def __init__(
        self,
        *,
        max_edges: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.node_feature_dim = 10
        self.edge_feature_dim = len(EDGE_TYPES) + 6
        self.global_feat_dim = 6
        self.max_object_nodes = len(self.object_names)
        self.max_receptacle_nodes = len(self.receptacle_names)
        self.max_nodes = 1 + self.max_object_nodes + self.max_receptacle_nodes
        self.max_edges = max_edges or self._default_max_edges()
        self._agent_idx = 0
        self._object_offset = 1
        self._receptacle_offset = 1 + self.max_object_nodes
        self.observation_space = spaces.Dict(
            nodes=spaces.Dict(
                features=spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_nodes, self.node_feature_dim),
                    dtype=np.float32,
                ),
                types=spaces.Box(
                    low=0,
                    high=len(NODE_TYPES),
                    shape=(self.max_nodes,),
                    dtype=np.int64,
                ),
                mask=spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_nodes,),
                    dtype=np.float32,
                ),
            ),
            edges=spaces.Dict(
                index=spaces.Box(
                    low=0,
                    high=self.max_nodes,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                features=spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_edges, self.edge_feature_dim),
                    dtype=np.float32,
                ),
                mask=spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_edges,),
                    dtype=np.float32,
                ),
            ),
            global_context=spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.global_feat_dim,),
                dtype=np.float32,
            ),
        )

    # ------------------------------------------------------------------ core observation API
    def _obs(self) -> Dict[str, object]:
        nodes, coords, node_mask = self._encode_nodes()
        edges = self._encode_edges(coords, node_mask)
        globals_vec = self._encode_globals()
        return {
            "nodes": nodes,
            "edges": edges,
            "global_context": globals_vec,
        }

    # ------------------------------------------------------------------ node encoding
    def _encode_nodes(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        grid_norm = max(1.0, float(self.grid_size - 1))
        node_features = np.zeros((self.max_nodes, self.node_feature_dim), dtype=np.float32)
        node_types = np.zeros((self.max_nodes,), dtype=np.int64)
        node_mask = np.zeros((self.max_nodes,), dtype=np.float32)
        raw_coords = np.full((self.max_nodes, 2), -1.0, dtype=np.float32)

        def assign_common(idx: int, coord: Coord | Tuple[float, float], node_type: int) -> None:
            node_types[idx] = node_type
            node_mask[idx] = 1.0
            raw_coords[idx] = np.asarray(coord, dtype=np.float32)
            node_features[idx, :2] = raw_coords[idx] / grid_norm
            node_features[idx, 2 + node_type] = 1.0

        # Agent node
        agent_idx = self._agent_idx
        assign_common(agent_idx, self.state.agent, NODE_TYPES.index("agent"))
        carrying_idx = (
            self.object_names.index(self.state.carrying)
            if self.state.carrying in self.object_names
            else -1
        )
        node_features[agent_idx, 5] = 1.0 if self.state.carrying is not None else 0.0
        node_features[agent_idx, 6] = 1.0 if self._coord_on_receptacle(self.state.agent) else 0.0
        node_features[agent_idx, 7] = 1.0 if self.state.carrying == self.target_object else 0.0
        node_features[agent_idx, 8] = (
            float(carrying_idx) / max(1.0, self.max_object_nodes - 1)
            if carrying_idx >= 0
            else -1.0
        )
        node_features[agent_idx, 9] = (
            float(self.object_names.index(self.target_object)) / max(1.0, self.max_object_nodes - 1)
        )

        # Object nodes
        for i, name in enumerate(self.object_names):
            idx = self._object_offset + i
            node_types[idx] = NODE_TYPES.index("object")
            if name not in self.active_objects:
                continue
            coord = self._object_position(name)
            assign_common(idx, coord, NODE_TYPES.index("object"))
            is_target = 1.0 if name == self.target_object else 0.0
            is_carried = 1.0 if self.state.carrying == name else 0.0
            tile_coord = (
                (int(coord[0]), int(coord[1])) if coord is not None else None
            )
            on_target_surface = (
                1.0
                if tile_coord is not None and tile_coord in self.receptacles[self.target_receptacle]
                else 0.0
            )
            node_features[idx, 5] = 1.0
            node_features[idx, 6] = is_target
            node_features[idx, 7] = is_carried
            node_features[idx, 8] = on_target_surface
            node_features[idx, 9] = float(i) / max(1.0, self.max_object_nodes - 1)

        # Receptacle nodes
        for j, name in enumerate(self.receptacle_names):
            idx = self._receptacle_offset + j
            node_types[idx] = NODE_TYPES.index("receptacle")
            centroid = self._receptacle_centroid(name)
            assign_common(idx, centroid, NODE_TYPES.index("receptacle"))
            occupancy = len(self._objects_on_receptacle(name))
            node_features[idx, 5] = occupancy / max(1.0, float(self.max_object_nodes))
            node_features[idx, 6] = 1.0 if name == self.target_receptacle else 0.0
            node_features[idx, 7] = len(self.receptacles[name]) / max(1.0, float(self.grid_size**2))
            node_features[idx, 8] = (
                1.0 if self.state.agent in self.receptacles[name] else 0.0
            )
            node_features[idx, 9] = float(j) / max(1.0, self.max_receptacle_nodes - 1)

        nodes = {
            "features": node_features,
            "types": node_types,
            "mask": node_mask,
        }
        return nodes, raw_coords, node_mask

    # ------------------------------------------------------------------ edge encoding
    def _encode_edges(
        self,
        coords: np.ndarray,
        node_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_attr = np.zeros((self.max_edges, self.edge_feature_dim), dtype=np.float32)
        edge_mask = np.zeros((self.max_edges,), dtype=np.float32)
        cursor = 0
        grid_norm = max(1.0, float(self.grid_size - 1))

        def coord_for(idx: int) -> np.ndarray | None:
            if node_mask[idx] == 0.0:
                return None
            return coords[idx]

        def discrete_tile(idx: int) -> Tuple[int, int] | None:
            arr = coord_for(idx)
            if arr is None:
                return None
            return (int(round(float(arr[0]))), int(round(float(arr[1]))))

        def add_edge(
            src: int,
            dst: int,
            edge_type: str,
            contact_flag: bool,
            target_flag: bool,
        ) -> None:
            nonlocal cursor
            if cursor >= self.max_edges:
                raise RuntimeError("Graph edge budget exceeded; increase max_edges.")
            src_coord = coord_for(src)
            dst_coord = coord_for(dst)
            if src_coord is None or dst_coord is None:
                return
            rel = dst_coord - src_coord
            rel_norm = rel / grid_norm
            manhattan = (np.abs(rel[0]) + np.abs(rel[1])) / (2.0 * grid_norm)
            euclid = float(np.linalg.norm(rel, ord=2)) / (np.sqrt(2.0) * grid_norm)
            feat = np.zeros((self.edge_feature_dim,), dtype=np.float32)
            type_idx = EDGE_TYPE_TO_IDX[edge_type]
            feat[type_idx] = 1.0
            offset = len(EDGE_TYPES)
            feat[offset] = rel_norm[0]
            feat[offset + 1] = rel_norm[1]
            feat[offset + 2] = manhattan
            feat[offset + 3] = euclid
            feat[offset + 4] = 1.0 if contact_flag else 0.0
            feat[offset + 5] = 1.0 if target_flag else 0.0
            edge_index[:, cursor] = (src, dst)
            edge_attr[cursor] = feat
            edge_mask[cursor] = 1.0
            cursor += 1

        agent_idx = self._agent_idx
        object_indices = {
            name: self._object_offset + i for i, name in enumerate(self.object_names)
        }
        receptacle_indices = {
            name: self._receptacle_offset + j
            for j, name in enumerate(self.receptacle_names)
        }

        # Agent <-> objects
        for name in self.active_objects:
            idx = object_indices[name]
            obj_tile = discrete_tile(idx)
            agent_contact = obj_tile is not None and self.state.agent == obj_tile
            is_target_obj = name == self.target_object
            add_edge(agent_idx, idx, "agent_object", agent_contact, is_target_obj)
            add_edge(idx, agent_idx, "object_agent", agent_contact, is_target_obj)

        # Agent <-> receptacles
        for rec_name in self.receptacle_names:
            rec_idx = receptacle_indices[rec_name]
            agent_on_rec = self.state.agent in self.receptacles[rec_name]
            is_target_rec = rec_name == self.target_receptacle
            add_edge(agent_idx, rec_idx, "agent_receptacle", agent_on_rec, is_target_rec)
            add_edge(rec_idx, agent_idx, "receptacle_agent", agent_on_rec, is_target_rec)

        # Object <-> receptacles
        for obj_name in self.active_objects:
            obj_idx = object_indices[obj_name]
            obj_tile = discrete_tile(obj_idx)
            for rec_name in self.receptacle_names:
                rec_idx = receptacle_indices[rec_name]
                on_surface = obj_tile is not None and obj_tile in self.receptacles[rec_name]
                target_pair = (obj_name == self.target_object) or (
                    rec_name == self.target_receptacle
                )
                add_edge(obj_idx, rec_idx, "object_receptacle", on_surface, target_pair)
                add_edge(rec_idx, obj_idx, "receptacle_object", on_surface, target_pair)

        # Object <-> object proximity (complete directed graph over active objects)
        for i, src_name in enumerate(self.active_objects):
            src_idx = object_indices[src_name]
            src_tile = discrete_tile(src_idx)
            for j, dst_name in enumerate(self.active_objects):
                if i == j:
                    continue
                dst_idx = object_indices[dst_name]
                dst_tile = discrete_tile(dst_idx)
                same_tile = src_tile is not None and dst_tile is not None and src_tile == dst_tile
                touches_target = src_name == self.target_object or dst_name == self.target_object
                add_edge(src_idx, dst_idx, "object_object", same_tile, touches_target)

        return {
            "index": edge_index,
            "features": edge_attr,
            "mask": edge_mask,
        }

    # ------------------------------------------------------------------ globals
    def _encode_globals(self) -> np.ndarray:
        obj_idx = self.object_names.index(self.target_object)
        rec_idx = self.receptacle_names.index(self.target_receptacle)
        max_obj_norm = max(1.0, self.max_object_nodes - 1)
        max_rec_norm = max(1.0, self.max_receptacle_nodes - 1)
        carrying = 1.0 if self.state.carrying is not None else 0.0
        carrying_target = 1.0 if self.state.carrying == self.target_object else 0.0
        on_target_surface = (
            1.0
            if self._objects_on_receptacle(self.target_receptacle)
            else 0.0
        )
        return np.asarray(
            [
                1.0 if self.task_type == "clear" else 0.0,
                obj_idx / max_obj_norm,
                rec_idx / max_rec_norm,
                carrying,
                carrying_target,
                on_target_surface,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ helpers
    def _receptacle_centroid(self, name: str) -> Tuple[float, float]:
        tiles = self.receptacles[name]
        xs = [coord[0] for coord in tiles]
        ys = [coord[1] for coord in tiles]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _default_max_edges(self) -> int:
        obj = self.max_object_nodes
        rec = self.max_receptacle_nodes
        agent_obj = 2 * obj
        agent_rec = 2 * rec
        obj_rec = 2 * obj * rec
        obj_obj = max(0, 2 * obj * (obj - 1))
        return agent_obj + agent_rec + obj_rec + obj_obj


__all__ = [
    "SimpleGridEnv",
    "OBJECT_NAMES",
    "RECEPTACLE_LIST",
]
