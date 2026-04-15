from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool

from .world import CONTENTS, ROOMS, RestaurantWorldConfig, RestaurantWorldState


NODE_TYPES = ("restaurant", "room", "robot", "container", "object")
EDGE_TYPES = (
    "contains_room",
    "room_in_restaurant",
    "room_contains_container",
    "container_in_room",
    "room_contains_object",
    "object_in_room",
    "container_supports_object",
    "object_on_container",
    "room_has_robot",
    "robot_in_room",
    "robot_holding_object",
    "object_held_by_robot",
)
NODE_TYPE_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}
EDGE_TYPE_INDEX = {name: idx for idx, name in enumerate(EDGE_TYPES)}
ATTRIBUTE_NAMES = (
    "dirty",
    "empty",
    "contains_water",
    "contains_coffee",
    "contains_fruit",
    "wash_source",
    "water_source",
    "coffee_source",
    "fruit_source",
)


@dataclass
class EncodedGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


@dataclass
class GraphRegressionExample:
    graph: EncodedGraph
    target: float


class TextEmbeddingProvider:
    def __init__(
        self,
        *,
        mode: str = "sbert",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        cache_path: Path | None = None,
    ) -> None:
        self.mode = mode
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)
        self.cache_path = cache_path
        self._cache: Dict[str, torch.Tensor] = {}
        self._model = None
        if self.cache_path is not None and self.cache_path.exists():
            payload = torch.load(self.cache_path, map_location="cpu")
            self._cache = {
                key: value.float().cpu()
                for key, value in payload.get("embeddings", {}).items()
            }

    def embed(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached.clone()
        if self.mode == "hash":
            vector = self._hash_embed(text)
        else:
            vector = self._sbert_embed(text)
        self._cache[text] = vector.clone().cpu()
        self._flush_cache()
        return vector

    def _sbert_embed(self, text: str) -> torch.Tensor:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for SBERT node features. "
                    "Install it or rerun with --text-encoder hash for smoke tests."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        vector = self._model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return vector.detach().float().cpu()

    def _hash_embed(self, text: str) -> torch.Tensor:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vec = torch.randn((self.embedding_dim,), generator=generator, dtype=torch.float32)
        return vec / vec.norm(p=2).clamp(min=1e-6)

    def _flush_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "embeddings": self._cache,
            },
            self.cache_path,
        )


def graph_feature_dim(text_dim: int) -> int:
    return text_dim + len(NODE_TYPES) + 2 + len(ATTRIBUTE_NAMES)


def edge_feature_dim() -> int:
    return len(EDGE_TYPES) + 2


def encode_state_as_graph(
    state: RestaurantWorldState,
    config: RestaurantWorldConfig,
    *,
    text_provider: TextEmbeddingProvider,
) -> EncodedGraph:
    container_names = sorted(config.containers)
    object_names = sorted(state.objects)
    node_count = 1 + len(ROOMS) + 1 + len(container_names) + len(object_names)
    text_dim = text_provider.embedding_dim
    node_dim = graph_feature_dim(text_dim)

    features = torch.zeros((node_count, node_dim), dtype=torch.float32)
    edges: List[List[int]] = []
    edge_features: List[torch.Tensor] = []

    restaurant_idx = 0
    room_offset = 1
    robot_idx = room_offset + len(ROOMS)
    container_offset = robot_idx + 1
    object_offset = container_offset + len(container_names)
    room_index = {room: room_offset + idx for idx, room in enumerate(ROOMS)}
    container_index = {name: container_offset + idx for idx, name in enumerate(container_names)}
    object_index = {name: object_offset + idx for idx, name in enumerate(object_names)}

    def set_node(
        idx: int,
        *,
        text: str,
        node_type: str,
        coord: Tuple[int, int],
        attrs: Sequence[float],
    ) -> None:
        text_vec = text_provider.embed(text)
        offset = 0
        features[idx, offset : offset + text_dim] = text_vec
        offset += text_dim
        features[idx, offset + NODE_TYPE_INDEX[node_type]] = 1.0
        offset += len(NODE_TYPES)
        features[idx, offset] = coord[0] / max(1.0, config.width - 1)
        features[idx, offset + 1] = coord[1] / max(1.0, config.height - 1)
        offset += 2
        features[idx, offset : offset + len(ATTRIBUTE_NAMES)] = torch.tensor(attrs, dtype=torch.float32)

    def edge_attr(edge_type: str, src_coord: Tuple[int, int], dst_coord: Tuple[int, int]) -> torch.Tensor:
        feat = torch.zeros((edge_feature_dim(),), dtype=torch.float32)
        feat[EDGE_TYPE_INDEX[edge_type]] = 1.0
        feat[len(EDGE_TYPES)] = (dst_coord[0] - src_coord[0]) / max(1.0, config.width - 1)
        feat[len(EDGE_TYPES) + 1] = (dst_coord[1] - src_coord[1]) / max(1.0, config.height - 1)
        return feat

    def add_edge(src: int, dst: int, edge_type: str, src_coord: Tuple[int, int], dst_coord: Tuple[int, int]) -> None:
        edges.append([src, dst])
        edge_features.append(edge_attr(edge_type, src_coord, dst_coord))

    set_node(
        restaurant_idx,
        text="restaurant",
        node_type="restaurant",
        coord=(config.width // 2, config.height // 2),
        attrs=[0.0] * len(ATTRIBUTE_NAMES),
    )

    room_coords = {
        "kitchen": (3, config.height // 2),
        "serving_room": (12, config.height // 2),
    }
    for room in ROOMS:
        idx = room_index[room]
        set_node(
            idx,
            text=room.replace("_", " "),
            node_type="room",
            coord=room_coords[room],
            attrs=[0.0] * len(ATTRIBUTE_NAMES),
        )
        add_edge(restaurant_idx, idx, "contains_room", room_coords[room], room_coords[room])
        add_edge(idx, restaurant_idx, "room_in_restaurant", room_coords[room], room_coords[room])

    robot_room = config.location_room(state.robot_location)
    set_node(
        robot_idx,
        text="service robot",
        node_type="robot",
        coord=config.container(state.robot_location).coord,
        attrs=[0.0] * len(ATTRIBUTE_NAMES),
    )
    add_edge(room_index[robot_room], robot_idx, "room_has_robot", room_coords[robot_room], config.container(state.robot_location).coord)
    add_edge(robot_idx, room_index[robot_room], "robot_in_room", config.container(state.robot_location).coord, room_coords[robot_room])

    for name in container_names:
        spec = config.container(name)
        idx = container_index[name]
        attrs = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if spec.wash_source else 0.0,
            1.0 if spec.water_source else 0.0,
            1.0 if spec.coffee_source else 0.0,
            1.0 if spec.fruit_source else 0.0,
        ]
        set_node(
            idx,
            text=f"{spec.category} {name.replace('_', ' ')}",
            node_type="container",
            coord=spec.coord,
            attrs=attrs,
        )
        add_edge(room_index[spec.room], idx, "room_contains_container", room_coords[spec.room], spec.coord)
        add_edge(idx, room_index[spec.room], "container_in_room", spec.coord, room_coords[spec.room])

    for name in object_names:
        obj = state.objects[name]
        idx = object_index[name]
        coord = config.container(obj.location).coord if obj.location in config.containers else config.container(state.robot_location).coord
        attrs = [
            1.0 if obj.dirty else 0.0,
            1.0 if obj.contents == "empty" else 0.0,
            1.0 if obj.contents == "water" else 0.0,
            1.0 if obj.contents == "coffee" else 0.0,
            1.0 if obj.contents == "fruit" else 0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        set_node(
            idx,
            text=f"{obj.category} {name.replace('_', ' ')}",
            node_type="object",
            coord=coord,
            attrs=attrs,
        )
        if state.holding == name:
            add_edge(robot_idx, idx, "robot_holding_object", config.container(state.robot_location).coord, coord)
            add_edge(idx, robot_idx, "object_held_by_robot", coord, config.container(state.robot_location).coord)
            add_edge(room_index[robot_room], idx, "room_contains_object", room_coords[robot_room], coord)
            add_edge(idx, room_index[robot_room], "object_in_room", coord, room_coords[robot_room])
        else:
            room = config.location_room(obj.location)
            add_edge(room_index[room], idx, "room_contains_object", room_coords[room], coord)
            add_edge(idx, room_index[room], "object_in_room", coord, room_coords[room])
            add_edge(container_index[obj.location], idx, "container_supports_object", config.container(obj.location).coord, coord)
            add_edge(idx, container_index[obj.location], "object_on_container", coord, config.container(obj.location).coord)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr_t = torch.stack(edge_features, dim=0) if edge_features else torch.zeros((0, edge_feature_dim()), dtype=torch.float32)
    return EncodedGraph(node_features=features, edge_index=edge_index, edge_attr=edge_attr_t)


def collate_graphs(examples: Sequence[GraphRegressionExample]) -> Dict[str, torch.Tensor]:
    node_features: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    batch: List[torch.Tensor] = []
    targets: List[float] = []
    offset = 0
    for graph_id, example in enumerate(examples):
        graph = example.graph
        num_nodes = int(graph.node_features.shape[0])
        node_features.append(graph.node_features)
        edge_indices.append(graph.edge_index + offset)
        edge_attrs.append(graph.edge_attr)
        batch.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        targets.append(float(example.target))
        offset += num_nodes
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch, dim=0),
        "targets": torch.tensor(targets, dtype=torch.float32),
    }


class AnticipatoryRestaurantGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(node_features)
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = self.activation(x)
        pooled = torch.cat(
            [
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)


def select_device(raw: str = "auto") -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: AnticipatoryRestaurantGNN,
    *,
    input_dim: int,
    edge_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    text_encoder_mode: str,
    text_model_name: str,
    text_embedding_dim: int,
    metrics: Dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "heads": heads,
        "dropout": dropout,
        "text_encoder_mode": text_encoder_mode,
        "text_model_name": text_model_name,
        "text_embedding_dim": text_embedding_dim,
        "metrics": metrics or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> AnticipatoryRestaurantGNN:
    payload = torch.load(path, map_location=map_location)
    model = AnticipatoryRestaurantGNN(
        input_dim=int(payload["input_dim"]),
        edge_dim=int(payload["edge_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        heads=int(payload.get("heads", 4)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
