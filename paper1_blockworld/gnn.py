from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from .world import WorldConfig, WorldState, block_color, region_color


NODE_TYPES = ("room", "location", "object")
COLORS = (
    "none",
    "white",
    "red",
    "blue",
    "green",
    "cyan",
    "pink",
    "orange",
    "brown",
)
TYPE_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}
COLOR_INDEX = {name: idx for idx, name in enumerate(COLORS)}


@dataclass
class EncodedGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor


@dataclass
class GraphRegressionExample:
    graph: EncodedGraph
    target: float


def graph_feature_dim() -> int:
    return len(NODE_TYPES) + len(COLORS) + 2


def encode_state_as_graph(
    state: WorldState,
    config: WorldConfig,
) -> EncodedGraph:
    region_names = list(config.all_regions)
    block_names = list(config.all_blocks)
    node_count = 1 + len(region_names) + len(block_names)
    features = torch.zeros((node_count, graph_feature_dim()), dtype=torch.float32)
    edges: List[List[int]] = []

    env_idx = 0
    region_offset = 1
    block_offset = 1 + len(region_names)

    def add_one_hot(row: int, offset: int, size: int, index: int) -> None:
        features[row, offset + index] = 1.0

    def set_coord(row: int, coord: tuple[int, int] | None) -> None:
        if coord is None:
            return
        x, y = coord
        features[row, len(NODE_TYPES) + len(COLORS)] = x / max(1.0, config.width - 1)
        features[row, len(NODE_TYPES) + len(COLORS) + 1] = y / max(1.0, config.height - 1)

    def add_edge(src: int, dst: int) -> None:
        edges.append([src, dst])

    add_one_hot(env_idx, 0, len(NODE_TYPES), TYPE_INDEX["room"])
    add_one_hot(env_idx, len(NODE_TYPES), len(COLORS), COLOR_INDEX["none"])
    set_coord(env_idx, (config.width // 2, config.height // 2))

    region_index_by_name: Dict[str, int] = {}
    for idx, region in enumerate(region_names):
        node_idx = region_offset + idx
        region_index_by_name[region] = node_idx
        add_one_hot(node_idx, 0, len(NODE_TYPES), TYPE_INDEX["location"])
        add_one_hot(
            node_idx,
            len(NODE_TYPES),
            len(COLORS),
            COLOR_INDEX[region_color(region)],
        )
        coord = config.region_coords[region]
        set_coord(node_idx, coord)
        add_edge(env_idx, node_idx)
        add_edge(node_idx, env_idx)

    for idx, block in enumerate(block_names):
        node_idx = block_offset + idx
        add_one_hot(node_idx, 0, len(NODE_TYPES), TYPE_INDEX["object"])
        add_one_hot(
            node_idx,
            len(NODE_TYPES),
            len(COLORS),
            COLOR_INDEX[block_color(block, config)],
        )
        coord = state.placements.get(block)
        set_coord(node_idx, coord)
        placement = state.placements.get(block)
        if placement is None:
            continue
        region = config.region_for_coord(placement)
        if region is None:
            continue
        region_idx = region_index_by_name[region]
        add_edge(region_idx, node_idx)
        add_edge(node_idx, region_idx)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return EncodedGraph(node_features=features, edge_index=edge_index)


def collate_graphs(
    examples: Sequence[GraphRegressionExample],
) -> Dict[str, torch.Tensor]:
    node_features = []
    edge_indices = []
    batch = []
    targets = []
    offset = 0
    for graph_id, example in enumerate(examples):
        graph = example.graph
        num_nodes = graph.node_features.shape[0]
        node_features.append(graph.node_features)
        edge_indices.append(graph.edge_index + offset)
        batch.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        targets.append(example.target)
        offset += num_nodes
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "batch": torch.cat(batch, dim=0),
        "targets": torch.tensor(targets, dtype=torch.float32),
    }


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv = SAGEConv(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class AnticipatoryGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: List[GraphSAGELayer] = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(GraphSAGELayer(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = node_features
        for layer in self.layers:
            x = self.activation(layer(x, edge_index))
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        pooled = torch.zeros(
            (num_graphs, x.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )
        pooled.index_add_(0, batch, x)
        counts = torch.zeros((num_graphs, 1), dtype=x.dtype, device=x.device)
        counts.index_add_(
            0,
            batch,
            torch.ones((batch.shape[0], 1), dtype=x.dtype, device=x.device),
        )
        pooled = pooled / counts.clamp(min=1.0)
        return self.output(pooled).squeeze(-1)


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
    model: AnticipatoryGNN,
    *,
    hidden_dim: int,
    num_layers: int,
    metrics: Dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": graph_feature_dim(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "metrics": metrics or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> AnticipatoryGNN:
    payload = torch.load(path, map_location=map_location)
    model = AnticipatoryGNN(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
