from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, BatchNorm
from torch_geometric.nn.pool import global_mean_pool, global_add_pool


class APCostEstimator(nn.Module):
    """GNN for anticipatory cost estimation (Talukder et al. 2024, Sec. V-A).

    4 TransformerConv layers, BatchNorm, LeakyReLU, mean + add global
    pooling, single linear head → scalar V_A.P.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = TransformerConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.conv4 = TransformerConv(hidden_dim, hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x, edge_index)))
        x = self.act(self.bn2(self.conv2(x, edge_index)))
        x = self.act(self.bn3(self.conv3(x, edge_index)))
        x = self.act(self.bn4(self.conv4(x, edge_index)))
        x = global_mean_pool(x, batch) + global_add_pool(x, batch)
        return self.head(x).squeeze(-1)


def _test():
    from torch_geometric.data import Data
    from gnn.graph_encoder import NODE_TYPES, SBERT_DIM, BINARY_ATTRS

    in_dim = SBERT_DIM + len(NODE_TYPES) + 2 + len(BINARY_ATTRS)
    model = APCostEstimator(in_dim, hidden_dim=64)

    x = torch.randn(18, in_dim)
    edge_index = torch.randint(0, 18, (2, 32))
    batch = torch.zeros(18, dtype=torch.long)
    out = model(x, edge_index, batch)
    assert out.shape == (1,), f"expected (1,) got {out.shape}"

    data = Data(x=x, edge_index=edge_index)
    out_batch = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
    assert out_batch.shape == (1,), f"expected (1,) got {out_batch.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"APCostEstimator(in_dim={in_dim}, hidden=64): {n_params:,} params, output shape OK")


if __name__ == "__main__":
    _test()
