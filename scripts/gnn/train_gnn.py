#!/usr/bin/env python3
"""Train APCostEstimator GNN on (graph, V_A.P.) pairs (Talukder et al. Sec. V-A).

Usage:
    python scripts/gnn/train_gnn.py \
        --data-path runs/toy3_2k.pt \
        --output-dir runs/gnn_train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from gnn.graph_encoder import NODE_TYPES, SBERT_DIM, BINARY_ATTRS
from gnn.model import APCostEstimator


def split_indices(n: int, train_frac: float = 0.8, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    split = int(n * train_frac)
    return perm[:split], perm[split:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train APCostEstimator GNN")
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--scheduler-step", type=int, default=1000)
    ap.add_argument("--scheduler-gamma", type=float, default=0.5)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    dataset = torch.load(args.data_path, weights_only=False)
    graphs = [d["graph"] for d in dataset]
    targets = torch.tensor([d["v_ap"] for d in dataset], dtype=torch.float32)
    print(f"Loaded {len(graphs)} samples from {args.data_path}")

    mean = targets.mean().item()
    std = targets.std().item()
    print(f"V_A.P.  mean={mean:.2f}  std={std:.2f}")

    train_idx, val_idx = split_indices(len(graphs), args.train_frac, args.seed)
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=args.batch_size)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    in_dim = SBERT_DIM + len(NODE_TYPES) + 2 + len(BINARY_ATTRS)
    model = APCostEstimator(in_dim, hidden_dim=args.hidden_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (in_dim={in_dim}, hidden={args.hidden_dim})")

    loss_fn = nn.L1Loss()
    optimizer = Adagrad(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                val_loss += loss_fn(out, batch.y).item() * batch.num_graphs
        val_loss /= len(val_idx)

        lr = scheduler.get_last_lr()[0]
        mark = "*" if val_loss < best_val_loss else " "
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

        print(f"Epoch {epoch:2d}{mark}  train={train_loss:.2f}  val={val_loss:.2f}  lr={lr:.6f}")

    print(f"\nBest val loss: {best_val_loss:.2f}")
    torch.save(model.state_dict(), args.output_dir / "last_model.pt")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
