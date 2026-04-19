"""Train APCostEstimator (TransformerConv-style message passing) for restaurant."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class GraphExample:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor


class TransformerConvStyleLayer(nn.Module):
    """Lightweight attention message passing layer (TransformerConv-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        q = self.q(x[dst])
        k = self.k(x[src])
        v = self.v(x[src])
        logits = (q * k).sum(dim=-1, keepdim=True) / np.sqrt(x.shape[-1])
        alpha = torch.sigmoid(logits)
        msg = alpha * v
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        out = self.o(out)
        return self.ln(x + out)


class APCostEstimator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(), nn.BatchNorm1d(hidden_dim))
        self.layers = nn.ModuleList([TransformerConvStyleLayer(hidden_dim) for _ in range(layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        pooled = torch.cat([h.mean(dim=0), h.sum(dim=0)], dim=0)
        return self.out(pooled).squeeze(-1)


def _load_dataset(path: Path) -> Dict[str, List[GraphExample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    out: Dict[str, List[GraphExample]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row["split"]
        ex = GraphExample(
            x=torch.tensor(row["node_features"], dtype=torch.float32),
            edge_index=torch.tensor(row["edge_index"], dtype=torch.int64),
            y=torch.tensor(float(row["target_ap_cost"]), dtype=torch.float32),
        )
        out.setdefault(split, []).append(ex)
    return out


def _eval(model: APCostEstimator, data: List[GraphExample], device: torch.device) -> float:
    if not data:
        return float("nan")
    model.eval()
    errs: List[float] = []
    with torch.no_grad():
        for ex in data:
            pred = model(ex.x.to(device), ex.edge_index.to(device))
            mae = torch.abs(pred - ex.y.to(device))
            errs.append(float(mae.item()))
    return float(np.mean(errs))


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _load_dataset(args.dataset_path)
    if not dataset["train"]:
        raise RuntimeError("Empty train split in dataset.")
    in_dim = int(dataset["train"][0].x.shape[-1])
    model = APCostEstimator(in_dim=in_dim, hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = optim.Adagrad(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.step_gamma)
    loss_fn = nn.L1Loss()

    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = dataset["train"][:]
        rng.shuffle(train_rows)
        losses: List[float] = []
        for ex in train_rows:
            pred = model(ex.x.to(device), ex.edge_index.to(device))
            loss = loss_fn(pred, ex.y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        scheduler.step()
        train_mae = float(np.mean(losses)) if losses else float("nan")
        val_mae = _eval(model, dataset.get("val", []), device)
        test_mae = _eval(model, dataset.get("test", []), device)
        row = {"epoch": float(epoch), "train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae}
        history.append(row)
        print(f"[epoch {epoch:03d}] train_mae={train_mae:.3f} val_mae={val_mae:.3f} test_mae={test_mae:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "apcost_estimator.pt"
    torch.save(model.state_dict(), ckpt)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"history": history}, fh, indent=2)
    with (args.output_dir / "train_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    print(f"Saved estimator -> {ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train restaurant APCostEstimator from planner-labeled dataset.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/restaurant_planner_dataset/paper2_planner_labels.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/restaurant_apcost_gnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--step-gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
