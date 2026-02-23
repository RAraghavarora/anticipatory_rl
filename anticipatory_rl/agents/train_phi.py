"""Train the tiny anticipatory potential network φ_θ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PhiModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the φ_θ anticipatory value model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("runs") / "phi_labels.npz",
        help="NPZ file containing 'states' and 'labels'.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden layer width for the MLP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "phi_model.pt",
        help="Path to save the trained model state_dict.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("runs") / "phi_training_metrics.csv",
        help="CSV file to store (step, loss). Set to '' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    data = np.load(args.dataset)
    states = torch.tensor(data["states"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.float32)
    input_dim = states.shape[1]

    torch.manual_seed(args.seed)
    model = PhiModel(input_dim, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    num_samples = states.shape[0]
    logs = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, num_samples, (args.batch_size,))
        batch_states = states[idx]
        batch_labels = labels[idx]

        preds = model(batch_states)
        loss = criterion(preds, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        logs.append((step, loss_value))
        if step % 10 == 0 or step == args.steps:
            print(f"Step {step:04d}: loss={loss_value:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dataset": str(args.dataset),
        },
        args.output,
    )
    print(f"Saved φ_θ weights to {args.output}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as fh:
            fh.write("step,loss\n")
            for step, loss_value in logs:
                fh.write(f"{step},{loss_value}\n")
        print(f"Wrote training metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
