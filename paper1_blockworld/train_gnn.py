from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .estimator import OracleFutureCostEstimator
from .gnn import (
    AnticipatoryGNN,
    GraphRegressionExample,
    collate_graphs,
    encode_state_as_graph,
    graph_feature_dim,
    save_checkpoint,
    select_device,
)
from .planner import FastDownwardBlockworldPlanner
from .world import Task, WorldConfig, WorldGenerator, WorldState


class GraphExampleDataset(Dataset):
    def __init__(self, examples: Sequence[GraphRegressionExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GraphRegressionExample:
        return self.examples[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GraphSAGE-style anticipatory-cost estimator for paper1 Blockworld."
    )
    parser.add_argument("--num-train-envs", type=int, default=250)
    parser.add_argument("--num-val-envs", type=int, default=0)
    parser.add_argument("--num-test-envs", type=int, default=150)
    parser.add_argument("--states-per-env", type=int, default=200)
    parser.add_argument("--tasks-per-environment", type=int, default=24)
    parser.add_argument(
        "--future-task-sample",
        default="all",
        help="Use 'all' for exact targets, otherwise sample this many tasks per label.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper1_blockworld/checkpoints"),
    )
    parser.add_argument(
        "--save-dataset",
        type=Path,
        default=None,
        help="Optional path to save the generated regression dataset metadata.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny train/val/test pass to validate the pipeline.",
    )
    return parser


def parse_future_task_sample(raw: str) -> int | None:
    if raw.strip().lower() == "all":
        return None
    return max(1, int(raw))


def collect_states_for_environment(
    generator: WorldGenerator,
    rng: random.Random,
    *,
    states_per_env: int,
    tasks_per_environment: int,
) -> Tuple[List[WorldState], List[Task]]:
    task_library = generator.sample_task_library(rng, count=tasks_per_environment)
    states = [generator.sample_initial_state(rng) for _ in range(states_per_env)]
    return states, task_library


def build_examples(
    *,
    num_envs: int,
    states_per_env: int,
    tasks_per_environment: int,
    future_task_sample: int | None,
    seed: int,
) -> Tuple[List[GraphRegressionExample], Dict[str, float]]:
    master_rng = random.Random(seed)
    examples: List[GraphRegressionExample] = []
    targets: List[float] = []

    for env_idx in range(num_envs):
        env_seed = master_rng.randint(0, 10**9)
        env_rng = random.Random(env_seed)
        config = WorldConfig.sample(env_rng)
        generator = WorldGenerator(config)
        planner = FastDownwardBlockworldPlanner(config)
        states, task_library = collect_states_for_environment(
            generator,
            env_rng,
            states_per_env=states_per_env,
            tasks_per_environment=tasks_per_environment,
        )
        estimator = OracleFutureCostEstimator(
            planner,
            future_task_sample=future_task_sample,
            estimator_seed=env_seed,
        )
        for state in states:
            target = estimator.estimate(state, task_library)
            graph = encode_state_as_graph(state, config)
            examples.append(GraphRegressionExample(graph=graph, target=target))
            targets.append(target)
        print(
            f"[dataset env {env_idx + 1}/{num_envs}] "
            f"states={len(states)} mean_target={mean(targets[-len(states):]):.1f}"
        )

    stats = {
        "num_examples": float(len(examples)),
        "target_mean": float(mean(targets)) if targets else 0.0,
        "target_min": float(min(targets)) if targets else 0.0,
        "target_max": float(max(targets)) if targets else 0.0,
    }
    return examples, stats


def evaluate_model(
    model: AnticipatoryGNN,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    abs_errors: List[float] = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(
                batch["node_features"].to(device),
                batch["edge_index"].to(device),
                batch["batch"].to(device),
            )
            targets = batch["targets"].to(device)
            abs_errors.extend(torch.abs(predictions - targets).cpu().tolist())
    return float(mean(abs_errors)) if abs_errors else 0.0


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_train_envs = 2
        args.num_val_envs = 1
        args.num_test_envs = 1
        args.states_per_env = 6
        args.tasks_per_environment = 8
        args.future_task_sample = "2"
        args.epochs = 1
        args.batch_size = 4
        args.hidden_dim = 64

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    future_task_sample = parse_future_task_sample(args.future_task_sample)

    train_examples, train_stats = build_examples(
        num_envs=args.num_train_envs,
        states_per_env=args.states_per_env,
        tasks_per_environment=args.tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=args.seed,
    )
    val_examples, _ = build_examples(
        num_envs=args.num_val_envs,
        states_per_env=args.states_per_env,
        tasks_per_environment=args.tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=args.seed + 1_000,
    )
    test_examples, _ = build_examples(
        num_envs=args.num_test_envs,
        states_per_env=args.states_per_env,
        tasks_per_environment=args.tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=args.seed + 2_000,
    )

    train_loader = DataLoader(
        GraphExampleDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        GraphExampleDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    test_loader = DataLoader(
        GraphExampleDataset(test_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    model = AnticipatoryGNN(
        input_dim=graph_feature_dim(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            predictions = model(
                batch["node_features"].to(device),
                batch["edge_index"].to(device),
                batch["batch"].to(device),
            )
            targets = batch["targets"].to(device)
            loss = torch.nn.functional.l1_loss(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_mae = float(mean(losses)) if losses else 0.0
        val_mae = evaluate_model(model, val_loader, device=device) if val_examples else train_mae
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_mae={train_mae:.3f} val_mae={val_mae:.3f}"
        )
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate_model(model, test_loader, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "paper1_anticipatory_gnn.pt"
    metrics = {
        "best_val_mae": best_val,
        "test_mae": test_mae,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
    }
    save_checkpoint(
        checkpoint_path,
        model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        metrics=metrics,
    )

    summary = {
        "settings": {
            "num_train_envs": args.num_train_envs,
            "num_val_envs": args.num_val_envs,
            "num_test_envs": args.num_test_envs,
            "states_per_env": args.states_per_env,
            "tasks_per_environment": args.tasks_per_environment,
            "future_task_sample": args.future_task_sample,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seed": args.seed,
        },
        "dataset": train_stats,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = args.output_dir / "paper1_anticipatory_gnn_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_dataset is not None:
        args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
        args.save_dataset.write_text(json.dumps(summary["dataset"], indent=2), encoding="utf-8")

    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Validation MAE: {best_val:.3f}")
    print(f"Test MAE: {test_mae:.3f}")


if __name__ == "__main__":
    main()
