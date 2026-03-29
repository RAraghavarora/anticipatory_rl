from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from accelerate import Accelerator
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=0,
        help="CPU worker processes for planner-labeled dataset generation. "
        "0 uses SLURM_CPUS_PER_TASK when available, otherwise 1.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="no",
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper1_blockworld/checkpoints"),
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=None,
        help="Optional path to cache the planner-labeled dataset for distributed runs.",
    )
    parser.add_argument(
        "--rebuild-dataset-cache",
        action="store_true",
        help="Regenerate the planner-labeled dataset even if the cache already exists.",
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


def _build_environment_examples(
    env_idx: int,
    env_seed: int,
    *,
    num_envs: int,
    states_per_env: int,
    tasks_per_environment: int,
    future_task_sample: int | None,
) -> Tuple[int, WorldConfig, List[WorldState], List[float], float]:
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
    env_targets: List[float] = []
    for state in states:
        target = estimator.estimate(state, task_library)
        env_targets.append(target)
    mean_target = float(mean(env_targets)) if env_targets else 0.0
    return env_idx, config, states, env_targets, mean_target


def _build_environment_examples_from_args(
    args: Tuple[int, int, int, int, int, int | None]
) -> Tuple[int, WorldConfig, List[WorldState], List[float], float]:
    env_idx, env_seed, num_envs, states_per_env, tasks_per_environment, future_task_sample = args
    return _build_environment_examples(
        env_idx,
        env_seed,
        num_envs=num_envs,
        states_per_env=states_per_env,
        tasks_per_environment=tasks_per_environment,
        future_task_sample=future_task_sample,
    )


def infer_dataset_workers(requested_workers: int, *, num_envs: int) -> int:
    if num_envs <= 1:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, num_envs))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, min(int(slurm_cpus), num_envs))
    return 1


def make_graph_examples(
    config: WorldConfig,
    states: Sequence[WorldState],
    targets: Sequence[float],
) -> List[GraphRegressionExample]:
    return [
        GraphRegressionExample(
            graph=encode_state_as_graph(state, config),
            target=target,
        )
        for state, target in zip(states, targets)
    ]


def build_examples(
    *,
    num_envs: int,
    states_per_env: int,
    tasks_per_environment: int,
    future_task_sample: int | None,
    seed: int,
    dataset_workers: int,
) -> Tuple[List[GraphRegressionExample], Dict[str, float]]:
    '''
    Build a dataset of (s_t, V_AP(s_t))
    '''
    
    if num_envs <= 0:
        return [], {
            "num_examples": 0.0,
            "target_mean": 0.0,
            "target_min": 0.0,
            "target_max": 0.0,
        }

    master_rng = random.Random(seed)
    env_seeds = [master_rng.randint(0, 10**9) for _ in range(num_envs)]
    worker_count = infer_dataset_workers(dataset_workers, num_envs=num_envs)
    print(
        f"Generating planner-labeled data for {num_envs} environments "
        f"with dataset_workers={worker_count}"
    )

    env_examples_by_idx: List[List[GraphRegressionExample] | None] = [None] * num_envs
    if worker_count == 1:
        for env_idx, env_seed in enumerate(env_seeds):
            _, config, states, env_targets, env_mean_target = _build_environment_examples(
                env_idx,
                env_seed,
                num_envs=num_envs,
                states_per_env=states_per_env,
                tasks_per_environment=tasks_per_environment,
                future_task_sample=future_task_sample,
            )
            env_examples = make_graph_examples(config, states, env_targets)
            env_examples_by_idx[env_idx] = env_examples
            print(
                f"[dataset env {env_idx + 1}/{num_envs}] "
                f"states={len(env_examples)} mean_target={env_mean_target:.1f}"
            )
    else:
        ctx = mp.get_context("spawn")
        worker_args = [
            (
                env_idx,
                env_seed,
                num_envs,
                states_per_env,
                tasks_per_environment,
                future_task_sample,
            )
            for env_idx, env_seed in enumerate(env_seeds)
        ]
        with ctx.Pool(processes=worker_count) as pool:
            for env_idx, config, states, env_targets, env_mean_target in pool.imap_unordered(
                _build_environment_examples_from_args,
                worker_args,
            ):
                env_examples = make_graph_examples(config, states, env_targets)
                env_examples_by_idx[env_idx] = env_examples
                print(
                    f"[dataset env {env_idx + 1}/{num_envs}] "
                    f"states={len(env_examples)} mean_target={env_mean_target:.1f}"
                )

    examples = [
        example
        for env_examples in env_examples_by_idx
        if env_examples is not None
        for example in env_examples
    ]
    targets = [example.target for example in examples]

    stats = {
        "num_examples": float(len(examples)),
        "target_mean": float(mean(targets)) if targets else 0.0,
        "target_min": float(min(targets)) if targets else 0.0,
        "target_max": float(max(targets)) if targets else 0.0,
    }
    return examples, stats


def build_dataset_bundle(
    *,
    num_train_envs: int,
    num_val_envs: int,
    num_test_envs: int,
    states_per_env: int,
    tasks_per_environment: int,
    future_task_sample: int | None,
    seed: int,
    dataset_workers: int,
) -> Dict[str, Any]:
    train_examples, train_stats = build_examples(
        num_envs=num_train_envs,
        states_per_env=states_per_env,
        tasks_per_environment=tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=seed,
        dataset_workers=dataset_workers,
    )
    val_examples, val_stats = build_examples(
        num_envs=num_val_envs,
        states_per_env=states_per_env,
        tasks_per_environment=tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=seed + 1_000,
        dataset_workers=dataset_workers,
    )
    test_examples, test_stats = build_examples(
        num_envs=num_test_envs,
        states_per_env=states_per_env,
        tasks_per_environment=tasks_per_environment,
        future_task_sample=future_task_sample,
        seed=seed + 2_000,
        dataset_workers=dataset_workers,
    )
    return {
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_examples": test_examples,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
    }


def default_dataset_cache_path(args: argparse.Namespace) -> Path:
    future_task_token = str(args.future_task_sample).strip().lower()
    filename = (
        "dataset_cache"
        f"_seed{args.seed}"
        f"_train{args.num_train_envs}"
        f"_val{args.num_val_envs}"
        f"_test{args.num_test_envs}"
        f"_states{args.states_per_env}"
        f"_tasks{args.tasks_per_environment}"
        f"_future{future_task_token}.pt"
    )
    return args.output_dir / filename


def load_torch_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_build_dataset_bundle(
    args: argparse.Namespace,
    accelerator: Accelerator,
    *,
    future_task_sample: int | None,
) -> Tuple[Dict[str, Any], Path]:
    cache_path = args.dataset_cache or default_dataset_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        if cache_path.exists() and not args.rebuild_dataset_cache:
            accelerator.print(f"Loading planner-labeled dataset cache from {cache_path}")
        else:
            accelerator.print(f"Building planner-labeled dataset cache at {cache_path}")
            dataset_bundle = build_dataset_bundle(
                num_train_envs=args.num_train_envs,
                num_val_envs=args.num_val_envs,
                num_test_envs=args.num_test_envs,
                states_per_env=args.states_per_env,
                tasks_per_environment=args.tasks_per_environment,
                future_task_sample=future_task_sample,
                seed=args.seed,
                dataset_workers=args.dataset_workers,
            )
            torch.save(dataset_bundle, cache_path)

    accelerator.wait_for_everyone()
    return load_torch_payload(cache_path), cache_path


def per_device_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size <= 1:
        return global_batch_size
    if global_batch_size < world_size or global_batch_size % world_size != 0:
        raise ValueError(
            "--batch-size is treated as the global batch size. "
            f"Received batch_size={global_batch_size} for world_size={world_size}; "
            "choose a batch size divisible by the number of processes."
        )
    return global_batch_size // world_size


def evaluate_model(
    model: AnticipatoryGNN,
    loader: DataLoader,
    *,
    accelerator: Accelerator,
) -> float:
    model.eval()
    abs_errors: List[float] = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(
                batch["node_features"],
                batch["edge_index"],
                batch["batch"],
            )
            targets = batch["targets"]
            gathered_abs_errors = accelerator.gather_for_metrics(
                torch.abs(predictions - targets)
            )
            abs_errors.extend(gathered_abs_errors.cpu().tolist())
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
        args.dataset_workers = 2
        args.rebuild_dataset_cache = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    force_cpu = (
        args.device == "cpu"
        and os.environ.get("LOCAL_RANK") is None
        and int(os.environ.get("WORLD_SIZE", "1")) == 1
    )
    accelerator = Accelerator(
        cpu=force_cpu,
        mixed_precision=args.mixed_precision,
    )
    future_task_sample = parse_future_task_sample(args.future_task_sample)
    dataset_bundle, dataset_cache_path = load_or_build_dataset_bundle(
        args,
        accelerator,
        future_task_sample=future_task_sample,
    )

    train_examples = dataset_bundle["train_examples"]
    val_examples = dataset_bundle["val_examples"]
    test_examples = dataset_bundle["test_examples"]
    train_stats = dataset_bundle["train_stats"]
    val_stats = dataset_bundle["val_stats"]
    test_stats = dataset_bundle["test_stats"]
    local_batch_size = per_device_batch_size(args.batch_size, accelerator.num_processes)
    pin_memory = accelerator.device.type == "cuda"

    train_loader = DataLoader(
        GraphExampleDataset(train_examples),
        batch_size=local_batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        GraphExampleDataset(val_examples),
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        GraphExampleDataset(test_examples),
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = AnticipatoryGNN(
        input_dim=graph_feature_dim(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
    )

    accelerator.print(
        "Accelerate setup: "
        f"world_size={accelerator.num_processes} "
        f"device={accelerator.device} "
        f"global_batch_size={args.batch_size} "
        f"per_device_batch_size={local_batch_size}"
    )
    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        abs_errors: List[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(
                batch["node_features"],
                batch["edge_index"],
                batch["batch"],
            )
            targets = batch["targets"]
            loss = torch.nn.functional.l1_loss(predictions, targets)
            accelerator.backward(loss)
            optimizer.step()
            gathered_abs_errors = accelerator.gather_for_metrics(
                torch.abs(predictions.detach() - targets.detach())
            )
            abs_errors.extend(gathered_abs_errors.cpu().tolist())

        train_mae = float(mean(abs_errors)) if abs_errors else 0.0
        val_mae = (
            evaluate_model(model, val_loader, accelerator=accelerator)
            if val_examples
            else train_mae
        )
        accelerator.print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_mae={train_mae:.3f} val_mae={val_mae:.3f}"
        )
        if val_mae < best_val:
            best_val = val_mae
            best_state = {
                key: value.detach().cpu()
                for key, value in accelerator.unwrap_model(model).state_dict().items()
            }

    if best_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_state)
    accelerator.wait_for_everyone()
    test_mae = evaluate_model(model, test_loader, accelerator=accelerator)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / "paper1_anticipatory_gnn.pt"
        metrics = {
            "best_val_mae": best_val,
            "test_mae": test_mae,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "world_size": accelerator.num_processes,
            "global_batch_size": args.batch_size,
            "per_device_batch_size": local_batch_size,
        }
        save_checkpoint(
            checkpoint_path,
            accelerator.unwrap_model(model),
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
                "dataset_workers": args.dataset_workers,
                "num_workers": args.num_workers,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "seed": args.seed,
                "mixed_precision": args.mixed_precision,
            },
            "dataset": {
                "cache_path": str(dataset_cache_path),
                "train": train_stats,
                "val": val_stats,
                "test": test_stats,
            },
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
