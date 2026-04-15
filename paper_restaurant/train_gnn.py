from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from accelerate import Accelerator  # type: ignore
except ImportError:  # pragma: no cover - exercised in local smoke envs without accelerate
    class Accelerator:  # type: ignore[override]
        def __init__(self, mixed_precision: str = "no") -> None:
            del mixed_precision
            self.is_main_process = True

        def wait_for_everyone(self) -> None:
            return None

        def prepare(self, *items):
            if len(items) == 1:
                return items[0]
            return items

        def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        def unwrap_model(self, model):
            return model

        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

from .estimator import OracleFutureCostEstimator
from .gnn import (
    AnticipatoryRestaurantGNN,
    GraphRegressionExample,
    TextEmbeddingProvider,
    collate_graphs,
    encode_state_as_graph,
    graph_feature_dim,
    edge_feature_dim,
    save_checkpoint,
)
from .planner import FastDownwardRestaurantPlanner
from .world import RestaurantTaskLibrary, RestaurantWorldConfig, RestaurantWorldGenerator, RestaurantWorldState


class GraphExampleDataset(Dataset):
    def __init__(self, examples: Sequence[GraphRegressionExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GraphRegressionExample:
        return self.examples[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TransformerConv anticipatory-cost estimator for paper-style restaurant worlds."
    )
    parser.add_argument("--num-train-envs", type=int, default=96)
    parser.add_argument("--num-val-envs", type=int, default=16)
    parser.add_argument("--states-per-env", type=int, default=64)
    parser.add_argument("--tasks-per-environment", type=int, default=72)
    parser.add_argument("--future-task-sample", default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_restaurant/checkpoints"))
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument("--dataset-workers", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--text-encoder", choices=("sbert", "hash"), default="sbert")
    parser.add_argument(
        "--text-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text-embedding-dim", type=int, default=384)
    parser.add_argument("--embedding-cache", type=Path, default=Path("paper_restaurant/cache/text_embeddings.pt"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def parse_future_task_sample(raw: str) -> int | None:
    if raw.strip().lower() == "all":
        return None
    return max(1, int(raw))


def infer_dataset_workers(requested_workers: int, *, num_envs: int) -> int:
    if num_envs <= 1:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, num_envs))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, min(int(slurm_cpus), num_envs))
    return 1


def collect_rollout_states(
    generator: RestaurantWorldGenerator,
    planner: FastDownwardRestaurantPlanner,
    rng: random.Random,
    task_library: RestaurantTaskLibrary,
    *,
    states_per_env: int,
) -> List[RestaurantWorldState]:
    state = generator.sample_initial_state(rng)
    states: List[RestaurantWorldState] = []
    for _ in range(states_per_env):
        states.append(state.clone())
        task = task_library.sample_task(rng)
        result = planner.plan_for_task(state, task)
        state = result.final_state.clone()
    return states


def _build_environment_payload(
    args: Tuple[int, int, int, int, int | None],
) -> Tuple[RestaurantWorldConfig, List[RestaurantWorldState], RestaurantTaskLibrary, List[float]]:
    env_idx, env_seed, states_per_env, tasks_per_environment, future_task_sample = args
    del env_idx
    env_rng = random.Random(env_seed)
    config = RestaurantWorldConfig.sample(env_rng)
    generator = RestaurantWorldGenerator(config)
    planner = FastDownwardRestaurantPlanner(config)
    task_library = generator.sample_task_library(env_rng, count=tasks_per_environment)
    states = collect_rollout_states(
        generator,
        planner,
        env_rng,
        task_library,
        states_per_env=states_per_env,
    )
    estimator = OracleFutureCostEstimator(
        planner,
        future_task_sample=future_task_sample,
        estimator_seed=env_seed,
    )
    labels = [float(estimator.estimate(state, task_library)) for state in states]
    return config, states, task_library, labels


def build_examples(
    *,
    split_name: str,
    num_envs: int,
    states_per_env: int,
    tasks_per_environment: int,
    future_task_sample: int | None,
    seed: int,
    dataset_workers: int,
    text_provider: TextEmbeddingProvider,
) -> Tuple[List[GraphRegressionExample], Dict[str, float]]:
    if num_envs <= 0:
        return [], {"num_examples": 0.0, "target_mean": 0.0, "target_min": 0.0, "target_max": 0.0}

    master_rng = random.Random(seed)
    env_seeds = [master_rng.randint(0, 10**9) for _ in range(num_envs)]
    worker_count = infer_dataset_workers(dataset_workers, num_envs=num_envs)
    payload_args = [
        (idx, env_seed, states_per_env, tasks_per_environment, future_task_sample)
        for idx, env_seed in enumerate(env_seeds)
    ]
    payloads = []
    if worker_count > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            payloads = pool.map(_build_environment_payload, payload_args)
    else:
        payloads = [_build_environment_payload(payload) for payload in payload_args]

    examples: List[GraphRegressionExample] = []
    targets: List[float] = []
    for config, states, _task_library, labels in payloads:
        for state, label in zip(states, labels):
            graph = encode_state_as_graph(state, config, text_provider=text_provider)
            examples.append(GraphRegressionExample(graph=graph, target=float(label)))
            targets.append(float(label))
    stats = {
        "num_examples": float(len(examples)),
        "target_mean": float(mean(targets)) if targets else 0.0,
        "target_min": float(min(targets)) if targets else 0.0,
        "target_max": float(max(targets)) if targets else 0.0,
    }
    print(f"[dataset:{split_name}] {stats}", flush=True)
    return examples, stats


def load_or_build_examples(args: argparse.Namespace) -> Tuple[List[GraphRegressionExample], List[GraphRegressionExample], Dict[str, float]]:
    cache_path = args.dataset_cache
    if cache_path is None:
        cache_path = args.output_dir / "restaurant_gnn_dataset.pt"
    if cache_path.exists() and not args.rebuild_dataset_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["train_examples"], payload["val_examples"], payload["stats"]

    text_provider = TextEmbeddingProvider(
        mode=args.text_encoder,
        model_name=args.text_model_name,
        embedding_dim=args.text_embedding_dim,
        cache_path=args.embedding_cache,
    )
    train_examples, train_stats = build_examples(
        split_name="train",
        num_envs=args.num_train_envs,
        states_per_env=args.states_per_env,
        tasks_per_environment=args.tasks_per_environment,
        future_task_sample=parse_future_task_sample(args.future_task_sample),
        seed=args.seed,
        dataset_workers=args.dataset_workers,
        text_provider=text_provider,
    )
    val_examples, val_stats = build_examples(
        split_name="val",
        num_envs=args.num_val_envs,
        states_per_env=max(8, args.states_per_env // 4),
        tasks_per_environment=args.tasks_per_environment,
        future_task_sample=parse_future_task_sample(args.future_task_sample),
        seed=args.seed + 10_000,
        dataset_workers=args.dataset_workers,
        text_provider=text_provider,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_examples": train_examples,
        "val_examples": val_examples,
        "stats": {"train": train_stats, "val": val_stats},
    }
    torch.save(payload, cache_path)
    return train_examples, val_examples, payload["stats"]


def evaluate(
    model: AnticipatoryRestaurantGNN,
    loader: DataLoader,
    accelerator: Accelerator,
) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    losses: List[float] = []
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["batch"],
            )
            loss = criterion(preds, batch["targets"])
            losses.append(float(accelerator.gather_for_metrics(loss.detach()).mean().item()))
    return float(mean(losses)) if losses else 0.0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke_test:
        args.num_train_envs = 2
        args.num_val_envs = 1
        args.states_per_env = 6
        args.tasks_per_environment = 16
        args.epochs = 1
        args.batch_size = 4
        args.dataset_workers = 1
        if args.text_encoder == "sbert":
            args.text_encoder = "hash"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    train_examples, val_examples, dataset_stats = load_or_build_examples(args)
    text_dim = train_examples[0].graph.node_features.shape[1] - (len(("restaurant", "room", "robot", "container", "object")) + 2 + 9)
    input_dim = int(train_examples[0].graph.node_features.shape[1])
    edge_dim = int(train_examples[0].graph.edge_attr.shape[1])
    model = AnticipatoryRestaurantGNN(
        input_dim=input_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    criterion = torch.nn.L1Loss()

    train_loader = DataLoader(
        GraphExampleDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        GraphExampleDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val = None
    history: List[Dict[str, float]] = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            preds = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["batch"],
            )
            loss = criterion(preds, batch["targets"])
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            loss_value = float(accelerator.gather_for_metrics(loss.detach()).mean().item())
            epoch_losses.append(loss_value)
            global_step += 1
        train_mae = float(mean(epoch_losses)) if epoch_losses else 0.0
        val_mae = evaluate(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"Epoch {epoch:02d}: train_mae={train_mae:.4f} val_mae={val_mae:.4f}")
        history.append({"epoch": float(epoch), "train_mae": train_mae, "val_mae": val_mae})
        if best_val is None or val_mae <= best_val:
            best_val = val_mae
            unwrapped = accelerator.unwrap_model(model)
            save_checkpoint(
                args.output_dir / "paper_restaurant_anticipatory_gnn.pt",
                unwrapped,
                input_dim=input_dim,
                edge_dim=edge_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.heads,
                dropout=args.dropout,
                text_encoder_mode=args.text_encoder,
                text_model_name=args.text_model_name,
                text_embedding_dim=text_dim,
                metrics={"best_val_mae": val_mae, "last_train_mae": train_mae},
            )

    if accelerator.is_main_process:
        metrics_path = args.output_dir / "training_metrics.json"
        metrics_path.write_text(
            json_dumps(
                {
                    "history": history,
                    "dataset_stats": dataset_stats,
                    "text_encoder": args.text_encoder,
                    "text_model_name": args.text_model_name,
                }
            ),
            encoding="utf-8",
        )
        print(f"Saved metrics to {metrics_path}")


def json_dumps(payload: Dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
