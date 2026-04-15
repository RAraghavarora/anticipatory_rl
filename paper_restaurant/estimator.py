from __future__ import annotations

import hashlib
from pathlib import Path
import random
from typing import Dict, Protocol, Sequence, Tuple

import torch

from .gnn import (
    GraphRegressionExample,
    TextEmbeddingProvider,
    collate_graphs,
    encode_state_as_graph,
    load_checkpoint,
    select_device,
)
from .planner import FastDownwardRestaurantPlanner
from .world import PaperRestaurantTask, RestaurantTaskLibrary, RestaurantWorldConfig, RestaurantWorldState


class FutureCostEstimator(Protocol):
    def estimate(self, state: RestaurantWorldState, task_library: RestaurantTaskLibrary) -> float:
        ...


def _stable_seed(parts: Tuple[object, ...]) -> int:
    digest = hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class OracleFutureCostEstimator:
    def __init__(
        self,
        planner: FastDownwardRestaurantPlanner,
        *,
        future_task_sample: int | None,
        estimator_seed: int,
    ) -> None:
        self.planner = planner
        self.future_task_sample = future_task_sample
        self.estimator_seed = estimator_seed
        self._cache: Dict[Tuple[Tuple[object, ...], Tuple[Tuple[str, float], ...]], float] = {}

    def estimate(self, state: RestaurantWorldState, task_library: RestaurantTaskLibrary) -> float:
        tasks, weights = self._select_tasks(state, task_library)
        weight_key = tuple((task.summary(), float(weight)) for task, weight in zip(tasks, weights))
        cache_key = (state.signature(), weight_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        weighted_cost = 0.0
        weight_sum = 0.0
        for task, weight in zip(tasks, weights):
            weighted_cost += float(weight) * self.planner.plan_for_task(state, task).cost
            weight_sum += float(weight)
        value = weighted_cost / max(weight_sum, 1e-8)
        self._cache[cache_key] = value
        return value

    def _select_tasks(
        self,
        state: RestaurantWorldState,
        task_library: RestaurantTaskLibrary,
    ) -> Tuple[Sequence[PaperRestaurantTask], Sequence[float]]:
        tasks = task_library.tasks
        weights = task_library.normalized_weights()
        if self.future_task_sample is None or self.future_task_sample >= len(tasks):
            return tasks, weights
        indices = list(range(len(tasks)))
        seeded = random.Random(_stable_seed((state.signature(), self.estimator_seed, len(tasks))))
        seeded.shuffle(indices)
        chosen = indices[: self.future_task_sample]
        chosen_tasks = [tasks[idx] for idx in chosen]
        chosen_weights = [weights[idx] for idx in chosen]
        return chosen_tasks, chosen_weights


class LearnedGNNFutureCostEstimator:
    def __init__(
        self,
        config: RestaurantWorldConfig,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        embedding_cache_path: Path | None = None,
    ) -> None:
        self.config = config
        self.device = select_device(device)
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.model = load_checkpoint(checkpoint_path, map_location=self.device).to(self.device)
        self.text_provider = TextEmbeddingProvider(
            mode=str(payload.get("text_encoder_mode", "sbert")),
            model_name=str(payload.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_dim=int(payload.get("text_embedding_dim", 384)),
            cache_path=embedding_cache_path,
        )
        self._cache: Dict[Tuple[object, ...], float] = {}

    def estimate(self, state: RestaurantWorldState, task_library: RestaurantTaskLibrary) -> float:
        del task_library
        cache_key = state.signature()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        graph = encode_state_as_graph(state, self.config, text_provider=self.text_provider)
        batch = collate_graphs([GraphRegressionExample(graph=graph, target=0.0)])
        with torch.no_grad():
            prediction = self.model(
                batch["node_features"].to(self.device),
                batch["edge_index"].to(self.device),
                batch["edge_attr"].to(self.device),
                batch["batch"].to(self.device),
            )[0]
        value = float(prediction.item())
        self._cache[cache_key] = value
        return value
