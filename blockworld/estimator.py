from __future__ import annotations

import hashlib
from pathlib import Path
import random
from statistics import mean
from typing import Dict, Protocol, Sequence, Tuple

import torch

from .gnn import (
    GraphRegressionExample,
    collate_graphs,
    encode_state_as_graph,
    load_checkpoint,
    select_device,
)
from .planner import FastDownwardBlockworldPlanner
from .world import Task, WorldConfig, WorldState


class FutureCostEstimator(Protocol):
    def estimate(self, state: WorldState, task_library: Sequence[Task]) -> float:
        ...


def _stable_seed(parts: Tuple[object, ...]) -> int:
    digest = hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class OracleFutureCostEstimator:
    def __init__(
        self,
        planner: FastDownwardBlockworldPlanner,
        *,
        future_task_sample: int | None,
        estimator_seed: int,
    ) -> None:
        self.planner = planner
        self.future_task_sample = future_task_sample
        self.estimator_seed = estimator_seed
        self._cache: Dict[Tuple[Tuple[object, ...], Tuple[Tuple[Tuple[str, str], ...], ...]], float] = {}

    def estimate(self, state: WorldState, task_library: Sequence[Task]) -> float:
        tasks = self._select_tasks(state, task_library)
        task_key = tuple(task.assignments for task in tasks)
        cache_key = (state.signature(), task_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        costs = [self.planner.plan_for_task(state, task).cost for task in tasks]
        value = mean(costs) if costs else 0.0
        self._cache[cache_key] = value
        return value

    def _select_tasks(
        self,
        state: WorldState,
        task_library: Sequence[Task],
    ) -> Sequence[Task]:
        if self.future_task_sample is None or self.future_task_sample >= len(task_library):
            return task_library
        indices = list(range(len(task_library)))
        seeded = random.Random(
            _stable_seed((state.signature(), self.estimator_seed, len(task_library)))
        )
        seeded.shuffle(indices)
        chosen = indices[: self.future_task_sample]
        return [task_library[idx] for idx in chosen]


class LearnedGNNFutureCostEstimator:
    def __init__(
        self,
        config: WorldConfig,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.device = select_device(device)
        self.model = load_checkpoint(checkpoint_path, map_location=self.device).to(self.device)
        self._cache: Dict[Tuple[object, ...], float] = {}

    def estimate(self, state: WorldState, task_library: Sequence[Task]) -> float:
        del task_library
        cache_key = state.signature()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        graph = encode_state_as_graph(state, self.config)
        batch = collate_graphs([GraphRegressionExample(graph=graph, target=0.0)])
        with torch.no_grad():
            prediction = self.model(
                batch["node_features"].to(self.device),
                batch["edge_index"].to(self.device),
                batch["batch"].to(self.device),
            )[0]
        value = float(prediction.item())
        self._cache[cache_key] = value
        return value
