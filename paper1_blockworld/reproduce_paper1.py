from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import itertools
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

from .estimator import (
    FutureCostEstimator,
    LearnedGNNFutureCostEstimator,
    OracleFutureCostEstimator,
)
from .planner import FastDownwardBlockworldPlanner, PlanResult
from .world import Task, WorldConfig, WorldGenerator, WorldState


BASELINES = (
    "myopic",
    "anticipatory",
    "prep_myopic",
    "prep_anticipatory",
)


@dataclass
class BaselineMetrics:
    total_cost: float = 0.0
    total_tasks: int = 0
    per_task_costs: List[float] | None = None

    def add_sequence(self, task_costs: Sequence[float]) -> None:
        self.total_cost += sum(task_costs)
        self.total_tasks += len(task_costs)
        if self.per_task_costs is None:
            self.per_task_costs = [0.0 for _ in task_costs]
        for idx, value in enumerate(task_costs):
            self.per_task_costs[idx] += value

    def average_cost(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_cost / self.total_tasks


class AnticipatoryExperiment:
    def __init__(
        self,
        config: WorldConfig,
        planner: FastDownwardBlockworldPlanner,
        *,
        estimator: FutureCostEstimator,
        candidate_goal_limit: int,
        preparation_iterations: int,
    ) -> None:
        self.config = config
        self.planner = planner
        self.estimator = estimator
        self.candidate_goal_limit = candidate_goal_limit
        self.preparation_iterations = preparation_iterations

    def estimate_future_cost(
        self,
        state: WorldState,
        task_library: Sequence[Task],
    ) -> float:
        return self.estimator.estimate(state, task_library)

    def prepare_state(
        self,
        initial_state: WorldState,
        task_library: Sequence[Task],
        rng: random.Random,
    ) -> WorldState:
        current = initial_state.clone()
        current_value = self.estimate_future_cost(current, task_library)
        for _ in range(self.preparation_iterations):
            sampled_task = rng.choice(task_library)
            plan = self.solve_anticipatory(current, sampled_task, task_library)
            plan_value = self.estimate_future_cost(plan.final_state, task_library)
            if plan_value < current_value:
                current = plan.final_state
                current_value = plan_value
        return current

    def solve_myopic(
        self,
        state: WorldState,
        task: Task,
    ) -> PlanResult:
        return self.planner.plan_for_task(state, task)

    def solve_anticipatory(
        self,
        state: WorldState,
        task: Task,
        task_library: Sequence[Task],
    ) -> PlanResult:
        best_result = self.planner.plan_for_task(state, task)
        best_total = best_result.cost + self.estimate_future_cost(
            best_result.final_state,
            task_library,
        )

        candidate_goals = self._candidate_goal_placements(
            state,
            task,
            best_result,
        )
        for placements in candidate_goals:
            result = self.planner.plan_to_placements(state, placements)
            total = result.cost + self.estimate_future_cost(
                result.final_state,
                task_library,
            )
            if total < best_total:
                best_result = result
                best_total = total
        return best_result

    def rollout_sequence(
        self,
        initial_state: WorldState,
        sequence: Sequence[Task],
        task_library: Sequence[Task],
        *,
        anticipatory: bool,
    ) -> List[float]:
        state = initial_state.clone()
        costs: List[float] = []
        for task in sequence:
            if anticipatory:
                plan = self.solve_anticipatory(state, task, task_library)
            else:
                plan = self.solve_myopic(state, task)
            costs.append(float(plan.cost))
            state = plan.final_state
        return costs

    def _candidate_goal_placements(
        self,
        state: WorldState,
        task: Task,
        base_result: PlanResult,
    ) -> List[Dict[str, Tuple[int, int]]]:
        task_goal_positions = task.goal_positions(self.config)
        extra_blocks = [
            block
            for block in base_result.moved_blocks
            if block not in task.blocks
        ]
        if not extra_blocks:
            return []

        parking_cells = WorldGenerator(self.config).candidate_parking_cells(state, task)
        base_extra_positions = [
            base_result.final_state.placements[block]
            for block in extra_blocks
            if block in base_result.final_state.placements
        ]
        for cell in base_extra_positions:
            if cell not in parking_cells and cell not in task_goal_positions.values():
                parking_cells.append(cell)

        if len(parking_cells) < len(extra_blocks):
            return []

        candidates: List[Dict[str, Tuple[int, int]]] = []
        seen = set()
        for cells in itertools.permutations(parking_cells, len(extra_blocks)):
            placements = dict(task_goal_positions)
            placements.update(zip(extra_blocks, cells))
            frozen = tuple(sorted(placements.items()))
            if frozen in seen:
                continue
            seen.add(frozen)
            candidates.append(placements)
            if len(candidates) >= self.candidate_goal_limit:
                break
        return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the small 2D Blockworld evaluation from paper1."
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--tasks-per-environment", type=int, default=20)
    parser.add_argument("--preparation-iterations", type=int, default=200)
    parser.add_argument(
        "--future-task-sample",
        default="8",
        help="Use 'all' for the exact one-step expectation, otherwise a sample count.",
    )
    parser.add_argument("--candidate-goal-limit", type=int, default=24)
    parser.add_argument(
        "--paper-settings",
        action="store_true",
        help="Use the paper1 evaluation settings (32 envs, 100 sequences, 10 tasks).",
    )
    parser.add_argument(
        "--estimator",
        choices=("oracle", "learned"),
        default="oracle",
        help="Use exact/sampled planner-based future cost or a trained GNN checkpoint.",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for --estimator learned.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the learned estimator.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the aggregated metrics JSON.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small end-to-end check.",
    )
    return parser


def parse_future_task_sample(raw: str) -> int | None:
    if raw.strip().lower() == "all":
        return None
    return max(1, int(raw))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke_test:
        args.num_envs = 1
        args.num_sequences = 1
        args.sequence_length = 2
        args.tasks_per_environment = 8
        args.preparation_iterations = 2
        args.future_task_sample = "2"
        args.candidate_goal_limit = 4
    if args.paper_settings:
        args.num_envs = 32
        args.num_sequences = 100
        args.sequence_length = 10
        args.tasks_per_environment = 20
        args.preparation_iterations = 200

    metrics = {name: BaselineMetrics() for name in BASELINES}
    master_rng = random.Random(args.seed)

    for env_idx in range(args.num_envs):
        env_rng = random.Random(master_rng.randint(0, 10**9))
        config = WorldConfig.sample(env_rng)
        generator = WorldGenerator(config)
        planner = FastDownwardBlockworldPlanner(config)
        if args.estimator == "learned":
            if args.gnn_checkpoint is None:
                raise ValueError("--gnn-checkpoint is required when --estimator learned.")
            estimator: FutureCostEstimator = LearnedGNNFutureCostEstimator(
                config,
                args.gnn_checkpoint,
                device=args.device,
            )
        else:
            estimator = OracleFutureCostEstimator(
                planner,
                future_task_sample=parse_future_task_sample(args.future_task_sample),
                estimator_seed=args.seed,
            )
        experiment = AnticipatoryExperiment(
            config,
            planner,
            estimator=estimator,
            candidate_goal_limit=args.candidate_goal_limit,
            preparation_iterations=args.preparation_iterations,
        )
        initial_state = generator.sample_initial_state(env_rng)
        task_library = generator.sample_task_library(
            env_rng,
            count=args.tasks_per_environment,
        )
        prepared_state = experiment.prepare_state(initial_state, task_library, env_rng)

        for _ in range(args.num_sequences):
            sequence = generator.sample_task_sequence(
                env_rng,
                task_library,
                args.sequence_length,
            )
            metrics["myopic"].add_sequence(
                experiment.rollout_sequence(
                    initial_state,
                    sequence,
                    task_library,
                    anticipatory=False,
                )
            )
            metrics["anticipatory"].add_sequence(
                experiment.rollout_sequence(
                    initial_state,
                    sequence,
                    task_library,
                    anticipatory=True,
                )
            )
            metrics["prep_myopic"].add_sequence(
                experiment.rollout_sequence(
                    prepared_state,
                    sequence,
                    task_library,
                    anticipatory=False,
                )
            )
            metrics["prep_anticipatory"].add_sequence(
                experiment.rollout_sequence(
                    prepared_state,
                    sequence,
                    task_library,
                    anticipatory=True,
                )
            )
        print(
            f"[env {env_idx + 1}/{args.num_envs}] "
            f"myopic={metrics['myopic'].average_cost():.1f} "
            f"ap={metrics['anticipatory'].average_cost():.1f} "
            f"prep+myopic={metrics['prep_myopic'].average_cost():.1f} "
            f"prep+ap={metrics['prep_anticipatory'].average_cost():.1f}"
        )

    summary = {
        "settings": {
            "num_envs": args.num_envs,
            "num_sequences": args.num_sequences,
            "sequence_length": args.sequence_length,
            "tasks_per_environment": args.tasks_per_environment,
            "preparation_iterations": args.preparation_iterations,
            "future_task_sample": args.future_task_sample,
            "candidate_goal_limit": args.candidate_goal_limit,
            "estimator": args.estimator,
            "gnn_checkpoint": str(args.gnn_checkpoint) if args.gnn_checkpoint else None,
            "seed": args.seed,
        },
        "results": {
            "N.L. Myopic": {
                "average_cost_per_task": metrics["myopic"].average_cost(),
            },
            f"A.P. ({args.estimator})": {
                "average_cost_per_task": metrics["anticipatory"].average_cost(),
            },
            "Prep + N.L. Myopic": {
                "average_cost_per_task": metrics["prep_myopic"].average_cost(),
            },
            f"Prep + A.P. ({args.estimator})": {
                "average_cost_per_task": metrics["prep_anticipatory"].average_cost(),
            },
        },
    }

    print("\nAverage cost per task")
    for label, payload in summary["results"].items():
        print(f"{label:26s} {payload['average_cost_per_task']:.2f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
