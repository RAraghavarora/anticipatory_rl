from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .candidates import expand_goal_candidates
from .estimator import FutureCostEstimator, LearnedGNNFutureCostEstimator, OracleFutureCostEstimator
from .planner import FastDownwardRestaurantPlanner, GoalCandidate, PlanResult
from .world import PaperRestaurantTask, RestaurantTaskLibrary, RestaurantWorldConfig, RestaurantWorldGenerator, RestaurantWorldState


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
        config: RestaurantWorldConfig,
        planner: FastDownwardRestaurantPlanner,
        *,
        estimator: FutureCostEstimator,
        candidate_goal_limit: int,
    ) -> None:
        self.config = config
        self.generator = RestaurantWorldGenerator(config)
        self.planner = planner
        self.estimator = estimator
        self.candidate_goal_limit = candidate_goal_limit

    def solve_myopic(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
    ) -> PlanResult:
        return self.planner.plan_for_task(state, task)

    def solve_anticipatory(
        self,
        state: RestaurantWorldState,
        task: PaperRestaurantTask,
        task_library: RestaurantTaskLibrary,
    ) -> PlanResult:
        base_candidates = self.planner.default_goal_candidates(state, task)
        candidates = expand_goal_candidates(
            self.generator,
            state,
            task,
            base_candidates,
            candidate_goal_limit=self.candidate_goal_limit,
        )
        best_result = None
        best_total = None
        for candidate in candidates:
            result = self.planner.plan_to_candidate(state, task, candidate)
            future = self.estimator.estimate(result.final_state, task_library)
            total = result.cost + future
            if best_total is None or total < best_total:
                best_total = total
                best_result = result
        if best_result is None:
            raise RuntimeError("No anticipatory candidate produced a plan result.")
        return best_result

    def rollout_sequence(
        self,
        initial_state: RestaurantWorldState,
        sequence: Sequence[PaperRestaurantTask],
        task_library: RestaurantTaskLibrary,
        *,
        anticipatory: bool,
    ) -> List[float]:
        state = initial_state.clone()
        costs: List[float] = []
        for task in sequence:
            plan = (
                self.solve_anticipatory(state, task, task_library)
                if anticipatory
                else self.solve_myopic(state, task)
            )
            costs.append(float(plan.cost))
            state = plan.final_state
        return costs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the supervised myopic vs anticipatory restaurant evaluation."
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--tasks-per-environment", type=int, default=72)
    parser.add_argument("--candidate-goal-limit", type=int, default=24)
    parser.add_argument("--future-task-sample", default="all")
    parser.add_argument("--paper-settings", action="store_true")
    parser.add_argument("--estimator", choices=("oracle", "learned"), default="oracle")
    parser.add_argument("--gnn-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=Path("paper_restaurant/cache/text_embeddings.pt"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def parse_future_task_sample(raw: str) -> int | None:
    if raw.strip().lower() == "all":
        return None
    return max(1, int(raw))


def _plot_cost_curve(
    summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    myo = summary["myopic"]["per_task_cost_curve"]
    ant = summary["anticipatory"]["per_task_cost_curve"]
    xs = list(range(1, len(myo) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, myo, marker="o", linewidth=1.6, color="#ea580c", label="Myopic")
    ax.plot(xs, ant, marker="o", linewidth=1.6, color="#2563eb", label="Anticipatory")
    ax.set_xlabel("Task Index In Sequence")
    ax.set_ylabel("Average Cost")
    ax.set_title("Restaurant Supervised Baselines")
    ax.grid(alpha=0.35)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke_test:
        args.num_envs = 2
        args.num_sequences = 1
        args.sequence_length = 4
        args.tasks_per_environment = 24
        if args.estimator == "learned" and args.gnn_checkpoint is None:
            parser.error("--gnn-checkpoint is required for learned smoke tests.")
    if args.paper_settings:
        args.num_envs = 500
        args.num_sequences = 1
        args.sequence_length = 40

    metrics = {
        "myopic": BaselineMetrics(),
        "anticipatory": BaselineMetrics(),
    }
    master_rng = random.Random(args.seed)
    future_task_sample = parse_future_task_sample(args.future_task_sample)

    for env_idx in range(args.num_envs):
        env_rng = random.Random(master_rng.randint(0, 10**9))
        config = RestaurantWorldConfig.sample(env_rng)
        generator = RestaurantWorldGenerator(config)
        planner = FastDownwardRestaurantPlanner(config)
        task_library = generator.sample_task_library(env_rng, count=args.tasks_per_environment)
        initial_state = generator.sample_initial_state(env_rng)
        if args.estimator == "learned":
            if args.gnn_checkpoint is None:
                raise ValueError("--gnn-checkpoint is required when --estimator learned.")
            estimator: FutureCostEstimator = LearnedGNNFutureCostEstimator(
                config,
                args.gnn_checkpoint,
                device=args.device,
                embedding_cache_path=args.embedding_cache,
            )
        else:
            estimator = OracleFutureCostEstimator(
                planner,
                future_task_sample=future_task_sample,
                estimator_seed=args.seed + env_idx,
            )
        experiment = AnticipatoryExperiment(
            config,
            planner,
            estimator=estimator,
            candidate_goal_limit=args.candidate_goal_limit,
        )
        for _ in range(args.num_sequences):
            sequence = generator.sample_task_sequence(env_rng, task_library, args.sequence_length)
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

    summary = {
        name: {
            "average_cost_per_task": metric.average_cost(),
            "total_cost": metric.total_cost,
            "total_tasks": metric.total_tasks,
            "per_task_cost_curve": [
                value / max(1, args.num_envs * args.num_sequences)
                for value in (metric.per_task_costs or [])
            ],
        }
        for name, metric in metrics.items()
    }
    summary["delta_anticipatory_minus_myopic"] = {
        "average_cost_per_task": summary["anticipatory"]["average_cost_per_task"] - summary["myopic"]["average_cost_per_task"],
        "total_cost": summary["anticipatory"]["total_cost"] - summary["myopic"]["total_cost"],
    }

    print()
    print("=" * 60)
    print(" PAPER-STYLE RESTAURANT SUPERVISED EVAL")
    print("=" * 60)
    print(f"  Myopic avg cost/task       : {summary['myopic']['average_cost_per_task']:.3f}")
    print(f"  Anticipatory avg cost/task : {summary['anticipatory']['average_cost_per_task']:.3f}")
    print(
        f"  Delta (ant - myopic)       : "
        f"{summary['delta_anticipatory_minus_myopic']['average_cost_per_task']:+.3f}"
    )

    output_json = args.output_json or Path("runs") / "paper_restaurant_supervised_eval.json"
    output_plot = args.output_plot or Path("runs") / "paper_restaurant_supervised_cost_curve.png"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_cost_curve(summary, output_plot)
    print(f"Wrote JSON -> {output_json}")
    print(f"Wrote plot -> {output_plot}")


if __name__ == "__main__":
    main()
