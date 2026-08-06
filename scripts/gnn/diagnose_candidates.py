#!/usr/bin/env python3
"""Diagnose whether GNN candidate selection is limited by augmentation gen / C_AP
or misranked by the trained GNN.

Follows a myopic post-consumption trajectory. At each non-auto state evaluates
the myopic candidate, generates focused augmentations, evaluates each augmented
candidate, computes exact one-step C_AP for every post-consumption state, and
reports pairwise ranking agreement / regret.

Usage:
    python scripts/gnn/diagnose_candidates.py \
        --sequence-path experiments/sequences/iid-eval-seq-00.json \
        --gnn-model runs/gnn_train/best_model.pt \
        --max-tasks 3 --max-augs 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_project_root = str(_SCRIPT_DIR.parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, str(_SCRIPT_DIR.parent / "restaurant"))
import toy_anticipatory_oracle as tao

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    PlannerResult,
    RestaurantPlannerState,
    consume_delivery_from_state,
    solve_restaurant_task_with_fd,
)
# ---------------------------------------------------------------------------
# Reused from eval_sequence (imported dynamically to avoid double-import mess)
# ---------------------------------------------------------------------------

from eval_sequence import (  # noqa: E402
    _evaluate_augmented_plan,
    _evaluate_plan,
    _generate_focused_augmentations,
    load_gnn_model,
)


# ---------------------------------------------------------------------------
# Exact C_AP computation (same semantics as _compute_v_ap in toy_anticipatory_oracle)
# ---------------------------------------------------------------------------

def _compute_exact_c_ap(
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    future_tasks: Sequence[tuple[RestaurantTask, float]],
    *,
    planner_path: Path,
    domain_path: Path,
    timeout_s: float,
    cache: dict[tuple, float],
) -> tuple[float, int]:
    """E_tau[cost(state, tau)] via FD, with caching. Returns (c_ap, fd_failures)."""
    key = tao._state_signature(state)
    cached = cache.get(key)
    if cached is not None:
        return cached, 0  # cached hits counted as 0 new failures

    fd_failures = 0
    total = 0.0
    for task, prob in future_tasks:
        if prob <= 0:
            continue
        if tao._task_is_auto_satisfied(state, task, env):
            cost = 0.0
        else:
            result = solve_restaurant_task_with_fd(
                env, state, task,
                planner_path=planner_path,
                domain_path=domain_path,
                search="astar(ff())",
                timeout_s=timeout_s,
            )
            if not result.success:
                cost = 1e6
                fd_failures += 1
            else:
                cost = result.plan_cost
        total += prob * cost

    cache[key] = total
    return total, fd_failures


# ---------------------------------------------------------------------------
# Pure helpers — testable without FD / GNN / env
# ---------------------------------------------------------------------------


def _select_gnn_argmin(candidates: list[dict]) -> dict:
    """Return candidate with lowest gnn_score."""
    return min(candidates, key=lambda c: c["gnn_score"])


def _select_exact_argmin(candidates: list[dict]) -> dict:
    """Return candidate with lowest exact_score."""
    return min(candidates, key=lambda c: c["exact_score"])


def _compute_regret(candidates: list[dict]) -> tuple[str, str, float]:
    """Return (gnn_strategy, exact_strategy, regret).  Empty → ('','',0)."""
    if not candidates:
        return ("", "", 0.0)
    gnn = _select_gnn_argmin(candidates)
    exact = _select_exact_argmin(candidates)
    regret = float(gnn["exact_score"] - exact["exact_score"])
    return (gnn["strategy"], exact["strategy"], regret)


def _compute_pairwise_ordering(candidates: list[dict]) -> tuple[int, int]:
    """Return (total_pairs, agreeing_pairs).  Ties in exact_score are skipped."""
    total = 0
    agree = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            exact_diff = a["exact_score"] - b["exact_score"]
            gnn_diff = a["gnn_score"] - b["gnn_score"]
            if exact_diff == 0:
                continue
            total += 1
            if (exact_diff > 0 and gnn_diff > 0) or (exact_diff < 0 and gnn_diff < 0):
                agree += 1
    return total, agree


def _has_exact_improvement(candidates: list[dict]) -> bool:
    """True if any augmented candidate improves on the myopic exact_score."""
    myopic_exact = next((c["exact_score"] for c in candidates if c["strategy"] == "myopic"), None)
    if myopic_exact is None:
        return False
    return any(c["exact_score"] < myopic_exact for c in candidates)


# ---------------------------------------------------------------------------
# Diagnostic runner
# ---------------------------------------------------------------------------


def run_diagnostic(
    args: argparse.Namespace,
) -> dict:
    device = torch.device("cpu")
    model = load_gnn_model(args.gnn_model, args.hidden_dim, device)
    env = RestaurantSymbolicEnv(config_path=args.config_path)
    env.reset(seed=args.seed)
    state = RestaurantPlannerState.from_env(env)

    with open(args.sequence_path, "r", encoding="utf-8") as fh:
        seq_data = json.load(fh)
    all_tasks_raw: list[dict] = seq_data["tasks"]
    if args.max_tasks is not None and args.max_tasks > 0:
        all_tasks_raw = all_tasks_raw[: args.max_tasks]

    tasks: list[RestaurantTask] = []
    for raw in all_tasks_raw:
        tasks.append(RestaurantTask(
            task_type=raw["task_type"],
            target_location=raw.get("target_location"),
            target_kind=raw.get("target_kind"),
            object_name=raw.get("object_name"),
        ))

    # Pre-enumerate the full task distribution (42 tasks) for exact C_AP
    future_tasks = env.enumerate_task_distribution()
    exact_cache: dict[tuple, float] = {}

    t_start = time.perf_counter()
    task_records: list[dict] = []
    total_fd_failures_exact = 0
    total_fd_failures_myopic = 0
    total_fd_failures_aug = 0
    tasks_failed_myopic = 0

    for idx, task in enumerate(tasks):
        task_type = str(task.task_type)
        fd_calls_this = 0
        fd_failures_exact_this = 0
        fd_failures_plan_this = 0

        # --- auto-satisfied: advance state, record, skip ---
        if tao._task_is_auto_satisfied(state, task, env):
            consume_delivery_from_state(state, task_type, task.target_location)
            task_records.append({
                "index": idx, "task_type": task_type, "auto": True,
                "generated": 0, "attempted": 0, "solved": 0, "valid": 0,
                "gnn_selected": "auto", "exact_selected": "auto",
                "argmin_agree": True, "exact_regret": 0.0,
                "fd_failures_exact": 0, "fd_failures_plan": 0,
                "candidates": [],
            })
            continue

        # --- Step 1: myopic FD plan ---
        result: PlannerResult = solve_restaurant_task_with_fd(
            env=env, state=state, task=task,
            planner_path=args.planner_path, domain_path=args.domain_path,
            search=args.search, timeout_s=args.fd_timeout_s,
        )
        fd_calls_this += 1

        if not result.success:
            total_fd_failures_myopic += 1
            tasks_failed_myopic += 1
            task_records.append({
                "index": idx, "task_type": task_type, "auto": False,
                "generated": 0, "attempted": 0, "solved": 0, "valid": 0,
                "myopic_fd_failed": True,
                "gnn_selected": "failed_myopic", "exact_selected": "failed_myopic",
                "argmin_agree": None, "exact_regret": None,
                "fd_failures_exact": 0, "fd_failures_plan": 1,
                "candidates": [],
            })
            continue

        # --- Step 2: evaluate myopic candidate ---
        initial_agent_location = state.agent_location
        try:
            myopic_candidate = _evaluate_plan(
                state, result.plan_actions, task, env,
                model, device, args.gamma, "myopic",
            )
        except ValueError:
            tasks_failed_myopic += 1
            task_records.append({
                "index": idx, "task_type": task_type, "auto": False,
                "generated": 0, "attempted": 0, "solved": 0, "valid": 0,
                "myopic_eval_failed": True,
                "gnn_selected": "failed_myopic", "exact_selected": "failed_myopic",
                "argmin_agree": None, "exact_regret": None,
                "fd_failures_exact": 0, "fd_failures_plan": 0,
                "candidates": [],
            })
            continue

        candidates: list[dict] = []

        # --- Record myopic candidate ---
        myopic_post = myopic_candidate["post"]
        exact_cap, exact_fails = _compute_exact_c_ap(
            myopic_post, env, future_tasks,
            planner_path=args.planner_path,
            domain_path=args.domain_path,
            timeout_s=args.fd_timeout_s,
            cache=exact_cache,
        )
        fd_failures_exact_this += exact_fails

        candidates.append({
            "task_idx": idx, "task_type": task_type,
            "strategy": "myopic",
            "clause_type": "", "p_add": "",
            "base_valid": True, "p_add_valid": None,
            "prefix_cost": myopic_candidate["prefix_cost"],
            "gnn_c_ap": myopic_candidate["v_ap"],
            "gnn_score": myopic_candidate["score"],
            "exact_c_ap": float(exact_cap),
            "exact_score": float(myopic_candidate["prefix_cost"] + (args.gamma ** myopic_candidate["actions"]) * exact_cap),
            "fd_failures_exact": exact_fails,
        })

        # --- Step 3: focused augmentations ---
        prefix = myopic_candidate["prefix"]
        clauses = _generate_focused_augmentations(
            prefix, state, initial_agent_location, env,
        )

        generated = 1 + len(clauses)           # all candidates proposed
        attempted = 1 + min(len(clauses), args.max_augs)

        # jar / machine counts from the full proposal list (before cap)
        jar_pos_count = sum(1 for c in clauses if c.clause_type == "jar_position")
        machine_water_count = sum(1 for c in clauses if c.clause_type == "machine_water")

        solved = 1
        valid = 1

        for clause in clauses[: args.max_augs]:
            aug_result = solve_restaurant_task_with_fd(
                env=env, state=state, task=task,
                planner_path=args.planner_path, domain_path=args.domain_path,
                search=args.search,
                extra_goal_clauses=[clause.pddl_clause],
                timeout_s=args.fd_timeout_s,
            )
            fd_calls_this += 1

            if not aug_result.success:
                fd_failures_plan_this += 1
                continue
            solved += 1

            try:
                aug_candidate = _evaluate_augmented_plan(
                    state, aug_result.plan_actions, task, clause,
                    env, model, device, args.gamma,
                    f"aug+{clause.clause_type}+{clause.object_name}",
                )
            except ValueError:
                continue
            valid += 1

            aug_post = aug_candidate["post"]
            exact_cap, exact_fails = _compute_exact_c_ap(
                aug_post, env, future_tasks,
                planner_path=args.planner_path,
                domain_path=args.domain_path,
                timeout_s=args.fd_timeout_s,
                cache=exact_cache,
            )
            fd_failures_exact_this += exact_fails

            candidates.append({
                "task_idx": idx, "task_type": task_type,
                "strategy": aug_candidate["strategy"],
                "clause_type": clause.clause_type,
                "p_add": clause.pddl_clause,
                "base_valid": True, "p_add_valid": True,
                "prefix_cost": aug_candidate["prefix_cost"],
                "gnn_c_ap": aug_candidate["v_ap"],
                "gnn_score": aug_candidate["score"],
                "exact_c_ap": float(exact_cap),
                "exact_score": float(aug_candidate["prefix_cost"] + (args.gamma ** aug_candidate["actions"]) * exact_cap),
                "fd_failures_exact": exact_fails,
            })

        total_fd_failures_exact += fd_failures_exact_this
        total_fd_failures_aug += fd_failures_plan_this

        # --- Step 4: per-task analysis ---
        gnn_strategy, exact_strategy, exact_regret = _compute_regret(candidates)
        argmin_agree = gnn_strategy == exact_strategy

        # --- Step 5: advance state along myopic post-consumption trajectory ---
        state = myopic_post

        task_records.append({
            "index": idx, "task_type": task_type, "auto": False,
            "generated": generated, "attempted": attempted,
            "solved": solved, "valid": valid,
            "jar_position_count": jar_pos_count,
            "machine_water_count": machine_water_count,
            "gnn_selected": gnn_strategy,
            "exact_selected": exact_strategy,
            "argmin_agree": argmin_agree,
            "exact_regret": round(float(exact_regret), 6),
            "fd_failures_exact": fd_failures_exact_this,
            "fd_failures_plan": fd_failures_plan_this,
            "fd_calls": fd_calls_this,
            "candidates": candidates,
        })

    wall_seconds = time.perf_counter() - t_start

    # --- Aggregate summary ---
    # Diagnosable = non-auto, has candidates, myopic didn't fail
    diagnosable = [tr for tr in task_records
                   if not tr.get("auto", False) and tr.get("exact_regret") is not None]

    all_candidates: list[dict] = []
    for tr in diagnosable:
        all_candidates.extend(tr.get("candidates", []))

    argmin_agreements = sum(1 for tr in diagnosable if tr.get("argmin_agree", False))
    argmin_rate = argmin_agreements / len(diagnosable) if diagnosable else 1.0

    regrets = [tr["exact_regret"] for tr in diagnosable]
    mean_regret = float(sum(regrets) / len(regrets)) if regrets else 0.0
    max_regret = float(max(regrets)) if regrets else 0.0

    # Tasks with exact augmented improvement over myopic (uses extracted helper)
    tasks_with_improvement = sum(1 for tr in diagnosable
                                 if _has_exact_improvement(tr.get("candidates", [])))

    # Pairwise ordering agreement (uses extracted helper)
    pairwise_total = 0
    pairwise_agree = 0
    for tr in diagnosable:
        pt, pa = _compute_pairwise_ordering(tr.get("candidates", []))
        pairwise_total += pt
        pairwise_agree += pa
    ordering_agreement = pairwise_agree / pairwise_total if pairwise_total > 0 else 1.0

    total_jar_pos = sum(tr.get("jar_position_count", 0) for tr in diagnosable)
    total_machine_water = sum(tr.get("machine_water_count", 0) for tr in diagnosable)
    total_generated = sum(tr.get("generated", 0) for tr in diagnosable)
    total_attempted = sum(tr.get("attempted", 0) for tr in diagnosable)
    total_solved = sum(tr.get("solved", 0) for tr in diagnosable)
    total_valid = sum(tr.get("valid", 0) for tr in diagnosable)

    summary = {
        "sequence_path": str(args.sequence_path),
        "gnn_model": str(args.gnn_model),
        "max_tasks": args.max_tasks,
        "max_augs": args.max_augs,
        "gamma": args.gamma,
        "tasks_total": len(task_records),
        "auto_tasks": len(task_records) - len([tr for tr in task_records if not tr.get("auto", False)]),
        "tasks_failed_myopic": tasks_failed_myopic,
        "tasks_diagnosed": len(diagnosable),
        "total_generated": total_generated,
        "total_attempted": total_attempted,
        "total_solved": total_solved,
        "total_valid": total_valid,
        "jar_position_proposals": total_jar_pos,
        "machine_water_proposals": total_machine_water,
        "tasks_with_exact_augmented_improvement": tasks_with_improvement,
        "argmin_agreement_rate": round(argmin_rate, 4),
        "mean_exact_regret": round(mean_regret, 6),
        "max_exact_regret": round(max_regret, 6),
        "pairwise_ordering_agreement": round(ordering_agreement, 4),
        "pairwise_pairs": pairwise_total,
        "total_fd_failures_exact": total_fd_failures_exact,
        "total_fd_failures_myopic": total_fd_failures_myopic,
        "total_fd_failures_aug": total_fd_failures_aug,
        "wall_seconds": round(wall_seconds, 2),
    }

    return {"summary": summary, "tasks": task_records}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GNN candidate ranking vs exact C_AP."
    )
    parser.add_argument("--sequence-path", type=Path, required=True)
    parser.add_argument("--gnn-model", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--config-path", type=Path,
                        default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--domain-path", type=Path,
                        default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path,
                        default=Path("downward/fast-downward.py"))
    parser.add_argument("--search", type=str, default="astar(ff())")
    parser.add_argument("--fd-timeout-s", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--max-augs", type=int, default=5)
    args = parser.parse_args()

    result = run_diagnostic(args)

    # Always print summary
    print(json.dumps(result["summary"], indent=2))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\nFull results written to {args.output_path}")


if __name__ == "__main__":
    main()
