#!/usr/bin/env python3
"""Evaluate trained GNN (APCostEstimator) on a persistent task sequence.

Anticipatory planning: for each task, evaluates myopic + augmented FD plans,
scores each via prefix_cost + gamma * V_AP(post), and selects the best.

Usage:
    # GNN anticipatory only
    python scripts/gnn/eval_sequence.py \\
        --sequence-path experiments/sequences/iid-eval-seq-00.json \\
        --gnn-model runs/gnn_train/best_model.pt \\
        --max-tasks 3 --max-augs 5 --seed 42

    # Paired: myopic vs GNN anticipatory
    python scripts/gnn/eval_sequence.py \\
        --sequence-path experiments/sequences/iid-eval-seq-00.json \\
        --gnn-model runs/gnn_train/best_model.pt \\
        --policy both --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "restaurant"))
import toy_anticipatory_oracle as tao

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    PlannerResult,
    RestaurantPlannerState,
    apply_plan,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from gnn.graph_encoder import BINARY_ATTRS, NODE_TYPES, SBERT_DIM, state_to_graph
from gnn.model import APCostEstimator

INPUT_DIM = SBERT_DIM + len(NODE_TYPES) + 2 + len(BINARY_ATTRS)  # 399

# ---------------------------------------------------------------------------
# GNN helpers
# ---------------------------------------------------------------------------


def load_gnn_model(checkpoint_path: Path, hidden_dim: int, device: torch.device) -> APCostEstimator:
    model = APCostEstimator(INPUT_DIM, hidden_dim=hidden_dim)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_v_ap(
    state: RestaurantPlannerState,
    model: APCostEstimator,
    env: RestaurantSymbolicEnv,
    device: torch.device,
) -> float:
    graph = state_to_graph(state, env)
    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        out = model(x, edge_index, batch)
    return float(out.item())


# ---------------------------------------------------------------------------
# Augmented-clause representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentedClause:
    """Immutable augmentation carrying the PDDL clause and metadata for validation.

    Covers the five clause forms: clean object, object filled with water,
    object at location, machine water at coffee machine, and (steelman only)
    a jar both filled with water and relocated to a consumer.  No generic
    predicate framework — just enough metadata to validate and describe each
    clause.

    Every clause_type listed below MUST have a matching branch in
    ``_p_add_is_satisfied``; without one the clause is silently rejected at
    verification and never scored.
    """
    pddl_clause: str
    clause_type: str        # "clean", "fill", "jar_position", "machine_water", "jar_prepared"
    object_name: str
    target_location: str    # empty string when not applicable


# ---------------------------------------------------------------------------
# p_add clause verification
# ---------------------------------------------------------------------------


def _p_add_is_satisfied(state: RestaurantPlannerState, clause: AugmentedClause) -> bool:
    """Check whether a single augmented-goal clause is satisfied in *state*."""
    obj = state.objects.get(clause.object_name)
    if clause.clause_type == "clean":
        return obj is not None and not obj.dirty
    elif clause.clause_type == "fill":
        return obj is not None and obj.filled_with == "water"
    elif clause.clause_type == "jar_position":
        return obj is not None and obj.location == clause.target_location
    elif clause.clause_type == "jar_prepared":
        # Conjunctive steelman clause: both PDDL conjuncts must hold.
        return (obj is not None and obj.filled_with == "water"
                and obj.location == clause.target_location)
    elif clause.clause_type == "machine_water":
        # Any concrete water-kind object at the target location matches the
        # abstract PDDL water constant (Talukder et al. 2024, Section 4).
        return any(
            o.kind == "water" and o.location == clause.target_location
            for o in state.objects.values()
        )
    return False


# ---------------------------------------------------------------------------
# Focused augmented-clause generation
# ---------------------------------------------------------------------------
#
# Talukder et al. describe a bounded-region rule for proposal but do not
# publish the exact region radius.  We reconstruct the focused region as the
# path locations (initial agent location + every move source/destination from
# the myopic prefix) plus every map location within one Dijkstra hop of any
# path location:
#
#     bounded_region = path_locations ∪ {loc | ∃ p∈path_locations : d(p, loc) ≤ 1.0}
#
# with d(·,·) = env._dijkstra_distance.  Directly manipulated objects and
# objects physically in the bounded region are eligible.  The rules:
#   • clean object     — dirty fillable/knife/plate
#   • fill with water   — empty cup/mug/jar, fountain in region
#   • jar → coffee      — both jar location and coffee station in region
#   • machine water     — non-fountain water-kind object depleted (location
#                          is None) when coffee station is in region
#   • jar prepared      — steelman only (``unbounded_jar=True``): jar filled
#                          with water AND relocated to a consumer, ignoring
#                          bounded_region entirely; prepended to the clause
#                          list so it survives the caller's --max-augs cap


def _generate_focused_augmentations(
    prefix: list[tuple[str, list[str]]],
    state: RestaurantPlannerState,
    initial_agent_location: str,
    env: RestaurantSymbolicEnv,
    *,
    unbounded_jar: bool = False,
) -> list[AugmentedClause]:
    # 1. Path locations: initial agent location + every move source/destination
    path_locations: set[str] = {initial_agent_location}
    for name, args in prefix:
        if name == "move":
            if len(args) >= 1:
                path_locations.add(args[0])  # source
            if len(args) >= 2:
                path_locations.add(args[1])  # destination

    # 2. Bounded region: path locations + one-hop
    bounded_region = set(path_locations)
    for path_loc in path_locations:
        for candidate in env.locations:
            if env._dijkstra_distance(path_loc, candidate) <= 1.0:
                bounded_region.add(candidate)

    # 3. Relevant objects: directly manipulated + physically in region
    relevant_objects: set[str] = set()
    for _name, args in prefix:
        for arg in args:
            if arg in state.objects:
                relevant_objects.add(arg)
    for obj_name, obj in state.objects.items():
        if obj.location in bounded_region:
            relevant_objects.add(obj_name)
        # Also include depleted non-fountain water objects even though they
        # have no physical location, because restoration is triggered solely
        # by the coffee station being in the bounded region.
        if obj.kind == "water" and obj_name != "water_fountain" and obj.location is None:
            relevant_objects.add(obj_name)

    # 4. Station presence in bounded region
    coffeemachine_locs = {
        loc for loc in env.locations if env._is_location(loc, "coffeemachine")
    }
    fountain_locs = {
        loc for loc in env.locations if env._is_location(loc, "fountain")
    }
    coffee_in_region = bool(bounded_region & coffeemachine_locs)
    fountain_in_region = bool(bounded_region & fountain_locs)

    # 5. Generate clauses (deterministic order, deduplicated)
    clauses: list[AugmentedClause] = []
    seen: set[str] = set()

    def _emit(clause: AugmentedClause) -> None:
        if clause.pddl_clause not in seen:
            seen.add(clause.pddl_clause)
            clauses.append(clause)

    for obj_name in sorted(relevant_objects):
        obj = state.objects.get(obj_name)
        if obj is None:
            continue

        # clean object (retain existing rule)
        if obj.kind in {"cup", "mug", "jar", "bowl", "knife", "plate"} and obj.dirty:
            _emit(AugmentedClause(
                pddl_clause=f"(not (is-dirty {obj_name}))",
                clause_type="clean",
                object_name=obj_name,
                target_location="",
            ))

        # object filled with water (retain existing rule)
        if obj.kind in {"cup", "mug", "jar"} and obj.filled_with is None and fountain_in_region:
            _emit(AugmentedClause(
                pddl_clause=f"(filled-with water {obj_name})",
                clause_type="fill",
                object_name=obj_name,
                target_location="",
            ))

        # jar positioning: only when both jar location and coffee station in region
        # Skip when the jar is already at the target coffee location.
        if obj.kind == "jar" and obj.location is not None:
            if obj.location in bounded_region and coffee_in_region:
                for cm_loc in sorted(coffeemachine_locs & bounded_region):
                    if obj.location == cm_loc:
                        continue  # already there — nothing to do
                    _emit(AugmentedClause(
                        pddl_clause=f"(is-at {obj_name} {cm_loc})",
                        clause_type="jar_position",
                        object_name=obj_name,
                        target_location=cm_loc,
                    ))

        # machine water at coffee: restore depleted non-fountain water
        if obj.kind == "water" and obj_name != "water_fountain":
            if obj.location is None and coffee_in_region:
                for cm_loc in sorted(coffeemachine_locs & bounded_region):
                    _emit(AugmentedClause(
                        pddl_clause=f"(is-at water {cm_loc})",
                        clause_type="machine_water",
                        object_name=obj_name,
                        target_location=cm_loc,
                    ))

    # 6. Steelman: unbounded jar fill+relocate, ignoring bounded_region and
    # coffee_in_region. The GNN's candidate generator otherwise cannot propose
    # "fetch a jar from a distant pantry, fill it, park it near a consumer";
    # this hands it that candidate so a later refusal is attributable to the
    # value horizon, not to candidate coverage. Conjunctive: the jar starts
    # empty, so a position-only clause would be unsatisfiable-useless.
    #
    # These are PREPENDED, not appended: callers truncate with
    # ``clauses[: args.max_augs]`` (default 10), and at a mid-chain state the
    # bounded rules alone already fill that budget, so appended steelman
    # clauses would be cut before ever reaching the planner — indistinguishable
    # from the baseline declining them.
    if unbounded_jar:
        consumer_locs = coffeemachine_locs | set(env.service_locations)
        jar_clauses: list[AugmentedClause] = []
        for obj_name in sorted(state.objects):
            obj = state.objects[obj_name]
            if obj.kind != "jar":
                continue
            for consumer_loc in sorted(consumer_locs):
                if obj.location == consumer_loc:
                    continue
                clause = AugmentedClause(
                    pddl_clause=(
                        f"(filled-with water {obj_name})\n      "
                        f"(is-at {obj_name} {consumer_loc})"
                    ),
                    clause_type="jar_prepared",
                    object_name=obj_name,
                    target_location=consumer_loc,
                )
                # Same dedup semantics as _emit, into a local list.
                if clause.pddl_clause in seen:
                    continue
                seen.add(clause.pddl_clause)
                jar_clauses.append(clause)
        clauses[:0] = jar_clauses

    return clauses


# ---------------------------------------------------------------------------
# Candidate plan scoring
# ---------------------------------------------------------------------------

_Candidate = Dict[str, Any]


def _evaluate_plan(
    state: RestaurantPlannerState,
    plan_actions: list[tuple[str, list[str]]],
    task: RestaurantTask,
    env: RestaurantSymbolicEnv,
    model: APCostEstimator,
    device: torch.device,
    gamma: float,
    strategy: str,
) -> _Candidate:
    terminal, prefix = tao.apply_plan_until_first_task_satisfied(
        state, plan_actions, task, env,
    )
    if not tao._task_is_auto_satisfied(terminal, task, env):
        raise ValueError(f"Terminal state does not satisfy task {task.task_type}")

    prefix_cost = planner_actions_paper2_cost(prefix, env)

    post = terminal.copy()
    consume_delivery_from_state(post, task.task_type, task.target_location)

    assert post.agent_location in env.location_index, (
        f"Agent at invalid location {post.agent_location}"
    )

    v_ap = predict_v_ap(post, model, env, device)

    discount = gamma ** len(prefix)
    score = prefix_cost + discount * v_ap

    return {
        "prefix": prefix,
        "prefix_cost": float(prefix_cost),
        "post": post,
        "strategy": strategy,
        "v_ap": v_ap,
        "score": float(score),
        "actions": len(prefix),
    }


def _evaluate_augmented_plan(
    state: RestaurantPlannerState,
    plan_actions: list[tuple[str, list[str]]],
    task: RestaurantTask,
    aug_clause: AugmentedClause,
    env: RestaurantSymbolicEnv,
    model: APCostEstimator,
    device: torch.device,
    gamma: float,
    strategy: str,
) -> _Candidate:
    """Apply the *complete* FD plan as an atomic anticipatory macro.

    This matches Talukder et al.'s ``Tail(plan)`` construction (Section 4):
    the full augmented plan executes without truncation at first-task
    satisfaction.  Both the base task *and* the extra ``p_add`` clause are
    verified at the FD-planned terminal state; delivery consumption follows
    afterward to produce the post-consumption next-task state that this
    project's GNN training labels score.  p_add need not remain true after
    consumption — only at the terminal is checked.
    """
    # 1. Apply full plan — no truncation at first task satisfaction
    full = apply_plan(state, plan_actions)

    # 2. Verify base task is satisfied
    if not tao._task_is_auto_satisfied(full, task, env):
        raise ValueError(
            f"Augmented plan terminal does not satisfy base task {task.task_type}"
        )

    # 3. Verify p_add clause is satisfied (uses record metadata, not regex)
    if not _p_add_is_satisfied(full, aug_clause):
        raise ValueError(
            f"Augmented plan terminal does not satisfy p_add clause: {aug_clause.pddl_clause}"
        )

    # 4. Charge complete plan cost
    full_cost = planner_actions_paper2_cost(plan_actions, env)

    # 5. Consume delivery only after both are satisfied
    post = full.copy()
    consume_delivery_from_state(post, task.task_type, task.target_location)

    assert post.agent_location in env.location_index, (
        f"Agent at invalid location {post.agent_location}"
    )

    # 6. GNN-score the post-consumption state (matching training-label semantics)
    v_ap = predict_v_ap(post, model, env, device)

    discount = gamma ** len(plan_actions)
    score = full_cost + discount * v_ap

    return {
        "prefix": plan_actions,       # full plan, not truncated
        "prefix_cost": float(full_cost),
        "post": post,
        "strategy": strategy,
        "v_ap": v_ap,
        "score": float(score),
        "actions": len(plan_actions),
    }


# ---------------------------------------------------------------------------
# Sequence runner — shared state-advance logic
# ---------------------------------------------------------------------------


_TaskRec = Dict[str, Any]


def run_sequence(
    args: argparse.Namespace,
    *,
    model: Optional[APCostEstimator] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run a persistent task sequence. If model is None, uses pure myopic FD.
    Otherwise uses GNN anticipatory planning with augmented candidates."""

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

    if model is not None and device is None:
        device = torch.device("cpu")

    t_start = time.perf_counter()
    records: list[_TaskRec] = []
    total_fd_calls = 0
    total_gnn_calls = 0
    completed = 0
    auto_count = 0
    total_cost = 0.0
    strategy_counts: dict[str, int] = {}
    search = args.search

    gnn_mode = model is not None

    for idx, task in enumerate(tasks):
        task_type = str(task.task_type)
        fd_calls = 0
        gnn_calls = 0

        # --- Step 1: auto-satisfied ---
        if tao._task_is_auto_satisfied(state, task, env):
            consume_delivery_from_state(state, task_type, task.target_location)
            records.append({
                "index": idx, "task_type": task_type,
                "auto": True, "success": True,
                "cost": 0.0, "actions": 0, "strategy": "auto",
                "v_ap": 0.0, "fd_calls": 0, "gnn_calls": 0,
                "trace": [], "augments_tried": 0, "augments_accepted": 0,
            })
            auto_count += 1
            completed += 1
            strategy_counts["auto"] = strategy_counts.get("auto", 0) + 1
            continue

        # --- Step 2: myopic FD plan ---
        result: PlannerResult = solve_restaurant_task_with_fd(
            env=env, state=state, task=task,
            planner_path=args.planner_path, domain_path=args.domain_path,
            search=search, timeout_s=args.fd_timeout_s,
        )
        fd_calls += 1

        if not result.success:
            records.append({
                "index": idx, "task_type": task_type,
                "auto": False, "success": False,
                "cost": float("inf"), "actions": 0,
                "strategy": "failed_myopic", "v_ap": 0.0,
                "fd_calls": fd_calls, "gnn_calls": 0,
                "trace": [], "augments_tried": 0, "augments_accepted": 0,
            })
            total_fd_calls += fd_calls
            continue

        # --- Pure myopic mode: apply prefix, consume, advance ---
        if not gnn_mode:
            terminal, prefix = tao.apply_plan_until_first_task_satisfied(
                state, result.plan_actions, task, env,
            )
            prefix_cost = planner_actions_paper2_cost(prefix, env)
            consume_delivery_from_state(terminal, task_type, task.target_location)
            state = terminal

            records.append({
                "index": idx, "task_type": task_type,
                "auto": False, "success": True,
                "cost": float(prefix_cost), "actions": len(prefix),
                "strategy": "myopic", "v_ap": 0.0,
                "fd_calls": fd_calls, "gnn_calls": 0,
                "trace": [f"{name}({', '.join(a)})" for name, a in prefix],
                "augments_tried": 0, "augments_accepted": 0,
            })
            completed += 1
            total_cost += prefix_cost
            total_fd_calls += fd_calls
            strategy_counts["myopic"] = strategy_counts.get("myopic", 0) + 1
            continue

        # --- GNN mode: evaluate myopic candidate ---
        # Capture initial location before the myopic prefix for focused sampling.
        initial_agent_location = state.agent_location

        try:
            myopic_candidate = _evaluate_plan(
                state, result.plan_actions, task, env,
                model, device, args.gamma, "myopic",
            )
        except ValueError:
            records.append({
                "index": idx, "task_type": task_type, "auto": False,
                "success": False, "cost": float("inf"), "actions": 0,
                "strategy": "failed_myopic", "v_ap": 0.0,
                "fd_calls": fd_calls, "gnn_calls": 0, "trace": [],
                "augments_tried": 0, "augments_accepted": 0,
            })
            total_fd_calls += fd_calls
            continue
        gnn_calls += 1

        best = myopic_candidate
        best_score = best["score"]

        # --- Step 3: augmented candidates (focused bounded-region sampling) ---
        prefix = best["prefix"]
        clauses = _generate_focused_augmentations(
            prefix, state, initial_agent_location, env,
            unbounded_jar=args.unbounded_jar_augmentation,
        )

        augments_tried = 0
        augments_accepted = 0
        # Per-clause-type outcomes, so "the planner timed out" and "the
        # candidate was scored and declined" are distinguishable in the output.
        aug_outcomes: dict[str, dict[str, int]] = {}
        for clause in clauses[: args.max_augs]:
            outcome = aug_outcomes.setdefault(clause.clause_type, {
                "attempted": 0, "solved": 0, "failed": 0,
                "invalid": 0, "scored": 0, "accepted": 0,
            })
            outcome["attempted"] += 1

            aug_result = solve_restaurant_task_with_fd(
                env=env, state=state, task=task,
                planner_path=args.planner_path, domain_path=args.domain_path,
                search=search, extra_goal_clauses=[clause.pddl_clause],
                timeout_s=args.fd_timeout_s,
            )
            fd_calls += 1
            augments_tried += 1

            if not aug_result.success:
                outcome["failed"] += 1
                continue
            outcome["solved"] += 1

            try:
                aug_candidate = _evaluate_augmented_plan(
                    state, aug_result.plan_actions, task, clause,
                    env, model, device, args.gamma,
                    f"aug+{clause.clause_type}+{clause.object_name}",
                )
            except ValueError:
                outcome["invalid"] += 1
                continue
            gnn_calls += 1
            outcome["scored"] += 1

            if aug_candidate["score"] < best_score:
                best = aug_candidate
                best_score = aug_candidate["score"]
                augments_accepted += 1
                outcome["accepted"] += 1

        # --- Step 4: advance state ---
        state = best["post"]

        records.append({
            "index": idx, "task_type": task_type,
            "auto": False, "success": True,
            "cost": best["prefix_cost"], "actions": best["actions"],
            "strategy": best["strategy"], "v_ap": best["v_ap"],
            "fd_calls": fd_calls, "gnn_calls": gnn_calls,
            "trace": [f"{name}({', '.join(a)})" for name, a in best["prefix"]],
            "augments_tried": augments_tried,
            "augments_accepted": augments_accepted,
            "aug_outcomes": aug_outcomes,
        })
        completed += 1
        total_cost += best["prefix_cost"]
        total_fd_calls += fd_calls
        total_gnn_calls += gnn_calls
        strategy_counts[best["strategy"]] = strategy_counts.get(best["strategy"], 0) + 1

    wall_seconds = time.perf_counter() - t_start
    attempted = len(tasks)
    mean_cost = total_cost / completed if completed > 0 else float("inf")

    # Roll up per-clause-type augmentation outcomes over the whole sequence.
    aug_outcome_totals: dict[str, dict[str, int]] = {}
    for rec in records:
        for ctype, counts in rec.get("aug_outcomes", {}).items():
            tot = aug_outcome_totals.setdefault(ctype, {})
            for key, val in counts.items():
                tot[key] = tot.get(key, 0) + val

    summary = {
        "sequence_path": str(args.sequence_path),
        "policy": "gnn_anticipatory" if gnn_mode else "myopic",
        "gnn_model": str(args.gnn_model) if args.gnn_model else None,
        "gamma": args.gamma,
        "attempted": attempted,
        "completed": completed,
        "auto_count": auto_count,
        "total_cost": float(total_cost),
        "mean_cost": float(mean_cost),
        "total_fd_calls": total_fd_calls,
        "total_gnn_calls": total_gnn_calls,
        "strategy_distribution": strategy_counts,
        "aug_outcomes_by_clause_type": aug_outcome_totals,
        "unbounded_jar_augmentation": bool(getattr(args, "unbounded_jar_augmentation", False)),
        "wall_seconds": wall_seconds,
    }

    return {"summary": summary, "tasks": records}


# ---------------------------------------------------------------------------
# Paired evaluation
# ---------------------------------------------------------------------------


def _pair_results(myopic: dict, gnn: dict) -> dict:
    mc = myopic["summary"]["total_cost"]
    gc = gnn["summary"]["total_cost"]
    return {
        "myopic": myopic,
        "gnn_anticipatory": gnn,
        "cost_delta": round(gc - mc, 6),
        "cost_reduction_pct": round((gc - mc) / mc * 100, 2) if mc > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN anticipatory planning on a task sequence."
    )
    parser.add_argument("--sequence-path", type=Path, required=True,
                        help="JSON task sequence")
    parser.add_argument("--gnn-model", type=Path, required=True,
                        help="Trained GNN checkpoint")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Optional JSON output file")
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
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-augs", type=int, default=10)
    parser.add_argument("--policy", type=str, default="gnn_anticipatory",
                        choices=["gnn_anticipatory", "myopic", "both"])
    parser.add_argument("--unbounded-jar-augmentation", action="store_true",
                        help="Steelman: also propose fill+relocate jar candidates "
                             "outside the bounded region (see _generate_focused_augmentations).")
    args = parser.parse_args()

    if args.policy == "both":
        device = torch.device("cpu")
        model = load_gnn_model(args.gnn_model, args.hidden_dim, device)

        myopic_result = run_sequence(args, model=None)
        gnn_result = run_sequence(args, model=model, device=device)
        output = _pair_results(myopic_result, gnn_result)
    else:
        if args.policy == "gnn_anticipatory":
            device = torch.device("cpu")
            model = load_gnn_model(args.gnn_model, args.hidden_dim, device)
            output = run_sequence(args, model=model, device=device)
        else:
            output = run_sequence(args, model=None)

    print(json.dumps(output["summary"] if args.policy != "both" else {
        "myopic": output["myopic"]["summary"],
        "gnn_anticipatory": output["gnn_anticipatory"]["summary"],
        "cost_delta": output["cost_delta"],
        "cost_reduction_pct": output["cost_reduction_pct"],
    }, indent=2))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()
