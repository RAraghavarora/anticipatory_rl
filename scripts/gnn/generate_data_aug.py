#!/usr/bin/env python3
"""Generate GNN training data including augmented candidate terminal states.

Variant of `generate_data.py` for the counterfactual-training ablation: at
each chain state, in addition to the myopic post-consumption state, also
generate the terminal states of the focused augmented candidates (Talukder
`tau ∪ p_add` plans, executed fully and consumed) and label every state with
exact one-step C_AP over all tasks.

This deviates from Talukder et al.'s Sec. IV-C recipe, where training states
are reachable only by myopic task solving.  It exists so the GNN can learn
the value of prepared states (filled jar, restored machine water, ...) that
myopic chains never visit; it is an ablation, not the faithful baseline.

Usage:
    python scripts/gnn/generate_data_aug.py \
        --config-path configs/restaurant/toy_level_3.yaml \
        --planner-path downward/fast-downward.py \
        --domain-path pddl/toy_restaurant_domain.pddl \
        --num-states 100 \
        --output-path runs/train_data_aug.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "restaurant"))

import toy_anticipatory_oracle as tao

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_plan,
    consume_delivery_from_state,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.envs.restaurant.task_sampling import sample_task
from eval_sequence import _generate_focused_augmentations
from generate_data import _compute_v_ap
from gnn.graph_encoder import state_to_graph


def _build_metadata(args: argparse.Namespace) -> dict:
    """Provenance for the generated dataset (saved as a JSON sidecar next to
    the .pt output, since train_gnn.py loads the .pt as a flat sample list)."""
    return {
        "config_path": str(args.config_path.resolve()),
        "config_hash": hashlib.sha256(args.config_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "num_states": args.num_states,
        "max_augs": args.max_augs,
        "timeout_s": args.timeout_s,
        "unbounded_jar_augmentation": args.unbounded_jar_augmentation,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate GNN training data with augmented states")
    ap.add_argument("--config-path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner-path", type=Path, required=True)
    ap.add_argument("--domain-path", type=Path, required=True)
    ap.add_argument("--num-states", type=int, default=2000)
    ap.add_argument("--max-augs", type=int, default=10,
                    help="max augmented candidates to solve per chain state")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--unbounded-jar-augmentation", action="store_true",
                    help="Steelman: also propose fill+relocate jar candidates "
                         "outside the bounded region (see _generate_focused_augmentations).")
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    env = RestaurantSymbolicEnv(config_path=args.config_path)
    env.reset(seed=args.seed)

    tasks = env.enumerate_task_distribution()
    print(f"Task distribution: {len(tasks)} tasks", flush=True)

    dataset = []
    v_aps = []
    state = RestaurantPlannerState.from_env(env)
    total_fd_calls = 0
    total_fail = 0
    n_skipped = 0
    n_aug_fail = 0
    n_dup = 0
    n_myopic = 0
    n_aug = 0

    print(
        f"Generating {args.num_states} chain states "
        f"(x{1 + args.max_augs} candidates) ...",
        flush=True,
    )
    t0 = time.time()

    for i in range(args.num_states):
        task = sample_task(env)

        result = solve_restaurant_task_with_fd(
            env=env,
            state=state,
            task=task,
            domain_path=args.domain_path,
            planner_path=args.planner_path,
            timeout_s=args.timeout_s,
            search="astar(ff())",
        )
        total_fd_calls += 1

        if not result.success:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  [FAIL solve] {task.task_type} — {result.error}", flush=True)
            continue

        initial_agent_location = state.agent_location

        # Chain advance via the myopic plan (faithful recipe).
        chain_state = apply_plan(state, result.plan_actions)
        consume_delivery_from_state(chain_state, task.task_type, task.target_location)

        # Candidate states to label: myopic terminal + augmented terminals.
        samples = [chain_state]
        seen_sigs = {tao._state_signature(chain_state)}

        clauses = _generate_focused_augmentations(
            result.plan_actions, state, initial_agent_location, env,
            unbounded_jar=args.unbounded_jar_augmentation,
        )
        for clause in clauses[: args.max_augs]:
            aug_result = solve_restaurant_task_with_fd(
                env=env,
                state=state,
                task=task,
                domain_path=args.domain_path,
                planner_path=args.planner_path,
                timeout_s=args.timeout_s,
                search="astar(ff())",
                extra_goal_clauses=[clause.pddl_clause],
            )
            total_fd_calls += 1
            if not aug_result.success:
                n_aug_fail += 1
                continue
            aug_state = apply_plan(state, aug_result.plan_actions)
            consume_delivery_from_state(aug_state, task.task_type, task.target_location)
            sig = tao._state_signature(aug_state)
            if sig in seen_sigs:
                n_dup += 1
                continue
            seen_sigs.add(sig)
            samples.append(aug_state)

        for idx_s, s in enumerate(samples):
            v_ap, n_calls, n_fail = _compute_v_ap(
                s, env, tasks, args.planner_path, args.domain_path, args.timeout_s,
            )
            total_fd_calls += n_calls
            total_fail += n_fail
            graph = state_to_graph(s, env)
            graph.y = torch.tensor(v_ap, dtype=torch.float32)
            dataset.append({"graph": graph, "v_ap": v_ap})
            v_aps.append(v_ap)
            if idx_s == 0:
                n_myopic += 1
        n_aug += len(samples) - 1

        # Advance the chain through the myopic terminal only.
        state = chain_state

        if (i + 1) % args.log_interval == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (len(dataset)) / max(elapsed, 0.1)
            eta = (args.num_states - i - 1) / max(rate, 0.01) / 60
            print(
                f"[{i+1}/{args.num_states}] "
                f"v_ap={v_ap:.1f}  samples={len(dataset)} (myopic={n_myopic} aug={n_aug}) "
                f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}m  "
                f"fd={total_fd_calls} fail={total_fail} aug_fail={n_aug_fail} dup={n_dup}",
                flush=True,
            )

    elapsed = time.time() - t0
    n_generated = len(dataset)
    print(f"\nGenerated {n_generated} samples in {elapsed:.1f}s", flush=True)
    print(f"  myopic: {n_myopic}, augmented: {n_aug}", flush=True)
    print(f"Total FD calls: {total_fd_calls} ({total_fd_calls / max(n_generated, 1):.1f} per sample)", flush=True)
    print(f"Total FD failures (labels): {total_fail}", flush=True)
    print(f"Skipped chain solves: {n_skipped}, augmented solve failures: {n_aug_fail}, duplicates dropped: {n_dup}", flush=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = args.output_path.with_suffix(".pt")
    torch.save(dataset, pt_path)
    print(f"Saved {n_generated} samples to {pt_path}", flush=True)

    if v_aps:
        npz_path = args.output_path.with_suffix(".npz")
        np.savez(npz_path, v_ap=v_aps)
        print(f"Saved v_ap array ({len(v_aps)},) to {npz_path}", flush=True)

    meta_path = args.output_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(_build_metadata(args), f, indent=2)
    print(f"Saved metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
