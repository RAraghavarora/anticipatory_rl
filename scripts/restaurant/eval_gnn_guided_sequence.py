#!/usr/bin/env python3
"""Ablation: the one-task GNN's C_AP estimator inside OUR cost-bounded search.

This separates the two components of the method. Our search contributes candidate
generation and the cost bound; the anticipatory DQN contributes a multi-task credit
horizon. Swapping in the GNN's C_AP keeps the former and removes the latter, so whatever
changes is attributable to the estimator alone.

Predicted outcome, from the exact C_AP computation in results/v5/ANALYSIS.md: it should
improve on the published GNN (whose bounded augmentation heuristic cannot even propose the
jar) but cannot cross the K=2 exact one-task optimum, because C_AP is a one-task
expectation. The most credit it can offer a prepared jar is 785.3 - 535.0 = 250.3 against
a 3,000 investment.

SIGN CONVENTION. The search selects argmax over its terminal value (cost_bounded sorts by
-v_ap), so the hook contract is HIGHER IS BETTER. C_AP is a COST, so it is negated here.
Passing the raw cost would invert the search and silently produce the worst plan in budget.

Usage:
  PYTHONPATH=. python scripts/restaurant/eval_gnn_guided_sequence.py \
      --gnn-model runs/v5_gnn_seeds/aug/s0/best_model.pt \
      --sequence-path experiments/sequences/iid-eval-seq-00.json \
      --config-path configs/restaurant/toy_level_5.yaml \
      --cost-ratio 3.0 --output-path out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts" / "restaurant"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn.graph_encoder import BINARY_ATTRS, NODE_TYPES, SBERT_DIM, state_to_graph
from gnn.model import APCostEstimator

import evaluate_bellman_novelty_sequence as ebns
import toy_q_guided_planner as tqp

INPUT_DIM = SBERT_DIM + len(NODE_TYPES) + 2 + len(BINARY_ATTRS)


def make_gnn_value_fn(model: APCostEstimator, device: torch.device):
    """Return a value_fn matching the planner hook contract (higher is better).

    Caches by the same state signature the DQN path uses, so the two estimators see the
    same candidate set and differ only in how they score it.
    """
    def value_fn(state, env, future_tasks, *, cache):
        key = tqp._state_signature(state)
        hit = cache.get(key)
        if hit is not None:
            return hit
        saved_state = copy.deepcopy(env.state)
        saved_task = env.task
        try:
            tqp._sync_env_from_planner_state(env, state)
            graph = state_to_graph(state, env)
            with torch.no_grad():
                batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
                c_ap = float(model(graph.x.to(device), graph.edge_index.to(device), batch).item())
        finally:
            env.state, env.task = saved_state, saved_task
        v = -c_ap                     # cost -> value; see SIGN CONVENTION above
        cache[key] = v
        return v
    return value_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gnn-model", type=Path, required=True)
    p.add_argument("--sequence-path", type=Path, required=True)
    p.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_5.yaml"))
    p.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    p.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--cost-ratio", type=float, default=3.0)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--max-expansions", type=int, default=20000)
    p.add_argument("--fd-timeout-s", type=float, default=60.0)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Only affects diagnostics; cost_bounded selection uses v_ap alone.")
    p.add_argument("--success-reward", type=float, default=74.01255445461501)
    # must match evaluate_bellman_novelty_sequence.py, or the two arms get different
    # candidate plans and the comparison is not like-for-like.
    p.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-path", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cpu")
    model = APCostEstimator(INPUT_DIM, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.gnn_model, map_location=device))
    model.to(device).eval()

    value_fn = make_gnn_value_fn(model, device)
    result = ebns.run_sequence(
        policy="cost_bounded",
        sequence_path=args.sequence_path,
        config_path=args.config_path,
        domain_path=args.domain_path,
        planner_path=args.planner_path,
        gamma=args.gamma,
        success_reward=args.success_reward,
        cost_ratio=args.cost_ratio,
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
        fd_timeout_s=args.fd_timeout_s,
        alias=args.alias,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        q_weights=None,
        value_fn=value_fn,
    )
    out = {"gnn_guided": result, "estimator": str(args.gnn_model),
           "note": "one-task GNN C_AP inside the cost-bounded Bellman+novelty search"}
    print(json.dumps(result["summary"], indent=2))
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWritten to {args.output_path}")


if __name__ == "__main__":
    main()
