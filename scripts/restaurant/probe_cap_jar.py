#!/usr/bin/env python3
"""Can a ONE-TASK expectation ever justify the jar investment?

The GNN baseline's value signal is C_AP(s) = sum_tau P(tau) * plan_cost(s, tau) -- the
expected cost of ONE next task. It therefore credits at most one task's worth of the
jar's saving, no matter how well the network is fit.

This computes C_AP EXACTLY with Fast Downward for the pre- and post-investment states and
compares the gap against what the investment costs. If gap < cost, then every method whose
value signal is a one-task expectation provably refuses the investment -- a statement about
the representation, not about any particular network's training.

That mirrors the K=2-optimal argument: prove the baseline cannot reach it, rather than
observing that one implementation didn't.

Usage:
    PYTHONPATH=. python scripts/restaurant/probe_cap_jar.py \
        --config-path configs/restaurant/toy_level_5.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)


def c_ap(env, state, *, planner_path: Path, domain_path: Path, timeout_s: float, label: str):
    """Exact C_AP(s) = sum_tau P(tau) * optimal_plan_cost(s, tau)."""
    total, unsolved = 0.0, []
    tasks = env.enumerate_task_distribution()
    t0 = time.perf_counter()
    for i, (task, p) in enumerate(tasks, 1):
        r = solve_restaurant_task_with_fd(
            env, state, task,
            planner_path=planner_path, domain_path=domain_path, timeout_s=timeout_s,
        )
        if not r.success:
            unsolved.append(task.task_type)
            continue
        # PlannerResult exposes plan_actions; drop any non-physical completion action so the
        # cost matches the paper2 metric used everywhere else.
        acts = [(n, a) for n, a in r.plan_actions if n != "auto_complete"]
        total += p * planner_actions_paper2_cost(acts, env)
        if i % 12 == 0:
            print(f"    [{label}] {i}/{len(tasks)} tasks, {time.perf_counter()-t0:.0f}s")
    if unsolved:
        print(f"    [{label}] WARNING unsolved: {unsolved}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_5.yaml"))
    ap.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    ap.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--place-at", type=str, default="servingtable",
                    help="Where the post-investment jar sits.")
    args = ap.parse_args()

    env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=0)
    env.reset(seed=0)
    jars = [n for n, o in env.state.objects.items() if o.kind == "jar"]
    assert len(jars) == 1, jars
    jar = jars[0]
    start_loc = env.state.objects[jar].location

    pre = RestaurantPlannerState.from_env(env)
    post = pre.copy()
    post.objects[jar].location = args.place_at
    post.objects[jar].filled_with = "water"

    print(f"jar={jar}  pre: at {start_loc}, filled={pre.objects[jar].filled_with}")
    print(f"          post: at {args.place_at}, filled=water")
    print(f"{len(env.enumerate_task_distribution())} enumerated tasks per state\n")

    kw = dict(planner_path=args.planner_path, domain_path=args.domain_path,
              timeout_s=args.timeout_s)
    print("computing C_AP(pre)...")
    cap_pre = c_ap(env, pre, label="pre", **kw)
    print("computing C_AP(post)...")
    cap_post = c_ap(env, post, label="post", **kw)

    # What it costs to get from pre to post: walk to the jar, fetch it, fill it, park it.
    # The first move starts from the AGENT's position, not the jar's -- using the jar's
    # location makes it a zero-cost no-op and understates the investment by a whole
    # 35-hop trip.
    agent_loc = pre.agent_location
    invest_actions = [
        ("move", [agent_loc, start_loc]),
        ("pick", [jar]),
        ("move", [start_loc, "fountain"]),
        ("fill", [jar]),
        ("move", ["fountain", args.place_at]),
        ("place", [args.place_at]),
    ]
    print(f"  (agent starts at {agent_loc}; jar at {start_loc})")
    invest = planner_actions_paper2_cost(invest_actions, env)

    gap = cap_pre - cap_post
    print(f"\n  C_AP(pre)  = {cap_pre:10.1f}")
    print(f"  C_AP(post) = {cap_post:10.1f}")
    print(f"  gap        = {gap:10.1f}   <- credit a one-task expectation can give")
    print(f"  investment = {invest:10.1f}   <- what it costs to reach post")
    print(f"\n  one-task expectation covers {100*gap/invest:.1f}% of the investment")
    if gap < invest:
        print(f"  => REFUSES. Needs {invest/max(gap,1e-9):.1f} tasks of credit, has 1.")
        print("     Any one-task-expectation method (GNN / Talukder) cannot justify the jar here,")
        print("     regardless of how well its value function is fit.")
    else:
        print("  => ACCEPTS. A one-task expectation suffices; this domain does NOT separate")
        print("     one-task-horizon methods from full-horizon ones.")


if __name__ == "__main__":
    main()
