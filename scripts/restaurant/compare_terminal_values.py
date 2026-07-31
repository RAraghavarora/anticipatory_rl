#!/usr/bin/env python3
"""Compare V_AP of jar-based vs fountain-based make_coffee terminal states
under myopic and anticipatory DQN checkpoints.

One search generates the physical terminal candidates (ant model drives
generation); every terminal is then scored under BOTH value functions, so the
comparison is controlled: same states, only the Q-network differs.

Init: canonical dry-machine make_coffee (water_machine consumed), so every
plan must resupply the machine via pour -- either fill(cup) at the fountain
("cup-based") or refill_water(jar) ("jar-based").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    planner_actions_paper2_cost,
)
from anticipatory_rl.utils import select_device
from toy_anticipatory_oracle import _enumerate_future_tasks
from toy_q_guided_planner import _compute_v_q_ap
import toy_bellman_novelty_planner as bnp


def _load_model(env, obs_dim, weights, hidden_dim, device):
    model = RestaurantQNetwork(
        input_dim=obs_dim,
        action_type_dim=int(env.action_space["action_type"].n),
        object_dim=int(env.action_space["object1"].n),
        location_dim=int(env.action_space["location"].n),
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model


def _is_jar_plan(prefix) -> bool:
    return any(
        name == "refill_water" and any("jar" in str(a) for a in args)
        for name, args in prefix
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--myo-weights", type=Path, required=True)
    p.add_argument("--ant-weights", type=Path, required=True)
    p.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_3.yaml"))
    p.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    p.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    p.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    p.add_argument("--fd-timeout-s", type=float, default=20.0)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--success-reward", type=float, default=81.06943684690286)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--max-expansions", type=int, default=5000)
    p.add_argument("--cost-ratio", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    env = RestaurantSymbolicEnv(
        config_path=args.config_path,
        rng_seed=args.seed,
        success_reward=args.success_reward,
    )
    obs, _ = env.reset(seed=args.seed)

    # Canonical dry-machine make_coffee init.
    init = RestaurantPlannerState.from_env(env)
    init.objects["water_machine"].location = None
    task = RestaurantTask(
        task_type="make_coffee", target_location="servingtable",
        target_kind=None, object_name=None,
    )
    env.set_task("make_coffee", target_location="servingtable")

    device = select_device()
    obs_dim = int(np.asarray(obs).shape[0])
    ant = _load_model(env, obs_dim, args.ant_weights, args.hidden_dim, device)
    myo = _load_model(env, obs_dim, args.myo_weights, args.hidden_dim, device)

    result = bnp.search_task(
        env=env, init_state=init.copy(), task=task,
        model=ant, device=device,
        gamma=args.gamma, success_reward=args.success_reward,
        max_depth=args.max_depth, max_expansions=args.max_expansions,
        scoring_mode="cost_bounded", cost_ratio=args.cost_ratio,
        planner_path=args.planner_path, domain_path=args.domain_path,
        alias=args.alias, fd_timeout_s=args.fd_timeout_s,
    )
    future_tasks = _enumerate_future_tasks(env, init)

    rows = []
    for t in result.all_terminals:
        # Fresh cache per model: cache key is state-only, values are model-specific.
        v_ant = _compute_v_q_ap(t.state, env, future_tasks, model=ant, device=device, cache={})
        v_myo = _compute_v_q_ap(t.state, env, future_tasks, model=myo, device=device, cache={})
        rows.append({
            "jar": _is_jar_plan(t.prefix),
            "source": t.source,
            "cost": planner_actions_paper2_cost(t.prefix, env),
            "v_ant": v_ant,
            "v_myo": v_myo,
            "prefix": " ".join(f"{n}({','.join(a)})" for n, a in t.prefix),
        })

    rows.sort(key=lambda r: r["cost"])
    print(f"\nDry-machine make_coffee: {len(rows)} terminals "
          f"({sum(r['jar'] for r in rows)} jar, {sum(not r['jar'] for r in rows)} cup/fountain)")
    print(f"{'jar':>3} {'src':>7} {'cost':>7} {'V_ant':>8} {'V_myo':>8}  prefix")
    for r in rows:
        print(f"{str(r['jar']):>3} {r['source']:>7} {r['cost']:>7.1f} "
              f"{r['v_ant']:>8.2f} {r['v_myo']:>8.2f}  {r['prefix'][:100]}")

    jar_rows = [r for r in rows if r["jar"]]
    cup_rows = [r for r in rows if not r["jar"]]
    if not jar_rows or not cup_rows:
        print("Missing one of the two plan classes; cannot compare.")
        return

    ref = next((r for r in rows if r["source"] == "myopic"), min(cup_rows, key=lambda r: r["cost"]))
    best_jar_cost = min(jar_rows, key=lambda r: r["cost"])

    print("\n=== Headline: V_AP at task termination (post-consumption) ===")
    print(f"{'terminal':<28} {'cost':>7} {'V_ant':>8} {'V_myo':>8}")
    print(f"{'cup/fountain (myopic ref)':<28} {ref['cost']:>7.1f} {ref['v_ant']:>8.2f} {ref['v_myo']:>8.2f}")
    print(f"{'jar (cheapest)':<28} {best_jar_cost['cost']:>7.1f} {best_jar_cost['v_ant']:>8.2f} {best_jar_cost['v_myo']:>8.2f}")
    print(f"{'Δ (jar - ref)':<28} {best_jar_cost['cost'] - ref['cost']:>7.1f} "
          f"{best_jar_cost['v_ant'] - ref['v_ant']:>8.2f} {best_jar_cost['v_myo'] - ref['v_myo']:>8.2f}")

    # cost_ratio only affects selection, not generation: sweep it offline.
    print(f"\n=== cost_ratio sweep (ref cost {ref['cost']:.1f}) ===")
    print(f"{'ratio':>5} {'budget':>7} {'elig':>4} | {'ant pick':>14} {'cost':>6} {'V_ant':>7} | {'myo pick':>14} {'cost':>6} {'V_myo':>7}")
    for ratio in (1.05, 1.10, 1.15, 1.20, 1.25, 1.35, 1.50, 2.00):
        budget = ratio * ref["cost"]
        eligible = [r for r in rows if r["cost"] <= budget + 1e-9]
        if not eligible:
            continue
        sel_ant = max(eligible, key=lambda r: (r["v_ant"], -r["cost"]))
        sel_myo = max(eligible, key=lambda r: (r["v_myo"], -r["cost"]))
        ka = "JAR" if sel_ant["jar"] else "fountain"
        km = "JAR" if sel_myo["jar"] else "fountain"
        print(f"{ratio:>5.2f} {budget:>7.1f} {len(eligible):>4} | {ka:>14} {sel_ant['cost']:>6.0f} {sel_ant['v_ant']:>7.2f} | {km:>14} {sel_myo['cost']:>6.0f} {sel_myo['v_myo']:>7.2f}")


if __name__ == "__main__":
    main()
