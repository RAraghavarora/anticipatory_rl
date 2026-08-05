#!/usr/bin/env python3
"""Debug harness: replay myopic oracle trajectory up to task N, then run both
myopic FD oracle and cost-bounded DQN-guided planner from the exact same
planner state. Prints side-by-side plans, costs, and end states.

Usage (on 5080):
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ant_rl
    cd /home/robocup/raghav/anticipatory_rl
    export PYTHONPATH=/home/robocup/raghav/anticipatory_rl
    python scripts/restaurant/debug_guided_vs_oracle.py \
        --seed 0 --seq 2 --tasks 5 7 \
        --q-weights results/canonical_planner/checkpoints/myopic/seed0/restaurant_dqn.pt \
        --gamma 0.97
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant.env import (
    RestaurantSymbolicEnv,
    RestaurantTask,
    RestaurantObjectState,
)
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.utils import select_device
from toy_anticipatory_oracle import (
    _task_is_auto_satisfied,
    apply_plan_until_first_task_satisfied,
)
import toy_bellman_novelty_planner as bnp


def _load_sequence(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["tasks"])


def _task_from_dict(d: Dict[str, Any]) -> RestaurantTask:
    return RestaurantTask(
        task_type=str(d["task_type"]),
        target_location=d.get("target_location"),
        target_kind=d.get("target_kind"),
        object_name=d.get("object_name"),
    )


def _fmt_state(state: RestaurantPlannerState, env: RestaurantSymbolicEnv) -> str:
    lines = [f"  agent at {state.agent_location}, holding={state.holding}"]
    # Interesting objects
    names = sorted(state.objects.keys())
    for name in names:
        obj = state.objects[name]
        if obj.kind in {"water", "water_source"}:
            continue
        parts = [f"loc={obj.location}"]
        if obj.dirty:
            parts.append("dirty")
        if obj.filled_with:
            parts.append(f"filled={obj.filled_with}")
        if obj.contained_in:
            parts.append(f"in={obj.contained_in}")
        lines.append(f"    {name} ({obj.kind}): {', '.join(parts)}")
    # Water sources
    for name in ("water_fountain", "water_machine", "jar_0"):
        if name in state.objects:
            obj = state.objects[name]
            lines.append(
                f"    {name}: loc={obj.location}, filled={obj.filled_with}, "
                f"contained_in={obj.contained_in}, dirty={obj.dirty}"
            )
    return "\n".join(lines)


def _fmt_plan(prefix: List[Tuple[str, List[str]]], env: RestaurantSymbolicEnv) -> Tuple[str, List[float]]:
    traces = []
    costs = []
    for a_name, a_args in prefix:
        trace_str = f"{a_name}({', '.join(a_args)})"
        cost = planner_actions_paper2_cost([(a_name, a_args)], env)
        costs.append(cost)
        traces.append(f"  {trace_str:<45} cost={cost:>6.1f}")
    return "\n".join(traces), costs


def _run_myopic(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
    search: Optional[str] = None,
) -> Tuple[bool, float, int, List[Tuple[str, List[str]]], RestaurantPlannerState, Optional[str]]:
    result = solve_restaurant_task_with_fd(
        env, init_state.copy(), task,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, search=search, timeout_s=fd_timeout_s,
    )
    if not result.success:
        return False, 0.0, 0, [], init_state, f"FD plan failure: {result.error}"

    terminal, prefix = apply_plan_until_first_task_satisfied(
        init_state.copy(), result.plan_actions, task, env,
    )
    consume_delivery_from_state(terminal, task.task_type, task.target_location)
    cost = planner_actions_paper2_cost(prefix, env)
    return True, cost, len(prefix), prefix, terminal, None


def _run_guided(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    *,
    model: RestaurantQNetwork,
    device: torch.device,
    gamma: float,
    success_reward: float,
    max_depth: int,
    max_expansions: int,
    cost_ratio: float,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
) -> Tuple[bool, float, int, List[Tuple[str, List[str]]], RestaurantPlannerState, Optional[str], Dict[str, Any]]:
    result = bnp.search_task(
        env=env, init_state=init_state.copy(), task=task,
        model=model, device=device,
        gamma=gamma, success_reward=success_reward,
        max_depth=max_depth, max_expansions=max_expansions,
        scoring_mode="cost_bounded", cost_ratio=cost_ratio,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, fd_timeout_s=fd_timeout_s,
    )
    if result.selected is None:
        return False, 0.0, 0, [], init_state, "no feasible candidate", {
            "reference_cost": result.reference_cost,
            "cost_budget": result.cost_budget,
            "eligible": result.eligible_terminal_count,
            "expansions": result.expansions,
        }

    sel = result.selected
    cost = planner_actions_paper2_cost(sel.prefix, env)
    diagnostics = {
        "v_ap": sel.v_ap,
        "reference_cost": result.reference_cost,
        "cost_budget": result.cost_budget,
        "eligible": result.eligible_terminal_count,
        "expansions": result.expansions,
        "selected_source": result.selected_source,
        "jar_ready": sel.jar_ready,
        "undiscounted_rl_cost": sel.undiscounted_rl_cost,
    }
    return True, cost, sel.depth, sel.prefix, sel.state, None, diagnostics


def _replay_oracle_trajectory(
    env: RestaurantSymbolicEnv,
    tasks: List[Dict[str, Any]],
    target_task_idx: int,
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
    search: Optional[str],
) -> RestaurantPlannerState:
    """Run myopic FD oracle for tasks 0..target-1 and return the planner state."""
    state = RestaurantPlannerState.from_env(env)
    for idx in range(target_task_idx):
        task = _task_from_dict(tasks[idx])
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )
        auto = _task_is_auto_satisfied(state, task, env)
        if auto:
            consume_delivery_from_state(state, task.task_type, task.target_location)
            continue
        success, _cost, _n, _trace, next_state, error = _run_myopic(
            env, state, task,
            planner_path=planner_path, domain_path=domain_path,
            alias=alias, fd_timeout_s=fd_timeout_s, search=search,
        )
        if not success:
            raise RuntimeError(f"Oracle failed at task {idx}: {error}")
        state = next_state
    return state


def _compare_task(
    env: RestaurantSymbolicEnv,
    state: RestaurantPlannerState,
    task: RestaurantTask,
    task_idx: int,
    *,
    model: RestaurantQNetwork,
    device: torch.device,
    gamma: float,
    success_reward: float,
    max_depth: int,
    max_expansions: int,
    cost_ratio: float,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
    search: Optional[str],
) -> None:
    print(f"\n{'='*80}")
    print(f"TASK {task_idx}: {task.task_type}")
    print(f"  target_location={task.target_location}  target_kind={task.target_kind}  object_name={task.object_name}")
    print(f"\nINIT STATE (from oracle trajectory):\n{_fmt_state(state, env)}")

    # Oracle from this state
    oracle_success, oracle_cost, oracle_n, oracle_trace, oracle_end, oracle_err = _run_myopic(
        env, state, task,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, fd_timeout_s=fd_timeout_s, search=search,
    )

    # Guided from this state
    guided_success, guided_cost, guided_n, guided_trace, guided_end, guided_err, guided_diag = _run_guided(
        env, state, task,
        model=model, device=device,
        gamma=gamma, success_reward=success_reward,
        max_depth=max_depth, max_expansions=max_expansions,
        cost_ratio=cost_ratio,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, fd_timeout_s=fd_timeout_s,
    )

    print(f"\n--- ORACLE (myopic FD, A*+blind) ---")
    if not oracle_success:
        print(f"FAIL: {oracle_err}")
    else:
        plan_str, costs = _fmt_plan(oracle_trace, env)
        print(f"cost={oracle_cost:.1f}  actions={oracle_n}")
        print(plan_str)
        print(f"  {'TOTAL':<45} {sum(costs):>6.1f}")
        print(f"\nEND STATE:\n{_fmt_state(oracle_end, env)}")

    print(f"\n--- GUIDED (cost-bounded myopic DQN) ---")
    if not guided_success:
        print(f"FAIL: {guided_err}")
        print(f"diagnostics: {guided_diag}")
    else:
        plan_str, costs = _fmt_plan(guided_trace, env)
        print(f"cost={guided_cost:.1f}  actions={guided_n}")
        print(f"v_ap={guided_diag['v_ap']:.3f}  reference_cost={guided_diag['reference_cost']:.1f}  "
              f"budget={guided_diag['cost_budget']:.1f}  eligible={guided_diag['eligible']}  "
              f"expansions={guided_diag['expansions']}  source={guided_diag['selected_source']}  "
              f"jar_ready={guided_diag['jar_ready']}")
        print(plan_str)
        print(f"  {'TOTAL':<45} {sum(costs):>6.1f}")
        print(f"\nEND STATE:\n{_fmt_state(guided_end, env)}")

    if oracle_success and guided_success:
        delta = guided_cost - oracle_cost
        print(f"\n--- DELTA: guided - oracle = {delta:+.1f} ---")
        if delta < -1e-3:
            print("GUIDED BEATS ORACLE from the same init state.")
        elif delta > 1e-3:
            print("ORACLE BEATS GUIDED from the same init state.")
        else:
            print("Costs match from the same init state.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq", type=int, default=2)
    parser.add_argument("--tasks", type=int, nargs="+", default=[5, 7])
    parser.add_argument("--q-weights", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--domain-path", type=Path, default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path, default=Path("downward/fast-downward.py"))
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--search", type=str, default="astar(blind())",
                        help="Oracle search string; default is optimal A*+blind.")
    parser.add_argument("--fd-timeout-s", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--success-reward", type=float, default=81.06943684690286)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--cost-ratio", type=float, default=1.25)
    args = parser.parse_args()

    seq_path = Path(f"experiments/sequences/iid-eval-seq-{args.seq:02d}.json")
    tasks = _load_sequence(seq_path)

    np.random.seed(args.seed)
    env = RestaurantSymbolicEnv(
        config_path=args.config_path,
        rng_seed=args.seed,
        success_reward=args.success_reward,
    )
    obs, _info = env.reset(seed=args.seed)

    device = select_device()
    model = RestaurantQNetwork(
        input_dim=int(np.asarray(obs).shape[0]),
        action_type_dim=int(env.action_space["action_type"].n),
        object_dim=int(env.action_space["object1"].n),
        location_dim=int(env.action_space["location"].n),
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(
        args.q_weights.expanduser().resolve(), map_location=device, weights_only=True,
    ))
    model.eval()

    for task_idx in sorted(args.tasks):
        if task_idx >= len(tasks):
            print(f"Task index {task_idx} out of range (sequence has {len(tasks)} tasks)")
            continue

        # Replay oracle trajectory fresh for each target task to avoid state contamination
        state = _replay_oracle_trajectory(
            env, tasks, task_idx,
            planner_path=args.planner_path, domain_path=args.domain_path,
            alias=args.alias, fd_timeout_s=args.fd_timeout_s, search=args.search,
        )
        task = _task_from_dict(tasks[task_idx])
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )

        _compare_task(
            env, state, task, task_idx,
            model=model, device=device,
            gamma=args.gamma, success_reward=args.success_reward,
            max_depth=args.max_depth, max_expansions=args.max_expansions,
            cost_ratio=args.cost_ratio,
            planner_path=args.planner_path, domain_path=args.domain_path,
            alias=args.alias, fd_timeout_s=args.fd_timeout_s, search=args.search,
        )


if __name__ == "__main__":
    main()
