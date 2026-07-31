#!/usr/bin/env python3
"""Lockstep replay of myopic-FD vs DQN-guided trajectories from saved eval JSONs.
Diffs the world state at every task boundary and characterizes the init-state
difference wherever guided's per-task cost beats myopic's.

Usage (on 5080):
    python scripts/restaurant/analyze_state_divergence.py --seeds 0 4 8 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_plan,
    consume_delivery_from_state,
    planner_actions_paper2_cost,
)
from toy_anticipatory_oracle import _task_is_auto_satisfied

_TRACE_RE = re.compile(r"^([a-z_\-]+)\((.*)\)$")


def _parse_trace(trace: List[str]) -> List[Tuple[str, List[str]]]:
    out = []
    for s in trace:
        m = _TRACE_RE.match(s.strip())
        if not m:
            raise ValueError(f"bad trace line: {s!r}")
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        out.append((m.group(1), args))
    return out


def _task_from_entry(e: Dict[str, Any]) -> RestaurantTask:
    return RestaurantTask(
        task_type=str(e["task_type"]),
        target_location=e.get("target_location"),
        target_kind=e.get("target_kind"),
        object_name=e.get("object_name"),
    )


def _features(state: RestaurantPlannerState) -> Dict[str, Any]:
    feats: Dict[str, Any] = {
        "agent_location": state.agent_location,
        "holding": state.holding,
    }
    for name, obj in sorted(state.objects.items()):
        feats[f"{name}.loc"] = obj.location
        feats[f"{name}.dirty"] = obj.dirty
        feats[f"{name}.filled"] = obj.filled_with
        feats[f"{name}.in"] = obj.contained_in
    return feats


def _diff(sm: RestaurantPlannerState, sg: RestaurantPlannerState) -> List[str]:
    fm, fg = _features(sm), _features(sg)
    return [f"{k}: myo={fm[k]} | gui={fg[k]}" for k in sorted(fm) if fm[k] != fg[k]]


def _advance(state: RestaurantPlannerState, entry: Dict[str, Any],
             env: RestaurantSymbolicEnv, warns: List[str], tag: str) -> RestaurantPlannerState:
    task = _task_from_entry(entry)
    env.set_task(task.task_type, target_location=task.target_location,
                 target_kind=task.target_kind, object_name=task.object_name)
    auto = _task_is_auto_satisfied(state, task, env)
    if auto != bool(entry["auto"]):
        warns.append(f"{tag} task {entry['index']}: auto mismatch replay={auto} stored={entry['auto']}")
    if entry["auto"]:
        consume_delivery_from_state(state, task.task_type, task.target_location)
        return state
    plan = _parse_trace(entry["trace"])
    cost = planner_actions_paper2_cost(plan, env)
    if abs(cost - float(entry["cost"])) > 1e-6:
        warns.append(f"{tag} task {entry['index']}: cost mismatch replay={cost} stored={entry['cost']}")
    state = apply_plan(state, plan)  # apply_plan returns a NEW state
    consume_delivery_from_state(state, task.task_type, task.target_location)
    return state


def _action_counts(tasks: List[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for e in tasks:
        for name, _args in _parse_trace(e["trace"]):
            c[name] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/bellman_myopic_g0.97"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--seqs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--config-path", type=Path,
                    default=Path("configs/restaurant/toy_level_3.yaml"))
    ap.add_argument("--verbose-wins", type=int, default=6,
                    help="max winning tasks to print diffs for per sequence")
    args = ap.parse_args()

    grand_diff_lines: Counter = Counter()
    grand_net_actions: Counter = Counter()
    grand = {"win_save": 0.0, "loss_cost": 0.0, "auto_flip_save": 0.0,
             "wins": 0, "losses": 0, "auto_flips": 0}

    for seed in args.seeds:
        for seq in args.seqs:
            path = args.results_dir / f"myopic_seed{seed}_iid-eval-seq-{seq:02d}.json"
            if not path.exists():
                print(f"[skip] {path}")
                continue
            data = json.loads(path.read_text())
            myo_tasks = data["myopic"]["tasks"]
            gui_tasks = data["guided"]["tasks"]

            env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=seed)
            env.reset(seed=seed)
            sm = RestaurantPlannerState.from_env(env)
            sg = sm.copy()

            warns: List[str] = []
            rows = []
            for t in range(len(myo_tasks)):
                em, eg = myo_tasks[t], gui_tasks[t]
                if (em["task_type"] != eg["task_type"]
                        or em.get("target_location") != eg.get("target_location")
                        or em.get("object_name") != eg.get("object_name")):
                    warns.append(f"task {t}: sequence mismatch between runs")
                rows.append({
                    "t": t, "task_type": em["task_type"],
                    "auto_m": bool(em["auto"]), "auto_g": bool(eg["auto"]),
                    "cost_m": float(em["cost"]), "cost_g": float(eg["cost"]),
                    "diff": _diff(sm, sg),
                })
                sm = _advance(sm, em, env, warns, f"seed{seed}/seq{seq}/myo")
                sg = _advance(sg, eg, env, warns, f"seed{seed}/seq{seq}/gui")

            tot_m = sum(r["cost_m"] for r in rows)
            tot_g = sum(r["cost_g"] for r in rows)
            wins = [r for r in rows if not r["auto_g"] and r["cost_g"] < r["cost_m"] - 1e-6]
            losses = [r for r in rows if not r["auto_g"] and r["cost_g"] > r["cost_m"] + 1e-6]
            flips = [r for r in rows if r["auto_g"] and not r["auto_m"]]  # guided gets free success; reverse dir shows in losses
            win_save = sum(r["cost_m"] - r["cost_g"] for r in wins)
            loss_cost = sum(r["cost_g"] - r["cost_m"] for r in losses)
            flip_save = sum(r["cost_m"] for r in flips)

            first_div = next((r["t"] for r in rows if r["diff"]), None)
            net = _action_counts(gui_tasks) - _action_counts(myo_tasks)
            net = Counter({k: v for k, v in net.items() if v != 0})

            print(f"\n{'='*78}")
            print(f"SEED {seed} SEQ {seq}: myo={tot_m:.0f} gui={tot_g:.0f} delta={tot_g-tot_m:+.0f} "
                  f"| auto m={sum(r['auto_m'] for r in rows)} g={sum(r['auto_g'] for r in rows)} "
                  f"| wins={len(wins)} (-{win_save:.0f}) losses={len(losses)} (+{loss_cost:.0f}) "
                  f"| auto-flips={len(flips)} (saved {flip_save:.0f})")
            print(f"first state divergence at task t={first_div}")
            if first_div is not None and first_div > 0:
                pm, pg = myo_tasks[first_div-1], gui_tasks[first_div-1]
                print(f"  created by task {first_div-1} ({pm['task_type']}):")
                print(f"    myo ({pm['cost']:.0f}): {pm['trace']}")
                print(f"    gui ({pg['cost']:.0f}): {pg['trace']}")
            if flips:
                print(f"  auto-flips (guided gets free success):")
                for r in flips:
                    print(f"    t={r['t']} {r['task_type']}: gui auto, myo paid {r['cost_m']:.0f}")
            print(f"  winning tasks (guided cheaper, non-auto):")
            for r in wins[: args.verbose_wins]:
                print(f"    t={r['t']} {r['task_type']}: {r['cost_m']:.0f} -> {r['cost_g']:.0f} "
                      f"({r['cost_g']-r['cost_m']:+.0f})")
                for line in r["diff"]:
                    print(f"      {line}")
            if len(wins) > args.verbose_wins:
                print(f"    ... {len(wins)-args.verbose_wins} more")
            print(f"  NET actions (gui - myo): {dict(net)}")
            if warns:
                print(f"  REPLAY WARNINGS ({len(warns)}): {warns[:5]}")

            for r in wins:
                for line in r["diff"]:
                    grand_diff_lines[line.split(":")[0]] += 1
            grand_net_actions.update(net)
            grand["win_save"] += win_save
            grand["loss_cost"] += loss_cost
            grand["auto_flip_save"] += flip_save
            grand["wins"] += len(wins)
            grand["losses"] += len(losses)
            grand["auto_flips"] += len(flips)

    print(f"\n{'='*78}")
    print(f"GRAND TOTAL over {len(args.seeds)} seeds x {len(args.seqs)} seqs")
    print(f"  wins={grand['wins']} saved={grand['win_save']:.0f} | "
          f"losses={grand['losses']} cost={grand['loss_cost']:.0f} | "
          f"auto-flips={grand['auto_flips']} saved={grand['auto_flip_save']:.0f}")
    print(f"  NET = {-grand['win_save'] + grand['loss_cost'] - grand['auto_flip_save']:+.0f}")
    print(f"\n  Diff features at winning-task boundaries (count of tasks where feature differed):")
    for k, v in grand_diff_lines.most_common(25):
        print(f"    {v:4d}  {k}")
    print(f"\n  NET actions across all (gui - myo): {dict(grand_net_actions.most_common())}")


if __name__ == "__main__":
    main()
