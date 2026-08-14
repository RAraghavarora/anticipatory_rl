#!/usr/bin/env python3
"""Bellman + novelty search prototype for deterministic restaurant tasks.

Searches over factored physical actions, using a trained DQN for
completion-heuristic (max-a Q) and terminal scoring (V_AP via weighted
task distribution). Includes a width-1/2 novelty lane for search diversity.

Usage:
    python scripts/restaurant/toy_bellman_novelty_planner.py \
        --q-weights runs/<checkpoint>/restaurant_dqn.pt \
        --planner-path downward/fast-downward.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.envs.restaurant.env import (  # noqa: E402
    ACTION_HEADS,
    ACTION_TYPES,
    RestaurantSymbolicEnv,
    RestaurantTask,
)
from anticipatory_rl.envs.restaurant.planner import (  # noqa: E402
    RestaurantPlannerState,
    _current_task_satisfied,
    apply_planner_action,
    consume_delivery_from_state,
    solve_restaurant_task_with_fd,
)
from anticipatory_rl.utils import select_device  # noqa: E402
from toy_anticipatory_oracle import (  # noqa: E402
    _enumerate_future_tasks,
    _state_signature,
    apply_plan_until_first_task_satisfied,
)
from toy_q_guided_planner import (  # noqa: E402
    _compute_v_q_ap,
    _sync_env_from_planner_state,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchNode:
    state: RestaurantPlannerState
    prefix: List[Tuple[str, List[str]]]
    depth: int
    G: float                  # discounted RL prefix return (negative costs only)
    undiscounted_cost: float  # running sum of positive RL costs
    done: bool = False


@dataclass
class TerminalCandidate:
    state: RestaurantPlannerState  # post-consumption
    prefix: List[Tuple[str, List[str]]]
    depth: int
    G_complete: float
    terminal_score: float
    undiscounted_rl_cost: float
    v_ap: float             # raw V_AP(s') after consumption
    source: str          # "myopic" | "search"
    jar_ready: bool = False

    def __lt__(self, other: "TerminalCandidate") -> bool:
        if abs(self.terminal_score - other.terminal_score) > 1e-12:
            return self.terminal_score > other.terminal_score
        if abs(self.undiscounted_rl_cost - other.undiscounted_rl_cost) > 1e-12:
            return self.undiscounted_rl_cost < other.undiscounted_rl_cost
        return self.depth < other.depth


# ---------------------------------------------------------------------------
# RL cost helpers
# ---------------------------------------------------------------------------

_X_COST_ATTR: Dict[str, str] = {
    "move": "_travel_cost",
    "pick": "pick_cost",
    "place": "place_cost",
    "wash": "wash_cost",
    "fill": "fill_cost",
    "make_coffee": "brew_cost",
    "make_fruit_bowl": "fruit_cost",
    "pour": "pour_cost",
    "refill_water": "refill_cost",
    "drain": "drain_cost",
}


def _factored_rl_cost(
    env: RestaurantSymbolicEnv,
    action_type: str,
    obj1_idx: int,
    loc_idx: int,
    obj2_idx: int,
) -> float:
    """Positive RL cost for a single factored env action."""
    if action_type == "move":
        dst = env.locations[loc_idx]
        return float(env._travel_cost(env.state.agent_location, dst))
    attr = _X_COST_ATTR.get(action_type)
    if attr is None:
        return 0.0
    return float(getattr(env, attr))


def _planner_action_rl_cost(
    env: RestaurantSymbolicEnv,
    action_name: str,
    action_args: List[str],
) -> float:
    """Positive RL cost for a single planner action (FD plan replay)."""
    if action_name == "move":
        return float(env._travel_cost(action_args[0], action_args[-1]))
    mapped = {
        "pick": "pick_cost", "place": "place_cost", "wash": "wash_cost",
        "fill": "fill_cost", "make-coffee": "brew_cost",
        "make-fruit-bowl": "fruit_cost", "pour": "pour_cost",
        "refill_water": "refill_cost", "drain": "drain_cost",
    }
    attr = mapped.get(action_name)
    if attr is None:
        return 0.0
    return float(getattr(env, attr))


# ---------------------------------------------------------------------------
# Factored env action -> planner tuple
# ---------------------------------------------------------------------------

def _factored_to_planner_action(
    env: RestaurantSymbolicEnv,
    action_type: str,
    obj1_idx: int,
    loc_idx: int,
    obj2_idx: int,
) -> Tuple[str, List[str]]:
    """Convert a valid factored env action to a planner (name, args) tuple."""
    obj1_name = env.object_names[obj1_idx] if obj1_idx < env.num_objects else "none"
    obj2_name = env.object_names[obj2_idx] if obj2_idx < env.num_objects else "none"
    loc_name = env.locations[loc_idx] if loc_idx < env.num_locations else "none"
    agent_loc = env.state.agent_location

    if action_type == "move":
        return ("move", [agent_loc, loc_name])
    if action_type == "pick":
        return ("pick", [obj1_name])
    if action_type == "place":
        held = env.state.holding or "none"
        return ("place", [held, loc_name])
    if action_type == "wash":
        return ("wash", [obj1_name])
    if action_type == "fill":
        return ("fill", [obj1_name, agent_loc])
    if action_type == "make_coffee":
        return ("make-coffee", [obj1_name, agent_loc])
    if action_type == "make_fruit_bowl":
        held = env.state.holding or "none"
        return ("make-fruit-bowl", [obj1_name, obj2_name, held, agent_loc])
    if action_type == "pour":
        return ("pour", [obj1_name, agent_loc])
    if action_type == "refill_water":
        return ("refill_water", [obj1_name, agent_loc, obj2_name])
    if action_type == "drain":
        return ("drain", [obj1_name])
    raise ValueError(f"Unknown action type for conversion: {action_type}")


# ---------------------------------------------------------------------------
# Novelty (width 1 / 2), Q-independent scoring
# ---------------------------------------------------------------------------

def _state_atoms(
    state: RestaurantPlannerState,
) -> List[Tuple]:
    """Grounded-style atoms (static object-kind excluded)."""
    atoms: List[Tuple] = [
        ("agent_at", state.agent_location),
        ("holding", state.holding or "none"),
        ("bread_spread", state.bread_spread or "none"),
    ]
    for name, obj in state.objects.items():
        loc = obj.location or "hand"
        atoms.append(("obj_at", name, loc))
        atoms.append(("obj_dirty", name, "dirty" if obj.dirty else "clean"))
        atoms.append(("obj_filled", name, obj.filled_with or "none"))
        atoms.append(("obj_contained", name, obj.contained_in or "none"))
    return atoms


def _atom_pairs(atoms: List[Tuple]) -> List[Tuple[Tuple, Tuple]]:
    n = len(atoms)
    return [(atoms[i], atoms[j]) for i in range(n) for j in range(i + 1, n)]


def _compute_novelty_width(
    atoms: List[Tuple],
    pairs: List[Tuple[Tuple, Tuple]],
    seen_atoms: set,
    seen_pairs: set,
) -> int | None:
    """Return 1 (width-1 novel), 2 (width-2 novel), or None (not novel)."""
    has_new = any(a not in seen_atoms for a in atoms)
    if has_new:
        return 1
    has_new_pair = any(p not in seen_pairs for p in pairs)
    if has_new_pair:
        return 2
    return None


# ---------------------------------------------------------------------------
# Jar-ready check (diagnostic, not influencing search)
# ---------------------------------------------------------------------------

def _jar_ready(state: RestaurantPlannerState) -> bool:
    # ponytail: this diagnostic is specific to toy_level_3's canonical machine name.
    return any(
        obj.kind == "jar" and obj.filled_with == "water"
        and obj.location == "coffeemachine"
        for obj in state.objects.values()
    )


# ---------------------------------------------------------------------------
# max-a Q(s, task) with dedicated cache
# ---------------------------------------------------------------------------

def _compute_max_q_single(
    state: RestaurantPlannerState,
    env: RestaurantSymbolicEnv,
    task: RestaurantTask,
    *,
    model: RestaurantQNetwork,
    device: torch.device,
    cache: Dict,
) -> float:
    return _compute_v_q_ap(
        state, env, [(task, 1.0)],
        model=model, device=device, cache=cache,
    )


# ---------------------------------------------------------------------------
# FD myopic fallback
# ---------------------------------------------------------------------------

def _myopic_fallback(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    future_tasks: List[Tuple[RestaurantTask, float]],
    gamma: float,
    success_reward: float,
    model: RestaurantQNetwork,
    device: torch.device,
    v_ap_cache: Dict,
    scoring_mode: str = "bellman",
    task_gamma: float = 0.95,
    value_fn=None,
    *,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    timeout_s: float,
) -> Optional[TerminalCandidate]:
    result = solve_restaurant_task_with_fd(
        env, init_state.copy(), task,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, timeout_s=timeout_s,
    )
    if not result.success:
        return None

    terminal, prefix = apply_plan_until_first_task_satisfied(
        init_state.copy(), result.plan_actions, task, env,
    )

    G = 0.0
    undiscounted = 0.0
    for i, (a_name, a_args) in enumerate(prefix):
        cost = _planner_action_rl_cost(env, a_name, a_args)
        G += (gamma ** i) * (-cost)
        undiscounted += cost

    return _terminal_from_node(
        SearchNode(
            state=terminal,
            prefix=list(prefix),
            depth=len(prefix),
            G=G,
            undiscounted_cost=undiscounted,
            done=True,
        ),
        env,
        task,
        future_tasks,
        gamma,
        success_reward,
        model,
        device,
        v_ap_cache,
        source="myopic",
        scoring_mode=scoring_mode,
        task_gamma=task_gamma,
        value_fn=value_fn,
    )


# ---------------------------------------------------------------------------
# Node expansion (fixed object2 enumeration)
# ---------------------------------------------------------------------------

_EXCLUDED_ACTIONS = frozenset({"auto_complete", "apply_spread"})


def _expand_node(
    node: SearchNode,
    env: RestaurantSymbolicEnv,
    task: RestaurantTask,
    gamma: float,
) -> List[SearchNode]:
    """Generate all physical successors. Updates undiscounted_cost incrementally."""
    _sync_env_from_planner_state(env, node.state)
    masks = env._build_action_masks()

    n_action_types = len(env.action_type_index)
    total_objects = env.num_objects + 1  # includes none sentinel
    children: List[SearchNode] = []

    def _make_child(
        action_name: str,
        o1: int,
        loc: int,
        o2: int,
    ) -> None:
        cost = _factored_rl_cost(env, action_name, o1, loc, o2)
        planner_action = _factored_to_planner_action(env, action_name, o1, loc, o2)

        next_state = node.state.copy()
        apply_planner_action(next_state, planner_action)

        new_G = node.G + (gamma ** node.depth) * (-cost)
        new_undis = node.undiscounted_cost + cost
        new_depth = node.depth + 1
        done = _current_task_satisfied(next_state, task, env)

        children.append(SearchNode(
            state=next_state,
            prefix=node.prefix + [planner_action],
            depth=new_depth,
            G=new_G,
            undiscounted_cost=new_undis,
            done=done,
        ))

    for type_idx in range(n_action_types):
        if masks["valid_action_type_mask"][type_idx] <= 0:
            continue
        action_name = ACTION_TYPES[type_idx]
        if action_name in _EXCLUDED_ACTIONS:
            continue

        heads = ACTION_HEADS.get(action_name, ())
        need_o1 = "object1" in heads
        need_loc = "location" in heads
        need_o2 = "object2" in heads

        valid_o1_all = np.where(masks["valid_object1_mask"][type_idx] > 0)[0]
        valid_loc_all = np.where(masks["valid_location_mask"][type_idx] > 0)[0]

        for o1 in valid_o1_all:
            if not need_o1 and o1 != env.none_object_index:
                continue
            o1_i = int(o1)

            for loc in valid_loc_all:
                if not need_loc and loc != env.none_location_index:
                    continue
                loc_i = int(loc)

                if need_o2:
                    for o2 in range(total_objects):
                        if masks["valid_object2_mask"][type_idx, o1_i, o2] > 0:
                            _make_child(action_name, o1_i, loc_i, o2)
                else:
                    o2 = env.none_object_index
                    if masks["valid_object2_mask"][type_idx, o1_i, o2] > 0:
                        _make_child(action_name, o1_i, loc_i, o2)

    return children


# ---------------------------------------------------------------------------
# Terminal scoring from a completed SearchNode
# ---------------------------------------------------------------------------

_VALID_MODES = frozenset({"bellman", "task_boundary", "cost_bounded"})


def _terminal_from_node(
    node: SearchNode,
    env: RestaurantSymbolicEnv,
    task: RestaurantTask,
    future_tasks: List[Tuple[RestaurantTask, float]],
    gamma: float,
    success_reward: float,
    model: RestaurantQNetwork,
    device: torch.device,
    v_ap_cache: Dict,
    source: str,
    scoring_mode: str = "bellman",
    task_gamma: float = 0.95,
    value_fn=None,
) -> TerminalCandidate:
    d = node.depth
    # G_complete still computed for diagnostics in both modes.
    G_complete = (
        success_reward
        if d == 0
        else node.G + (gamma ** (d - 1)) * success_reward
    )
    consumed = node.state.copy()
    consume_delivery_from_state(consumed, task.task_type, task.target_location)
    if value_fn is None:
        v_ap = _compute_v_q_ap(
            consumed, env, future_tasks,
            model=model, device=device, cache=v_ap_cache,
        )
    else:
        # Alternative terminal estimator (e.g. the one-task GNN's C_AP). The contract is
        # HIGHER IS BETTER, matching _compute_v_q_ap's return semantics, because
        # cost_bounded selection sorts by -v_ap. A cost-valued estimator must therefore be
        # negated by the caller; passing a raw cost here would invert the search.
        v_ap = value_fn(consumed, env, future_tasks, cache=v_ap_cache)
    if scoring_mode == "bellman":
        score = G_complete + (gamma ** d) * v_ap
    elif scoring_mode == "task_boundary":
        score = -node.undiscounted_cost + task_gamma * v_ap
    else:  # cost_bounded
        score = v_ap

    return TerminalCandidate(
        state=consumed,
        prefix=list(node.prefix),
        depth=d,
        G_complete=G_complete,
        terminal_score=score,
        undiscounted_rl_cost=node.undiscounted_cost,
        v_ap=v_ap,
        source=source,
        jar_ready=_jar_ready(consumed),
    )


# ---------------------------------------------------------------------------
# Terminal selection (factored for testability)
# ---------------------------------------------------------------------------

def _select_terminal(
    terminals: List[TerminalCandidate],
    scoring_mode: str,
    cost_ratio: float = 1.25,
    tolerance: float = 1e-9,
) -> Tuple[Optional[TerminalCandidate], Optional[float], Optional[float], int]:
    """Select best terminal according to *scoring_mode*.

    Returns (selected, reference_cost, cost_budget, eligible_count).
    *reference_cost* / *cost_budget* / *eligible_count* are meaningful only
    for ``cost_bounded``; other modes return ``None`` / ``None`` / len(terminals).
    """
    if not terminals:
        return None, None, None, 0

    if scoring_mode in ("bellman", "task_boundary"):
        terminals.sort()
        return terminals[0], None, None, len(terminals)

    # cost_bounded
    myopic = [t for t in terminals if t.source == "myopic"]
    if myopic:
        reference_cost = float(myopic[0].undiscounted_rl_cost)
    else:
        # Fallback: minimum undiscounted cost across all search terminals.
        reference_cost = float(min(t.undiscounted_rl_cost for t in terminals))

    cost_budget = cost_ratio * reference_cost

    eligible = [
        t for t in terminals
        if t.undiscounted_rl_cost <= cost_budget + tolerance
    ]
    if not eligible:
        return None, reference_cost, cost_budget, 0

    eligible.sort(key=lambda t: (-t.v_ap, t.undiscounted_rl_cost, t.depth))
    return eligible[0], reference_cost, cost_budget, len(eligible)


def _best_jar_terminal(
    terminals: List[TerminalCandidate],
) -> Optional[TerminalCandidate]:
    jar_terminals = [candidate for candidate in terminals if candidate.jar_ready]
    return min(jar_terminals) if jar_terminals else None


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    selected: Optional[TerminalCandidate]
    all_terminals: List[TerminalCandidate]
    expansions: int
    expansions_bellman: int
    expansions_novelty: int
    stale_novelty_pops: int
    action_trace: List[str]
    generated_jar_ready: bool   # any search terminal had jar-ready
    selected_jar_ready: bool    # selected terminal was jar-ready
    selected_source: str
    # cost_bounded diagnostics (None when scoring_mode != "cost_bounded")
    scoring_mode: str = "bellman"
    cost_ratio: float = 1.25
    reference_cost: Optional[float] = None
    cost_budget: Optional[float] = None
    eligible_terminal_count: Optional[int] = None


def search_task(
    env: RestaurantSymbolicEnv,
    init_state: RestaurantPlannerState,
    task: RestaurantTask,
    model: RestaurantQNetwork,
    device: torch.device,
    *,
    gamma: float,
    success_reward: float,
    max_depth: int,
    max_expansions: int,
    scoring_mode: str = "bellman",
    task_gamma: float = 0.95,
    cost_ratio: float = 1.25,
    value_fn=None,
    planner_path: Path,
    domain_path: Path,
    alias: str,
    fd_timeout_s: float,
) -> SearchResult:
    if scoring_mode not in _VALID_MODES:
        raise ValueError(f"Invalid scoring_mode '{scoring_mode}'; must be one of {sorted(_VALID_MODES)}")
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if not (0.0 < task_gamma <= 1.0):
        raise ValueError(f"task_gamma must be in (0, 1], got {task_gamma}")
    if scoring_mode == "cost_bounded" and cost_ratio < 1.0:
        raise ValueError(f"cost_ratio must be >= 1.0 for cost_bounded mode, got {cost_ratio}")

    future_tasks = _enumerate_future_tasks(env, init_state)

    max_q_cache: Dict = {}
    v_ap_cache: Dict = {}

    # (state_sig, depth) -> best G seen (for dedup before pushing)
    seen: Dict[Tuple, float] = {}
    # (state_sig, depth) -> best G actually expanded (expansion ledger)
    expanded: Dict[Tuple, float] = {}

    seen_atoms: set = set()
    seen_pairs: set = set()

    # Bellman heap: (-priority, counter, node)
    bellman_heap: List[Tuple[float, int, SearchNode]] = []
    # Novelty heap: (width, depth, counter, node) — Q-independent
    novelty_heap: List[Tuple[int, int, int, SearchNode]] = []
    _node_counter = 0

    terminals: List[TerminalCandidate] = []

    # --- Myopic FD fallback ---
    fb = _myopic_fallback(
        env, init_state, task, future_tasks, gamma, success_reward,
        model, device, v_ap_cache,
        scoring_mode=scoring_mode, task_gamma=task_gamma, value_fn=value_fn,
        planner_path=planner_path, domain_path=domain_path,
        alias=alias, timeout_s=fd_timeout_s,
    )
    if fb is not None:
        terminals.append(fb)

    expansions = 0
    expansions_bellman = 0
    expansions_novelty = 0
    stale_novelty_pops = 0
    scheduler_phase = 0  # 0,1,2 = bellman, 3 = novelty

    def _push_to_heaps(child: SearchNode) -> None:
        nonlocal _node_counter
        if child.depth >= max_depth:
            return
        key = (_state_signature(child.state), child.depth)
        prev = seen.get(key)
        if prev is not None and child.G <= prev:
            return
        seen[key] = child.G

        # Bellman priority
        if value_fn is None:
            max_q = _compute_max_q_single(
                child.state, env, task,
                model=model, device=device, cache=max_q_cache,
            )
        else:
            # No task-conditioned analogue exists for a task-agnostic estimator like the
            # GNN's C_AP, so the same state value drives the Bellman lane's priority. This
            # makes the run independent of any DQN checkpoint, at the cost of a lane that
            # orders by expected next-task value rather than by current-task max-Q.
            max_q = value_fn(child.state, env, future_tasks, cache=max_q_cache)
        priority = child.G + (gamma ** child.depth) * max_q
        _node_counter += 1
        heapq.heappush(bellman_heap, (-priority, _node_counter, child))

        # Novelty: Q-independent width-based key
        atoms = _state_atoms(child.state)
        pairs = _atom_pairs(atoms)
        width = _compute_novelty_width(atoms, pairs, seen_atoms, seen_pairs)
        if width is not None:
            _node_counter += 1
            heapq.heappush(novelty_heap, (width, child.depth, _node_counter, child))

    def _process_child(child: SearchNode, source: str) -> None:
        if child.done:
            key = (_state_signature(child.state), child.depth)
            if key in seen and seen[key] >= child.G:
                return
            seen[key] = child.G
            term = _terminal_from_node(
                child, env, task, future_tasks, gamma, success_reward,
                model, device, v_ap_cache, source=source,
                scoring_mode=scoring_mode, task_gamma=task_gamma, value_fn=value_fn,
            )
            terminals.append(term)
        else:
            _push_to_heaps(child)

    def _expand(node: SearchNode) -> List[SearchNode]:
        return _expand_node(node, env, task, gamma)

    # --- Root node ---
    _sync_env_from_planner_state(env, init_state)
    root = SearchNode(
        state=init_state.copy(), prefix=[], depth=0, G=0.0,
        undiscounted_cost=0.0, done=False,
    )
    _push_to_heaps(root)

    while expansions < max_expansions and (bellman_heap or novelty_heap):
        if scheduler_phase < 3:
            # --- Bellman lane ---
            scheduler_phase = (scheduler_phase + 1) % 4
            if not bellman_heap:
                continue

            _, _nid, node = heapq.heappop(bellman_heap)
            sig = (_state_signature(node.state), node.depth)

            # Expansion ledger: skip if already expanded with >= G
            if expanded.get(sig, float("-inf")) >= node.G:
                continue
            if node.depth >= max_depth:
                continue

            expanded[sig] = node.G
            expansions += 1
            expansions_bellman += 1

            for child in _expand(node):
                _process_child(child, source="search")

        else:
            # --- Novelty lane ---
            scheduler_phase = 0
            if not novelty_heap:
                continue

            popped = False
            while novelty_heap:
                old_width, _old_depth, _nid, node = heapq.heappop(novelty_heap)
                sig = (_state_signature(node.state), node.depth)

                # Expansion ledger check
                if expanded.get(sig, float("-inf")) >= node.G:
                    continue
                if node.depth >= max_depth:
                    continue

                # Recompute novelty against current tables
                atoms = _state_atoms(node.state)
                pairs = _atom_pairs(atoms)
                new_width = _compute_novelty_width(atoms, pairs, seen_atoms, seen_pairs)

                if new_width is None:
                    # No longer novel — drop
                    stale_novelty_pops += 1
                    continue

                if new_width > old_width:
                    # Worsened (was 1, now 2): reinsert with updated width
                    _node_counter += 1
                    heapq.heappush(
                        novelty_heap,
                        (new_width, node.depth, _node_counter, node),
                    )
                    continue

                # Still novel with same or better width: expand
                for a in atoms:
                    seen_atoms.add(a)
                for p in pairs:
                    seen_pairs.add(p)

                expanded[sig] = node.G
                expansions += 1
                expansions_novelty += 1
                popped = True

                for child in _expand(node):
                    _process_child(child, source="search")
                break

            if not popped:
                scheduler_phase = 0

    # --- Select best ---
    selected, ref_cost, cost_budget, eligible_count = _select_terminal(
        terminals, scoring_mode, cost_ratio=cost_ratio,
    )

    if selected is None:
        return SearchResult(
            selected=None, all_terminals=terminals, expansions=expansions,
            expansions_bellman=expansions_bellman,
            expansions_novelty=expansions_novelty,
            stale_novelty_pops=stale_novelty_pops,
            action_trace=[],
            generated_jar_ready=False, selected_jar_ready=False,
            selected_source="none",
            scoring_mode=scoring_mode, cost_ratio=cost_ratio,
            reference_cost=ref_cost, cost_budget=cost_budget,
            eligible_terminal_count=eligible_count,
        )

    action_trace = [
        f"{a_name}({', '.join(a_args)})" for a_name, a_args in selected.prefix
    ]

    search_terminals = [t for t in terminals if t.source == "search"]
    gen_jar = any(t.jar_ready for t in search_terminals) if search_terminals else False

    return SearchResult(
        selected=selected, all_terminals=terminals,
        expansions=expansions,
        expansions_bellman=expansions_bellman,
        expansions_novelty=expansions_novelty,
        stale_novelty_pops=stale_novelty_pops,
        action_trace=action_trace,
        generated_jar_ready=gen_jar,
        selected_jar_ready=selected.jar_ready,
        selected_source=selected.source,
        scoring_mode=scoring_mode, cost_ratio=cost_ratio,
        reference_cost=ref_cost, cost_budget=cost_budget,
        eligible_terminal_count=eligible_count,
    )


# ---------------------------------------------------------------------------
# CLI diagnostic
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bellman+novelty search diagnostic for restaurant tasks."
    )
    parser.add_argument("--q-weights", type=Path, required=True)
    parser.add_argument("--config-path", type=Path,
                        default=Path("configs/restaurant/toy_level_3.yaml"))
    parser.add_argument("--domain-path", type=Path,
                        default=Path("pddl/toy_restaurant_domain.pddl"))
    parser.add_argument("--planner-path", type=Path,
                        default=Path("downward/fast-downward.py"))
    parser.add_argument("--alias", type=str, default="seq-sat-lama-2011")
    parser.add_argument("--fd-timeout-s", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--success-reward", type=float, default=81.06943684690286)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--scoring-mode", type=str, default="bellman",
                        choices=["bellman", "task_boundary", "cost_bounded"])
    parser.add_argument("--task-gamma", type=float, default=0.95)
    parser.add_argument("--cost-ratio", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = RestaurantSymbolicEnv(
        config_path=args.config_path, rng_seed=args.seed,
        success_reward=args.success_reward,
    )
    obs, _info = env.reset(seed=args.seed)

    # Diagnostic setup: water_machine.location=None, make_coffee(servingtable)
    wm = env.state.objects.get("water_machine")
    if wm is not None:
        wm.location = None
    env.set_task("make_coffee", target_location="servingtable")

    init_state = RestaurantPlannerState.from_env(env)

    device = select_device()
    model = RestaurantQNetwork(
        input_dim=int(np.asarray(obs).shape[0]),
        action_type_dim=int(env.action_space["action_type"].n),
        object_dim=int(env.action_space["object1"].n),
        location_dim=int(env.action_space["location"].n),
        hidden_dim=args.hidden_dim,
    ).to(device)
    weights_path = args.q_weights.expanduser().resolve()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    t0 = time.perf_counter()
    result = search_task(
        env=env, init_state=init_state, task=env.task,
        model=model, device=device,
        gamma=args.gamma, success_reward=env.success_reward,
        max_depth=args.max_depth, max_expansions=args.max_expansions,
        scoring_mode=args.scoring_mode, task_gamma=args.task_gamma,
        cost_ratio=args.cost_ratio,
        planner_path=args.planner_path, domain_path=args.domain_path,
        alias=args.alias, fd_timeout_s=args.fd_timeout_s,
    )
    wall_s = time.perf_counter() - t0

    output: Dict[str, Any] = {
        "scoring_mode": args.scoring_mode,
        "task_gamma": args.task_gamma,
        "cost_ratio": args.cost_ratio,
        "wall_seconds": round(wall_s, 4),
        "expansions": result.expansions,
        "expansions_bellman": result.expansions_bellman,
        "expansions_novelty": result.expansions_novelty,
        "stale_novelty_pops": result.stale_novelty_pops,
        "num_terminals": len(result.all_terminals),
        "generated_jar_ready": result.generated_jar_ready,
    }

    if result.reference_cost is not None:
        output["reference_cost"] = round(result.reference_cost, 6)
    if result.cost_budget is not None:
        output["cost_budget"] = round(result.cost_budget, 6)
    if result.eligible_terminal_count is not None:
        output["eligible_terminal_count"] = result.eligible_terminal_count

    if result.selected is not None:
        sel = result.selected
        output["selected_depth"] = sel.depth
        output["selected_G_complete"] = round(sel.G_complete, 6)
        output["selected_terminal_score"] = round(sel.terminal_score, 6)
        output["selected_undiscounted_rl_cost"] = round(sel.undiscounted_rl_cost, 6)
        output["selected_v_ap"] = round(sel.v_ap, 6)
        output["selected_source"] = sel.source
        output["selected_jar_ready"] = sel.jar_ready
        output["action_trace"] = result.action_trace
        output["num_actions"] = len(result.action_trace)

        best_jar = _best_jar_terminal(result.all_terminals)
        if best_jar is not None:
            jar_entry: Dict[str, Any] = {
                "depth": best_jar.depth,
                "G_complete": round(best_jar.G_complete, 6),
                "terminal_score": round(best_jar.terminal_score, 6),
                "score_gap_vs_selected": round(best_jar.terminal_score - sel.terminal_score, 6),
                "undiscounted_rl_cost": round(best_jar.undiscounted_rl_cost, 6),
                "v_ap": round(best_jar.v_ap, 6),
                "source": best_jar.source,
                "action_trace": [
                    f"{name}({', '.join(action_args)})"
                    for name, action_args in best_jar.prefix
                ],
            }
            if args.scoring_mode == "cost_bounded":
                budget = result.cost_budget
                jar_entry["eligible"] = (
                    budget is not None
                    and best_jar.undiscounted_rl_cost <= budget + 1e-9
                )
            output["best_jar_terminal"] = jar_entry
    else:
        output["selected"] = None
        output["action_trace"] = []
        output["selected_source"] = "none"
        output["selected_jar_ready"] = False

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
