"""Focused tests for toy_bellman_novelty_planner search functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    apply_planner_action,
)

# Import the bellman novelty planner module.
_SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "restaurant"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import toy_bellman_novelty_planner as bnp  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(env: RestaurantSymbolicEnv, task_type: str, **kwargs: Any) -> RestaurantTask:
    """Create a RestaurantTask with defaults from env."""
    defaults: Dict[str, Any] = {
        "task_type": task_type,
        "target_location": None,
        "target_kind": None,
        "object_name": None,
    }
    defaults.update(kwargs)
    return RestaurantTask(
        task_type=defaults["task_type"],
        target_location=defaults["target_location"],
        target_kind=defaults["target_kind"],
        object_name=defaults["object_name"],
    )


def _make_search_node(
    state: RestaurantPlannerState,
    prefix: List[Tuple[str, List[str]]],
    depth: int,
    G: float,
    undiscounted_cost: float,
    done: bool = False,
) -> bnp.SearchNode:
    return bnp.SearchNode(
        state=state.copy(),
        prefix=list(prefix),
        depth=depth,
        G=G,
        undiscounted_cost=undiscounted_cost,
        done=done,
    )


# ---------------------------------------------------------------------------
# 1. Symbolic expansion reachability
# ---------------------------------------------------------------------------

def test_expand_node_reaches_fill_jar(env: RestaurantSymbolicEnv) -> None:
    """From seed-0 state, a move→pick→move prefix exposes fill(jar_0)."""
    task = _make_task(env, "make_coffee", target_location="servingtable")
    env.set_task("make_coffee", target_location="servingtable")

    # Build a planner state three actions deep.
    ps = RestaurantPlannerState.from_env(env)
    prefix: List[Tuple[str, List[str]]] = []
    G = 0.0
    undis = 0.0
    gamma = 0.99

    # Action 1: move countertop → shelf
    apply_planner_action(ps, ("move", ["countertop", "shelf"]))
    cost1 = bnp._planner_action_rl_cost(env, "move", ["countertop", "shelf"])
    G += (-cost1)  # gamma^0 = 1
    undis += cost1
    prefix.append(("move", ["countertop", "shelf"]))

    # Action 2: pick jar_0
    apply_planner_action(ps, ("pick", ["jar_0"]))
    cost2 = bnp._planner_action_rl_cost(env, "pick", ["jar_0"])
    G += (gamma ** 1) * (-cost2)
    undis += cost2
    prefix.append(("pick", ["jar_0"]))

    # Action 3: move shelf → fountain
    apply_planner_action(ps, ("move", ["shelf", "fountain"]))
    cost3 = bnp._planner_action_rl_cost(env, "move", ["shelf", "fountain"])
    G += (gamma ** 2) * (-cost3)
    undis += cost3
    prefix.append(("move", ["shelf", "fountain"]))

    node = _make_search_node(ps, prefix, depth=3, G=G, undiscounted_cost=undis)

    # Expand.  _expand_node calls _sync_env_from_planner_state internally.
    children = bnp._expand_node(node, env, task, gamma=gamma)

    # Find the fill child.
    fill_children = [
        c for c in children
        if c.prefix[-1][0] == "fill" and c.prefix[-1][1] == ["jar_0", "fountain"]
    ]
    assert len(fill_children) == 1, f"Expected exactly 1 fill(jar_0) child, got {len(fill_children)}"

    fc = fill_children[0]
    assert fc.state.objects["jar_0"].filled_with == "water", "fill should set jar_0 filled_with='water'"
    assert fc.depth == 4
    expected_G = G + (gamma ** 3) * (-bnp._factored_rl_cost(env, "fill",
        env.object_names.index("jar_0"), env.none_location_index, env.none_object_index))
    assert abs(fc.G - expected_G) < 1e-9, f"G mismatch: {fc.G} vs {expected_G}"
    # undiscounted cost should include the fill cost
    fill_cost = bnp._factored_rl_cost(env, "fill",
        env.object_names.index("jar_0"), env.none_location_index, env.none_object_index)
    assert abs(fc.undiscounted_cost - (undis + fill_cost)) < 1e-9


def test_expand_node_includes_object2_actions(env: RestaurantSymbolicEnv) -> None:
    """At a state where refill_water is valid, object2 dimension is enumerated."""
    task = _make_task(env, "serve_water", target_location="servingtable")
    env.set_task("serve_water", target_location="servingtable")

    # Construct a planner state: agent at countertop holding cup_0,
    # jar_0 at countertop filled with water.
    ps = RestaurantPlannerState.from_env(env)
    ps.agent_location = "countertop"
    ps.holding = "cup_0"
    ps.objects["cup_0"].location = None  # held
    ps.objects["jar_0"].location = "countertop"
    ps.objects["jar_0"].filled_with = "water"

    # Sync env, build masks, assert refill_water IS valid (deliberate construction).
    bnp._sync_env_from_planner_state(env, ps)
    masks = env._build_action_masks()
    refill_idx = env.action_type_index.get("refill_water")
    assert refill_idx is not None
    assert masks["valid_action_type_mask"][refill_idx] > 0, (
        "refill_water must be valid: agent holding cup_0 at countertop, "
        "jar_0 at countertop filled with water"
    )

    node = _make_search_node(ps, [], depth=0, G=0.0, undiscounted_cost=0.0)
    children = bnp._expand_node(node, env, task, gamma=0.99)

    refill_kids = [c for c in children if c.prefix[-1][0] == "refill_water"]
    assert len(refill_kids) > 0, "Expected at least one refill_water child"

    # Every refill_water child should have jar_0 as object2.
    for c in refill_kids:
        action_name, action_args = c.prefix[-1]
        assert action_name == "refill_water"
        assert action_args[0] == "cup_0"
        assert action_args[2] == "jar_0", f"Expected jar_0 as obj2, got {action_args}"


# ---------------------------------------------------------------------------
# 2. _terminal_from_node completion semantics
# ---------------------------------------------------------------------------

def test_terminal_from_node_coffee(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TerminalCandidate computes G_complete and score correctly; coffee consumed once."""
    gamma = 0.95
    success_reward = 81.06943684690286
    v_ap_value = 42.0

    monkeypatch.setattr(bnp, "_compute_v_q_ap", lambda *a, **kw: v_ap_value)

    task = _make_task(env, "make_coffee", target_location="servingtable")

    # Set up a state where cup_0 is at servingtable with coffee and dirty.
    ps = RestaurantPlannerState.from_env(env)
    ps.objects["cup_0"].location = "servingtable"
    ps.objects["cup_0"].filled_with = "coffee"
    ps.objects["cup_0"].dirty = True

    # Pretend prefix of 2 actions: move(coffeemachine), make_coffee(cup_0)
    prefix = [("move", ["countertop", "coffeemachine"]), ("make-coffee", ["cup_0", "coffeemachine"])]
    G_prefix = -5.0   # arbitrary already-accumulated G
    undis = 6.5
    d = 2

    node = _make_search_node(ps, prefix, depth=d, G=G_prefix, undiscounted_cost=undis)
    future_tasks: List[Tuple[RestaurantTask, float]] = [(task, 1.0)]

    result = bnp._terminal_from_node(
        node, env, task, future_tasks, gamma=gamma, success_reward=success_reward,
        model=None, device=None, v_ap_cache={}, source="search",
    )

    # G_complete = node.G + gamma^(d-1) * success_reward
    expected_G_complete = G_prefix + (gamma ** (d - 1)) * success_reward
    assert abs(result.G_complete - expected_G_complete) < 1e-9

    # score = G_complete + gamma^d * V_AP
    expected_score = expected_G_complete + (gamma ** d) * v_ap_value
    assert abs(result.terminal_score - expected_score) < 1e-9

    # Coffee consumed exactly once: cup_0 filled_with should be None.
    assert result.state.objects["cup_0"].filled_with is None, "Coffee should be consumed"
    # Undiscounted cost preserved.
    assert abs(result.undiscounted_rl_cost - undis) < 1e-9
    assert result.source == "search"
    assert result.depth == d


def test_terminal_from_node_jar_ready_diagnostic(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """jar_ready flag reflects jar at coffeemachine with water."""
    monkeypatch.setattr(bnp, "_compute_v_q_ap", lambda *a, **kw: 0.0)
    task = _make_task(env, "make_coffee", target_location="servingtable")

    # State where make_coffee was just completed AND jar_0 is at coffeemachine
    # filled with water.
    ps = RestaurantPlannerState.from_env(env)
    ps.objects["cup_0"].location = "servingtable"
    ps.objects["cup_0"].filled_with = "coffee"
    ps.objects["cup_0"].dirty = True
    ps.objects["jar_0"].location = "coffeemachine"
    ps.objects["jar_0"].filled_with = "water"

    node = _make_search_node(ps, [("make-coffee", ["cup_0", "coffeemachine"])], depth=1,
                              G=-1.0, undiscounted_cost=1.0)
    future_tasks: List[Tuple[RestaurantTask, float]] = []

    result = bnp._terminal_from_node(
        node, env, task, future_tasks, gamma=0.95, success_reward=81.0,
        model=None, device=None, v_ap_cache={}, source="search",
    )
    assert result.jar_ready is True, "jar_0 at coffeemachine with water => jar_ready"


# ---------------------------------------------------------------------------
# 3. Search terminates when heaps exhaust
# ---------------------------------------------------------------------------

def test_search_terminates_on_heap_exhaustion(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No infinite loop when all heaps drain and no fallback exists."""
    monkeypatch.setattr(bnp, "_myopic_fallback", lambda *a, **kw: None)
    monkeypatch.setattr(bnp, "_expand_node", lambda node, env, task, gamma: [])
    # Prevent _compute_max_q_single from needing a real model.
    monkeypatch.setattr(bnp, "_compute_max_q_single", lambda *a, **kw: -10.0)

    task = _make_task(env, "make_coffee", target_location="servingtable")
    ps = RestaurantPlannerState.from_env(env)

    result = bnp.search_task(
        env=env, init_state=ps, task=task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.99, success_reward=81.0,
        max_depth=20, max_expansions=100,
        planner_path=Path("nonexistent"), domain_path=Path("nonexistent"),
        alias="none", fd_timeout_s=1.0,
    )

    assert result.selected is None, "No reachable terminals => None"
    assert result.expansions <= 1, "At most root expanded; empty children"
    assert len(result.all_terminals) == 0
    assert result.selected_source == "none"


# ---------------------------------------------------------------------------
# 4. Novelty lane expansion under the real scheduler
# ---------------------------------------------------------------------------

def test_novelty_lane_expands(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Novelty lane gets at least one expansion after bellman phases 0–2."""
    monkeypatch.setattr(bnp, "_myopic_fallback", lambda *a, **kw: None)
    monkeypatch.setattr(bnp, "_compute_v_q_ap", lambda *a, **kw: -10.0)
    # Uniform Q so bellman and novelty ordering diverge; novelty finds unexpanded nodes.
    monkeypatch.setattr(bnp, "_compute_max_q_single", lambda *a, **kw: -10.0)

    task = _make_task(env, "make_coffee", target_location="servingtable")
    env.set_task("make_coffee", target_location="servingtable")
    ps = RestaurantPlannerState.from_env(env)

    # max_expansions=50 gives enough rounds for the novelty lane to both expand
    # novel nodes and encounter stale pops (atoms seen by prior bellman expansions).
    result = bnp.search_task(
        env=env, init_state=ps, task=task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.99, success_reward=81.0,
        max_depth=15, max_expansions=50,
        planner_path=Path("nonexistent"), domain_path=Path("nonexistent"),
        alias="none", fd_timeout_s=1.0,
    )

    assert result.expansions_novelty > 0, (
        f"Expected at least one novelty-lane expansion; got "
        f"bellman={result.expansions_bellman}, novelty={result.expansions_novelty}"
    )
    assert result.expansions >= 2, "Should expand root + at least one child"
    assert result.stale_novelty_pops > 0, (
        f"Expected stale novelty pops; atoms seen by prior novelty expansions "
        f"should render some queued novelty nodes non-novel. "
        f"stale={result.stale_novelty_pops}"
    )


# ---------------------------------------------------------------------------
# 4b. _compute_novelty_width discrimination
# ---------------------------------------------------------------------------

def test_compute_novelty_width() -> None:
    """Width 1 with unseen atom; 2 when atoms seen but pair unseen; None when both seen."""
    a1 = ("agent_at", "countertop")
    a2 = ("agent_at", "shelf")
    a3 = ("holding", "cup_0")
    atoms = [a1, a2, a3]
    pairs = bnp._atom_pairs(atoms)  # (a1,a2), (a1,a3), (a2,a3)

    # Case 1: unseen atom → width 1.
    assert bnp._compute_novelty_width(atoms, pairs, set(), set()) == 1

    # Case 2: all atoms seen, but a pair is unseen → width 2.
    seen_atoms = {a1, a2, a3}
    # seen_pairs only has (a1,a2); (a1,a3) and (a2,a3) are unseen.
    seen_pairs = {(a1, a2)}
    assert bnp._compute_novelty_width(atoms, pairs, seen_atoms, seen_pairs) == 2

    # Case 3: all atoms and all pairs seen → None (not novel).
    seen_pairs_all = set(pairs)
    assert bnp._compute_novelty_width(atoms, pairs, seen_atoms, seen_pairs_all) is None


# ---------------------------------------------------------------------------
# 5. Queue ledger: dominated lower-G (state_sig, depth) is not expanded
# ---------------------------------------------------------------------------

def test_ledger_dominates_lower_g(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two paths to same (state_sig, depth): only the higher-G one is expanded."""
    monkeypatch.setattr(bnp, "_myopic_fallback", lambda *a, **kw: None)
    monkeypatch.setattr(bnp, "_compute_v_q_ap", lambda *a, **kw: 0.0)
    monkeypatch.setattr(bnp, "_compute_max_q_single", lambda *a, **kw: -10.0)

    task = _make_task(env, "make_coffee", target_location="servingtable")
    ps_root = RestaurantPlannerState.from_env(env)

    # Build a second planner state (agent moved to shelf, nothing else changed).
    ps_s1 = ps_root.copy()
    ps_s1.agent_location = "shelf"

    root_sig = bnp._state_signature(ps_root)

    def _controlled_expand(node, env, task, gamma):
        if bnp._state_signature(node.state) == root_sig and node.depth == 0:
            c1 = bnp.SearchNode(
                state=ps_s1.copy(), prefix=[("move", ["countertop", "shelf"])],
                depth=1, G=-1.0, undiscounted_cost=0.25, done=False,
            )
            c2 = bnp.SearchNode(
                state=ps_s1.copy(), prefix=[("move", ["countertop", "coffeemachine"])],
                depth=1, G=-5.0, undiscounted_cost=5.0, done=False,
            )
            return [c1, c2]
        return []

    monkeypatch.setattr(bnp, "_expand_node", _controlled_expand)

    result = bnp.search_task(
        env=env, init_state=ps_root, task=task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.99, success_reward=81.0,
        max_depth=20, max_expansions=10,
        planner_path=Path("nonexistent"), domain_path=Path("nonexistent"),
        alias="none", fd_timeout_s=1.0,
    )

    # Root expanded (1 bellman) + c1 expanded (phase 1 bellman) = 2 expansions.
    # c2 was dominated (same state_sig, lower G) and never pushed — no 3rd expansion.
    assert result.expansions == 2, (
        f"Expected 2 expansions (root + better-G child); got {result.expansions}"
    )


# ---------------------------------------------------------------------------
# 6. Real-expansion jar-setup discovery (NOT a ready-jar start)
# ---------------------------------------------------------------------------

def test_search_discovers_jar_setup(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From root with jar_0 empty at shelf, search discovers fill + place at coffeemachine.

    This is a genuine preparation-discovery test — jar_0 starts empty at shelf
    (NOT already water-filled at the target).  Guided Q/V_AP stubs reward
    progressive jar-preparedness strongly enough to steer the search through
    pick→move to fountain→fill→move to coffeemachine→place.  The task is
    pick_place so completion depends only on jar location, not water content;
    the stubs make water-filling part of the discovered plan.
    """
    # Task: pick_place jar_0 to coffeemachine.  Once jar_0 is placed there filled
    # with water, the task completes and _jar_ready returns True.
    task = _make_task(env, "pick_place", object_name="jar_0", target_location="coffeemachine")
    env.set_task("pick_place", object_name="jar_0", target_location="coffeemachine")
    ps = RestaurantPlannerState.from_env(env)

    # No myopic fallback — all terminals must come from search.
    monkeypatch.setattr(bnp, "_myopic_fallback", lambda *a, **kw: None)

    # Guided Q: reward progressive jar-preparedness.
    def _guided_max_q(state, _env, _task, *, model, device, cache):
        jar = state.objects.get("jar_0")
        if jar is None:
            return -10.0
        if jar.filled_with == "water" and jar.location == "coffeemachine":
            return 50.0
        if jar.filled_with == "water":
            return 20.0
        if jar.location is None:  # held
            return 5.0
        return -10.0

    monkeypatch.setattr(bnp, "_compute_max_q_single", _guided_max_q)

    # Guided V_AP: heavily reward terminal states where jar is water-ready at coffeemachine.
    def _guided_v_ap(state, _env, _future_tasks, *, model, device, cache):
        jar = state.objects.get("jar_0")
        if jar is not None and jar.filled_with == "water" and jar.location == "coffeemachine":
            return 100.0
        return 0.0

    monkeypatch.setattr(bnp, "_compute_v_q_ap", _guided_v_ap)

    result = bnp.search_task(
        env=env, init_state=ps, task=task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.99, success_reward=81.0,
        max_depth=15, max_expansions=300,
        planner_path=Path("nonexistent"), domain_path=Path("nonexistent"),
        alias="none", fd_timeout_s=1.0,
    )

    assert result.selected is not None, "Must find a terminal"
    assert result.selected.source == "search", "Terminal must come from search, not myopic"
    assert result.generated_jar_ready, "At least one search terminal must be jar-ready"
    assert result.selected_jar_ready, "Selected terminal must be jar-ready"

    # The selected terminal consumed the delivery (jar placed at coffeemachine).
    sel = result.selected
    jar = sel.state.objects["jar_0"]
    assert jar.filled_with == "water", "Jar must be water-filled in terminal state"
    assert jar.location == "coffeemachine", "Jar must be at coffeemachine in terminal state"


# ---------------------------------------------------------------------------
# 7. Terminal ordering: score > cost > depth
# ---------------------------------------------------------------------------

def test_terminal_candidate_ordering() -> None:
    """Higher score wins; tiebreak by lower undiscounted cost, then fewer actions."""
    _dummy_state = RestaurantPlannerState(agent_location="", holding=None, objects={})

    def make_candidate(score: float, cost: float, depth: int) -> bnp.TerminalCandidate:
        return bnp.TerminalCandidate(
            state=_dummy_state,
            prefix=[],
            depth=depth,
            G_complete=score - 10.0,  # dummy
            terminal_score=score,
            undiscounted_rl_cost=cost,
            v_ap=0.0,
            source="search",
            jar_ready=False,
        )

    a = make_candidate(100.0, 50.0, 5)
    b = make_candidate(200.0, 100.0, 10)
    c = make_candidate(200.0, 50.0, 10)
    d = make_candidate(200.0, 50.0, 3)

    candidates = [a, b, c, d]
    candidates.sort()

    # Sorted: highest score first. Among equal score, lower cost. Among equal cost, fewer actions.
    expected_order = [
        (200.0, 50.0, 3),   # d — best: highest score, lowest cost, fewest actions
        (200.0, 50.0, 10),  # c
        (200.0, 100.0, 10), # b
        (100.0, 50.0, 5),   # a
    ]
    actual = [(t.terminal_score, t.undiscounted_rl_cost, t.depth) for t in candidates]
    assert actual == expected_order, f"Order wrong: {actual}"


# ---------------------------------------------------------------------------
# 8. Task-boundary scoring formula
# ---------------------------------------------------------------------------

def test_terminal_from_node_task_boundary_formula(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scoring_mode="task_boundary" uses -undiscounted_rl_cost + task_gamma * V_AP."""
    v_ap_value = 75.0
    task_gamma = 0.9
    gamma = 0.95

    monkeypatch.setattr(bnp, "_compute_v_q_ap", lambda *a, **kw: v_ap_value)

    task = _make_task(env, "make_coffee", target_location="servingtable")

    ps = RestaurantPlannerState.from_env(env)
    ps.objects["cup_0"].location = "servingtable"
    ps.objects["cup_0"].filled_with = "coffee"
    ps.objects["cup_0"].dirty = True

    prefix = [("move", ["countertop", "coffeemachine"]), ("make-coffee", ["cup_0", "coffeemachine"])]
    undis = 7.5   # undiscounted RL cost sum
    G_prefix = -5.0
    d = 2

    node = _make_search_node(ps, prefix, depth=d, G=G_prefix, undiscounted_cost=undis)
    future_tasks: List[Tuple[RestaurantTask, float]] = [(task, 1.0)]

    result = bnp._terminal_from_node(
        node, env, task, future_tasks, gamma=gamma, success_reward=81.0,
        model=None, device=None, v_ap_cache={}, source="search",
        scoring_mode="task_boundary", task_gamma=task_gamma,
    )

    expected_score = -undis + task_gamma * v_ap_value
    assert abs(result.terminal_score - expected_score) < 1e-9, (
        f"task_boundary score mismatch: {result.terminal_score} vs {expected_score}"
    )
    assert abs(result.v_ap - v_ap_value) < 1e-9, "v_ap must be stored"
    assert abs(result.undiscounted_rl_cost - undis) < 1e-9


# ---------------------------------------------------------------------------
# 9. Scoring-mode preference reversal
# ---------------------------------------------------------------------------

def test_scoring_mode_preference_reversal() -> None:
    """Bellman prefers shorter lower-V_AP plan; task_boundary prefers higher-V_AP jar plan."""
    _dummy_state = RestaurantPlannerState(agent_location="", holding=None, objects={})
    gamma = 0.95
    task_gamma = 0.95

    # Candidate A: short plan, low V_AP.
    A = bnp.TerminalCandidate(
        state=_dummy_state,
        prefix=[],
        depth=1,
        G_complete=100.0,
        terminal_score=0.0,   # placeholder, recomputed below
        undiscounted_rl_cost=2.0,
        v_ap=20.0,
        source="search",
        jar_ready=False,
    )
    # Candidate B: long plan, high V_AP (e.g. jar-prepared).
    B = bnp.TerminalCandidate(
        state=_dummy_state,
        prefix=[],
        depth=8,
        G_complete=30.0,
        terminal_score=0.0,
        undiscounted_rl_cost=15.0,
        v_ap=80.0,
        source="search",
        jar_ready=True,
    )

    bellman_A = A.G_complete + (gamma ** A.depth) * A.v_ap
    bellman_B = B.G_complete + (gamma ** B.depth) * B.v_ap
    assert bellman_A > bellman_B, f"Bellman must prefer A: {bellman_A} > {bellman_B}"

    tb_A = -A.undiscounted_rl_cost + task_gamma * A.v_ap
    tb_B = -B.undiscounted_rl_cost + task_gamma * B.v_ap
    assert tb_B > tb_A, f"Task-boundary must prefer B: {tb_B} > {tb_A}"


# ---------------------------------------------------------------------------
# 10. _select_terminal — cost_bounded mode
# ---------------------------------------------------------------------------

def test_select_terminal_cost_bounded_fixed_candidates() -> None:
    """cost_bounded: jar selected (cost 25.5 <= 28.125), runaway (cost 30) excluded."""
    _dummy = RestaurantPlannerState(agent_location="", holding=None, objects={})

    def _c(source: str, cost: float, v_ap: float, depth: int, jar: bool = False) -> bnp.TerminalCandidate:
        return bnp.TerminalCandidate(
            state=_dummy, prefix=[], depth=depth,
            G_complete=v_ap - cost, terminal_score=v_ap,
            undiscounted_rl_cost=cost, v_ap=v_ap,
            source=source, jar_ready=jar,
        )

    myopic = _c("myopic", cost=22.5, v_ap=10.0, depth=4)
    jar = _c("search", cost=25.5, v_ap=80.0, depth=10, jar=True)
    runaway = _c("search", cost=30.0, v_ap=90.0, depth=12)

    terminals = [myopic, jar, runaway]
    selected, ref_cost, budget, eligible = bnp._select_terminal(
        terminals, scoring_mode="cost_bounded", cost_ratio=1.25,
    )

    assert ref_cost == 22.5, f"reference cost from myopic: {ref_cost}"
    assert abs(budget - 28.125) < 1e-9, f"budget: {budget}"
    assert eligible == 2, f"eligible count: {eligible}"
    assert selected is not None
    assert selected is jar, f"Expected jar selected, got source={selected.source} cost={selected.undiscounted_rl_cost}"


def test_select_terminal_cost_bounded_no_fallback() -> None:
    """When no myopic fallback exists, reference is min candidate cost."""
    _dummy = RestaurantPlannerState(agent_location="", holding=None, objects={})

    def _c(cost: float, v_ap: float, depth: int) -> bnp.TerminalCandidate:
        return bnp.TerminalCandidate(
            state=_dummy, prefix=[], depth=depth,
            G_complete=v_ap - cost, terminal_score=v_ap,
            undiscounted_rl_cost=cost, v_ap=v_ap,
            source="search", jar_ready=False,
        )

    t1 = _c(cost=10.0, v_ap=5.0, depth=3)
    t2 = _c(cost=15.0, v_ap=80.0, depth=8)
    t3 = _c(cost=20.0, v_ap=90.0, depth=12)

    terminals = [t1, t2, t3]
    selected, ref_cost, budget, eligible = bnp._select_terminal(
        terminals, scoring_mode="cost_bounded", cost_ratio=1.25,
    )

    assert ref_cost == 10.0  # min across all terminals
    assert abs(budget - 12.5) < 1e-9
    assert eligible == 1  # only t1 (10 <= 12.5)
    assert selected is t1


def test_select_terminal_cost_bounded_invalid_ratio() -> None:
    """cost_ratio < 1.0 is rejected by search_task (value error)."""
    with pytest.raises(ValueError, match="cost_ratio"):
        bnp.search_task(
            env=None, init_state=None, task=None,  # type: ignore[arg-type]
            model=None, device=None,  # type: ignore[arg-type]
            gamma=0.99, success_reward=81.0,
            max_depth=20, max_expansions=100,
            scoring_mode="cost_bounded", cost_ratio=0.8,
            planner_path=Path("nonexistent"), domain_path=Path("nonexistent"),
            alias="none", fd_timeout_s=1.0,
        )


def test_best_jar_terminal_is_not_insertion_order() -> None:
    state = RestaurantPlannerState(agent_location="", holding=None, objects={})
    lower = bnp.TerminalCandidate(
        state=state, prefix=[], depth=4, G_complete=0.0,
        terminal_score=10.0, undiscounted_rl_cost=5.0, v_ap=10.0,
        source="search", jar_ready=True,
    )
    higher = bnp.TerminalCandidate(
        state=state, prefix=[], depth=5, G_complete=0.0,
        terminal_score=20.0, undiscounted_rl_cost=6.0, v_ap=20.0,
        source="search", jar_ready=True,
    )

    assert bnp._best_jar_terminal([lower, higher]) is higher
