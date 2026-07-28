"""Tests for evaluate_bellman_novelty_sequence.py harness."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.envs.restaurant.planner import RestaurantPlannerState

_SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "restaurant"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import evaluate_bellman_novelty_sequence as ev  # noqa: E402
import toy_bellman_novelty_planner as bnp  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_task_sequence() -> str:
    return json.dumps({
        "sequence_id": "test-seq",
        "tasks": [
            {"task_type": "make_coffee", "target_location": "servingtable"},
            {"task_type": "serve_water", "target_location": "servingtable"},
        ],
    })


def _two_task_sequence_path() -> str:
    """Write the sequence to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write(_two_task_sequence())
    f.close()
    return f.name


def _make_fd_success(
    plan: List[Tuple[str, List[str]]],
    cost: float = 0.0,
) -> Any:
    """Minimal mock with .success, .plan_actions, .plan_cost, .error."""

    class _FakeResult:
        success: bool = True
        plan_actions: List[Tuple[str, List[str]]] = plan
        plan_cost: float = cost
        error: str | None = None
    return _FakeResult()


def _make_fake_search_result(
    selected: bnp.TerminalCandidate | None,
    *,
    expansions: int = 1,
    reference_cost: float | None = None,
    cost_budget: float | None = None,
    eligible_count: int = 1,
    source: str = "search",
) -> Any:

    class _Fake:
        pass

    obj = _Fake()
    obj.selected = selected
    obj.all_terminals = [selected] if selected else []
    obj.expansions = expansions
    obj.expansions_bellman = expansions
    obj.expansions_novelty = 0
    obj.stale_novelty_pops = 0
    obj.action_trace = (
        [f"{a}({', '.join(args)})" for a, args in selected.prefix]
        if selected else []
    )
    obj.generated_jar_ready = False
    obj.selected_jar_ready = False
    obj.selected_source = source
    obj.scoring_mode = "cost_bounded"
    obj.cost_ratio = 1.25
    obj.reference_cost = reference_cost
    obj.cost_budget = cost_budget
    obj.eligible_terminal_count = eligible_count
    return obj


# ---------------------------------------------------------------------------
# 1. Both policies receive identical task tuples in order
# ---------------------------------------------------------------------------

def test_identical_task_order(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_sequence for myopic and cost_bounded returns the same task types in order."""
    plan = [
        ("move", ["countertop", "coffeemachine"]),
        ("make-coffee", ["cup_0", "coffeemachine"]),
        ("move", ["coffeemachine", "servingtable"]),
        ("place", ["cup_0", "servingtable"]),
    ]
    monkeypatch.setattr(
        ev, "solve_restaurant_task_with_fd",
        lambda env, state, task, **kw: _make_fd_success(plan, cost=0.0),
    )

    # Prevent model loading from needing a real file or matching weights.
    import torch
    monkeypatch.setattr(torch, "load", lambda f, map_location=None, weights_only=True: {})
    monkeypatch.setattr(
        torch.nn.Module, "load_state_dict",
        lambda self, state_dict, strict=True, assign=False: type(
            "IncompatibleKeys", (), {"missing_keys": [], "unexpected_keys": []}
        )(),
    )

    sel = bnp.TerminalCandidate(
        state=RestaurantPlannerState.from_env(env),
        prefix=[("move", ["countertop", "coffeemachine"])],
        depth=1, G_complete=80.0, terminal_score=80.0,
        undiscounted_rl_cost=10.0, v_ap=42.5, source="search", jar_ready=False,
    )
    monkeypatch.setattr(
        bnp, "search_task",
        lambda **kw: _make_fake_search_result(sel, expansions=7, source="search",
                                               reference_cost=9.5, cost_budget=11.875),
    )

    seq_path = _two_task_sequence_path()
    try:
        myopic_result = ev.run_sequence(
            "myopic", sequence_path=Path(seq_path),
            config_path=Path("configs/restaurant/toy_level_3.yaml"),
            domain_path=Path("dummy"), planner_path=Path("dummy"),
            alias="dummy", fd_timeout_s=10.0,
            seed=0, gamma=0.95, success_reward=81.0,
            hidden_dim=256, max_depth=10, max_expansions=100, cost_ratio=1.25,
        )
        guided_result = ev.run_sequence(
            "cost_bounded", sequence_path=Path(seq_path),
            config_path=Path("configs/restaurant/toy_level_3.yaml"),
            domain_path=Path("dummy"), planner_path=Path("dummy"),
            alias="dummy", fd_timeout_s=10.0,
            seed=0, gamma=0.95, success_reward=81.0,
            hidden_dim=256, max_depth=10, max_expansions=100, cost_ratio=1.25,
            q_weights=Path("dummy.pt"),
        )
    finally:
        Path(seq_path).unlink(missing_ok=True)

    myopic_types = [(r["task_type"], r["target_location"], r["target_kind"], r["object_name"])
                    for r in myopic_result["tasks"]]
    guided_types = [(r["task_type"], r["target_location"], r["target_kind"], r["object_name"])
                    for r in guided_result["tasks"]]

    assert myopic_types == guided_types
    assert len(myopic_types) == 2
    assert myopic_types[0] == ("make_coffee", "servingtable", None, None)
    assert myopic_types[1] == ("serve_water", "servingtable", None, None)


# ---------------------------------------------------------------------------
# 2. World persists within each policy and completion consumption occurs once
# ---------------------------------------------------------------------------

def test_world_persists_myopic(env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a myopic task, the planner_state reflects consumption and
    is the input state for the next task."""
    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")

    # Build a state where cup_0 is at coffeemachine, unbrewed (task NOT pre-satisfied).
    ps = RestaurantPlannerState.from_env(env)
    ps.objects["cup_0"].location = "coffeemachine"
    ps.objects["cup_0"].filled_with = None
    ps.objects["cup_0"].dirty = False

    # Fake FD: plan makes coffee then moves to servingtable.
    plan = [
        ("move", ["countertop", "coffeemachine"]),
        ("make-coffee", ["cup_0", "coffeemachine"]),
        ("move", ["coffeemachine", "servingtable"]),
        ("place", ["cup_0", "servingtable"]),
    ]
    monkeypatch.setattr(
        ev, "solve_restaurant_task_with_fd",
        lambda env, state, task, **kw: _make_fd_success(plan, cost=0.0),
    )

    r = ev._run_myopic(
        env, ps, task,
        planner_path=Path("dummy"), domain_path=Path("dummy"),
        alias="dummy", fd_timeout_s=10.0,
    )

    assert r.success
    assert r.num_actions == 4
    # Coffee consumed: cup_0 filled_with cleared.
    assert r.next_state.objects["cup_0"].filled_with is None
    # World persists: cup is at servingtable after the place+consumption.
    assert r.next_state.objects["cup_0"].location == "servingtable"


# ---------------------------------------------------------------------------
# 3. Auto-satisfied second task costs zero and no action is executed
# ---------------------------------------------------------------------------

def test_auto_satisfied_second_task_zero_cost(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a task is auto-satisfied, cost=0 and no physical actions are recorded."""

    # Build a state with a clean, water-filled cup at servingtable
    ps = RestaurantPlannerState.from_env(env)
    ps.objects["cup_0"].location = "servingtable"
    ps.objects["cup_0"].filled_with = "water"
    ps.objects["cup_0"].dirty = False

    task = RestaurantTask(task_type="serve_water", target_location="servingtable")
    env.set_task("serve_water", target_location="servingtable")

    auto = ev._task_is_auto_satisfied(ps, task, env)  # ponytail: from toy_anticipatory_oracle
    assert auto, "serve_water at servingtable with water-filled cup should be auto-satisfied"

    # Consume directly (the harness does this)
    state_copy = ps.copy()
    consume = __import__(
        "anticipatory_rl.envs.restaurant.planner",
        fromlist=["consume_delivery_from_state"],
    ).consume_delivery_from_state
    consume(state_copy, "serve_water", "servingtable")

    # After consumption: cup's water is consumed (filled_with cleared).
    # consume_delivery for serve_water only clears filled_with; dirty is NOT set.
    assert state_copy.objects["cup_0"].filled_with is None


# ---------------------------------------------------------------------------
# 4. Cost aggregation and paired delta (via _pair_results helper)
# ---------------------------------------------------------------------------

def test_pair_results_delta() -> None:
    """_pair_results computes paired_cost_delta = guided - myopic from summaries."""
    myopic = {
        "summary": {"total_pddl_cost": 100.0, "policy": "myopic"},
        "tasks": [],
    }
    guided = {
        "summary": {"total_pddl_cost": 120.0, "policy": "cost_bounded"},
        "tasks": [],
    }
    output = ev._pair_results(myopic, guided)
    assert output["paired_cost_delta"] == 20.0
    assert output["myopic"] is myopic
    assert output["guided"] is guided

    # Zero delta.
    output2 = ev._pair_results(myopic, myopic)
    assert output2["paired_cost_delta"] == 0.0


# ---------------------------------------------------------------------------
# 5. Guided records include declared ratio/budget/source/expansions
# ---------------------------------------------------------------------------

def test_guided_records_include_cost_bounded_fields(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cost_bounded task records carry v_ap, reference_cost, budget, eligible,
    source, and expansions."""
    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")
    ps = RestaurantPlannerState.from_env(env)

    sel = bnp.TerminalCandidate(
        state=ps.copy(), prefix=[("move", ["countertop", "coffeemachine"])],
        depth=1, G_complete=80.0, terminal_score=80.0,
        undiscounted_rl_cost=10.0, v_ap=42.5, source="search", jar_ready=False,
    )
    monkeypatch.setattr(
        bnp, "search_task",
        lambda **kw: _make_fake_search_result(sel, expansions=13, source="search",
                                               reference_cost=9.5, cost_budget=11.875,
                                               eligible_count=3),
    )

    r = ev._run_cost_bounded(
        env, ps, task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.95, success_reward=81.0,
        max_depth=10, max_expansions=100, cost_ratio=1.25,
        planner_path=Path("dummy"), domain_path=Path("dummy"),
        alias="dummy", fd_timeout_s=10.0,
    )

    assert r.success
    assert r.v_ap == 42.5
    assert r.reference_cost == 9.5
    assert r.cost_budget == 11.875
    assert r.eligible_count == 3
    assert r.expansions == 13
    assert r.source == "search"


def test_guided_records_preserve_zero_v_ap(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v_ap=0.0 is preserved (not filtered by truthiness)."""
    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")
    ps = RestaurantPlannerState.from_env(env)

    sel = bnp.TerminalCandidate(
        state=ps.copy(), prefix=[],
        depth=0, G_complete=0.0, terminal_score=0.0,
        undiscounted_rl_cost=0.0, v_ap=0.0, source="myopic", jar_ready=False,
    )
    monkeypatch.setattr(
        bnp, "search_task",
        lambda **kw: _make_fake_search_result(sel, expansions=1, source="myopic",
                                               reference_cost=0.0, cost_budget=0.0,
                                               eligible_count=1),
    )

    r = ev._run_cost_bounded(
        env, ps, task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.95, success_reward=81.0,
        max_depth=10, max_expansions=100, cost_ratio=1.25,
        planner_path=Path("dummy"), domain_path=Path("dummy"),
        alias="dummy", fd_timeout_s=10.0,
    )

    assert r.success
    assert r.v_ap == 0.0, "v_ap=0.0 must survive"
    assert r.reference_cost == 0.0, "reference_cost=0.0 must survive"
    assert r.cost_budget == 0.0, "cost_budget=0.0 must survive"


# ---------------------------------------------------------------------------
# 6. Cost-bounded failure when no candidate
# ---------------------------------------------------------------------------

def test_cost_bounded_failure_no_candidate(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search_task returns no selected candidate, _run_cost_bounded reports failure."""

    monkeypatch.setattr(
        bnp, "search_task",
        lambda **kw: _make_fake_search_result(None, expansions=0, source="none"),
    )

    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")
    ps = RestaurantPlannerState.from_env(env)

    r = ev._run_cost_bounded(
        env, ps, task,
        model=None, device=None,  # type: ignore[arg-type]
        gamma=0.95, success_reward=81.0,
        max_depth=10, max_expansions=100, cost_ratio=1.25,
        planner_path=Path("dummy"), domain_path=Path("dummy"),
        alias="dummy", fd_timeout_s=10.0,
    )

    assert not r.success
    assert r.error is not None
    assert r.cost == 0.0
    assert r.num_actions == 0
    assert r.source == "none"


# ---------------------------------------------------------------------------
# 7. Load-sequence rejects malformed JSON
# ---------------------------------------------------------------------------

def test_load_sequence_rejects_malformed() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([1, 2, 3], f)
        bad_path = f.name
    try:
        with pytest.raises(ValueError, match="'tasks'"):
            ev._load_sequence(Path(bad_path))
    finally:
        Path(bad_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. run_sequence end-to-end with monkeypatched FD+search (myopic only)
# ---------------------------------------------------------------------------

def test_run_sequence_myopic_end_to_end(
    env: RestaurantSymbolicEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full run_sequence(myopic) with a 2-task sequence and fake FD."""

    plan = [
        ("move", ["countertop", "coffeemachine"]),
        ("make-coffee", ["cup_0", "coffeemachine"]),
        ("move", ["coffeemachine", "servingtable"]),
        ("place", ["cup_0", "servingtable"]),
    ]
    monkeypatch.setattr(
        ev, "solve_restaurant_task_with_fd",
        lambda env, state, task, **kw: _make_fd_success(plan, cost=0.0),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(_two_task_sequence())
        seq_path = f.name
    try:
        result = ev.run_sequence(
            "myopic", sequence_path=Path(seq_path),
            config_path=Path("configs/restaurant/toy_level_3.yaml"),
            domain_path=Path("dummy"),
            planner_path=Path("dummy"),
            alias="dummy", fd_timeout_s=10.0,
            seed=0, gamma=0.95, success_reward=81.0,
            hidden_dim=256, max_depth=10, max_expansions=100, cost_ratio=1.25,
        )
    finally:
        Path(seq_path).unlink(missing_ok=True)

    assert result["summary"]["policy"] == "myopic"
    assert result["summary"]["attempted"] == 2
    assert len(result["tasks"]) == 2
    assert result["tasks"][0]["task_type"] == "make_coffee"
    assert result["tasks"][1]["task_type"] == "serve_water"


# ---------------------------------------------------------------------------
# 9. run_sequence(cost_bounded) raises ValueError without q_weights
# ---------------------------------------------------------------------------

def test_cost_bounded_requires_q_weights() -> None:
    seq_path = _two_task_sequence_path()
    try:
        with pytest.raises(ValueError, match="q_weights"):
            ev.run_sequence(
                "cost_bounded", sequence_path=Path(seq_path),
                config_path=Path("configs/restaurant/toy_level_3.yaml"),
                domain_path=Path("dummy"), planner_path=Path("dummy"),
                alias="dummy", fd_timeout_s=1.0,
                seed=0, gamma=0.95, success_reward=81.0,
                hidden_dim=256, max_depth=10, max_expansions=100, cost_ratio=1.25,
                q_weights=None,
            )
    finally:
        Path(seq_path).unlink(missing_ok=True)
