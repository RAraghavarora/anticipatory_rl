"""Tests for the unbounded-jar steelman augmentation (task 1 of
gnn-steelman-jar-augmentation).

`_generate_focused_augmentations`'s bounded-region rule cannot propose "fetch
a jar from a distant pantry, fill it, park it near a consumer" because the
jar's location and the consumer are never both within one hop of the myopic
plan's path. The `unbounded_jar` flag hands the baseline that candidate
anyway (fill AND relocate, conjunctive, since the jar starts empty), so a
later refusal is attributable to the value horizon, not to candidate
coverage. With the flag off, behaviour must be unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_GNN_DIR = _REPO_ROOT / "scripts" / "gnn"
for _p in (_REPO_ROOT, _SCRIPT_GNN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_sequence import (  # noqa: E402
    _generate_focused_augmentations,
    _p_add_is_satisfied,
)

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    RestaurantTask,
    build_restaurant_problem_text,
)


def _make_env() -> RestaurantSymbolicEnv:
    env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_5.yaml")
    env.reset(seed=0)
    return env


def test_flag_off_no_clause_mentions_jar():
    """toy_level_5 jar_0 starts at the far pantry (distance 35 from every
    other location), so with an empty myopic prefix rooted at the agent's
    start location, the jar is never in the bounded region and no clause
    (clean/fill/jar_position) mentions it. This is the observable stand-in
    for "the flag-off clause list is unchanged by the patch"."""
    env = _make_env()
    state = RestaurantPlannerState.from_env(env)
    assert state.objects["jar_0"].location == "pantry"

    clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env,
    )
    assert all(c.object_name != "jar_0" for c in clauses)


def test_flag_on_emits_conjunctive_jar_prepared_clause():
    env = _make_env()
    state = RestaurantPlannerState.from_env(env)

    clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env,
        unbounded_jar=True,
    )
    jar_clauses = [c for c in clauses if c.clause_type == "jar_prepared"]
    assert jar_clauses, "expected >=1 jar_prepared clause with the flag on"

    for c in jar_clauses:
        assert c.object_name == "jar_0"
        assert "filled-with water" in c.pddl_clause
        assert "is-at" in c.pddl_clause

    expected_targets = {
        loc for loc in env.locations if env._is_location(loc, "coffeemachine")
    } | set(env.service_locations)
    assert {c.target_location for c in jar_clauses} == expected_targets

    # Reused by _generate_focused_augmentations's _emit for dedup: no duplicate
    # pddl_clause strings among the jar_prepared clauses.
    assert len({c.pddl_clause for c in jar_clauses}) == len(jar_clauses)

    task = RestaurantTask(task_type="serve_water", target_location="servingtable")
    problem_text = build_restaurant_problem_text(
        env, state, task, extra_goal_clauses=[jar_clauses[0].pddl_clause],
    )
    goal_block = problem_text[problem_text.index("(:goal"):]
    assert f"(filled-with water {jar_clauses[0].object_name})" in goal_block
    assert (
        f"(is-at {jar_clauses[0].object_name} {jar_clauses[0].target_location})"
        in goal_block
    )


def test_flag_default_is_false():
    """Backward compatibility for callers in generate_data_aug.py /
    diagnose_candidates.py that don't pass unbounded_jar at all."""
    env = _make_env()
    state = RestaurantPlannerState.from_env(env)

    default_clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env,
    )
    explicit_off_clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env, unbounded_jar=False,
    )
    assert [c.pddl_clause for c in default_clauses] == [
        c.pddl_clause for c in explicit_off_clauses
    ]
    # Equality alone is vacuous (both calls take the same branch): assert the
    # default really is the flag-off branch.
    assert all(c.clause_type != "jar_prepared" for c in default_clauses)


def test_jar_prepared_clause_is_admitted_at_a_satisfying_terminal():
    """A jar_prepared clause must VERIFY, not just be generated.

    `_evaluate_augmented_plan` rejects any candidate whose clause fails
    `_p_add_is_satisfied`, and `run_sequence` swallows that rejection with
    `except ValueError: continue` — so a missing branch silently drops every
    steelman candidate before it is ever scored, producing output identical to
    a genuine decline. No Fast Downward needed: the terminal state is
    fabricated directly.
    """
    env = _make_env()
    state = RestaurantPlannerState.from_env(env)
    clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env, unbounded_jar=True,
    )
    clause = next(c for c in clauses if c.clause_type == "jar_prepared")

    # Terminal state a successful augmented plan would reach: jar filled with
    # water AND standing at the clause's consumer location.
    terminal = state.copy()
    jar = terminal.objects[clause.object_name]
    jar.filled_with = "water"
    jar.location = clause.target_location
    assert _p_add_is_satisfied(terminal, clause) is True

    # Both conjuncts are required: neither half alone satisfies the clause.
    only_filled = state.copy()
    only_filled.objects[clause.object_name].filled_with = "water"
    assert _p_add_is_satisfied(only_filled, clause) is False

    only_moved = state.copy()
    only_moved.objects[clause.object_name].location = clause.target_location
    assert _p_add_is_satisfied(only_moved, clause) is False


def test_jar_prepared_clauses_survive_the_max_augs_cap():
    """Callers truncate with `clauses[: max_augs]`; if the steelman clauses are
    appended they are cut first at exactly the mid-chain states where they
    matter, reproducing the same false 'declined' conclusion."""
    env = _make_env()
    state = RestaurantPlannerState.from_env(env)
    clauses = _generate_focused_augmentations(
        [], state, env.state.agent_location, env, unbounded_jar=True,
    )
    n_jar = sum(1 for c in clauses if c.clause_type == "jar_prepared")
    assert n_jar > 0
    assert all(c.clause_type == "jar_prepared" for c in clauses[:n_jar]), (
        "jar_prepared clauses must be prepended so --max-augs cannot cut them"
    )
