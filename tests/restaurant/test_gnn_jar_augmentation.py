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

from eval_sequence import _generate_focused_augmentations  # noqa: E402

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
    build_restaurant_problem_text(
        env, state, task, extra_goal_clauses=[jar_clauses[0].pddl_clause],
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
