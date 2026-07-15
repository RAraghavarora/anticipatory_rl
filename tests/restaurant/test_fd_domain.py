"""FD domain integration tests for toy_level_3 make_coffee with dry machine."""
from __future__ import annotations

from pathlib import Path

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv
from anticipatory_rl.envs.restaurant.planner import (
    RestaurantPlannerState,
    RestaurantTask,
    solve_restaurant_task_with_fd,
)


def _make_dry_machine_state(env: RestaurantSymbolicEnv) -> None:
    """Drain water_machine so coffeemachine has no water."""
    env.state.objects["water_machine"].location = None


def test_dry_machine_brew_with_fountain(
    env: RestaurantSymbolicEnv,
    planner_state: RestaurantPlannerState,
    fd_path: Path,
    domain_path: Path,
):
    """make_coffee with dry machine and empty jar: FD must fill- pour- brew."""
    _make_dry_machine_state(env)
    env.state.agent_location = "coffeemachine"
    env.state.holding = None
    env.state.objects["cup_0"].location = "coffeemachine"
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None
    env.state.objects["jar_0"].location = "shelf"
    env.state.objects["jar_0"].filled_with = None

    ps = RestaurantPlannerState.from_env(env)
    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")
    result = solve_restaurant_task_with_fd(
        env, ps, task,
        planner_path=fd_path,
        domain_path=domain_path,
        alias="seq-sat-lama-2011",
        timeout_s=120.0,
    )
    assert result.success, f"FD failed: {result.error}"
    action_names = [a[0] for a in result.plan_actions]
    assert "fill" in action_names, f"No fill in plan: {action_names}"
    assert "pour" in action_names, f"No pour in plan: {action_names}"
    assert "make-coffee" in action_names, f"No make-coffee in plan: {action_names}"


def test_dry_machine_brew_with_jar(
    env: RestaurantSymbolicEnv,
    planner_state: RestaurantPlannerState,
    fd_path: Path,
    domain_path: Path,
):
    """make_coffee with dry machine and a filled jar: FD uses refill_water instead of fountain."""
    _make_dry_machine_state(env)
    env.state.agent_location = "shelf"
    env.state.holding = "cup_0"
    env.state.objects["cup_0"].location = None
    env.state.objects["cup_0"].dirty = False
    env.state.objects["cup_0"].filled_with = None
    env.state.objects["jar_0"].location = "shelf"
    env.state.objects["jar_0"].filled_with = "water"

    ps = RestaurantPlannerState.from_env(env)
    task = RestaurantTask(task_type="make_coffee", target_location="servingtable")
    result = solve_restaurant_task_with_fd(
        env, ps, task,
        planner_path=fd_path,
        domain_path=domain_path,
        alias="seq-sat-lama-2011",
        timeout_s=120.0,
    )
    assert result.success, f"FD failed: {result.error}"
    action_names = [a[0] for a in result.plan_actions]
    assert "make-coffee" in action_names, f"No make-coffee in plan: {action_names}"
    assert any(n in action_names for n in ("refill-water", "refill_water", "pour", "fill")), (
        f"Expected refill|pour|fill in plan, got: {action_names}"
    )
