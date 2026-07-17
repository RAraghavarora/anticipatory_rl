import pytest
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


def _evaluate_dqn_transition_logic(env: RestaurantSymbolicEnv, current_world_state_key: tuple, current_task_snapshot: dict, next_info: dict) -> tuple[bool, bool, bool, bool]:
    world_unchanged = env._action_mask_state_key() == current_world_state_key
    auto_success_flag = bool(next_info.get("auto_success", False))
    task_equality = next_info.get("task") == current_task_snapshot
    
    # Buggy Logic (Before Fix)
    done_buggy = bool(auto_success_flag and world_unchanged and task_equality)
    
    # Fixed Logic
    done_fixed = bool(auto_success_flag and world_unchanged)
    
    return auto_success_flag, world_unchanged, done_buggy, done_fixed


def test_multitask_degenerate_loop_terminalized():
    """Test that a multi-task (A->B->A) auto-success loop on a stagnant world is terminal."""
    env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    env.reset(seed=42)
    
    for obj in env.state.objects.values():
        if obj.kind == "cup":
            obj.location = "countertop"
            obj.dirty = False
            obj.filled_with = None
        if obj.location == "servingtable":
            obj.location = "shelf"
            
    env.set_task("wash_objects", target_kind="cup")
    env._pending_auto_success = True
    current_task_snapshot = dict(env._info(success=False).get("task", {}))
    current_world_state_key = env._action_mask_state_key()
    
    # Hack resampler to pick a DIFFERENT auto-satisfied task
    env._resample_task = lambda: env.set_task("clear_containers", target_location="servingtable")
    
    action = {"action_type": 0, "object1": 0, "location": 0, "object2": 0}
    next_obs, reward, success, truncated, next_info = env.step(action)
    
    auto_success_flag, world_unchanged, done_buggy, done_fixed = _evaluate_dqn_transition_logic(
        env, current_world_state_key, current_task_snapshot, next_info
    )
    
    assert auto_success_flag is True
    assert world_unchanged is True
    
    assert done_buggy is False  # Fails to stop the bootstrap
    assert done_fixed is True   # Successfully terminalizes the loop


def test_consumable_auto_success_bootstraps():
    """Test that a world-changing auto-success preserves the anticipatory bootstrap.
    
    This test is load-bearing on env._action_mask_state_key() accurately
    capturing object states like filled_with, ensuring we don't over-terminalize.
    """
    env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    env.reset(seed=42)
    
    for obj in env.state.objects.values():
        if obj.kind == "cup":
            obj.location = "servingtable"
            obj.dirty = False
            obj.filled_with = "coffee"
            
    env.set_task("make_coffee", target_location="servingtable")
    env._pending_auto_success = True
    current_task_snapshot = dict(env._info(success=False).get("task", {}))
    current_world_state_key = env._action_mask_state_key()
    
    env._resample_task = lambda: env.set_task("pick_place", target_location="countertop", object_name="apple_0")
    
    action = {"action_type": 0, "object1": 0, "location": 0, "object2": 0}
    next_obs, reward, success, truncated, next_info = env.step(action)
    
    auto_success_flag, world_unchanged, done_buggy, done_fixed = _evaluate_dqn_transition_logic(
        env, current_world_state_key, current_task_snapshot, next_info
    )
    
    assert auto_success_flag is True
    assert world_unchanged is False  # Coffee was consumed
    
    assert done_fixed is False  # Successfully allows bootstrapping


def test_action_driven_success_bootstraps():
    """Test that a standard action-driven task completion preserves the anticipatory bootstrap."""
    env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
    env.reset(seed=42)
    
    # Setup state for 1-step pick_place completion
    env.state.agent_location = "servingtable"
    env.state.holding = "apple_0"
    env.state.objects["apple_0"].location = "servingtable"
    env.state.objects["apple_0"].contained_in = None
    
    env.set_task("pick_place", target_location="servingtable", object_name="apple_0")
    env._pending_auto_success = False
    
    current_task_snapshot = dict(env._info(success=False).get("task", {}))
    current_world_state_key = env._action_mask_state_key()
    
    env._resample_task = lambda: env.set_task("wash_objects", target_kind="cup")
    
    # Take the 'place' action at 'servingtable'
    action = {
        "action_type": env.action_type_index["place"],
        "location": env.location_index["servingtable"],
        "object1": 0,
        "object2": 0
    }
    
    next_obs, reward, success, truncated, next_info = env.step(action)
    
    auto_success_flag, world_unchanged, done_buggy, done_fixed = _evaluate_dqn_transition_logic(
        env, current_world_state_key, current_task_snapshot, next_info
    )
    
    assert success is True
    assert auto_success_flag is False
    assert world_unchanged is False  # Action changed the world state
    
    assert done_fixed is False  # Bootstraps normally
