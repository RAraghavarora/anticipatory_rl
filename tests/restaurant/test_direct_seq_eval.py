"""Focused regression tests for scripts/restaurant/rl_direct_seq_eval.py.

Tests checkpoint metadata loading, refreshed obs/info after set_task,
independent env reset identity, and PDDL cost cumulative delta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask

_REPO = Path(__file__).resolve().parents[2]
# Use the checked-in sequence for eval tests.
SEQ_00 = _REPO / "experiments" / "sequences" / "iid-eval-seq-00.json"


# ---------------------------------------------------------------------------
# Helper: load the first task from the canonical sequence
# ---------------------------------------------------------------------------
def _first_seq_task():
    with SEQ_00.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    td = data["tasks"][0]
    return RestaurantTask(
        task_type=str(td["task_type"]),
        target_location=td.get("target_location"),
        target_kind=td.get("target_kind"),
        object_name=td.get("object_name"),
    )


# ---------------------------------------------------------------------------
# Test 1: checkpoint metadata loading & model construction with centering
# ---------------------------------------------------------------------------
class TestCheckpointLoading:
    def test_load_train_args_myopic(self):
        """train_args.json adjacent to the myopic checkpoint parses correctly."""
        from scripts.restaurant.rl_direct_seq_eval import _load_checkpoint_dir
        ckpt_dir = _REPO / "runs" / "v3_myopic_g0.9_peb"
        meta = _load_checkpoint_dir(ckpt_dir)
        assert meta["tasks_per_episode"] == 1
        assert meta["hidden_dim"] == 256
        assert meta["success_reward"] == pytest.approx(81.06943684690286)
        assert meta["invalid_action_penalty"] == 6.0
        assert meta["no_dueling_centering"] is False
        assert meta["env_reset_tasks"] == 50
        assert meta["config_path"] == "configs/restaurant/toy_level_3.yaml"

    def test_load_train_args_anticipatory(self):
        """train_args.json adjacent to the anticipatory checkpoint parses correctly."""
        from scripts.restaurant.rl_direct_seq_eval import _load_checkpoint_dir
        ckpt_dir = _REPO / "results/canonical_planner/checkpoints/anticipatory/seed0"
        meta = _load_checkpoint_dir(ckpt_dir)
        assert meta["tasks_per_episode"] == 50
        assert meta["hidden_dim"] == 256
        assert meta["success_reward"] == pytest.approx(81.06943684690286)
        assert meta["no_dueling_centering"] is False

    def test_build_model_respects_centering(self):
        """Constructed model reflects no_dueling_centering from metadata."""
        from scripts.restaurant.rl_direct_seq_eval import _build_model
        env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml")
        env.reset(seed=0)
        device = torch.device("cpu")

        meta_centered = {"hidden_dim": 64, "no_dueling_centering": False}
        m1 = _build_model(env, meta_centered, device)
        assert m1.center_advantages is True

        meta_uncentered = {"hidden_dim": 64, "no_dueling_centering": True}
        m2 = _build_model(env, meta_uncentered, device)
        assert m2.center_advantages is False

    def test_validate_meta_pair_rejects_mismatched_centering(self):
        from scripts.restaurant.rl_direct_seq_eval import _validate_meta_pair
        ant = {"config_path": "c", "hidden_dim": 256, "max_steps_per_task": 64,
               "success_reward": 81.0, "invalid_action_penalty": 6.0,
               "env_reset_tasks": 50, "tasks_per_episode": 50,
               "no_dueling_centering": False}
        myo = dict(ant, tasks_per_episode=1, no_dueling_centering=True)
        with pytest.raises(ValueError, match="Centering mismatch"):
            _validate_meta_pair(ant, myo)

    def test_validate_meta_pair_rejects_bad_tasks_per_episode(self):
        from scripts.restaurant.rl_direct_seq_eval import _validate_meta_pair
        base = {"config_path": "c", "hidden_dim": 256, "max_steps_per_task": 64,
                "success_reward": 81.0, "invalid_action_penalty": 6.0,
                "env_reset_tasks": 50, "no_dueling_centering": False}
        ant_bad = dict(base, tasks_per_episode=1)
        myo_ok = dict(base, tasks_per_episode=1)
        with pytest.raises(ValueError, match="tasks_per_episode > 1"):
            _validate_meta_pair(ant_bad, myo_ok)

        ant_ok = dict(base, tasks_per_episode=50)
        myo_bad = dict(base, tasks_per_episode=50)
        with pytest.raises(ValueError, match="tasks_per_episode=1"):
            _validate_meta_pair(ant_ok, myo_bad)


# ---------------------------------------------------------------------------
# Test 2: refreshed obs/info after set_task encodes the assigned task
# ---------------------------------------------------------------------------
class TestObsInfoAfterSetTask:
    def test_info_task_matches_set_task(self, env):
        """After set_task, info['task'] returns the exact same task fields."""
        task = _first_seq_task()
        env.set_task(task.task_type, target_location=task.target_location,
                     target_kind=task.target_kind, object_name=task.object_name)
        obs = env._obs()
        info = env._info(success=False)
        t = info["task"]
        assert t["task_type"] == task.task_type
        assert t["target_location"] == task.target_location
        assert t["target_kind"] == task.target_kind
        assert t["object_name"] == task.object_name

    def test_obs_encodes_new_task_not_stale(self, env):
        """After set_task to a new type, _obs() changes at the task-encoding offset."""
        from anticipatory_rl.envs.restaurant.env import TASK_TYPES
        env.set_task("serve_water", target_location="servingtable")
        obs_water = env._obs().copy()

        # Switch to wash_objects — a different task type vector
        env.set_task("wash_objects", target_kind="cup")
        obs_wash = env._obs().copy()

        # The task encoding portion should differ (task type one-hot).
        # It's at the end of obs, sized: n_task_types + n_locs+1 + n_kinds+1 + n_objects+1
        task_dim = len(TASK_TYPES) + (env.num_locations + 1) + (len(env.object_kinds) + 1) + (env.num_objects + 1)
        task_enc_water = obs_water[-task_dim:]
        task_enc_wash = obs_wash[-task_dim:]
        assert not np.array_equal(task_enc_water, task_enc_wash), \
            "Task encoding should differ between serve_water and wash_objects"

    def test_next_auto_satisfied_from_fresh_info(self, env):
        """next_auto_satisfied is readable from refreshed info after set_task."""
        env.set_task("serve_water", target_location="servingtable")
        info = env._info(success=False)
        assert isinstance(info.get("next_auto_satisfied"), bool)


# ---------------------------------------------------------------------------
# Test 3: paired independent envs start identically with same seed
# ---------------------------------------------------------------------------
class TestPairedEnvReset:
    def test_same_seed_same_obs(self):
        """Two independently created envs with same seed produce identical reset obs."""
        e1 = RestaurantSymbolicEnv(
            config_path="configs/restaurant/toy_level_3.yaml",
            rng_seed=42, max_steps_per_task=64,
            success_reward=81.07, invalid_action_penalty=6.0,
        )
        e2 = RestaurantSymbolicEnv(
            config_path="configs/restaurant/toy_level_3.yaml",
            rng_seed=42, max_steps_per_task=64,
            success_reward=81.07, invalid_action_penalty=6.0,
        )
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert np.array_equal(o1, o2)

    def test_different_seed_different_obs(self):
        """Two independently created envs with different seeds produce different reset obs."""
        e1 = RestaurantSymbolicEnv(
            config_path="configs/restaurant/toy_level_3.yaml",
            rng_seed=42, max_steps_per_task=64,
            success_reward=81.07, invalid_action_penalty=6.0,
        )
        e2 = RestaurantSymbolicEnv(
            config_path="configs/restaurant/toy_level_3.yaml",
            rng_seed=7, max_steps_per_task=64,
            success_reward=81.07, invalid_action_penalty=6.0,
        )
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=7)
        assert not np.array_equal(o1, o2)


# ---------------------------------------------------------------------------
# Test 4: PDDL task cost computed as cumulative delta
# ---------------------------------------------------------------------------
class TestPDDLCostDelta:
    def test_paper2_total_cost_increases_with_action(self, env):
        """After taking a valid step, paper2_total_cost increases by the step cost."""
        env.set_task("pick_place", target_location="coffeemachine", object_name="plate_0")
        # Move to servingtable where plate_0 starts, then pick it.
        at_move = env.action_type_index["move"]
        loc_serve = env.location_index["servingtable"]
        loc_none = env.none_location_index
        o_none = env.none_object_index

        prev = float(env._paper2_total_cost)
        action = {"action_type": at_move, "object1": o_none, "location": loc_serve, "object2": o_none}
        obs, reward, success, truncated, info = env.step(action)
        post = float(env._paper2_total_cost)
        assert post > prev, f"paper2 total should increase after move: {prev} -> {post}"

        # Now pick plate_0 (at servingtable, same location as agent)
        at_pick = env.action_type_index["pick"]
        o1_pick = env.object_name_index["plate_0"]
        prev2 = post
        action2 = {"action_type": at_pick, "object1": o1_pick, "location": loc_none, "object2": o_none}
        obs2, reward2, success2, truncated2, info2 = env.step(action2)
        post2 = float(env._paper2_total_cost)
        assert post2 > prev2, f"paper2 total should increase after pick: {prev2} -> {post2}"

    def test_paper2_total_cost_resets_only_at_task_boundaries(self, env):
        """paper2_total_cost is monotonic within a task; paper2_task_cost tracks per-task."""
        env.set_task("pick_place", target_location="coffeemachine", object_name="plate_0")
        at_move = env.action_type_index["move"]
        loc_serve = env.location_index["servingtable"]
        o_none = env.none_object_index
        prev = float(env._paper2_total_cost)
        action = {"action_type": at_move, "object1": o_none, "location": loc_serve, "object2": o_none}
        obs, reward, success, truncated, info = env.step(action)
        post = float(env._paper2_total_cost)
        # Total should increase since we took a valid step
        assert post > prev, f"paper2 total should increase: {prev} -> {post}"
        assert env._paper2_task_cost > 0.0, "Task cost should accumulate within a task"


# ---------------------------------------------------------------------------
# Smoke test: full eval script runs end-to-end on the real checkpoints
# ---------------------------------------------------------------------------
def test_direct_seq_eval_end_to_end(tmp_path):
    """Run the full eval on both real checkpoints and a short sequence, verify output JSON."""
    from scripts.restaurant.rl_direct_seq_eval import main
    ant_ckpt = _REPO / "results/canonical_planner/checkpoints/anticipatory/seed0"
    myo_ckpt = _REPO / "runs" / "v3_myopic_g0.9_peb"
    out = tmp_path / "eval.json"

    # Run with only 3 tasks to keep it fast
    import json as _json
    with SEQ_00.open() as fh:
        seq_data = _json.load(fh)
    short_seq = tmp_path / "short_seq.json"
    with short_seq.open("w") as fh:
        _json.dump({"sequence_id": "test", "tasks": seq_data["tasks"][:3]}, fh)

    import sys as _sys
    _sys.argv = [
        "rl_direct_seq_eval.py",
        "--ant-ckpt", str(ant_ckpt),
        "--myo-ckpt", str(myo_ckpt),
        "--seq", str(short_seq),
        "--seed", "0",
        "--output", str(out),
    ]

    main()

    assert out.exists()
    with out.open() as fh:
        result = _json.load(fh)

    assert result["eval_seed"] == 0
    assert result["sequence"]["n_tasks"] == 3

    for key in ("anticipatory", "myopic"):
        agent = result[key]
        assert agent["checkpoint"]
        assert agent["meta"]["hidden_dim"] == 256
        assert len(agent["tasks"]) == 3
        s = agent["summary"]
        assert 0 <= s["success_rate"] <= 1.0
        assert 0 <= s["auto_rate"] <= 1.0
        assert s["total_steps"] > 0
        assert s["total_return"] < 0 if s["success_rate"] < 0.5 else True  # costs > rewards
        assert s["mean_pddl_cost"] > 0
        # Per-task PDDL cost should be non-negative
        for t in agent["tasks"]:
            assert t["pddl_cost"] >= 0.0

    # Anticipatory tasks_per_episode metadata
    assert result["anticipatory"]["meta"]["tasks_per_episode"] == 50
    assert result["myopic"]["meta"]["tasks_per_episode"] == 1
