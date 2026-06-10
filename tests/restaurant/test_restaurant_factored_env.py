from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from anticipatory_rl.agents.restaurant import dqn as restaurant_dqn
from anticipatory_rl.envs.restaurant.env import (
    ACTION_TYPE_TO_INDEX,
    CONFIG_PATH,
    RestaurantObjectState,
    RestaurantState,
    RestaurantSymbolicEnv,
)


def _make_clean_state(env: RestaurantSymbolicEnv, *, agent_location: str, holding: str | None = None) -> RestaurantState:
    objects = {}
    for name, kind in env.object_specs:
        objects[name] = RestaurantObjectState(
            name=name,
            kind=kind,
            location="pantry_shelf" if "pantry_shelf" in env.location_index else env._default_agent_location(),
            dirty=False,
            filled_with=None,
            contained_in=None,
        )
    if holding is not None:
        objects[holding].location = None
    return RestaurantState(agent_location=agent_location, holding=holding, objects=objects, bread_spread=None)


class RestaurantFactoredEnvTests(unittest.TestCase):
    def test_reset_samples_non_auto_tasks(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        for seed in range(20):
            _, info = env.reset(seed=seed)
            # First task after reset must never be auto-satisfied (regardless of type).
            self.assertFalse(info["next_auto_satisfied"])

    def test_factored_action_masks_exposed(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        _, info = env.reset(seed=0)

        self.assertEqual(env.action_space["action_type"].n, len(ACTION_TYPE_TO_INDEX))
        self.assertEqual(info["valid_action_type_mask"].shape, (len(ACTION_TYPE_TO_INDEX),))
        self.assertEqual(
            info["valid_object1_mask"].shape,
            (len(ACTION_TYPE_TO_INDEX), env.action_space["object1"].n),
        )
        self.assertEqual(
            info["valid_location_mask"].shape,
            (len(ACTION_TYPE_TO_INDEX), env.action_space["location"].n),
        )
        self.assertEqual(
            info["valid_object2_mask"].shape,
            (
                len(ACTION_TYPE_TO_INDEX),
                env.action_space["object1"].n,
                env.action_space["object2"].n,
            ),
        )
        auto_idx = ACTION_TYPE_TO_INDEX["auto_complete"]
        self.assertEqual(info["valid_action_type_mask"][auto_idx], 0.0)
        self.assertEqual(info["valid_action_mask"][auto_idx], 0.0)

    def test_auto_success_is_flagged_without_executing_action(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        env.reset(seed=0)
        object_name = next(name for name, obj in env.state.objects.items() if obj.location is not None)
        object_location = env.state.objects[object_name].location
        env.set_task("pick_place", target_location=object_location, object_name=object_name, task_source="test")

        _, reward, success, truncated, next_info = env.step(
            {
                "action_type": ACTION_TYPE_TO_INDEX["move"],
                "object1": env.none_object_index,
                "location": env.none_location_index,
                "object2": env.none_object_index,
            }
        )

        self.assertEqual(reward, env.success_reward)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertTrue(next_info["auto_success"])

    def test_auto_complete_replay_action_uses_null_heads(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        _, info = env.reset(seed=0)
        masks = restaurant_dqn.extract_masks(info)

        action, replay_masks = restaurant_dqn._auto_complete_replay_action_and_masks(env, masks)
        auto_idx = ACTION_TYPE_TO_INDEX["auto_complete"]

        self.assertEqual(action["action_type"], auto_idx)
        self.assertEqual(action["object1"], env.none_object_index)
        self.assertEqual(action["location"], env.none_location_index)
        self.assertEqual(action["object2"], env.none_object_index)
        self.assertEqual(replay_masks["valid_action_type_mask"].sum(), 1.0)
        self.assertEqual(replay_masks["valid_action_type_mask"][auto_idx], 1.0)
        self.assertEqual(replay_masks["valid_object1_mask"][auto_idx, env.none_object_index], 1.0)
        self.assertEqual(replay_masks["valid_location_mask"][auto_idx, env.none_location_index], 1.0)
        self.assertEqual(replay_masks["valid_object2_mask"][auto_idx, env.none_object_index, env.none_object_index], 1.0)

    def test_auto_complete_q_composition_is_action_type_only(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        obs, info = env.reset(seed=0)
        masks = restaurant_dqn.extract_masks(info)
        action, replay_masks = restaurant_dqn._auto_complete_replay_action_and_masks(env, masks)
        q_net = restaurant_dqn.RestaurantQNetwork(
            len(obs),
            env.action_space["action_type"].n,
            env.action_space["object1"].n,
            env.action_space["location"].n,
            hidden_dim=32,
        )

        q_value = q_net(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
            action_types=torch.tensor([[action["action_type"]]], dtype=torch.int64),
            object1=torch.tensor([[action["object1"]]], dtype=torch.int64),
            location=torch.tensor([[action["location"]]], dtype=torch.int64),
            object2=torch.tensor([[action["object2"]]], dtype=torch.int64),
            action_type_masks=torch.tensor(replay_masks["valid_action_type_mask"], dtype=torch.float32).unsqueeze(0),
            object1_masks=torch.tensor(replay_masks["valid_object1_mask"], dtype=torch.float32).unsqueeze(0),
            location_masks=torch.tensor(replay_masks["valid_location_mask"], dtype=torch.float32).unsqueeze(0),
            object2_masks=torch.tensor(replay_masks["valid_object2_mask"], dtype=torch.float32).unsqueeze(0),
        )

        self.assertEqual(tuple(q_value.shape), (1, 1))

    def test_drain_mask_requires_held_water_at_fountain(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        env.reset(seed=0)
        env.state = _make_clean_state(env, agent_location="water_station", holding="cup_small")
        env.state.objects["cup_small"].filled_with = "water"

        info = env._info(success=False)
        drain_idx = ACTION_TYPE_TO_INDEX["drain"]
        cup_idx = env.object_name_index["cup_small"]

        self.assertEqual(info["valid_action_type_mask"][drain_idx], 1.0)
        self.assertEqual(info["valid_object1_mask"][drain_idx, cup_idx], 1.0)

        env.state.agent_location = "sink"
        info = env._info(success=False)
        self.assertEqual(info["valid_action_type_mask"][drain_idx], 0.0)

    def test_train_smoke_with_factored_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                args = Namespace(
                    total_steps=2,
                    replay_size=8,
                    batch_size=1,
                    hidden_dim=64,
                    gamma=0.99,
                    lr=3e-4,
                    epsilon_start=1.0,
                    epsilon_final=0.05,
                    epsilon_decay=10,
                    target_update=1,
                    tau=1.0,
                    max_grad_norm=1.0,
                    tasks_per_episode=1,
                    env_reset_tasks=1,
                    episode_step_limit=10,
                    max_steps_per_task=4,
                    success_reward=15.0,
                    invalid_action_penalty=6.0,
                    travel_cost_scale=1.0,
                    pick_cost=1.0,
                    place_cost=1.0,
                    wash_cost=2.0,
                    fill_cost=1.0,
                    brew_cost=2.0,
                    fruit_cost=2.0,
                    config_path=Path(CONFIG_PATH),
                    seed=0,
                    run_label="pytest_restaurant_factored",
                    output_name="restaurant_dqn.pt",
                    post_train_eval_tasks=2,
                    post_train_eval_max_steps=8,
                    post_train_log_trajectories=1,
                    post_train_plot_trajectories=2,
                    diagnostics=False,
                    diagnostics_interval=1000,
                    no_dueling_centering=False,
                    seed_replay_oracle=0,
                    protect_demo_fraction=0.0,
                    per_alpha=0.0,
                    per_beta=0.4,
                    per_eps=1e-6,
                    per_clip=10.0,
                )
                checkpoint = restaurant_dqn.train(args)
                self.assertTrue(checkpoint.exists())
                summary = Path(tmpdir) / "runs" / "pytest_restaurant_factored" / "train_summary.json"
                self.assertTrue(summary.exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
