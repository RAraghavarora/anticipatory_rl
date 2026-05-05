from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from anticipatory_rl.agents.restaurant import dqn as restaurant_dqn
from anticipatory_rl.envs.restaurant.env import (
    ACTION_TYPE_TO_INDEX,
    ACTION_TYPES,
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
    def test_reset_samples_non_auto_pick_place_tasks(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        for seed in range(20):
            _, info = env.reset(seed=seed)
            self.assertEqual(info["task"]["task_type"], "pick_place")
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

    def test_flat_action_catalog_uses_only_move_pick_place(self) -> None:
        env = RestaurantSymbolicEnv(config_path=CONFIG_PATH)
        _, info = env.reset(seed=0)
        catalog = restaurant_dqn.FlatActionCatalog(env)

        self.assertEqual(catalog.num_actions, env.num_objects + (2 * env.num_locations))
        flat_mask = catalog.project_mask(restaurant_dqn._extract_masks(info))
        self.assertEqual(flat_mask.shape, (catalog.num_actions,))
        self.assertGreater(int(np.sum(flat_mask > 0.0)), 0)

        allowed = {"move", "pick", "place"}
        for action in catalog.actions:
            self.assertIn(ACTION_TYPES[action.action_type], allowed)

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
                    boundary_mode="myopic",
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
                )
                checkpoint = restaurant_dqn.train(args)
                self.assertTrue(checkpoint.exists())
                summary = Path(tmpdir) / "runs" / "pytest_restaurant_factored" / "train_summary.json"
                self.assertTrue(summary.exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
