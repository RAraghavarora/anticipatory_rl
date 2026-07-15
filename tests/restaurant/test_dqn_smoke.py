"""Smoke test: instantiate DQN, forward pass, backward pass, check finite gradients."""
from __future__ import annotations

import numpy as np

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES


def test_dqn_forward_and_backward(env):
    import torch
    from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork

    obs_dim = env.observation_space.shape[0]
    n_actions = len(ACTION_TYPES)

    q_net = RestaurantQNetwork(obs_dim, n_actions, env.num_objects + 1, env.num_locations + 1, hidden_dim=64)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

    states = []
    action_types = []
    objects1 = []
    locations = []
    objects2 = []
    at_masks = []
    o1_masks = []
    loc_masks = []
    o2_masks = []

    for _ in range(4):
        masks = env._compute_action_masks()
        at_mask = masks["valid_action_type_mask"]
        at = int(np.random.choice(np.flatnonzero(at_mask > 0.5)))
        o1 = int(np.random.choice(np.flatnonzero(masks["valid_object1_mask"][at] > 0.5)))
        loc = int(np.random.choice(np.flatnonzero(masks["valid_location_mask"][at] > 0.5)))
        o2 = int(np.random.choice(np.flatnonzero(masks["valid_object2_mask"][at, o1] > 0.5)))

        states.append(env._obs())
        action_types.append([at])
        objects1.append([o1])
        locations.append([loc])
        objects2.append([o2])
        at_masks.append(at_mask)
        o1_masks.append(masks["valid_object1_mask"])
        loc_masks.append(masks["valid_location_mask"])
        o2_masks.append(masks["valid_object2_mask"])

    def t(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=torch.device("cpu"))

    q_values = q_net.forward(
        t(np.stack(states)),
        action_types=t(np.array(action_types)).long(),
        object1=t(np.array(objects1)).long(),
        location=t(np.array(locations)).long(),
        object2=t(np.array(objects2)).long(),
        action_type_masks=t(np.stack(at_masks)),
        object1_masks=t(np.stack(o1_masks)),
        location_masks=t(np.stack(loc_masks)),
        object2_masks=t(np.stack(o2_masks)),
    )

    assert q_values.shape == (4, 1), f"Unexpected Q shape: {q_values.shape}"
    assert q_values.isfinite().all(), "Q values contain nan/inf"

    loss = q_values.mean()
    optimizer.zero_grad()
    loss.backward()

    for name, param in q_net.named_parameters():
        if param.grad is not None:
            assert param.grad.isfinite().all(), f"Gradient not finite for {name}"

    optimizer.step()
