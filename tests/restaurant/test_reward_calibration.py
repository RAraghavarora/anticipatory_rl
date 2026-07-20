"""Regression tests for reward calibration formula."""
from __future__ import annotations

import pytest


# Inline the formula from scripts/restaurant/calibrate_reward.py so the test
# does not import the script (which would run the oracle at import time).
def _calibrate_R_star(successful_plan_rewards, gamma, margin):
    D_list = []
    for rewards in successful_plan_rewards:
        T = len(rewards)
        costs = [-r for r in rewards]
        D = sum(costs[t] * (gamma ** (t - (T - 1))) for t in range(T))
        D_list.append(D)
    max_D = max(D_list) if D_list else 0.0
    return (1.0 + margin) * max_D


def test_calibration_formula():
    gamma = 0.95
    margin = 0.2
    plan_a = [-1.0, -1.0, -1.0]
    plan_b = [-2.0, -1.0]
    R_star = _calibrate_R_star([plan_a, plan_b], gamma=gamma, margin=margin)
    expected_D = max(
        sum((-r) * (gamma ** (t - 2)) for t, r in enumerate(plan_a)),
        sum((-r) * (gamma ** (t - 1)) for t, r in enumerate(plan_b)),
    )
    assert R_star == pytest.approx((1.0 + margin) * expected_D)


def test_calibration_formula_empty():
    assert _calibrate_R_star([], gamma=0.95, margin=0.2) == 0.0
