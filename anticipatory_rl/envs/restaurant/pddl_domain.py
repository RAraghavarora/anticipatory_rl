"""PDDL action cost reference for the restaurant environment."""

from __future__ import annotations

from typing import Dict


# PDDL action costs from the author's domain (exact match)
# These are the canonical costs used for planner comparison and paper2_cost
PDDL_ACTION_COSTS: Dict[str, float] = {
    "pick": 100,
    "place": 100,
    "wash": 200,
    "fill": 1000,
    "make-coffee": 50,
    "make-fruit-bowl": 100,
    "apply-spread": 100,
    "pour": 200,
    "refill_water": 50,
    "drain": 50,
}


def get_pddl_cost(action_name: str) -> float:
    """Get the PDDL cost for a given action name.

    Args:
        action_name: The PDDL action name (e.g., "pick", "wash", "fill")

    Returns:
        The cost associated with the action, or 0 if not found.
    """
    return PDDL_ACTION_COSTS.get(action_name, 0.0)



