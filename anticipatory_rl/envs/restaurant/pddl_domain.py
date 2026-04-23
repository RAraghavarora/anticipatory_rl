"""PDDL-aligned domain specification for the restaurant environment.

This module provides a centralized specification of the restaurant domain that
matches the author's PDDL domain and clarifications. It serves as the source of
truth for action definitions, costs, object types, and task types across both
the RL environment and the planner.

Key author clarifications:
- Task distribution P(τ) is uniform
- Action costs vary by action (fill: 1000, wash: 200, make-coffee: 50)
- Movement cost uses ProcTHOR occupancy grids with 0.1 resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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


# Mapping from PDDL action names to RL macro-action names
PDDL_TO_RL_ACTION_MAP: Dict[str, str] = {
    "pick": "pick",
    "place": "place",
    "wash": "wash",
    "fill": "fill_water",
    "make-coffee": "make_coffee",
    "make-fruit-bowl": "make_fruit_bowl",
    "apply-spread": "apply_spread",
    "pour": "pour",
    "refill_water": "refill_water",
    "drain": "drain",
}


# Object types from the PDDL domain (exact match)
PDDL_OBJECT_TYPES: Tuple[str, ...] = (
    "cup",
    "mug",
    "jar",
    "coffeegrinds",
    "water",
    "bread",
    "knife",
    "plate",
    "bowl",
    "spread",
    "apple",
)


# Location types from the PDDL domain
PDDL_LOCATION_TYPES: Tuple[str, ...] = (
    "counter",
    "sink",
    "dish_rack",
    "water_station",
    "coffee_machine",
    "fruit_station",
    "table",
    "pass_counter",
    "bus_tub",
    "pantry_shelf",
    "service_shelf",
    "host_stand",
)


# Task types from the PDDL domain (derived from actions)
PDDL_TASK_TYPES: Tuple[str, ...] = (
    "serve_water",
    "make_coffee",
    "make_fruit_bowl",
    "clear_containers",
    "wash_objects",
    "pick_place",
)


# Contents types from the PDDL domain
PDDL_CONTENTS: Tuple[str, ...] = (
    "empty",
    "water",
    "coffee",
    "spread",
    "apple",
)


# Service locations (where serving tasks are performed)
PDDL_SERVICE_LOCATIONS: Tuple[str, ...] = (
    "pass_counter",
    "table_left",
    "table_center",
    "table_right",
)


# Wash-ready locations (where clean objects are stored)
PDDL_WASH_READY_LOCATIONS: Tuple[str, ...] = (
    "dish_rack",
    "prep_counter",
    "kitchen_counter",
)


# Dirty drop locations (where dirty objects are placed)
PDDL_DIRTY_DROP_LOCATIONS: Tuple[str, ...] = (
    "sink",
    "bus_tub",
)


@dataclass(frozen=True)
class PDDLActionSpec:
    """Specification of a PDDL action with preconditions, effects, and cost."""

    name: str
    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    cost: float

    @property
    def rl_action_name(self) -> str:
        """Get the corresponding RL macro-action name."""
        return PDDL_TO_RL_ACTION_MAP.get(self.name, self.name)


# PDDL action specifications (exact match to PDDL domain)
PDDL_ACTIONS: Tuple[PDDLActionSpec, ...] = (
    PDDLActionSpec(
        name="pick",
        preconditions=("hand-is-free", "is-pickable ?obj", "is-at ?obj ?loc", "rob-at ?loc"),
        effects=("not is-at ?obj ?loc", "is-holding ?obj", "not hand-is-free"),
        cost=PDDL_ACTION_COSTS["pick"],
    ),
    PDDLActionSpec(
        name="place",
        preconditions=("not hand-is-free", "rob-at ?loc", "is-holding ?obj"),
        effects=("is-at ?obj ?loc", "not is-holding ?obj", "hand-is-free"),
        cost=PDDL_ACTION_COSTS["place"],
    ),
    PDDLActionSpec(
        name="wash",
        preconditions=("rob-at dishwasher", "is-at ?i dishwasher", "is-dirty ?i"),
        effects=("not is-dirty ?i"),
        cost=PDDL_ACTION_COSTS["wash"],
    ),
    PDDLActionSpec(
        name="fill",
        preconditions=(
            "rob-at ?loc",
            "is-at ?liquid ?loc",
            "is-holding ?cnt",
            "not is-dirty ?cnt",
            "is-fountain ?loc",
            "is-liquid ?liquid",
            "is-fillable ?cnt",
        ),
        effects=("filled-with ?liquid ?cnt"),
        cost=PDDL_ACTION_COSTS["fill"],
    ),
    PDDLActionSpec(
        name="make-coffee",
        preconditions=(
            "rob-at coffeemachine",
            "is-at water coffeemachine",
            "is-at coffeegrinds coffeemachine",
            "not is-jar ?c",
            "is-fillable ?c",
            "not is-dirty ?c",
            "is-at ?c coffeemachine",
            "not filled-with water ?c",
            "not filled-with coffee ?c",
        ),
        effects=("filled-with coffee ?c", "is-dirty ?c", "not is-at water coffeemachine"),
        cost=PDDL_ACTION_COSTS["make-coffee"],
    ),
    PDDLActionSpec(
        name="make-fruit-bowl",
        preconditions=(
            "rob-at countertop",
            "is-at ?a countertop",
            "is-at ?b countertop",
            "is-holding ?k",
            "not is-dirty ?k",
            "not is-dirty ?b",
            "is-slicable ?a",
            "is-container ?b",
        ),
        effects=("is-in ?a ?b", "is-dirty ?k", "is-dirty ?b"),
        cost=PDDL_ACTION_COSTS["make-fruit-bowl"],
    ),
    PDDLActionSpec(
        name="apply-spread",
        preconditions=(
            "rob-at countertop",
            "is-at bread countertop",
            "is-at ?s countertop",
            "is-holding ?k",
            "not is-dirty ?k",
            "is-spread ?s",
            "is-spreadable bread",
            "not spread-applied bread ?s",
        ),
        effects=("spread-applied bread ?s", "is-dirty ?k"),
        cost=PDDL_ACTION_COSTS["apply-spread"],
    ),
    PDDLActionSpec(
        name="pour",
        preconditions=(
            "rob-at ?loc",
            "is-liquid ?liquid",
            "is-fillable ?loc",
            "filled-with ?liquid ?cnt",
            "is-holding ?cnt",
        ),
        effects=("is-at ?liquid ?loc", "not filled-with ?liquid ?cnt"),
        cost=PDDL_ACTION_COSTS["pour"],
    ),
    PDDLActionSpec(
        name="refill_water",
        preconditions=(
            "rob-at ?loc",
            "is-at ?jr ?loc",
            "is-holding ?cnt",
            "is-jar ?jr",
            "is-fillable ?cnt",
            "not is-dirty ?cnt",
            "filled-with water ?jr",
        ),
        effects=("filled-with ?liquid ?cnt"),
        cost=PDDL_ACTION_COSTS["refill_water"],
    ),
    PDDLActionSpec(
        name="drain",
        preconditions=(
            "filled-with water ?cnt",
            "is-holding ?cnt",
            "rob-at fountain",
        ),
        effects=("not filled-with water ?cnt"),
        cost=PDDL_ACTION_COSTS["drain"],
    ),
)


def get_pddl_cost(action_name: str) -> float:
    """Get the PDDL cost for a given action name.

    Args:
        action_name: The PDDL action name (e.g., "pick", "wash", "fill")

    Returns:
        The cost associated with the action, or 0 if not found.
    """
    return PDDL_ACTION_COSTS.get(action_name, 0.0)


def get_rl_action_cost(action_name: str) -> float:
    """Get the PDDL cost for an RL macro-action name.

    Args:
        action_name: The RL macro-action name (e.g., "pick", "wash", "fill_water")

    Returns:
        The cost associated with the action, or 0 if not found.
    """
    # Map RL action names back to PDDL names
    rl_to_pddl = {v: k for k, v in PDDL_TO_RL_ACTION_MAP.items()}
    pddl_name = rl_to_pddl.get(action_name, action_name)
    return PDDL_ACTION_COSTS.get(pddl_name, 0.0)
