from __future__ import annotations

import random

from paper1_blockworld.motion import LazyPRMMotionPlanner
from paper1_blockworld.planner import FastDownwardBlockworldPlanner
from paper1_blockworld.world import (
    COLORED_REGIONS,
    WHITE_REGIONS,
    Task,
    WorldConfig,
    WorldGenerator,
    WorldState,
)


def make_barrier_config() -> WorldConfig:
    anchors = (
        (4, 0),
        (4, 2),
        (4, 4),
        (4, 6),
        (4, 8),
        (0, 0),
        (0, 2),
        (0, 4),
        (0, 6),
        (8, 0),
    )
    return WorldConfig(
        region_layout=tuple(zip(COLORED_REGIONS + WHITE_REGIONS, anchors)),
        block_colors=tuple(
            (block, COLORED_REGIONS[idx] if idx < 5 else "white")
            for idx, block in enumerate(("a", "b", "c", "d", "e", "f", "g", "h"))
        ),
    )


def test_sampled_regions_do_not_overlap() -> None:
    rng = random.Random(0)
    config = WorldConfig.sample(rng)
    occupied = set()
    for tiles in config.region_tiles.values():
        assert len(tiles) == 4
        for x, y in tiles:
            assert 0 <= x < config.width
            assert 0 <= y < config.height
            assert (x, y) not in occupied
            occupied.add((x, y))


def test_initial_state_uses_one_block_per_region_and_robot_on_floor() -> None:
    rng = random.Random(1)
    config = WorldConfig.sample(rng)
    state = WorldGenerator(config).sample_initial_state(rng)
    occupied_regions = state.occupied_regions(config)
    assert len(state.placements) == len(config.all_blocks)
    assert len(occupied_regions) == len(config.all_blocks)
    assert state.robot in config.floor_cells


def test_region_goal_satisfaction_accepts_any_tile() -> None:
    config = WorldConfig()
    region = config.nonwhite_regions[0]
    task = Task((("a", region),))
    for tile in config.tiles_for_region(region):
        state = WorldState(robot=(2, 2), placements={"a": tile}, holding=None)
        assert state.is_task_satisfied(task, config)
    outside_tile = next(coord for coord in config.floor_cells if coord not in config.tiles_for_region(region))
    bad_state = WorldState(robot=(2, 2), placements={"a": outside_tile}, holding=None)
    assert not bad_state.is_task_satisfied(task, config)


def test_prm_scaling_matches_path_length() -> None:
    config = WorldConfig()
    state = WorldState(robot=(2, 2), placements={}, holding=None)
    prm = LazyPRMMotionPlanner(config, state)
    path = prm.shortest_path((2, 2), (2, 3))
    assert path is not None
    assert path.length == 1.0
    assert path.cost == 25


def test_prm_reports_no_path_across_full_barrier() -> None:
    config = make_barrier_config()
    state = WorldState(robot=(2, 9), placements={}, holding=None)
    prm = LazyPRMMotionPlanner(config, state)
    assert prm.shortest_path((2, 9), (8, 9)) is None


def test_planner_rejects_two_blocks_in_same_region_goal() -> None:
    config = WorldConfig()
    planner = FastDownwardBlockworldPlanner(config)
    region = config.nonwhite_regions[0]
    tiles = config.tiles_for_region(region)
    state = WorldState(robot=(2, 2), placements={}, holding=None)
    try:
        planner.plan_to_placements(state, {"a": tiles[0], "b": tiles[1]})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected duplicate-region goal placements to be rejected.")


def test_planner_solves_region_goal_task() -> None:
    rng = random.Random(5)
    config = WorldConfig.sample(rng)
    generator = WorldGenerator(config)
    state = generator.sample_initial_state(rng)
    task = generator.sample_task_library(rng, count=1)[0]
    planner = FastDownwardBlockworldPlanner(config)
    result = planner.plan_for_task(state, task)
    assert result.cost > 0
    assert result.final_state.is_task_satisfied(task, config)
