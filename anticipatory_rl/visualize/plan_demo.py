"""Replay a symbolic plan in MiniWorld and save an animation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from anticipatory_rl.envs.miniworld_env import MiniWorldGridRearrange
from anticipatory_rl.controllers.pddl_controller import FrameRecorder

ACTIONS = {
    "move_up": 0,
    "move_down": 1,
    "move_left": 2,
    "move_right": 3,
    "pick": 4,
    "place": 5,
}


def parse_plan(plan_text: str) -> List[List[str]]:
    commands = []
    for raw in plan_text.splitlines():
        line = raw.strip().lower()
        if not line or line.startswith(";"):
            continue
        tokens = line.strip("()").split()
        commands.append(tokens)
    return commands


def grid_from_loc(loc: str) -> tuple[int, int]:
    _, suffix = loc.split("_")
    return (int(suffix[0]), int(suffix[1]))


def action_sequence(plan_cmds: List[List[str]]) -> List[str]:
    moves = {
        (1, 0): "move_right",
        (-1, 0): "move_left",
        (0, 1): "move_down",
        (0, -1): "move_up",
    }
    sequence: List[str] = []
    current = (0, 0)
    for tokens in plan_cmds:
        name = tokens[0]
        if name == "move":
            target = grid_from_loc(tokens[3])
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            action = moves.get((dx, dy))
            if action is None:
                raise ValueError(f"Unsupported move from {current} to {target}")
            sequence.append(action)
            current = target
        elif name.startswith("pick"):
            sequence.append("pick")
        elif name.startswith("place") or name.startswith("stack") or name.startswith("drop"):
            sequence.append("place")
        else:
            raise ValueError(f"Unknown plan token '{name}'")
    return sequence


def run(plan_text: str, placements: Dict[str, str], output: Path, fps: int = 12) -> None:
    env = MiniWorldGridRearrange(render_mode=None)
    env.reset()
    env.apply_object_placements(placements)
    recorder = FrameRecorder(output.with_suffix(".mp4"), fps=fps)
    recorder.capture(env, "start")
    commands = parse_plan(plan_text)
    actions = action_sequence(commands)
    for idx, act_name in enumerate(actions, start=1):
        env.act(act_name)
        recorder.capture(env, f"{idx}. {act_name}")
    recorder.save()
    env.close()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a symbolic plan in MiniWorld.")
    parser.add_argument("--plan", type=Path, required=True, help="Path to plan text file (sas_plan style).")
    parser.add_argument(
        "--placements",
        type=str,
        required=True,
        help="Comma-separated assignments like 'water_bottle:kitchen_counter,...'.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs") / "plan_demo.mp4", help="Video output path.")
    parser.add_argument("--fps", type=int, default=12, help="Video FPS.")
    return parser


def parse_placements(spec: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for pair in spec.split(","):
        if not pair:
            continue
        name, region = pair.split(":")
        result[name.strip()] = region.strip()
    return result


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    plan_text = Path(args.plan).read_text()
    placements = parse_placements(args.placements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(plan_text, placements, args.output, fps=args.fps)
    print(f"Saved plan replay to {args.output}")


if __name__ == "__main__":
    main()
