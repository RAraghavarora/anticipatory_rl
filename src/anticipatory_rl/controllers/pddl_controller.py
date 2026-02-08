"""Controller that keeps MiniWorld and the PDDL planner in sync.

The controller performs three duties:

1. Encode the live MiniWorld world state as a fresh PDDL problem instance.
2. Call Fast Downward (or another planner) to obtain a symbolic plan.
3. Map the symbolic actions back into MiniWorld's discrete controls.

This allows us to re-use the existing `gridworld_domain.pddl` definition while
letting the MiniWorld simulator remain the source of truth for geometry and
rendering.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from anticipatory_rl.envs.miniworld_env import MiniWorldGridRearrange

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pillow is optional; we fall back to raw frames if missing.
    Image = ImageDraw = ImageFont = None

try:
    import imageio.v3 as iio
except ImportError:  # optional dependency, only needed when recording.
    iio = None


# Mapping tables keep the freshly themed PDDL names aligned with the historical
# names still used inside the MiniWorld environment.
ENV_TO_PDDL_RECEPTACLE = {
    "charging_pad": "kitchen_table",
    "storage_a": "kitchen_counter",
    "storage_b": "dining_table",
    "assembly_zone": "study_table",
    "shipping": "shelf",
}

ENV_TO_PDDL_OBJECT = {
    "blue_crate": "water_bottle",
    "red_crate": "tiffin_box",
    "sensor_pack": "apple",
    "shipping_label": "soda_can",
    "toolkit": "drinking_glass",
}

PDDL_TO_ENV_OBJECT = {v: k for k, v in ENV_TO_PDDL_OBJECT.items()}


def _object_to_pddl(name: str) -> str:
    return ENV_TO_PDDL_OBJECT.get(name, name)


def _receptacle_to_pddl(name: str) -> str:
    return ENV_TO_PDDL_RECEPTACLE.get(name, name)


def _loc_name(coord: Tuple[int, int]) -> str:
    x, y = coord
    return f"loc_{x}{y}"


def _loc_to_coord(loc: str) -> Tuple[int, int]:
    _, suffix = loc.split("_")
    if len(suffix) != 2:
        raise ValueError(f"Unexpected location id '{loc}'")
    return int(suffix[0]), int(suffix[1])


def _load_text(path: Path) -> str:
    return Path(path).read_text()


def _extract_section(text: str, start_token: str, end_token: str) -> str:
    start = text.index(start_token)
    end = text.index(end_token, start)
    return text[start:end]


@dataclass
class PDDLProblemTemplate:
    """Lightweight parser for splitting the template problem file."""

    header: str
    objects_block: str
    goal_block: str
    static_init_lines: List[str]

    @classmethod
    def from_file(cls, problem_path: Path) -> "PDDLProblemTemplate":
        text = _load_text(problem_path)
        objects_block = _extract_section(text, "  (:objects", "  (:init")
        goal_block = text[text.index("  (:goal") :].rstrip()
        header = text[: text.index("  (:objects")]
        init_block = _extract_section(text, "  (:init", "  (:goal")

        static_lines: List[str] = []
        for raw_line in init_block.splitlines()[1:]:  # skip "(:init"
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(";;" ) and ("region" in line or "adjacency" in line):
                static_lines.append(line)
            if line.startswith("(belongs") or line.startswith("(adjacent"):
                static_lines.append(line)

        return cls(header=header, objects_block=objects_block, goal_block=goal_block, static_init_lines=static_lines)

    def build_problem(self, dynamic_facts: Sequence[str]) -> str:
        init_lines = list(dynamic_facts)
        if self.static_init_lines:
            init_lines.append("")
            init_lines.extend(self.static_init_lines)
        init_body = "\n".join(f"    {line}" for line in init_lines)
        return "".join(
            [
                self.header,
                self.objects_block,
                "  (:init\n",
                f"{init_body}\n",
                "  )\n",
                self.goal_block,
                "\n",
            ]
        )


class MiniWorldStateEncoder:
    """Extracts symbolic facts from the live MiniWorld environment."""

    def __init__(self, env: MiniWorldGridRearrange) -> None:
        self.env = env

    def encode(self) -> List[str]:
        facts: List[str] = []

        agent_loc = _loc_name(self.env.agent_grid)
        facts.append(f"(agent-at klara {agent_loc})")

        carrying = getattr(self.env, "carrying", None)
        if carrying:
            facts.append(f"(holding klara {_object_to_pddl(carrying)})")
            facts.append(f"(clear {_object_to_pddl(carrying)})")
        else:
            facts.append("(handfree klara)")

        tile_contents: Dict[Tuple[int, int], List[str]] = getattr(self.env, "tile_contents", {})
        clear_locations = { _loc_name((x, y)) for x in range(self.env.grid_size) for y in range(self.env.grid_size) }
        clear_objects: set[str] = set()

        for coord, stack in tile_contents.items():
            if not stack:
                continue
            clear_locations.discard(_loc_name(coord))
            for idx, env_name in enumerate(stack):
                obj = _object_to_pddl(env_name)
                if idx == 0:
                    facts.append(f"(on {obj} {_loc_name(coord)})")
                else:
                    below = _object_to_pddl(stack[idx - 1])
                    facts.append(f"(on {obj} {below})")

                region_env = self.env.tile_to_receptacle.get(coord)
                if region_env:
                    facts.append(f"(in {obj} {_receptacle_to_pddl(region_env)})")

            clear_objects.add(_object_to_pddl(stack[-1]))

        for loc in sorted(clear_locations):
            facts.append(f"(clear {loc})")

        for obj in sorted(clear_objects):
            facts.append(f"(clear {obj})")

        return facts


class FastDownwardPlanner:
    def __init__(self, planner_path: Path, search: str = "astar(lmcut())") -> None:
        self.planner_path = planner_path.resolve()
        self.search = search

    def plan(self, domain_path: Path, problem_path: Path, plan_path: Path) -> None:
        workdir = plan_path.parent
        workdir.mkdir(parents=True, exist_ok=True)
        domain_abs = domain_path.resolve()
        problem_abs = problem_path.resolve()
        cmd = [
            sys.executable,
            str(self.planner_path),
            str(domain_abs),
            str(problem_abs),
            "--search",
            self.search,
        ]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Planner failed with code {}.\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                    proc.returncode, proc.stdout, proc.stderr
                )
            )

        plan_candidates = sorted(workdir.glob("sas_plan*"))
        if not plan_candidates:
            raise FileNotFoundError(
                "Fast Downward completed but no plan file was produced (expected sas_plan*)"
            )
        plan_contents = plan_candidates[0].read_text()
        plan_path.write_text(plan_contents)


def parse_plan(plan_text: str) -> List[Tuple[str, List[str]]]:
    actions: List[Tuple[str, List[str]]] = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip().lower()
        if not line or not line.startswith("(") or line.startswith(";"):
            continue
        tokens = line.strip("()").split()
        actions.append((tokens[0], tokens[1:]))
    return actions


class MiniWorldPDDLController:
    def __init__(
        self,
        env: MiniWorldGridRearrange,
        *,
        domain_path: Path,
        problem_template: Path,
        planner_path: Path,
    ) -> None:
        self.env = env
        self.template = PDDLProblemTemplate.from_file(problem_template)
        self.encoder = MiniWorldStateEncoder(env)
        self.planner = FastDownwardPlanner(planner_path)
        self.domain_path = domain_path

    def build_problem_text(self) -> str:
        dynamic_facts = self.encoder.encode()
        return self.template.build_problem(dynamic_facts)

    def compute_plan(self) -> List[Tuple[str, List[str]]]:
        problem_text = self.build_problem_text()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            problem_path = tmpdir_path / "problem.pddl"
            plan_path = tmpdir_path / "plan.soln"
            problem_path.write_text(problem_text)
            self.planner.plan(self.domain_path, problem_path, plan_path)
            if not plan_path.exists():
                raise FileNotFoundError("Planner finished without producing a plan")
            return parse_plan(plan_path.read_text())

    def execute_plan(
        self,
        plan: Sequence[Tuple[str, List[str]]],
        *,
        render: bool = False,
        recorder: "FrameRecorder | None" = None,
    ) -> None:
        for step_id, (name, args) in enumerate(plan, start=1):
            if name == "move":
                self._apply_move(args)
            elif name in {"pick-from-location", "pick-from-stack"}:
                self._apply_primitive("pick")
            elif name in {"place-on-location", "stack-on-object"}:
                self._apply_primitive("place")
            else:
                raise ValueError(f"Unsupported plan action '{name}'")

            if render:
                self.env.render()
            description = f"{step_id}. {name} {' '.join(args)}".strip()
            if recorder:
                recorder.capture(self.env, description)
            print(f"Executed step {step_id}: {name} {' '.join(args)}")

    def _apply_move(self, args: Sequence[str]) -> None:
        if len(args) != 3:
            raise ValueError(f"move expects 3 arguments, got {args}")
        _, _, target_loc = args
        target = _loc_to_coord(target_loc)
        current = self.env.agent_grid
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        move_lookup = {
            (1, 0): "move_right",
            (-1, 0): "move_left",
            (0, 1): "move_down",
            (0, -1): "move_up",
        }
        action_name = move_lookup.get((dx, dy))
        if not action_name:
            raise ValueError(f"Agent cannot move from {current} to {target}")
        self._apply_primitive(action_name)

    def _apply_primitive(self, action_name: str) -> None:
        self.env.act(action_name)


class FrameRecorder:
    """Utility to capture annotated top-down frames and write a video."""

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self.frames: List[np.ndarray] = []
        if iio is None:
            raise RuntimeError(
                "Recording requires imageio>=2.26. Install with `pip install imageio imageio-ffmpeg`."
            )

    def capture(self, env: MiniWorldGridRearrange, caption: str) -> None:
        frame = env.render_top_view(env.vis_fb)
        img = np.array(frame, copy=True)
        annotated = self._annotate_frame(img, caption)
        self.frames.append(annotated)

    def _annotate_frame(self, frame: np.ndarray, caption: str) -> np.ndarray:
        if Image is None:
            return frame
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        text = caption if caption else ""
        font = ImageFont.load_default()
        padding = 4
        text_size = draw.textbbox((0, 0), text, font=font)
        width = text_size[2] - text_size[0]
        height = text_size[3] - text_size[1]
        draw.rectangle(
            [0, 0, width + padding * 2, height + padding * 2],
            fill=(0, 0, 0, 160),
        )
        draw.text((padding, padding), text, font=font, fill=(255, 255, 255))
        return np.array(image)

    def save(self) -> None:
        if not self.frames:
            print("[recorder] No frames captured; skipping save.")
            return
        iio.imwrite(self.path, self.frames, fps=self.fps)
        print(f"[recorder] Saved {len(self.frames)} frames to {self.path}")


def run_controller(args: argparse.Namespace) -> None:
    env = MiniWorldGridRearrange(render_mode="human" if args.render else None)
    env.reset(seed=args.seed)
    recorder = FrameRecorder(Path(args.record), fps=args.fps) if args.record else None
    if recorder:
        recorder.capture(env, "reset")
    controller = MiniWorldPDDLController(
        env,
        domain_path=Path(args.domain),
        problem_template=Path(args.problem_template),
        planner_path=Path(args.planner),
    )
    plan = controller.compute_plan()
    if not plan:
        print("Planner returned an empty plan.")
        return
    print(f"Planner returned {len(plan)} steps. Executing now...")
    controller.execute_plan(plan, render=args.render, recorder=recorder)
    if recorder:
        recorder.save()
    env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDDL-driven controller for MiniWorld rearrangement tasks.")
    parser.add_argument("--domain", default="assets/pddl/gridworld_domain.pddl", help="Path to the domain PDDL file.")
    parser.add_argument(
        "--problem-template",
        default="assets/pddl/gridworld_problem.pddl",
        help="Template problem file that defines objects/goals/static facts.",
    )
    parser.add_argument(
        "--planner",
        default="downward/fast-downward.py",
        help="Fast Downward entry point (or compatible planner script).",
    )
    parser.add_argument("--seed", type=int, default=None, help="MiniWorld reset seed.")
    parser.add_argument("--render", action="store_true", help="Render the MiniWorld top view while executing the plan.")
    parser.add_argument("--record", type=str, default=None, help="Optional path to save a top-down annotated video (mp4/gif).")
    parser.add_argument("--fps", type=int, default=12, help="Playback FPS when recording video.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()
