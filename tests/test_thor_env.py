import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from anticipatory_rl.envs.thor_rearrangement_env import ThorRearrangementEnv, ThorTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out the ThorRearrangementEnv with random actions.")
    parser.add_argument("--scene", type=str, default=None, help="AI2-THOR scene to load (e.g., FloorPlan1).")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum primitive actions per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Seed controlling task sampling.")
    parser.add_argument("--width", type=int, default=400, help="Camera width.")
    parser.add_argument("--height", type=int, default=300, help="Camera height.")
    parser.add_argument(
        "--object-types",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of pickupable object types to sample from.",
    )
    parser.add_argument(
        "--receptacle-types",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of receptacle types to sample as goals.",
    )
    parser.add_argument(
        "--house-json",
        type=Path,
        default=None,
        help="ProcTHOR house JSON file to load instead of a built-in scene.",
    )
    parser.add_argument(
        "--house-glob",
        type=str,
        default=None,
        help="Glob that resolves to multiple ProcTHOR houses (e.g., data/procthor/*.json).",
    )
    return parser.parse_args()


def collect_house_paths(args: argparse.Namespace) -> Optional[List[Path]]:
    if args.house_glob:
        return sorted(Path().glob(args.house_glob))
    if args.house_json:
        return [args.house_json]
    return None


def main() -> None:
    args = parse_args()
    scene_pool = [args.scene] if args.scene else None
    house_paths = collect_house_paths(args)
    env = ThorRearrangementEnv(
        scene_pool=scene_pool,
        procthor_house_paths=house_paths,
        frame_width=args.width,
        frame_height=args.height,
        max_task_steps=args.max_steps,
        object_types=args.object_types,
        receptacle_types=args.receptacle_types,
        rng_seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        task = info.get("task")
        if isinstance(task, ThorTask):
            task_desc = f"{task.object_type} -> {task.receptacle_type}"
        else:
            task_desc = "N/A"
        print(f"Episode {episode:02d} scene={info.get('scene')} task={task_desc}")
        for step in range(args.max_steps):
            action = int(rng.integers(env.action_space.n))
            obs, reward, success, horizon, info = env.step(action)
            print(
                f"  step={step:03d} action={action} reward={reward:.2f} success={success} horizon={horizon} err={info.get('error')}"
            )
            if success or horizon:
                break

    env.close()


if __name__ == "__main__":
    main()
