#!/usr/bin/env python3
"""Decode and explain restaurant RL observation vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv


def _argmax_name(vec: np.ndarray, names: List[str], *, empty_name: str | None = None) -> str:
    idx = int(np.argmax(vec))
    if empty_name is not None and idx >= len(names):
        return empty_name
    return names[idx]


def decode_observation(env: RestaurantSymbolicEnv, obs: np.ndarray) -> Dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float32)
    p = 0

    # Agent location
    agent_vec = obs[p : p + env.num_locations]
    p += env.num_locations
    agent_location = _argmax_name(agent_vec, list(env.locations))

    # Holding
    held_vec = obs[p : p + env.num_objects + 1]
    p += env.num_objects + 1
    held_object = _argmax_name(held_vec, list(env.object_names), empty_name="none")

    # Per-object blocks
    objects: List[Dict[str, Any]] = []
    for obj_name in env.object_names:
        loc_vec = obs[p : p + env.num_locations + 1]
        p += env.num_locations + 1
        dirty_val = float(obs[p])
        p += 1
        contents_vec = obs[p : p + len(env.contents)]
        p += len(env.contents)
        kind_vec = obs[p : p + len(env.object_kinds)]
        p += len(env.object_kinds)

        loc_idx = int(np.argmax(loc_vec))
        if loc_idx == env.num_locations:
            location = "__held__"
        else:
            location = env.locations[loc_idx]

        objects.append(
            {
                "name": obj_name,
                "location": location,
                "dirty": bool(dirty_val > 0.5),
                "contents": _argmax_name(contents_vec, list(env.contents)),
                "kind": _argmax_name(kind_vec, list(env.object_kinds)),
            }
        )

    # Task encoding
    task_type_vec = obs[p : p + len(env.task_types)]
    p += len(env.task_types)
    target_loc_vec = obs[p : p + env.num_locations + 1]
    p += env.num_locations + 1
    target_kind_vec = obs[p : p + len(env.object_kinds) + 1]
    p += len(env.object_kinds) + 1

    task_type = _argmax_name(task_type_vec, list(env.task_types))
    tloc_idx = int(np.argmax(target_loc_vec))
    target_location = None if tloc_idx == env.num_locations else env.locations[tloc_idx]
    tkind_idx = int(np.argmax(target_kind_vec))
    target_kind = None if tkind_idx == len(env.object_kinds) else env.object_kinds[tkind_idx]

    if p != obs.shape[0]:
        raise ValueError(f"Decode mismatch: consumed {p} of {obs.shape[0]} dims.")

    return {
        "obs_dim": int(obs.shape[0]),
        "agent_location": agent_location,
        "holding": held_object,
        "task": {
            "task_type": task_type,
            "target_location": target_location,
            "target_kind": target_kind,
        },
        "objects": objects,
    }


def build_explanation(decoded: Dict[str, Any], info: Dict[str, Any], env: RestaurantSymbolicEnv) -> str:
    task = decoded["task"]
    hold = decoded["holding"]
    valid_mask = np.asarray(info.get("valid_action_mask", np.zeros((env.action_space.n,), dtype=np.float32)))
    action_labels = env.get_action_meanings()
    valid_actions = [action_labels[i] for i in np.flatnonzero(valid_mask > 0.0)]

    lines: List[str] = []
    lines.append("Observation explanation:")
    lines.append(f"- The agent is at `{decoded['agent_location']}` and is holding `{hold}`.")
    lines.append(
        f"- Current task encoding is `{task['task_type']}` "
        f"(target_location={task['target_location']}, target_kind={task['target_kind']})."
    )
    lines.append(
        f"- Object state channels encode each object's location, dirty flag, contents, and kind "
        f"for {len(decoded['objects'])} objects."
    )
    lines.append(
        f"- Valid action mask currently enables {len(valid_actions)} / {len(action_labels)} actions."
    )
    if valid_actions:
        preview = ", ".join(valid_actions[:10])
        lines.append(f"- First valid actions: {preview}{' ...' if len(valid_actions) > 10 else ''}")
    lines.append(
        "- `paper2_cost_*` in info tracks planner-style cost accumulation, while observation vector "
        "contains only state/task features."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and explain restaurant RL observation vectors.")
    parser.add_argument("--config-path", type=Path, default=Path("anticipatory_rl/configs/restaurant_symbolic.yaml"))
    parser.add_argument("--layout-corpus", type=Path, default=None, help="Optional paper2 layout corpus JSON.")
    parser.add_argument("--layout-index", type=int, default=0, help="Layout index when using --layout-corpus.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write decoded JSON.")
    args = parser.parse_args()

    env = RestaurantSymbolicEnv(config_path=args.config_path, rng_seed=args.seed)
    reset_options: Dict[str, Any] = {}
    if args.layout_corpus is not None:
        payload = json.loads(args.layout_corpus.read_text(encoding="utf-8"))
        layouts = payload["layouts"] if isinstance(payload, dict) else payload
        layout = layouts[args.layout_index % len(layouts)]
        reset_options["layout"] = layout
        if isinstance(layout.get("task_library"), list):
            reset_options["task_library"] = layout["task_library"]

    obs, info = env.reset(seed=args.seed, options=reset_options if reset_options else None)
    decoded = decode_observation(env, obs)
    explanation = build_explanation(decoded, info, env)

    output = {
        "decoded_observation": decoded,
        "info": {
            "task": info.get("task"),
            "next_auto_satisfied": info.get("next_auto_satisfied"),
            "layout_id": info.get("layout_id"),
            "paper2_cost_step": info.get("paper2_cost_step"),
            "paper2_cost_task": info.get("paper2_cost_task"),
            "paper2_cost_total": info.get("paper2_cost_total"),
        },
        "explanation": explanation,
    }

    print(json.dumps(output["decoded_observation"], indent=2))
    print()
    print(explanation)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nWrote decoded observation + explanation -> {args.output_json}")


if __name__ == "__main__":
    main()
