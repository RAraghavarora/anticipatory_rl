#!/usr/bin/env python3
"""Paired greedy evaluation of anticipatory & myopic DQN checkpoints on a fixed task sequence.

Each agent uses a single reset; the world persists across all sequence tasks.
Reports per-task success, steps, RL return, and PDDL cost as cumulative delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from anticipatory_rl.agents.restaurant.dqn import RestaurantQNetwork
from anticipatory_rl.agents.restaurant.restaurant_dqn_infer import _sample_structured_action
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv, RestaurantTask
from anticipatory_rl.utils import select_device


def _load_checkpoint_dir(ckpt_dir: Path, fallback: Optional[Path] = None) -> Dict[str, Any]:
    """Load train_args.json from a checkpoint directory or a .pt file's parent dir.

    Some runs were archived with only the checkpoint and train_summary.json, so `fallback`
    allows pointing at the train_args.json of a run sharing the same env and architecture.
    Only env/architecture fields are read; a wrong hidden_dim fails the state_dict load.
    """
    if ckpt_dir.suffix == ".pt":
        ckpt_dir = ckpt_dir.parent
    args_path = ckpt_dir / "train_args.json"
    if not args_path.exists():
        if fallback is None:
            raise FileNotFoundError(f"train_args.json not found in {ckpt_dir}")
        print(f"[warn] no train_args.json in {ckpt_dir}; using {fallback}")
        args_path = fallback
    with args_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_config_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (_REPO / p).resolve()
    return p


def _build_model(env: RestaurantSymbolicEnv, meta: Dict[str, Any], device: torch.device) -> RestaurantQNetwork:
    obs_dim = env.observation_space.shape[0]
    action_type_dim = int(env.action_space["action_type"].n)
    object_dim = int(env.action_space["object1"].n)
    location_dim = int(env.action_space["location"].n)
    hidden_dim = int(meta["hidden_dim"])
    center = not bool(meta.get("no_dueling_centering", False))
    return RestaurantQNetwork(
        obs_dim, action_type_dim, object_dim, location_dim,
        hidden_dim=hidden_dim, center_advantages=center,
    ).to(device)


def _validate_meta_pair(ant_meta: Dict[str, Any], myo_meta: Dict[str, Any]) -> None:
    fields = ["config_path", "hidden_dim", "max_steps_per_task", "success_reward",
              "invalid_action_penalty", "env_reset_tasks"]
    for f in fields:
        av, mv = ant_meta.get(f), myo_meta.get(f)
        if av != mv:
            raise ValueError(f"Mismatched {f}: ant={av} myo={mv}. Paired eval requires identical settings.")
    if myo_meta.get("tasks_per_episode", 1) != 1:
        raise ValueError(f"Myopic checkpoint must have tasks_per_episode=1, got {myo_meta.get('tasks_per_episode')}")
    if ant_meta.get("tasks_per_episode", 1) <= 1:
        raise ValueError(f"Anticipatory checkpoint must have tasks_per_episode > 1, got {ant_meta.get('tasks_per_episode')}")
    ant_c = not bool(ant_meta.get("no_dueling_centering", False))
    myo_c = not bool(myo_meta.get("no_dueling_centering", False))
    if ant_c != myo_c:
        raise ValueError(f"Centering mismatch: ant={ant_c} myo={myo_c}")


def _ensure_sequence_consistent(meta: Dict[str, Any], seq_len: int) -> None:
    """Warn if sequence length != env_reset_tasks (the intended one-world horizon)."""
    env_reset = meta.get("env_reset_tasks")
    if env_reset is not None and seq_len != env_reset:
        print(f"[warn] Sequence length ({seq_len}) != env_reset_tasks ({env_reset}). Using sequence length.")


def _load_sequence(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["tasks"])


def _eval_one_agent(
    meta: Dict[str, Any],
    ckpt_path: Path,
    tasks: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    device = select_device()

    config_path = _resolve_config_path(meta["config_path"])
    env = RestaurantSymbolicEnv(
        config_path=config_path,
        rng_seed=seed,
        max_steps_per_task=int(meta["max_steps_per_task"]),
        success_reward=float(meta["success_reward"]),
        invalid_action_penalty=float(meta["invalid_action_penalty"]),
    )
    env.reset(seed=seed)

    model = _build_model(env, meta, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    action_gen = torch.Generator(device="cpu")
    action_gen.manual_seed(seed)

    records: List[Dict[str, Any]] = []
    succ_cnt = 0
    auto_cnt = 0
    trunc_cnt = 0
    total_steps = 0
    total_return = 0.0
    initial_paper2 = float(env._paper2_total_cost)
    paper2_costs: List[float] = []

    for idx, tdict in enumerate(tasks):
        task = RestaurantTask(
            task_type=str(tdict["task_type"]),
            target_location=tdict.get("target_location"),
            target_kind=tdict.get("target_kind"),
            object_name=tdict.get("object_name"),
        )
        env.set_task(
            task.task_type,
            target_location=task.target_location,
            target_kind=task.target_kind,
            object_name=task.object_name,
        )
        obs = env._obs()
        info = env._info(success=False)

        # Assert the task in info matches the assigned task.
        info_task = info["task"]
        assert info_task["task_type"] == task.task_type, \
            f"Task {idx}: info task_type mismatch: {info_task['task_type']} != {task.task_type}"
        assert info_task.get("target_location") == task.target_location, \
            f"Task {idx}: target_location mismatch: {info_task.get('target_location')} != {task.target_location}"
        assert info_task.get("target_kind") == task.target_kind, \
            f"Task {idx}: target_kind mismatch: {info_task.get('target_kind')} != {task.target_kind}"
        assert info_task.get("object_name") == task.object_name, \
            f"Task {idx}: object_name mismatch: {info_task.get('object_name')} != {task.object_name}"

        auto = bool(info.get("next_auto_satisfied", False))
        prev_paper2 = float(env._paper2_total_cost)

        task_steps = 0
        task_return = 0.0
        success = False
        truncated = False

        while True:
            action = _sample_structured_action(
                model, obs, info,
                temperature=0.0, generator=action_gen, device=device,
            )
            obs, reward, success, truncated, info = env.step(action)
            task_steps += 1
            task_return += float(reward)
            if success or truncated:
                break

        post_paper2 = float(env._paper2_total_cost)
        paper2_delta = post_paper2 - prev_paper2
        paper2_costs.append(paper2_delta)
        total_steps += task_steps
        total_return += task_return
        if success:
            succ_cnt += 1
        if truncated:
            trunc_cnt += 1
        if auto:
            auto_cnt += 1

        records.append({
            "task_idx": idx,
            "task_type": task.task_type,
            "target_location": task.target_location,
            "target_kind": task.target_kind,
            "object_name": task.object_name,
            "success": success,
            "truncated": truncated,
            "auto_satisfied": auto,
            "steps": task_steps,
            "return": task_return,
            "pddl_cost": paper2_delta,
        })

    n = len(records)
    return {
        "checkpoint": str(ckpt_path),
        "meta": {
            "config_path": meta["config_path"],
            "hidden_dim": int(meta["hidden_dim"]),
            "success_reward": float(meta["success_reward"]),
            "invalid_action_penalty": float(meta["invalid_action_penalty"]),
            "max_steps_per_task": int(meta["max_steps_per_task"]),
            "env_reset_tasks": meta.get("env_reset_tasks"),
            "tasks_per_episode": meta.get("tasks_per_episode"),
            "no_dueling_centering": bool(meta.get("no_dueling_centering", False)),
            "seed": int(meta.get("seed", -1)),
        },
        "tasks": records,
        "summary": {
            "n_tasks": n,
            "success_count": succ_cnt,
            "success_rate": float(succ_cnt / max(1, n)),
            "truncation_count": trunc_cnt,
            "truncation_rate": float(trunc_cnt / max(1, n)),
            "auto_count": auto_cnt,
            "auto_rate": float(auto_cnt / max(1, n)),
            "total_steps": total_steps,
            "mean_steps": float(total_steps / max(1, n)),
            "total_return": total_return,
            "mean_return": float(total_return / max(1, n)),
            "total_pddl_cost": sum(paper2_costs),
            "mean_pddl_cost": float(sum(paper2_costs) / max(1, n)),
            "initial_pddl_total": initial_paper2,
            "final_pddl_total": float(env._paper2_total_cost),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paired greedy DQN evaluation on a fixed canonical task sequence."
    )
    p.add_argument("--ant-ckpt", type=Path, required=True, help="Anticipatory checkpoint .pt or directory")
    p.add_argument("--myo-ckpt", type=Path, default=None,
                   help="Myopic checkpoint .pt or directory. Omit to evaluate the "
                        "anticipatory agent alone (the paired comparison is then skipped).")
    p.add_argument("--train-args", type=Path, default=None,
                   help="train_args.json to fall back on when a checkpoint dir lacks one")
    p.add_argument("--seq", type=Path, required=True, help="Canonical task sequence JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = p.parse_args()

    ant_meta = _load_checkpoint_dir(args.ant_ckpt, args.train_args)
    myo_meta = (_load_checkpoint_dir(args.myo_ckpt, args.train_args)
                if args.myo_ckpt is not None else None)
    if myo_meta is not None:
        _validate_meta_pair(ant_meta, myo_meta)

    # Resolve actual .pt paths
    ant_pt = args.ant_ckpt if args.ant_ckpt.suffix == ".pt" else args.ant_ckpt / "restaurant_dqn.pt"
    if not ant_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ant_pt}")
    myo_pt = None
    if args.myo_ckpt is not None:
        myo_pt = args.myo_ckpt if args.myo_ckpt.suffix == ".pt" else args.myo_ckpt / "restaurant_dqn.pt"
        if not myo_pt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {myo_pt}")

    tasks = _load_sequence(args.seq)
    _ensure_sequence_consistent(ant_meta, len(tasks))

    ant_result = _eval_one_agent(ant_meta, ant_pt, tasks, args.seed)

    output = {
        "sequence": {"path": str(args.seq), "n_tasks": len(tasks)},
        "eval_seed": args.seed,
        "anticipatory": ant_result,
    }
    if myo_pt is not None:
        output["myopic"] = _eval_one_agent(myo_meta, myo_pt, tasks, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
