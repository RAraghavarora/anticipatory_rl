#!/usr/bin/env python3
"""Gate a demo .pt before spending a training run on it.

Demos exist to seed one rare behaviour: the jar investment. If they don't contain it,
no number of training steps recovers it, so check before training rather than after.

Usage:
    PYTHONPATH=. python scripts/restaurant/check_demos.py demos/foo.pt \
        [--config-path configs/restaurant/toy_level_5.yaml] [--success-reward 74.0125]
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import torch

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("demo_path", type=Path)
    p.add_argument("--config-path", type=Path, default=None,
                   help="If given, verify the demo was built from this exact config.")
    p.add_argument("--success-reward", type=float, default=None,
                   help="If given, verify the demo baked in this R*.")
    args = p.parse_args()

    blob = torch.load(args.demo_path, map_location="cpu", weights_only=False)
    meta, trans = blob["metadata"], blob["transitions"]
    objs, locs = meta["object_names"], meta["locations"]

    print(f"=== {args.demo_path.name} ===")
    for k in ("K", "credit_horizon", "n_outcomes", "env_reset_tasks", "stored",
              "success_reward", "seed", "max_steps_per_task"):
        print(f"  {k:20s} {meta.get(k)}")

    problems: list[str] = []

    # 1. stored count must match what is actually in the file
    if len(trans) != meta.get("stored"):
        problems.append(f"metadata says stored={meta.get('stored')} but file has {len(trans)}")

    # 2. wrong config is silent and fatal -- a level_3 demo trains a level_5 agent on the
    #    wrong jar economics entirely
    if args.config_path is not None:
        want = hashlib.sha256(args.config_path.read_bytes()).hexdigest()
        if want != meta.get("config_hash"):
            problems.append(f"config_hash mismatch: demo was NOT built from {args.config_path}")
        else:
            print(f"  config_hash          matches {args.config_path}")

    # 3. R* must match what training will use, or demo rewards are on a different scale
    if args.success_reward is not None:
        got = float(meta.get("success_reward", float("nan")))
        if abs(got - args.success_reward) > 1e-6:
            problems.append(f"success_reward mismatch: demo={got} vs expected={args.success_reward}")
        else:
            print(f"  success_reward       matches {args.success_reward}")

    # decode factored actions; match /jar/ generically -- never a specific index
    jar_ids = {i for i, n in enumerate(objs) if "jar" in n}
    acts: Counter = Counter()
    jar_hits: list[str] = []
    boundaries = 0
    rewards = []
    for t in trans:
        at = int(t["action_type"])
        o1, o2, loc = int(t["object1"]), int(t["object2"]), int(t["location"])
        name = ACTION_TYPES[at] if at < len(ACTION_TYPES) else f"?{at}"
        acts[name] += 1
        rewards.append(float(t["reward"]))
        boundaries += int(float(t.get("task_boundary", 0.0))) if "task_boundary" in t.keys() else 0
        if o1 in jar_ids or o2 in jar_ids:
            o1n = objs[o1] if o1 < len(objs) else str(o1)
            o2n = objs[o2] if o2 < len(objs) else str(o2)
            lname = locs[loc] if loc < len(locs) else str(loc)
            jar_hits.append(f"{name}(o1={o1n}, o2={o2n}, loc={lname})")

    print(f"\n  transitions          {len(trans)}")
    print(f"  reward  min/max      {min(rewards):.2f} / {max(rewards):.2f}")
    print(f"  action mix           {dict(acts.most_common())}")

    print(f"\n  jar transitions      {len(jar_hits)}")
    for h in Counter(jar_hits).most_common(12):
        print(f"     {h[1]:3d}x  {h[0]}")

    # The investment: pick(jar) then refill_water(cup, jar) later.
    # Do NOT look for "pantry" in the action -- pick(object) and refill_water(container, jar)
    # take no location argument (it shows as the none_location sentinel), and place(location)
    # records the location but not the object. The agent's position lives in the state, not
    # the action. Since the jar starts at the pantry, any pick of it implies the trip.
    fetched = [h for h in jar_hits if h.startswith("pick")]
    used = [h for h in jar_hits if h.startswith("refill_water")]
    filled = [h for h in jar_hits if h.startswith("fill")]
    print(f"\n  jar picked (investment) {len(fetched)}")
    print(f"  jar filled              {len(filled)}")
    print(f"  refill_water via jar    {len(used)}")

    if not fetched:
        problems.append("NO pick of the jar -- the investment is absent; "
                        "training on this cannot learn it")
    if not used:
        problems.append("NO refill_water using the jar -- the payoff half is absent")
    if fetched and used:
        print(f"\n  -> investment demonstrated {len(fetched)}x across "
              f"{max(1, meta['n_outcomes'] // max(1, meta['env_reset_tasks']))} episodes")

    print()
    if problems:
        print("FAIL")
        for x in problems:
            print(f"  - {x}")
        raise SystemExit(1)
    print("PASS: demos contain the jar investment and its payoff")


if __name__ == "__main__":
    main()
