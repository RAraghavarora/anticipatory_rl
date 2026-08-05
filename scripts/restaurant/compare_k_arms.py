#!/usr/bin/env python3
"""Compare clairvoyant-oracle K arms, controlling for planner convergence.

seq-sat-lama-2011 is an anytime satisficing planner: window cost depends on how
much search budget it got. Larger K => larger joint search space => more windows
hit the timeout => worse plans. So a raw total-cost comparison across K conflates
lookahead depth with compute budget.

This reports (a) the timeout rate per arm, and (b) paired per-window deltas split
by whether BOTH arms converged, so the lookahead effect is separable from the
budget artifact.

Usage:
    python scripts/restaurant/compare_k_arms.py runs/level4_seq01_k*_t600.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_arm(path: Path, timeout_s: float) -> dict:
    d = json.loads(path.read_text())
    # A window that used ~all its budget never converged; its plan is whatever
    # lama had at the cutoff. Tolerance absorbs teardown overhead past the limit.
    windows = {
        w["index"]: {
            "cost": w.get("first_segment_cost") or 0.0,
            "timeout": (w.get("solve_time_s") or 0.0) >= timeout_s - 1.0,
        }
        for w in d["windows"]
    }
    return {
        "path": path,
        "K": d["K"],
        "sequence_id": d.get("sequence_id"),
        "total": d.get("whole_sequence_physical_cost"),
        "complete": d.get("sequence_complete"),
        "windows": windows,
    }


def paired(base: dict, other: dict) -> None:
    common = sorted(set(base["windows"]) & set(other["windows"]))
    groups = {
        "both converged": [i for i in common
                           if not base["windows"][i]["timeout"] and not other["windows"][i]["timeout"]],
        "both timed out": [i for i in common
                           if base["windows"][i]["timeout"] and other["windows"][i]["timeout"]],
    }
    print(f"\nK={other['K']} vs K={base['K']}  ({len(common)} common windows)")
    for label, idxs in groups.items():
        if not idxs:
            print(f"  {label:15s} n=0")
            continue
        a = sum(base["windows"][i]["cost"] for i in idxs)
        b = sum(other["windows"][i]["cost"] for i in idxs)
        pct = f"{100 * (b - a) / a:+.2f}%" if a else "n/a"
        # Zero-cost windows are auto_success: they tie by construction and dilute
        # the signal, so report how many windows actually carry any.
        live = [i for i in idxs if base["windows"][i]["cost"] or other["windows"][i]["cost"]]
        wins = sum(1 for i in live if other["windows"][i]["cost"] < base["windows"][i]["cost"])
        loss = sum(1 for i in live if other["windows"][i]["cost"] > base["windows"][i]["cost"])
        print(f"  {label:15s} n={len(idxs):3d}  K={base['K']} {a:9.0f}  K={other['K']} {b:9.0f}"
              f"  delta {b - a:+9.0f} ({pct})")
        print(f"  {'':15s} nonzero-cost windows={len(live)}  "
              f"K={other['K']} cheaper {wins}, worse {loss}, tied {len(live) - wins - loss}")
        if live:
            deltas = sorted(other["windows"][i]["cost"] - base["windows"][i]["cost"] for i in live)
            print(f"  {'':15s} drop largest single win -> delta {sum(deltas) - deltas[0]:+.0f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--timeout-s", type=float, default=600.0,
                   help="budget the runs were given; used to classify windows as timed out")
    args = p.parse_args()

    arms = sorted((load_arm(path, args.timeout_s) for path in args.results), key=lambda a: a["K"])
    seqs = {a["sequence_id"] for a in arms}
    assert len(seqs) == 1, f"arms span different sequences: {seqs}"

    print(f"sequence: {seqs.pop()}   budget: {args.timeout_s}s")
    for a in arms:
        w = a["windows"]
        n_to = sum(x["timeout"] for x in w.values())
        print(f"  K={a['K']}  total={a['total']:>10}  complete={a['complete']}  "
              f"timed out {n_to}/{len(w)} ({100 * n_to / len(w):.0f}%)")

    # Every pair, not just vs the lowest K: the K=3-vs-K=2 contrast is the one that
    # separates this work from one-task-lookahead prior work.
    for i, base in enumerate(arms):
        for other in arms[i + 1:]:
            paired(base, other)


if __name__ == "__main__":
    main()
