"""One-off correction: myopic_dqn_beta1 seed=16 used a checkpoint that diverged
late in training (Q-value instability). Retrained checkpoint now lives at
runs/v3_myopic_g0.97_peb_s16/. This script regenerates the guided-planner and
greedy-RL results for that seed from the new checkpoint and splices them into
results/canonical_planner/{planner,greedy_rl}/*.csv, replacing the old rows.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Resume support: a prior run of this script died (tmux server killed on the
# remote host) partway through part_a_planner. Any raw output already
# rewritten after this cutoff is trusted as correct (generated from the new
# checkpoint) and is loaded from disk instead of recomputed.
RESUME_CUTOFF = datetime.datetime(2026, 8, 6, 0, 0, 0)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from evaluate_bellman_novelty_sequence import run_sequence
from rl_direct_seq_eval import _eval_one_agent, _load_checkpoint_dir

CANON = _REPO / "results" / "canonical_planner"
NEW_CKPT_DIR = _REPO / "runs" / "v3_myopic_g0.97_peb_s16"
SEQ_DIR = _REPO / "experiments" / "sequences"
SEQ_IDS = [f"iid-eval-seq-{i:02d}" for i in range(10)]
SEED = 16
SUCCESS_REWARD = 81.06943684690286


def part_a_planner():
    """Re-run cost_bounded (guided) eval for seed16, all 10 sequences."""
    raw_dir = CANON / "planner" / "raw" / "myopic_dqn_guided"
    run_rows, task_rows = [], []
    manifest = json.loads((CANON / "manifest.json").read_text())
    seq_meta = {s["sequence_id"]: s for s in manifest["canonical_sequences"]}

    for idx, seq_id in enumerate(SEQ_IDS):
        seq_path = SEQ_DIR / f"{seq_id}.json"
        out_path = raw_dir / f"myopic_seed16_{seq_id}.json"

        mtime = datetime.datetime.fromtimestamp(out_path.stat().st_mtime) if out_path.exists() else None
        if mtime is not None and mtime >= RESUME_CUTOFF:
            result = json.loads(out_path.read_text())
            print(f"[planner] resume: reusing already-correct {out_path} (mtime {mtime})")
        else:
            result = run_sequence(
                policy="cost_bounded",
                sequence_path=seq_path,
                config_path=_REPO / "configs" / "restaurant" / "toy_level_3.yaml",
                domain_path=_REPO / "pddl" / "toy_restaurant_domain.pddl",
                planner_path=_REPO / "downward" / "fast-downward.py",
                alias="seq-sat-lama-2011",
                fd_timeout_s=20.0,
                seed=0,
                gamma=0.97,
                success_reward=SUCCESS_REWARD,
                hidden_dim=256,
                max_depth=20,
                max_expansions=5000,
                cost_ratio=1.0,
                q_weights=NEW_CKPT_DIR / "restaurant_dqn.pt",
            )
            out_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"[planner] wrote {out_path}")

        smy = result["summary"]
        sm = seq_meta[seq_id]
        run_id = f"myopic_dqn_beta1__seed16__seq{idx:02d}"
        source_json = f"results/bellman_myopic_beta1.0/myopic_seed16_{seq_id}.json"
        run_rows.append({
            "method_id": "myopic_dqn_beta1", "run_id": run_id,
            "sequence_index": idx, "sequence_id": seq_id, "sequence_sha256": sm["sha256"],
            "checkpoint_seed": 16, "eval_seed": 0, "beta": 1.0, "gamma": 0.97, "K": "",
            "task_count": sm["task_count"], "completed_count": smy["completed"],
            "total_cost_pddl": smy["total_pddl_cost"], "mean_cost_pddl": round(smy["total_pddl_cost"] / sm["task_count"], 4),
            "total_actions": smy["total_actions"], "auto_success_count": smy["auto_count"],
            "auto_success_rate": smy["auto_rate"], "source_json": source_json,
            "source_sha256": "",  # filled after write
        })
        for t_idx, rec in enumerate(result["tasks"]):
            task_rows.append({
                "method_id": "myopic_dqn_beta1", "run_id": run_id,
                "sequence_index": idx, "sequence_id": seq_id, "sequence_sha256": sm["sha256"],
                "task_index": t_idx, "global_task_index": t_idx,
                "checkpoint_seed": 16, "eval_seed": 0, "beta": 1.0, "gamma": 0.97, "K": "",
                "task_type": rec["task_type"], "target_location": rec["target_location"],
                "target_kind": rec["target_kind"], "object_name": rec["object_name"],
                "task_cost_pddl": rec["cost"], "action_count": rec["actions"],
                "auto_success": int(rec["auto"]), "success": int(rec["success"]),
                "source_json": source_json,
            })

    import hashlib
    for row in run_rows:
        seq_id = row["sequence_id"]
        p = raw_dir / f"myopic_seed16_{seq_id}.json"
        row["source_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Splice into planner/run_summary.csv and planner/task_results.csv
    for fname, new_rows in [("run_summary.csv", run_rows), ("task_results.csv", task_rows)]:
        path = CANON / "planner" / fname
        df = pd.read_csv(path, dtype={"checkpoint_seed": "Int64"})
        keep = ~((df["method_id"] == "myopic_dqn_beta1") & (df["checkpoint_seed"] == 16))
        df = df[keep]
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        seed_order = {0: 0, 4: 1, 8: 2, 16: 3}
        method_order = {"myopic_fd_optimal": 0, "clairvoyant_k3_lama": 1, "myopic_dqn_beta1": 2, "anticipatory_dqn_beta1_25": 3}
        df["_m"] = df["method_id"].map(method_order)
        df["_s"] = df["checkpoint_seed"].map(lambda v: seed_order.get(v, -1) if pd.notna(v) else -1)
        df = df.sort_values(["_m", "_s", "sequence_index"]).drop(columns=["_m", "_s"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"[planner] rewrote {path} ({len(df)} rows)")

    return {row["source_json"]: row["source_sha256"] for row in run_rows}


def part_b_greedy():
    """Re-run greedy (direct policy) eval for seed16, all 10 sequences."""
    meta = _load_checkpoint_dir(NEW_CKPT_DIR)
    ckpt_pt = NEW_CKPT_DIR / "restaurant_dqn.pt"

    seed_rows, run_rows, task_rows = [], [], []
    total_succ = total_auto = total_steps = 0
    total_cost = 0.0
    total_n = 0

    for idx, seq_id in enumerate(SEQ_IDS):
        seq_path = SEQ_DIR / f"{seq_id}.json"
        tasks = json.loads(seq_path.read_text())["tasks"]
        result = _eval_one_agent(meta, ckpt_pt, tasks, seed=0)
        s = result["summary"]
        run_rows.append({
            "method_id": "myopic_dqn_greedy", "checkpoint_seed": 16, "checkpoint_variant": "final",
            "sequence_index": idx, "mean_cost_pddl": s["mean_pddl_cost"], "success_rate": s["success_rate"],
        })
        for t in result["tasks"]:
            task_rows.append({
                "method_id": "myopic_dqn_greedy", "checkpoint_seed": 16, "checkpoint_variant": "final",
                "sequence_index": idx, "task_index": t["task_idx"], "task_cost_pddl": t["pddl_cost"],
            })
        total_succ += s["success_count"]
        total_auto += s["auto_count"]
        total_steps += s["total_steps"]
        total_cost += s["total_pddl_cost"]
        total_n += s["n_tasks"]
        print(f"[greedy] {seq_id}: success={s['success_rate']:.2f} cost={s['mean_pddl_cost']:.1f}")

    seed_rows.append({
        "method_id": "myopic_dqn_greedy", "checkpoint_seed": 16, "checkpoint_variant": "final",
        "task_count": total_n, "success_rate": round(total_succ / total_n, 4),
        "auto_success_rate": round(total_auto / total_n, 4), "mean_steps": round(total_steps / total_n, 4),
        "mean_cost_pddl": round(total_cost / total_n, 4),
    })

    # Splice run_summary.csv and task_results.csv (drop ALL seed16 rows: old "final" + old "best")
    for fname, new_rows in [("run_summary.csv", run_rows), ("task_results.csv", task_rows)]:
        path = CANON / "greedy_rl" / fname
        df = pd.read_csv(path)
        keep = ~((df["method_id"] == "myopic_dqn_greedy") & (df["checkpoint_seed"] == 16))
        df = df[keep]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"[greedy] rewrote {path} ({len(df)} rows)")

    # seed_summary.csv: drop old seed16 rows (final + best), add new final-only row
    path = CANON / "greedy_rl" / "seed_summary.csv"
    df = pd.read_csv(path)
    df = df[~((df["method_id"] == "myopic_dqn_greedy") & (df["checkpoint_seed"] == 16))]
    df = pd.concat([df, pd.DataFrame(seed_rows)], ignore_index=True)
    df = df.sort_values(["method_id", "checkpoint_seed"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[greedy] rewrote {path} ({len(df)} rows)")

    # method_summary.csv: recompute myopic_dqn_greedy aggregate across seeds {0,4,8,16}, all "final"
    myo = df[(df["method_id"] == "myopic_dqn_greedy")]
    agg = {
        "method_id": "myopic_dqn_greedy", "seed_count": len(myo), "tasks_per_seed": int(myo["task_count"].iloc[0]),
        "success_rate_mean": myo["success_rate"].mean(), "success_rate_sd": myo["success_rate"].std(ddof=1),
        "auto_success_rate_mean": myo["auto_success_rate"].mean(), "auto_success_rate_sd": myo["auto_success_rate"].std(ddof=1),
        "mean_steps_mean": myo["mean_steps"].mean(), "mean_steps_sd": myo["mean_steps"].std(ddof=1),
        "mean_cost_pddl_mean": myo["mean_cost_pddl"].mean(), "mean_cost_pddl_sd": myo["mean_cost_pddl"].std(ddof=1),
    }
    ms_path = CANON / "greedy_rl" / "method_summary.csv"
    ms = pd.read_csv(ms_path)
    ms = ms[ms["method_id"] != "myopic_dqn_greedy"]
    ms = pd.concat([ms, pd.DataFrame([agg])], ignore_index=True)
    ms.to_csv(ms_path, index=False)
    print(f"[greedy] rewrote {ms_path}")
    print("[greedy] new myopic_dqn_greedy aggregate:", agg)


if __name__ == "__main__":
    new_source_hashes = part_a_planner()
    part_b_greedy()
    print("\nNew source_result_sha256 entries for manifest.json:")
    print(json.dumps(new_source_hashes, indent=2))
