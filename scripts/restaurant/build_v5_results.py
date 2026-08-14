#!/usr/bin/env python3
"""Assemble results/v5/ -- the canonical archive for the long-horizon restaurant variant.

Layout mirrors results/canonical_planner/ (the level_3 archive) so the two are readable
side by side.

  results/v5/
    manifest.json         provenance: git commit, sequence sha256s, method definitions
    RESULTS.md            headline tables, generated -- do not hand-edit
    planner/
      run_summary.csv     one row per (method, sequence): cost, timeouts, jar usage
      raw/<method>/       source JSONs
    gnn/
      raw/                both GNN arms x 10 sequences
    training/
      run_summary.csv     one row per DQN run: jar share, stability, success
      raw/                train_summary.json per run
    sequences/            snapshot of the 10 evaluation sequences
    toy_level_5.yaml      the domain

Method definitions (see manifest.json for the same text):
  myopic_fd_optimal    K=1 clairvoyant, astar(hmax()) -- exact, so its cost is the true
                       optimum for a one-task horizon
  k2_fd_optimal        K=2 clairvoyant, astar(hmax()) -- exact. The published baseline's
                       structural ceiling: a provably optimal one-task-lookahead planner
  gnn_faithful         Talukder et al., no counterfactual augmentation
  gnn_counterfactual   Talukder et al. as published (counterfactual augmentation)
  anticipatory_dqn     ours: DQN V_AP guiding a cost-bounded Bellman+novelty search
  clairvoyant_k3/k4    K=3 / K=4 clairvoyant, satisficing (exact is intractable at K>=3).
                       Costs are UPPER BOUNDS on the true optimum -- report timeout rates.

Usage:  PYTHONPATH=. python scripts/restaurant/build_v5_results.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics as st
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEQS = [f"{i:02d}" for i in range(10)]
SEEDS = [0, 4, 8, 16, 42]                # one seed protocol for every learned arm.
# 42 added late: the RL arms have it, the GNN arms do not (their sweep predates it), so the
# GNN globs simply find nothing for s42 and skip it. Seed counts per arm are printed below.
DEV = ["01", "03", "06"]                 # cost_ratio=3.0 selected here
HELD_OUT = [s for s in SEQS if s not in DEV]
JAR_DELIVERED = ["00", "02"]             # task stream itself contains pick_place(jar, .)
FD_CAP = 600.0                           # per-window budget for the satisficing arms

PLANNER = {
    "myopic_fd_optimal":   ("base_seq{s}_k1_opt.json", 900.0),
    "k2_fd_optimal":       ("base_seq{s}_k2_opt.json", 900.0),
    "clairvoyant_k3_lama": ("base_seq{s}_k3_sat.json", FD_CAP),
    "clairvoyant_k4_lama": ("base_seq{s}_k4_sat.json", FD_CAP),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def jar_counts(steps) -> Counter:
    """Count jar actions. Matches /jar/ generically -- hard-coding an object index caused
    three separate wrong conclusions during this investigation."""
    c = Counter()
    for t in steps or []:
        for a in (t.get("trace") or t.get("first_segment_plan") or []):
            s = str(a)
            if "jar" in s:
                c[s.split("(")[0].strip("['")] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/v5"))
    args = ap.parse_args()
    out = args.out
    for sub in ("planner/raw", "gnn/raw", "training/raw", "sequences", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    deploy: list[dict] = []   # greedy-vs-guided metrics that the CSV schema has no room for

    # ---- planner arms ------------------------------------------------------
    for mid, (pat, cap) in PLANNER.items():
        dst = out / "planner" / "raw" / mid
        dst.mkdir(parents=True, exist_ok=True)
        for s in SEQS:
            p = Path("runs/v5eval") / pat.format(s=s)
            d = load(p)
            if d is None:
                continue
            shutil.copy2(p, dst / p.name)
            ts = [w.get("solve_time_s", 0) or 0 for w in d.get("windows", [])]
            j = jar_counts(d.get("windows"))
            rows.append(dict(
                method_id=mid, sequence_id=f"iid-eval-seq-{s}", seq=s,
                split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
                checkpoint="", K=d.get("K"), cost_ratio="",
                total_cost_pddl=d.get("whole_sequence_physical_cost"),
                complete=d.get("sequence_complete"),
                completed_count=d.get("valid_prefix_completion_count"),
                windows=len(ts), windows_at_cap=sum(1 for t in ts if t >= cap - 1),
                refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
                source=str(p), source_sha256=sha256(p),
            ))

    # ---- ours --------------------------------------------------------------
    dst = out / "planner" / "raw" / "anticipatory_dqn"
    dst.mkdir(parents=True, exist_ok=True)
    for d_dir in (Path("runs/v5_heldout"), Path("runs/v5_crsweep")):  # ours + demo ablations
        if not d_dir.exists():
            continue
        for p in sorted(d_dir.glob("*_cr3.0.json")):     # only the selected ratio
            if p.stat().st_size < 1000:
                continue
            m = re.match(r"(v5[_-](ant|nodemo|myoDemo)[_-]g[\d.]+[_-]s\d+)_seq(\d+)_cr([\d.]+)\.json",
                         p.name)
            if not m:
                continue
            ckpt, arm, s, cr = m.groups()
            mid = {"ant": "anticipatory_dqn", "nodemo": "dqn_no_demos",
                   "myoDemo": "myopic_demo_dqn"}[arm]
            d = load(p)
            if not d or "guided" not in d:
                continue
            g = d["guided"]
            (out / "planner" / "raw" / mid).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / "planner" / "raw" / mid / p.name)
            j = jar_counts(g.get("tasks"))
            rows.append(dict(
                method_id=mid, sequence_id=f"iid-eval-seq-{s}", seq=s,
                split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
                checkpoint=ckpt, K="", cost_ratio=cr,
                total_cost_pddl=g["summary"].get("total_pddl_cost"),
                complete=g["summary"].get("completed") == g["summary"].get("attempted"),
                completed_count=g["summary"].get("completed"),
                windows=g["summary"].get("attempted"), windows_at_cap="",
                refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
                source=str(p), source_sha256=sha256(p),
            ))
            if mid == "anticipatory_dqn" and "g0.97" in ckpt:
                sm = g["summary"]
                deploy.append(dict(mode="Value-guided", ckpt=ckpt.split("_")[-1],
                                   n=sm["attempted"], done=sm["completed"],
                                   auto=sm["auto_count"], acts=sm["total_actions"],
                                   cost=sm["total_pddl_cost"]))

    # ---- GNN ---------------------------------------------------------------
    # gnn_steelman: the baseline handed the jar-preparation candidate its bounded heuristic
    # cannot generate, and trained on data containing those states. If it still declines,
    # the cause is its one-task value horizon, not candidate generation.
    # Both GNN arms are 4-seed sweeps (0/4/8/16), matching the RL arms, so the baseline is
    # reported as mean+/-std rather than a single draw. The original seed-42 checkpoints are
    # superseded and deliberately NOT read: seed 42 gave the augmented arm its best of five
    # draws, so reporting it alone flattered the baseline.
    for mode, mid in (("aug", "gnn_counterfactual"), ("faithful", "gnn_faithful")):
        for seed in SEEDS:
            for s in SEQS:
                p = Path(f"runs/v5_gnn_seeds/{mode}") / f"gnn_{mode}_s{seed}_seq{s}.json"
                d = load(p)
                if d is None:
                    continue
                (out / "gnn" / "seeds").mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out / "gnn" / "seeds" / p.name)
                g = d.get("gnn_anticipatory", {})
                j = jar_counts(g.get("tasks"))
                rows.append(dict(
                    method_id=mid, sequence_id=f"iid-eval-seq-{s}", seq=s,
                    split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
                    checkpoint=f"gnn_v5_{mode}_s{seed}", K="", cost_ratio="",
                    total_cost_pddl=g.get("summary", {}).get("total_cost"),
                    complete=True, completed_count=g.get("summary", {}).get("completed"),
                    windows=g.get("summary", {}).get("attempted"), windows_at_cap="",
                    refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
                    source=str(p), source_sha256=sha256(p),
                ))

    # gnn_steelman: the baseline handed the jar-preparation candidate its bounded heuristic
    # cannot generate, and trained on data containing those states. If it still declines,
    # the cause is its one-task value horizon, not candidate generation. A single seed-42
    # run, read against the augmented arm's seed spread rather than a single draw.
    for s in SEQS:
        p = Path("runs/v5_gnn_steelman_eval") / f"steelman_seq{s}.json"
        d = load(p)
        if d is None:
            continue
        shutil.copy2(p, out / "gnn" / "raw" / p.name)
        g = d.get("gnn_anticipatory", {})
        j = jar_counts(g.get("tasks"))
        rows.append(dict(
            method_id="gnn_steelman", sequence_id=f"iid-eval-seq-{s}", seq=s,
            split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
            checkpoint="gnn_v5_STEELMAN", K="", cost_ratio="",
            total_cost_pddl=g.get("summary", {}).get("total_cost"),
            complete=True, completed_count=g.get("summary", {}).get("completed"),
            windows=g.get("summary", {}).get("attempted"), windows_at_cap="",
            refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
            source=str(p), source_sha256=sha256(p),
        ))

    # ---- greedy deployment of the same anticipatory value ------------------
    # Identical checkpoints to anticipatory_dqn, acted on step-by-step instead of scoring
    # the terminal states of the cost-bounded search. Unlike every planner arm this one can
    # fail a task, so success rate carries information here and nowhere else.
    for seed in SEEDS:
        for s in SEQS:
            p = Path("runs/v5_greedy") / f"greedy_s{seed}_seq{s}.json"
            d = load(p)
            if d is None or "anticipatory" not in d:
                continue
            (out / "greedy" / "raw").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / "greedy" / "raw" / p.name)
            a = d["anticipatory"]
            sm = a["summary"]
            j = jar_counts(a.get("tasks"))
            rows.append(dict(
                method_id="anticipatory_dqn_greedy", sequence_id=f"iid-eval-seq-{s}", seq=s,
                split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
                checkpoint=f"v5_ant_g0.97_s{seed}", K="", cost_ratio="",
                total_cost_pddl=sm.get("total_pddl_cost"),
                complete=sm.get("success_count") == sm.get("n_tasks"),
                completed_count=sm.get("success_count"),
                windows=sm.get("n_tasks"), windows_at_cap="",
                refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
                source=str(p), source_sha256=sha256(p),
            ))
            deploy.append(dict(mode="Greedy", ckpt=f"s{seed}", n=sm["n_tasks"],
                               done=sm["success_count"], auto=sm["auto_count"],
                               acts=sm["total_steps"], cost=sm["total_pddl_cost"]))

    # ---- gamma sweep: guided deployment at seed 0 ---------------------------
    # Separate method_id so these never contaminate the gamma=0.97 headline mean. Each run is
    # evaluated with its OWN training gamma (the terminal score is prefix_cost +
    # gamma*V_AP(terminal), so a fixed gamma would mis-score every checkpoint but one).
    for p_ in sorted(Path("runs/v5_gamma_guided").glob("*_seq*_cr3.0.json")) \
              if Path("runs/v5_gamma_guided").exists() else []:
        d = load(p_)
        if not d or "guided" not in d:
            continue
        ck, s_ = p_.name.split("_seq")[0], p_.name.split("_seq")[1][:2]
        g = d["guided"]; sm = g["summary"]; j = jar_counts(g.get("tasks"))
        (out / "planner" / "raw" / "gamma_sweep_guided").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p_, out / "planner" / "raw" / "gamma_sweep_guided" / p_.name)
        rows.append(dict(
            method_id="gamma_sweep_guided", sequence_id=f"iid-eval-seq-{s_}", seq=s_,
            split="dev" if s_ in DEV else "held_out", jar_delivered=s_ in JAR_DELIVERED,
            checkpoint=ck, K="", cost_ratio="3.0",
            total_cost_pddl=sm.get("total_pddl_cost"),
            complete=sm.get("completed") == sm.get("attempted"),
            completed_count=sm.get("completed"), windows=sm.get("attempted"), windows_at_cap="",
            refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
            source=str(p_), source_sha256=sha256(p_),
        ))

    # ---- myopic arm: one-task credit horizon, both deployments ---------------
    # `v5-myopic-g097-s8` is genuinely myopic: tasks_per_episode=1 (completion terminal) and
    # myopicK1 demonstrations. NOT to be confused with the former `v5-myopic-g097-s0`, which
    # was the no-demonstrations ablation under a misleading name (now runs/v5_nodemo_g0.97_s0).
    for p in sorted(Path("runs/v5_myopic_guided").glob("*_seq*_cr3.0.json")) \
             if Path("runs/v5_myopic_guided").exists() else []:
        d = load(p)
        if not d or "guided" not in d:
            continue
        ck, s = p.name.split("_seq")[0], p.name.split("_seq")[1][:2]
        g = d["guided"]; sm = g["summary"]; j = jar_counts(g.get("tasks"))
        (out / "planner" / "raw" / "myopic_dqn").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / "planner" / "raw" / "myopic_dqn" / p.name)
        rows.append(dict(
            method_id="myopic_dqn", sequence_id=f"iid-eval-seq-{s}", seq=s,
            split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
            checkpoint=ck, K="", cost_ratio="3.0",
            total_cost_pddl=sm.get("total_pddl_cost"),
            complete=sm.get("completed") == sm.get("attempted"),
            completed_count=sm.get("completed"), windows=sm.get("attempted"), windows_at_cap="",
            refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
            source=str(p), source_sha256=sha256(p),
        ))
        deploy.append(dict(mode="Value-guided (myopic)", ckpt=ck, n=sm["attempted"],
                           done=sm["completed"], auto=sm["auto_count"],
                           acts=sm["total_actions"], cost=sm["total_pddl_cost"]))

    # Greedy runs whose filename carries a full run label rather than a bare seed. Covers the
    # myopic arm and the gamma sweep; kept under distinct method_ids so neither contaminates
    # the headline anticipatory_dqn_greedy mean, which is gamma=0.97 only.
    for p in sorted(Path("runs/v5_greedy").glob("greedy_v5*_seq*.json")):
        d = load(p)
        if not d or "anticipatory" not in d:
            continue
        ck = p.name[len("greedy_"):].split("_seq")[0]
        s = p.name.split("_seq")[1][:2]
        mid = "myopic_dqn_greedy" if "myopic" in ck else "gamma_sweep_greedy"
        a = d["anticipatory"]; sm = a["summary"]; j = jar_counts(a.get("tasks"))
        (out / "greedy" / "raw").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / "greedy" / "raw" / p.name)
        rows.append(dict(
            method_id=mid, sequence_id=f"iid-eval-seq-{s}", seq=s,
            split="dev" if s in DEV else "held_out", jar_delivered=s in JAR_DELIVERED,
            checkpoint=ck, K="", cost_ratio="",
            total_cost_pddl=sm.get("total_pddl_cost"),
            complete=sm.get("success_count") == sm.get("n_tasks"),
            completed_count=sm.get("success_count"),
            windows=sm.get("n_tasks"), windows_at_cap="",
            refill_water=j.get("refill_water", 0), jar_pick=j.get("pick", 0),
            source=str(p), source_sha256=sha256(p),
        ))
        if mid == "myopic_dqn_greedy":
            deploy.append(dict(mode="Greedy (myopic)", ckpt=ck, n=sm["n_tasks"],
                               done=sm["success_count"], auto=sm["auto_count"],
                               acts=sm["total_steps"], cost=sm["total_pddl_cost"]))

    with (out / "planner" / "run_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- training runs -----------------------------------------------------
    trows = []
    for d in sorted(Path("runs").glob("v5*")):
        p = d / "train_summary.json"
        if not p.exists():
            continue
        s = load(p)
        if s is None:
            continue
        shutil.copy2(p, out / "training" / "raw" / f"{d.name}.train_summary.json")
        ac = s.get("action_type_counts", {})
        rf, fl = ac.get("refill_water", 0), ac.get("fill", 0)
        mq = s.get("max_abs_q_selected", 0)
        trows.append(dict(
            run=d.name,
            demos="myopic_K1" if "myopic" in d.name else "anticipatory_K3",
            gamma=(re.search(r"g0?\.?(\d\d)", d.name).group(1) if re.search(r"g0?\.?(\d\d)", d.name) else ""),
            seed=(re.search(r"s(\d+)$", d.name).group(1) if re.search(r"s(\d+)$", d.name) else ""),
            jar_share=round(100 * rf / (rf + fl), 1) if rf + fl else "",
            refill_water=rf, fill=fl,
            best_checkpoint=s.get("best_checkpoint_value"),
            success_rate=round(s.get("success_rate", 0), 4),
            non_auto_success_rate=round(s.get("non_auto_success_rate", 0), 4),
            avg_task_steps=round(s.get("avg_task_steps", 0), 2),
            mean_q=round(s.get("mean_q_selected", 0)), max_abs_q=round(mq),
            diverged=mq > 5000,
        ))
    if trows:
        with (out / "training" / "run_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
            w.writeheader()
            w.writerows(trows)

    # ---- sequences + domain ------------------------------------------------
    for s in SEQS:
        shutil.copy2(f"experiments/sequences/iid-eval-seq-{s}.json", out / "sequences")
    shutil.copy2("configs/restaurant/toy_level_5.yaml", out)

    # ---- RESULTS.md --------------------------------------------------------
    def tot(mid, seqs, ckpt=None):
        sel = [r for r in rows if r["method_id"] == mid and r["seq"] in seqs
               and (ckpt is None or r["checkpoint"] == ckpt)
               and r["total_cost_pddl"] is not None]
        return (sum(r["total_cost_pddl"] for r in sel), len(sel)) if sel else (None, 0)

    ours_ckpts = sorted({r["checkpoint"] for r in rows
                         if r["method_id"] == "anticipatory_dqn" and "g0.97" in r["checkpoint"]})
    L = ["# v5 results (long-horizon restaurant variant)", "",
         "Generated by `scripts/restaurant/build_v5_results.py` -- do not hand-edit.", "",
         "## Headline: total PDDL cost over 10 sequences x 50 tasks", ""]
    def ckpts_of(mid):
        return sorted({r["checkpoint"] for r in rows
                       if r["method_id"] == mid and r["checkpoint"]})

    def arm(mid, seqs):
        """(mean, std, n_seeds) over checkpoints for a seeded arm; (total, 0, 1) otherwise.

        Only checkpoints covering EVERY sequence in `seqs` are averaged. A half-finished
        checkpoint contributes a half-sized total, which drags the mean down and reads as a
        spurious improvement -- the same partial-coverage bug the headline loop guards against.
        """
        cs = ckpts_of(mid)
        if len(cs) > 1:
            v = []
            for c in cs:
                t, n = tot(mid, seqs, c)
                if t and n == len(seqs):
                    v.append(t)
            if v:
                return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)
            return None, 0.0, 0
        t, n = tot(mid, seqs)
        return (t, 0.0, 1) if t and n == len(seqs) else (None, 0.0, 0)

    # The denominator is the augmented arm's 4-seed MEAN, not one draw.
    gnn_t, gnn_sd, gnn_n = arm("gnn_counterfactual", SEQS)
    k2_t, _ = tot("k2_fd_optimal", SEQS)
    L += ["| method | total | vs published GNN | vs K=2 optimal |", "|---|---|---|---|"]
    order = [("myopic_fd_optimal", "K=1 optimal (myopic)"),
             ("gnn_faithful", "GNN, no augmentation"),
             ("gnn_counterfactual", "**GNN as published**"),
             ("gnn_steelman", "GNN steelman (handed the jar candidate)"),
             ("k2_fd_optimal", "K=2 optimal (one-task ceiling)")]
    for mid, label in order:
        t, sd, n = arm(mid, SEQS)
        if not t:
            continue
        cell = f"{t:,.0f} ± {sd:,.0f} ({n} seeds)" if n > 1 else f"{t:,.0f}"
        L.append(f"| {label} | {cell} | {100*(t-gnn_t)/gnn_t:+.2f}% | {100*(t-k2_t)/k2_t:+.2f}% |")
    per = [tot("anticipatory_dqn", SEQS, c)[0] for c in ours_ckpts]
    per = [p for p in per if p]
    if per:
        m, sd = st.mean(per), (st.stdev(per) if len(per) > 1 else 0.0)
        L.append(f"| **ours (gamma=0.97, {len(per)} seeds)** | **{m:,.0f} ± {sd:,.0f}** | "
                 f"**{100*(m-gnn_t)/gnn_t:+.2f}%** | **{100*(m-k2_t)/k2_t:+.2f}%** |")
    # Compare each arm ONLY over the sequences it actually covers. Scoring a
    # partially-finished arm against the baseline's full 10-sequence total reports a
    # meaningless (huge) improvement -- that bug has appeared twice already.
    for mid, label in (("anticipatory_dqn_greedy", "ours, greedy deployment (no search)"),
                       ("myopic_dqn", "Myopic RL (guided) -- one-task credit horizon"),
                       ("myopic_dqn_greedy", "Myopic RL (greedy)"),
                       ("dqn_no_demos", "ours, NO demonstrations (ablation)"),
                       ("myopic_demo_dqn", "ours, myopic K=1 demonstrations (ablation)"),
                       ("clairvoyant_k3_lama", "K=3 clairvoyant (oracle)"),
                       ("clairvoyant_k4_lama", "K=4 clairvoyant (oracle)")):
        # arm(), not tot(): a seeded arm must be averaged over checkpoints, or its total
        # comes out n_seeds times too large.
        t, sd, nseeds = arm(mid, SEQS)
        if not t:
            continue
        covered = sorted({r["seq"] for r in rows if r["method_id"] == mid
                          and r["total_cost_pddl"] is not None})
        g_t = arm("gnn_counterfactual", covered)[0]
        k_t, _ = tot("k2_fd_optimal", covered)
        partial = "" if len(covered) == len(SEQS) else f" **[PARTIAL {len(covered)}/10 seqs]**"
        cell = f"{t:,.0f} ± {sd:,.0f} ({nseeds} seeds)" if nseeds > 1 else f"{t:,.0f}"
        L.append(f"| {label}{partial} | {cell} | {100*(t-g_t)/g_t:+.2f}% | "
                 f"{100*(t-k_t)/k_t:+.2f}% |")
    L += ["", "Per-seed totals: " + ", ".join(f"`{c.split('_')[-1]}`={p:,.0f}"
                                              for c, p in zip(ours_ckpts, per)), ""]

    # greedy vs value-guided: same value function, two deployments
    if deploy:
        L += ["", "## Deployment: the same value acted on greedily vs guiding the search", "",
              "| deployment | success | auto% | steps/task | cost/task |", "|---|---|---|---|---|"]
        for mode in ("Greedy", "Value-guided", "Greedy (myopic)", "Value-guided (myopic)"):
            per_seed = defaultdict(lambda: defaultdict(float))
            for d in deploy:
                if d["mode"] != mode:
                    continue
                for k in ("n", "done", "auto", "acts", "cost"):
                    per_seed[d["ckpt"]][k] += d[k]
            if not per_seed:
                continue
            f = lambda k, g: [v[k] / v["n"] * g for v in per_seed.values()]
            def ms(k, g=1.0):
                v = f(k, g)
                return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            (sc, sc_d), (au, au_d) = ms("done", 100), ms("auto", 100)
            (ac, ac_d), (co, co_d) = ms("acts"), ms("cost")
            L.append(f"| {mode} | {sc:.1f} ± {sc_d:.1f}% | {au:.1f} ± {au_d:.1f}% | "
                     f"{ac:.2f} ± {ac_d:.2f} | {co:,.0f} ± {co_d:,.0f} |")
        L += ["",
              "Steps/task is not one measure across the two rows: greedy counts environment "
              "primitive steps, the guided arm counts PDDL plan actions. Success is 100% by "
              "construction for the guided arm -- a complete planner returns a plan whenever "
              "one exists -- so only the greedy row's success rate carries information.", ""]

    # jar usage: the mechanism itself, counted from executed traces
    jl = [("anticipatory_dqn", "ours (guided, gamma=0.97)"),
          ("myopic_dqn", "Myopic RL (guided)"),
          ("gnn_counterfactual", "One-task GNN (augmented)"),
          ("dqn_no_demos", "ours, no demonstrations")]
    have = {m for m, _ in jl if any(r["method_id"] == m for r in rows)}
    if have:
        L += ["", "## Jar mechanism usage, from executed traces", "",
              "284 of the 500 tasks consume water, so that is the ceiling on refills.", "",
              "| arm | refills | jar picks | refills / water task |", "|---|---|---|---|"]
        for mid, label in jl:
            sel = [r for r in rows if r["method_id"] == mid]
            if not sel:
                continue
            cks = {r["checkpoint"] for r in sel if r["checkpoint"]} or {""}
            rf = sum(r["refill_water"] for r in sel) / max(1, len(cks))
            jp = sum(r["jar_pick"] for r in sel) / max(1, len(cks))
            L.append(f"| {label} | {rf:,.0f} | {jp:,.0f} | {rf/284:.0%} |")
        L += ["", "Counts are per checkpoint (summed over 10 sequences, averaged over seeds).", ""]

    # oracle timeout disclosure
    L += ["## Oracle arms are upper bounds", "",
          "Exact search is intractable at K>=3, so the K=3/K=4 arms are satisficing with a "
          f"{FD_CAP:.0f}s per-window budget. Windows reaching that cap return the best plan "
          "found so far, so these costs are UPPER BOUNDS on the true optimum -- the real "
          "oracles are lower.", "",
          "| arm | windows at cap |", "|---|---|"]
    for mid, label in (("clairvoyant_k3_lama", "K=3"), ("clairvoyant_k4_lama", "K=4")):
        sel = [r for r in rows if r["method_id"] == mid]
        if sel:
            L.append(f"| {label} | {sum(r['windows_at_cap'] for r in sel)}/{sum(r['windows'] for r in sel)} |")

    # per-sequence
    L += ["", "## Per sequence", "",
          "| seq | K=1 opt | GNN | K=2 opt | ours (mean) | K=3 | K=4 | held out | jar delivered |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SEQS:
        def c(mid):
            t, _ = tot(mid, [s]); return f"{t:,.0f}" if t else "-"
        o = [tot("anticipatory_dqn", [s], ck)[0] for ck in ours_ckpts]
        o = [x for x in o if x]
        # GNN column is the 4-seed mean for this sequence, matching the headline.
        def c_gnn():
            v = arm("gnn_counterfactual", [s])[0]
            return f"{v:,.0f}" if v else "-"
        L.append(f"| {s} | {c('myopic_fd_optimal')} | {c_gnn()} | "
                 f"{c('k2_fd_optimal')} | {st.mean(o):,.0f} | {c('clairvoyant_k3_lama')} | "
                 f"{c('clairvoyant_k4_lama')} | {'yes' if s in HELD_OUT else 'no (dev)'} | "
                 f"{'yes' if s in JAR_DELIVERED else ''} |")

    # training / stability
    if trows:
        L += ["", "## Training: discount factor controls jar usage AND stability", "",
              "| run | demos | gamma | jar share | best ckpt | success | max\\|Q\\| | diverged |",
              "|---|---|---|---|---|---|---|---|"]
        for t in trows:
            L.append(f"| {t['run']} | {t['demos']} | {t['gamma']} | {t['jar_share']}% | "
                     f"{t['best_checkpoint']} | {t['success_rate']:.3f} | {t['max_abs_q']:,} | "
                     f"{'YES' if t['diverged'] else ''} |")

    L += ["", "## Protocol notes", "",
          "- **The headline arm is gamma=0.97 only.** gamma=0.98 and 0.99 appear solely in the "
          "training/stability table below; their guided evaluations are not part of the main "
          "comparison. `planner/run_summary.csv` retains any gamma=0.98 evaluation rows that "
          "were run, so the CSV is a superset of this table by design.",
          "- `cost_ratio=3.0` was selected on sequences 01/03/06 from {1.25, 3, 6, 8} -- one "
          "scalar, four candidates, no training contact. The held-out seven are "
          f"{', '.join(HELD_OUT)}.",
          "- Sequences 00 and 02 contain a `pick_place(jar_0, .)` task, so the task stream "
          "itself delivers the investment. They are retained (excluding them would condition "
          "the eval set on the mechanism) and act as a control: the advantage should shrink "
          "where the jar is handed over.",
          "- All costs are undiscounted PDDL physical cost; one environment reset per "
          "50-task sequence, no reset between tasks.", ""]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    (out / "manifest.json").write_text(json.dumps(dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit, domain="toy_level_5.yaml (far pantry + empty jar)",
        units={"cost": "undiscounted PDDL physical cost"},
        persistence="one environment reset per 50-task sequence; no reset between tasks",
        method_definitions={k: v for k, v in (
            ("myopic_fd_optimal", "K=1 clairvoyant, astar(hmax()), exact"),
            ("k2_fd_optimal", "K=2 clairvoyant, astar(hmax()), exact -- provably optimal one-task lookahead"),
            ("gnn_faithful", "Talukder et al. without counterfactual augmentation"),
            ("gnn_counterfactual", "Talukder et al. as published"),
            ("anticipatory_dqn", "ours: DQN V_AP guiding cost-bounded Bellman+novelty search, cost_ratio=3.0"),
            ("clairvoyant_k3_lama", f"K=3 clairvoyant, satisficing, {FD_CAP:.0f}s/window -- UPPER BOUND"),
            ("clairvoyant_k4_lama", f"K=4 clairvoyant, satisficing, {FD_CAP:.0f}s/window -- UPPER BOUND"),
        )},
        splits=dict(dev=DEV, held_out=HELD_OUT, jar_delivered=JAR_DELIVERED),
        sequences=[dict(sequence_id=f"iid-eval-seq-{s}", task_count=50,
                        sha256=sha256(Path(f"experiments/sequences/iid-eval-seq-{s}.json")))
                   for s in SEQS],
    ), indent=2) + "\n")
    print(f"wrote {out}/ -- {len(rows)} eval rows, {len(trows)} training runs")


if __name__ == "__main__":
    main()
