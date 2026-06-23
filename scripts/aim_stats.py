"""Export Aim metrics to CSV using query_metrics — the only SDK path that resolves
run.run_label correctly (meta_attrs_tree.collect() is buggy/cached).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from aim import Repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Aim metrics to wide CSV.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Path to Aim repo (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aim_metrics_wide.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        dest="labels",
        help="Run label(s) to export (can be passed multiple times).",
    )
    parser.add_argument(
        "--include-action-type-q",
        action="store_true",
        help="Include per-action-type q_by_action_type metrics (slower)",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="List available run labels in the repo and exit.",
    )
    return parser.parse_args()


def list_run_labels(repo: Repo) -> set[str]:
    """Best-effort discovery of run labels present in the Aim repo."""
    labels: set[str] = set()
    try:
        # Aim may expose query_runs(); fall back to scanning metrics.
        try:
            for run in repo.query_runs("run.run_label"):
                label = run.get("run_label")
                if label:
                    labels.add(str(label))
        except Exception:
            pass

        if not labels:
            # Scan a cheap metric to collect run labels.
            q = repo.query_metrics("metric.name == 'loss'")
            for rm in q.iter_runs():
                label = rm.run.get("run_label")
                if label:
                    labels.add(str(label))
    except Exception as exc:
        print(f"[aim_stats] Could not list labels: {exc}")
    return labels


def discover_run_hashes(repo: Repo, labels: list[str], metrics: list[str]) -> dict[str, str]:
    """Map run hash -> label for the given labels, trying several anchor metrics."""
    hash_to_label: dict[str, str] = {}
    for label in labels:
        found_for_label = False
        for anchor in metrics:
            q = repo.query_metrics(f'run.run_label == "{label}" and metric.name == "{anchor}"')
            for rm in q.iter_runs():
                hash_to_label[rm.run.hash] = label
                found_for_label = True
            if found_for_label:
                break
        if not found_for_label:
            print(f"[aim_stats] Warning: no runs found for label '{label}'")
    return hash_to_label


def main() -> None:
    args = parse_args()

    repo = Repo(str(args.repo))
    print(f"[aim_stats] Using Aim repo: {args.repo.resolve()}")

    if args.list_labels:
        labels = list_run_labels(repo)
        print(f"[aim_stats] Found {len(labels)} run label(s):")
        for label in sorted(labels):
            print(f"  - {label}")
        return

    labels = args.labels or ["rest_v2_2_ant_option3_seed0"]

    METRICS = [
        "td_abs_mean",
        "loss",
        "greedy_success_rolling",
        "success_rate_rolling",
        "avg_task_return_rolling",
        "q_selected_mean",
        "q_selected_abs_max",
        "target_mean",
        "target_abs_max",
        "avg_loss_rolling",
    ]
    if args.include_action_type_q:
        METRICS.append("q_by_action_type")

    hash_to_label = discover_run_hashes(repo, labels, METRICS)
    print(f"Found {len(hash_to_label)} run hash(es) across {len(set(hash_to_label.values()))} label(s)")
    print(hash_to_label)

    if not hash_to_label:
        print(
            "[aim_stats] No runs matched the requested labels. "
            "Run with --list-labels to see available labels, "
            "or check that the Aim repo path is correct."
        )
        return

    metrics_list = '", "'.join(METRICS)
    hashes_list = '", "'.join(hash_to_label)
    query = repo.query_metrics(f'metric.name in ["{metrics_list}"] and run.hash in ["{hashes_list}"]')

    raw_records: list[dict] = []
    for run_metrics in query.iter_runs():
        rh = run_metrics.run.hash
        label = hash_to_label.get(rh, f"Run: {rh}")

        for metric in run_metrics:
            # Use dataframe() instead of sparse_numpy() — sparse_numpy() returns
            # Aim's internal storage IDs (huge int64 values, not training steps),
            # which made the old `rank(method="dense")` produce a non-chronological
            # "step" column and corrupt the export. dataframe() exposes the real
            # user-provided `step` (training step) alongside `value`.
            mdf = metric.dataframe()
            ctx = metric.context.to_dict()
            action = ctx.get("action_type", "")
            metric_label = f"{metric.name}_{action}" if action else metric.name

            for _, row in mdf.iterrows():
                raw_records.append({
                    "run_label": label,
                    "run_hash": rh,
                    "step": int(row["step"]),
                    "metric_label": metric_label,
                    "value": float(row["value"]),
                })

    if not raw_records:
        print("[aim_stats] No metric records found for the selected runs.")
        return

    df = pd.DataFrame(raw_records)

    df_wide = df.pivot_table(
        index=["run_label", "run_hash", "step"],
        columns="metric_label",
        values="value",
        aggfunc="first",
    )

    # Forward fill missing values so episodic metrics aren't mostly empty.
    # Group by both run_label and run_hash so distinct runs never cross-contaminate.
    df_wide = df_wide.groupby(level=["run_label", "run_hash"]).ffill().reset_index()

    df_wide.to_csv(args.output, index=False)
    print(f"Done – {len(df_wide):,} rows × {len(df_wide.columns)} columns → {args.output}")

    if "q_selected_abs_max" in df_wide.columns:
        max_row = df_wide.loc[df_wide["q_selected_abs_max"].idxmax()]
        print(
            f"Sanity: q_selected_abs_max max = {max_row['q_selected_abs_max']:.4f} "
            f"at step {int(max_row['step'])} (run_hash {max_row['run_hash']})"
        )


if __name__ == "__main__":
    main()
