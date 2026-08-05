"""Cumulative PDDL cost vs. number of tasks, from results/canonical_planner/task_results.csv.

At each task_index, cost is AVERAGED across the 10 canonical sequences, then
cumsum'd over task_index -- so the y-axis is the expected cumulative cost of
completing N tasks in one 50-task sequence. For the two FD baselines (no seed)
this gives one deterministic curve. For the two DQN methods this gives one
curve per checkpoint seed; the plotted band is mean +/- std across those 4
seed curves, so the only source of variance shown is seed, not sequence.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "results/canonical_planner/task_results.csv"
OUT_PATH = "results/canonical_planner/cumulative_cost_plot.png"

FD_METHODS = {
    "myopic_fd_optimal": ("Myopic FD (optimal)", "#000000"),
    "clairvoyant_k3_lama": ("Clairvoyant FD (K=3)", "#E69F00"),
}
DQN_METHODS = {
    "myopic_dqn_beta1": ("Myopic DQN", "#56B4E9"),
    "anticipatory_dqn_beta1_25": ("Anticipatory DQN", "#009E73"),
}


def cumulative_curve(df, method_id, seed=None):
    sub = df[df["method_id"] == method_id]
    if seed is not None:
        sub = sub[sub["checkpoint_seed"] == seed]
    averaged = sub.groupby("task_index")["task_cost_pddl"].mean().sort_index()
    return averaged.cumsum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_id, (label, color) in FD_METHODS.items():
        curve = cumulative_curve(df, method_id)
        ax.plot(curve.index + 1, curve.values, "--", color=color, linewidth=2, label=label)

    for method_id, (label, color) in DQN_METHODS.items():
        seeds = sorted(df.loc[df["method_id"] == method_id, "checkpoint_seed"].dropna().unique())
        curves = pd.concat(
            [cumulative_curve(df, method_id, seed) for seed in seeds], axis=1
        )
        mean = curves.mean(axis=1)
        std = curves.std(axis=1)
        x = mean.index + 1
        ax.plot(x, mean.values, "-", color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Cumulative PDDL cost (mean over 10 sequences)")
    ax.set_title("Cumulative cost vs. tasks completed")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
