from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from anticipatory_rl.envs.three_box_env import GRID_SIZE, REC_A, REC_B, REC_C, REC_RGB, ThreeBoxEnv


def _cell_xy(coord: tuple[int, int]) -> tuple[float, float]:
    x, y = coord
    return float(x), float(GRID_SIZE - 1 - y)


def render_initial_state(output_path: Path, seed: int = 42) -> Path:
    env = ThreeBoxEnv(prob_a=0.2)
    obs, info = env.reset(seed=seed)
    del obs

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("#EAEAF2")
    floor = Rectangle(
        (-0.08, -0.08),
        GRID_SIZE + 0.16,
        GRID_SIZE + 0.16,
        facecolor="#F4F4F8",
        edgecolor="#8A8A94",
        linewidth=1.5,
    )
    ax.add_patch(floor)

    for i in range(GRID_SIZE + 1):
        ax.plot([i, i], [0, GRID_SIZE], color="#B7B7C5", linewidth=1)
        ax.plot([0, GRID_SIZE], [i, i], color="#B7B7C5", linewidth=1)

    receptacles = [("A", REC_A), ("B", REC_B), ("C", REC_C)]
    for label, coord in receptacles:
        x, y = _cell_xy(coord)
        tile = Rectangle(
            (x, y),
            1,
            1,
            facecolor=[c / 255 for c in REC_RGB[label]],
            edgecolor="#4C4C56",
            linewidth=1.5,
        )
        ax.add_patch(tile)
        ax.text(
            x + 0.5,
            y + 0.62,
            label,
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
            color="white",
        )
        if label == "C":
            ax.text(
                x + 0.5,
                y + 0.18,
                "apple spawn",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )

    apple_x, apple_y = _cell_xy(REC_C)
    apple = Circle(
        (apple_x + 0.5, apple_y + 0.34),
        0.12,
        facecolor="#e6321e",
        edgecolor="#38130f",
        linewidth=1.4,
    )
    ax.add_patch(apple)

    agent_x, agent_y = _cell_xy(info["agent"])
    tri = Polygon(
        [
            (agent_x + 0.28, agent_y + 0.24),
            (agent_x + 0.28, agent_y + 0.76),
            (agent_x + 0.78, agent_y + 0.50),
        ],
        closed=True,
        facecolor="#b41e1e",
        edgecolor="#3c0707",
        linewidth=1.5,
    )
    ax.add_patch(tri)

    arrow = FancyArrowPatch(
        (agent_x + 0.9, agent_y + 0.5),
        (apple_x + 0.75, apple_y + 0.55),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#55555F",
        connectionstyle="arc3,rad=-0.25",
    )
    ax.add_patch(arrow)

    ax.text(
        agent_x + 0.5,
        agent_y - 0.12,
        "agent start",
        ha="center",
        va="top",
        fontsize=10,
        color="#44444D",
    )

    ax.set_xlim(-0.25, GRID_SIZE + 0.25)
    ax.set_ylim(-0.25, GRID_SIZE + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")

    side.set_facecolor("white")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    side.text(0.03, 0.94, "State Summary", fontsize=15, weight="bold", color="#26262C", transform=side.transAxes)
    side.text(0.03, 0.84, "Grid: 5 x 5", fontsize=12, color="#3E3E45", transform=side.transAxes)
    side.text(0.03, 0.77, "Apple: on C", fontsize=12, color="#3E3E45", transform=side.transAxes)
    side.text(0.03, 0.70, "Agent: random non-receptacle tile", fontsize=12, color="#3E3E45", transform=side.transAxes)

    side.text(0.03, 0.50, "Task 2 Distribution", fontsize=15, weight="bold", color="#26262C", transform=side.transAxes)
    side.text(0.03, 0.42, "Move to A: 20%", fontsize=12, color="#3E3E45", transform=side.transAxes)
    side.text(0.03, 0.35, "Move to B: 80%", fontsize=12, color="#3E3E45", transform=side.transAxes)

    side.text(0.03, 0.22, "Receptacles", fontsize=15, weight="bold", color="#26262C", transform=side.transAxes)
    side.text(0.03, 0.14, "A: bottom-left target", fontsize=12, color="#3E3E45", transform=side.transAxes)
    side.text(0.03, 0.08, "B: bottom-right target", fontsize=12, color="#3E3E45", transform=side.transAxes)
    side.text(0.03, 0.02, "C: top-center spawn", fontsize=12, color="#3E3E45", transform=side.transAxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the initial Three-Box environment state.")
    parser.add_argument("--output", type=Path, default=Path("runs") / "three_box" / "initial_state.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = render_initial_state(args.output, seed=args.seed)
    print(f"Saved image to {path}")


if __name__ == "__main__":
    main()