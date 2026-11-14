"""Draw the README figures from the small result files committed in ``results/``.

Usage: uv run python scripts/make_figures.py      # writes docs/figures/*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from chessbot.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "figures"

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

STAGE_LABELS = {
    "random": "Random mover",
    "greedy": "Greedy (1-ply material)",
    "alphabeta-d2": "Alpha-beta, depth 2",
    "alphabeta-d3": "Alpha-beta, depth 3",
    "supervised-raw": "Network, no search",
    "supervised-mcts100": "Network + MCTS",
    "selfplay-v2-mcts100": "Self-play network + MCTS",
}


def _frame(width: float, height: float, title: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def strength_by_stage() -> Path:
    with (RESULTS / "summary.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["stage"] in STAGE_LABELS]
    rows.sort(key=lambda r: list(STAGE_LABELS).index(r["stage"]), reverse=True)
    fig, ax = _frame(8.2, 4.2, "Playing strength by stage (Stockfish scale, 95% interval)")
    for y, r in enumerate(rows):
        rating, low, high = float(r["rating"]), float(r["low"]), float(r["high"])
        ax.plot([low, high], [y, y], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(rating, y, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        floor = r["stage"] in ("random", "greedy")
        text = f"below ~{high:.0f}" if r["stage"] == "greedy" else f"{rating:.0f}"
        text = "at the 400 floor" if r["stage"] == "random" else text
        ax.annotate(text, (high if not floor else max(high, rating), y),
                    textcoords="offset points", xytext=(10, 0), va="center", color=INK,
                    fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows)), [STAGE_LABELS[r["stage"]] for r in rows], color=SECONDARY,
                  fontsize=10)
    ax.set_xlim(300, 1900)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("Elo against limited-strength Stockfish", color=MUTED, fontsize=9)
    return _save(fig, "strength_by_stage.png")


def score_vs_stockfish() -> Path:
    series = [
        ("supervised-raw", "Network, no search", BLUE),
        ("alphabeta-d3", "Alpha-beta, depth 3", ORANGE),
        ("supervised-mcts100", "Network + MCTS", AQUA),
        ("selfplay-v2-mcts100", "Self-play network + MCTS", YELLOW),
    ]
    fig, ax = _frame(8.2, 4.4, "Score against each Stockfish level (100 games per point)")
    for stage, label, color in series:
        levels = json.loads((RESULTS / f"{stage}.json").read_text())["levels"]
        xs = [lv["opponent_elo"] for lv in levels]
        ys = [100 * lv["score"] for lv in levels]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=3, label=label)
        ax.plot(xs, ys, "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
    ax.axhline(50, color=AXIS, linewidth=1, zorder=1)
    ax.annotate("50% = equal strength", (1900, 50), textcoords="offset points", xytext=(0, 5),
                ha="right", color=MUTED, fontsize=9)
    ax.set_xticks([1320, 1500, 1700, 1900])
    ax.set_xlim(1260, 1960)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("Stockfish UCI_Elo setting", color=MUTED, fontsize=9)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY)
    legend.set_zorder(5)
    return _save(fig, "score_vs_stockfish.png")


def training_curve() -> Path:
    rows = [json.loads(line) for line in
            (RESULTS / "supervised_metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] / 1000 for r in rows]
    fig, ax = _frame(8.2, 4.0, "Supervised pretraining: move-prediction accuracy on held-out games")
    for key, label, color in (("top5", "Top-5", ORANGE), ("top1", "Top-1", BLUE)):
        ys = [100 * r[key] for r in rows]
        ax.plot(steps, ys, color=color, linewidth=2, zorder=3)
        ax.plot(steps[-1], ys[-1], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.annotate(f"{label}  {ys[-1]:.1f}%", (steps[-1], ys[-1]), textcoords="offset points",
                    xytext=(10, 0), va="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlim(0, steps[-1] * 1.17)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("Training steps (thousands, batch size 256)", color=MUTED, fontsize=9)
    return _save(fig, "training_curve.png")


def selfplay_gate() -> Path:
    rows = [json.loads(line) for line in
            (RESULTS / "selfplay_iterations.jsonl").read_text().splitlines()]
    xs = [r["iteration"] for r in rows]
    ys = [100 * r["gate_score"] for r in rows]
    fig, ax = _frame(8.2, 4.0, "Self-play: candidate vs current best (100 gate games each)")
    ax.axhline(55, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("55% needed to promote", (xs[-1] + 0.4, 55), textcoords="offset points",
                xytext=(0, 5), ha="right", color=SECONDARY, fontsize=9)
    ax.vlines(xs, 50, ys, color=BLUE, linewidth=2, zorder=2)
    ax.plot(xs, ys, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3)
    for r, x, y in zip(rows, xs, ys, strict=True):
        if r["promoted"]:
            ax.annotate("promoted", (x, y), textcoords="offset points", xytext=(0, 22),
                        ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.axhline(50, color=AXIS, linewidth=1, zorder=1)
    ax.set_xticks(xs)
    ax.set_xlim(0.4, xs[-1] + 0.6)
    ax.set_ylim(35, 65)
    ax.set_yticks(range(35, 66, 5), [f"{v}%" for v in range(35, 66, 5)])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("Iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("Candidate's score", color=MUTED, fontsize=9)
    return _save(fig, "selfplay_gate.png")


def main() -> None:
    for draw in (strength_by_stage, score_vs_stockfish, training_curve, selfplay_gate):
        print(f"wrote {draw()}")


if __name__ == "__main__":
    main()
