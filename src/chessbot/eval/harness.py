"""Evaluate a player against limited-strength Stockfish, fit a rating, and keep the results.

Per stage the harness writes ``results/<stage>.json`` (updated after every level, so partial
progress is visible), and upserts a row into ``results/summary.csv``. ``plot_summary`` turns
that table into the Elo-by-stage chart.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from chessbot.baselines.base import Bot
from chessbot.eval.fit import LevelResult, RatingFit, fit_rating
from chessbot.eval.match import MatchResult, StockfishBot, play_match

SUMMARY_FIELDS = ["stage", "rating", "low", "high", "games", "levels", "finished"]


@dataclass(frozen=True)
class StageResult:
    stage: str
    levels: list[LevelResult]
    fit: RatingFit
    movetime: float
    seconds: float
    complete: bool

    @property
    def games(self) -> int:
        return sum(lv.games for lv in self.levels)


def _write_stage(results_dir: Path, result: StageResult) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": result.stage,
        "complete": result.complete,
        "movetime": result.movetime,
        "seconds": round(result.seconds, 1),
        "rating_stockfish_scale": asdict(result.fit),
        "levels": [asdict(lv) | {"score": lv.score} for lv in result.levels],
    }
    (results_dir / f"{result.stage}.json").write_text(json.dumps(payload, indent=2))


def upsert_summary(results_dir: Path, result: StageResult) -> None:
    """Insert or replace this stage's row, keeping every other stage in its original order."""
    path = results_dir / "summary.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    row = {
        "stage": result.stage,
        "rating": f"{result.fit.rating:.1f}",
        "low": f"{result.fit.low:.1f}",
        "high": f"{result.fit.high:.1f}",
        "games": str(result.games),
        "levels": " ".join(str(lv.opponent_elo) for lv in result.levels),
        "finished": time.strftime("%Y-%m-%d %H:%M"),
    }
    for i, existing in enumerate(rows):
        if existing["stage"] == result.stage:
            rows[i] = row
            break
    else:
        rows.append(row)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_stage(
    stage: str,
    bot: Bot,
    levels: tuple[int, ...],
    games_per_level: int,
    movetime: float,
    results_dir: Path,
    stockfish_path: str = "stockfish",
    log: Callable[[str], None] = print,
) -> StageResult:
    """Play ``games_per_level`` games against Stockfish at each Elo in ``levels``."""
    start = time.time()
    done: list[LevelResult] = []
    for elo in levels:
        with StockfishBot(elo, movetime, stockfish_path) as sf:

            def progress(n: int, running: MatchResult, elo: int = elo) -> None:
                if n % 10 == 0 or n == games_per_level:
                    log(f"[{stage}] vs stockfish-{elo}: {n}/{games_per_level}  "
                        f"+{running.wins} ={running.draws} -{running.losses}")

            result = play_match(bot, sf, games_per_level, progress=progress)
        done.append(LevelResult(elo, result.wins, result.draws, result.losses))
        fit = fit_rating(done)
        log(f"[{stage}] level {elo} done: score {result.score:.1%}; running rating "
            f"{fit.rating:.0f} [{fit.low:.0f}, {fit.high:.0f}]")
        _write_stage(
            results_dir,
            StageResult(stage, list(done), fit, movetime, time.time() - start, False),
        )

    final = StageResult(stage, done, fit_rating(done), movetime, time.time() - start, True)
    _write_stage(results_dir, final)
    upsert_summary(results_dir, final)
    return final


def read_summary(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_summary(results_dir: Path) -> Path:
    """Elo (Stockfish scale) per stage with 95% intervals -> ``results/elo_by_stage.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_summary(results_dir)
    if not rows:
        raise FileNotFoundError("no results yet: run an evaluation first")

    surface, ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    x = list(range(len(rows)))
    ratings = [float(r["rating"]) for r in rows]
    low = [float(r["rating"]) - float(r["low"]) for r in rows]
    high = [float(r["high"]) - float(r["rating"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(rows) + 2), 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.errorbar(x, ratings, yerr=[low, high], fmt="none", ecolor=blue, elinewidth=2,
                capsize=4, capthick=2, zorder=2)
    ax.plot(x, ratings, color=blue, linewidth=2, zorder=3)
    ax.plot(x, ratings, "o", color=blue, markersize=9, markeredgecolor=surface,
            markeredgewidth=2, zorder=4)
    for xi, r, hi in zip(x, ratings, high, strict=True):
        ax.annotate(f"{r:.0f}", (xi, r + hi), textcoords="offset points", xytext=(0, 7),
                    ha="center", color=ink, fontsize=10, fontweight="bold")
    ax.set_xticks(x, [r["stage"] for r in rows], color=muted, fontsize=9)
    ax.tick_params(axis="y", colors=muted, labelsize=9)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.margins(y=0.15)
    ax.yaxis.grid(True, color=grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_ylabel("Elo (Stockfish scale)", color=muted, fontsize=10)
    ax.set_title("Strength by stage (95% bootstrap interval)", loc="left", color=ink,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = results_dir / "elo_by_stage.png"
    fig.savefig(out, dpi=160, facecolor=surface)
    plt.close(fig)
    return out
