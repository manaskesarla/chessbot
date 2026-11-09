import shutil
from pathlib import Path

import chess
import pytest

from chessbot.baselines import AlphaBetaBot, GreedyBot, RandomBot
from chessbot.eval.fit import LevelResult, RatingFit, fit_rating
from chessbot.eval.harness import (
    StageResult,
    evaluate_stage,
    plot_summary,
    read_summary,
    upsert_summary,
)
from chessbot.eval.openings import opening_boards
from chessbot.eval.players import make_bot


def test_even_results_recover_opponent_rating() -> None:
    fit = fit_rating([LevelResult(1500, 50, 0, 50)])
    assert fit.rating == pytest.approx(1500, abs=1)
    assert fit.low < 1500 < fit.high


def test_rating_combines_levels_and_is_monotonic_in_score() -> None:
    weak = fit_rating([LevelResult(1320, 30, 10, 60), LevelResult(1500, 10, 10, 80)])
    strong = fit_rating([LevelResult(1320, 70, 10, 20), LevelResult(1500, 50, 10, 40)])
    assert weak.rating < strong.rating
    assert 1100 < weak.rating < 1500 < strong.rating < 1900


def test_consistent_results_across_levels_agree() -> None:
    # 75% at 1320 implies ~1510; 50% at 1500 implies 1500: the joint fit lands in between.
    fit = fit_rating([LevelResult(1320, 75, 0, 25), LevelResult(1500, 50, 0, 50)])
    assert 1480 < fit.rating < 1530


def test_more_games_narrow_the_interval() -> None:
    few = fit_rating([LevelResult(1500, 10, 0, 10)])
    many = fit_rating([LevelResult(1500, 100, 0, 100)])
    assert (many.high - many.low) < (few.high - few.low)


def test_perfect_score_stays_finite_and_deterministic() -> None:
    a = fit_rating([LevelResult(1320, 100, 0, 0)])
    b = fit_rating([LevelResult(1320, 100, 0, 0)])
    assert a == b and a.rating > 1320 and a.high < 5000


def test_fit_requires_games() -> None:
    with pytest.raises(ValueError):
        fit_rating([])


def test_opening_book_is_legal_and_unique() -> None:
    boards = opening_boards()
    assert len(boards) >= 50
    assert len({b.fen() for b in boards}) == len(boards)


def test_make_bot_specs() -> None:
    assert isinstance(make_bot("random"), RandomBot)
    assert isinstance(make_bot("greedy"), GreedyBot)
    assert make_bot("alphabeta:2").name == "alphabeta-d2"
    assert isinstance(make_bot("alphabeta:3"), AlphaBetaBot)
    with pytest.raises(ValueError):
        make_bot("nonsense")


def stage(name: str, rating: float) -> StageResult:
    return StageResult(name, [LevelResult(1320, 5, 0, 5)], RatingFit(rating, rating - 50,
                       rating + 60), 0.05, 1.0, True)


def test_summary_upsert_keeps_order_and_replaces(tmp_path: Path) -> None:
    upsert_summary(tmp_path, stage("a", 1000))
    upsert_summary(tmp_path, stage("b", 1200))
    upsert_summary(tmp_path, stage("a", 1100))
    rows = read_summary(tmp_path)
    assert [r["stage"] for r in rows] == ["a", "b"]
    assert float(rows[0]["rating"]) == 1100


def test_plot_is_written(tmp_path: Path) -> None:
    upsert_summary(tmp_path, stage("supervised", 1300))
    upsert_summary(tmp_path, stage("+mcts", 1600))
    out = plot_summary(tmp_path)
    assert out.exists() and out.stat().st_size > 5_000


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_evaluate_stage_end_to_end_tiny(tmp_path: Path) -> None:
    logs: list[str] = []
    result = evaluate_stage("tiny", RandomBot(0), (1320,), 2, 0.01, tmp_path, log=logs.append)
    assert result.complete and result.games == 2
    assert (tmp_path / "tiny.json").exists() and (tmp_path / "summary.csv").exists()
    assert any("level 1320 done" in line for line in logs)
    assert chess.Board().is_valid()
