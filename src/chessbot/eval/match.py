"""Play games between bots, including Stockfish at a limited strength."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import chess
import chess.engine

from chessbot.baselines.base import Bot
from chessbot.eval.elo import EloEstimate, elo_difference
from chessbot.eval.openings import opening_boards

MAX_PLIES = 400  # games longer than this are adjudicated as draws


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    def elo(self) -> EloEstimate:
        return elo_difference(self.wins, self.draws, self.losses)

    def __str__(self) -> str:
        e = self.elo()
        return (
            f"+{self.wins} ={self.draws} -{self.losses} "
            f"({self.score:.1%}, Elo diff {e.diff:+.0f} [{e.low:+.0f}, {e.high:+.0f}])"
        )


class StockfishBot:
    """Stockfish limited to ``elo`` (UCI_LimitStrength), with a fixed per-move budget."""

    def __init__(self, elo: int, movetime: float = 0.05, path: str = "stockfish") -> None:
        resolved = shutil.which(path)
        if resolved is None:
            raise FileNotFoundError(f"Stockfish not found: {path}")
        self._engine = chess.engine.SimpleEngine.popen_uci(resolved)
        low = self._engine.options["UCI_Elo"].min
        high = self._engine.options["UCI_Elo"].max
        if low is None or high is None or not low <= elo <= high:
            self._engine.quit()
            raise ValueError(f"UCI_Elo must be in [{low}, {high}], got {elo}")
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self._limit = chess.engine.Limit(time=movetime)
        self.name = f"stockfish-{elo}"

    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(board, self._limit)
        assert result.move is not None
        return result.move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> StockfishBot:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def play_game(
    white: Bot, black: Bot, start: chess.Board | None = None
) -> tuple[float, chess.Board]:
    """Play one game. Returns White's score (1, 0.5 or 0) and the final board."""
    board = start.copy() if start is not None else chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        bot = white if board.turn == chess.WHITE else black
        move = bot.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError(f"{bot.name} played illegal move {move} in {board.fen()}")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5, board
    return (1.0 if outcome.winner == chess.WHITE else 0.0), board


def play_match(
    a: Bot,
    b: Bot,
    games: int,
    openings: list[chess.Board] | None = None,
    progress: Callable[[int, MatchResult], None] | None = None,
) -> MatchResult:
    """``a`` vs ``b`` over ``games`` games (even). Each opening is played with both colours.

    ``progress(games_done, running_result)`` is called after every game if given.
    """
    if games % 2:
        raise ValueError("games must be even so colours are balanced")
    book = openings if openings is not None else opening_boards()
    wins = draws = losses = 0
    for i in range(games // 2):
        start = book[i % len(book)]
        for a_is_white in (True, False):
            white, black = (a, b) if a_is_white else (b, a)
            white_score, _ = play_game(white, black, start)
            a_score = white_score if a_is_white else 1.0 - white_score
            if a_score == 1.0:
                wins += 1
            elif a_score == 0.0:
                losses += 1
            else:
                draws += 1
            if progress is not None:
                progress(wins + draws + losses, MatchResult(wins, draws, losses))
    return MatchResult(wins, draws, losses)
