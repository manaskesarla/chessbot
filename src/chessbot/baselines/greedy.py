from __future__ import annotations

import random

import chess

from chessbot.baselines.evaluation import material

MATE_SCORE = 100_000


class GreedyBot:
    """Picks the move giving the best material balance after one ply (random tie-break)."""

    name = "greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        best_score = -MATE_SCORE - 1
        best_moves: list[chess.Move] = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                score = MATE_SCORE
            else:
                score = -material(board)  # material() is from the opponent's view now
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self._rng.choice(best_moves)
