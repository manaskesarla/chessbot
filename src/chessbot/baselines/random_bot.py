from __future__ import annotations

import random

import chess


class RandomBot:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))
