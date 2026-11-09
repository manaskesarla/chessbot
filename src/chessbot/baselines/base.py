"""Common bot interface so any bot can play any other bot (or Stockfish)."""

from __future__ import annotations

from typing import Protocol

import chess


class Bot(Protocol):
    name: str

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a legal move for ``board.turn``. ``board`` must not be modified."""
        ...
