"""A Bot that plays by running MCTS with a network (or any evaluator)."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from chessbot.config import MCTSConfig
from chessbot.mcts.evaluator import Evaluator, NetEvaluator
from chessbot.mcts.search import MCTS, SearchResult
from chessbot.model import load_checkpoint
from chessbot.smoke import pick_device


class MCTSBot:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 100,
        config: MCTSConfig | None = None,
        name: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.simulations = simulations
        self.mcts = MCTS(evaluator, config, np.random.default_rng(seed))
        self.name = name or f"mcts-{simulations}"
        self.last_result: SearchResult | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        simulations: int = 100,
        config: MCTSConfig | None = None,
        name: str | None = None,
    ) -> MCTSBot:
        device = pick_device()
        evaluator = NetEvaluator(load_checkpoint(path, device), device)
        return cls(evaluator, simulations, config, name or f"{path.parent.name}+mcts{simulations}")

    def choose_move(self, board: chess.Board) -> chess.Move:
        # Search a copy (keeping move history for repetition detection) so the caller's board
        # is never touched.
        self.last_result = self.mcts.search(board.copy(), self.simulations)
        return self.last_result.best_move()
