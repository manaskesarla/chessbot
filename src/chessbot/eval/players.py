"""Build any player from a short text spec, so one command can evaluate every stage.

    random | greedy | alphabeta:<depth> | net:<checkpoint> | mcts:<checkpoint>:<simulations>
"""

from __future__ import annotations

from pathlib import Path

from chessbot.baselines import AlphaBetaBot, Bot, GreedyBot, RandomBot
from chessbot.config import REPO_ROOT, MCTSConfig, load_config


def _checkpoint(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def make_bot(spec: str, seed: int = 0) -> Bot:
    kind, _, rest = spec.partition(":")
    if kind == "random":
        return RandomBot(seed)
    if kind == "greedy":
        return GreedyBot(seed)
    if kind == "alphabeta":
        return AlphaBetaBot(int(rest or 3), seed)
    if kind == "net":
        from chessbot.model.player import NetBot

        return NetBot.from_checkpoint(_checkpoint(rest))
    if kind == "mcts":
        from chessbot.mcts import MCTSBot

        path, _, sims = rest.rpartition(":")
        return MCTSBot.from_checkpoint(
            _checkpoint(path), int(sims), load_config(MCTSConfig, "mcts")
        )
    raise ValueError(f"unknown player spec {spec!r}")
