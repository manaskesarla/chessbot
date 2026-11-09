"""Strength of a network against baselines and limited-strength Stockfish.

Usage: uv run python scripts/eval_net.py CHECKPOINT [games_per_opponent] [mcts_simulations]
With mcts_simulations = 0 (default) the raw network plays with no search.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from chessbot.baselines import AlphaBetaBot, Bot, GreedyBot, RandomBot
from chessbot.config import MCTSConfig, load_config
from chessbot.eval import StockfishBot, play_match
from chessbot.mcts import MCTSBot
from chessbot.model.player import NetBot


def main() -> None:
    checkpoint = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    net: Bot
    if sims:
        net = MCTSBot.from_checkpoint(checkpoint, sims, load_config(MCTSConfig, "mcts"), "net+mcts")
    else:
        net = NetBot.from_checkpoint(checkpoint, name="net")

    opponents: list[Bot] = [RandomBot(0), GreedyBot(0), AlphaBetaBot(2, 0), AlphaBetaBot(3, 0)]
    mode = f"MCTS {sims} sims" if sims else "raw net, no search"
    print(f"--- {checkpoint} ({mode}), {games} games per opponent ---")
    for opp in opponents:
        start = time.time()
        result = play_match(net, opp, games)
        print(f"{net.name} vs {opp.name:<14} {result}   [{time.time() - start:.0f}s]")
    for elo in (1320, 1500, 1700):
        with StockfishBot(elo) as sf:
            start = time.time()
            result = play_match(net, sf, games)
            print(f"{net.name} vs {sf.name:<14} {result}   [{time.time() - start:.0f}s]")


if __name__ == "__main__":
    main()
