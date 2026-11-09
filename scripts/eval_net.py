"""Strength of a raw network (no search) against baselines and limited-strength Stockfish.

Usage: uv run python scripts/eval_net.py checkpoints/supervised/best.pt [games_per_opponent]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from chessbot.baselines import AlphaBetaBot, Bot, GreedyBot, RandomBot
from chessbot.eval import StockfishBot, play_match
from chessbot.model.player import NetBot


def main() -> None:
    checkpoint = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    net = NetBot.from_checkpoint(checkpoint, name="net")

    opponents: list[Bot] = [RandomBot(0), GreedyBot(0), AlphaBetaBot(2, 0), AlphaBetaBot(3, 0)]
    print(f"--- {checkpoint} (raw net, no search), {games} games per opponent ---")
    for opp in opponents:
        start = time.time()
        print(f"net vs {opp.name:<14} {play_match(net, opp, games)}   [{time.time() - start:.0f}s]")
    for elo in (1320, 1500):
        with StockfishBot(elo) as sf:
            start = time.time()
            result = play_match(net, sf, games)
            print(f"net vs {sf.name:<14} {result}   [{time.time() - start:.0f}s]")


if __name__ == "__main__":
    main()
