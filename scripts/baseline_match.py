"""Round-robin between the baselines, then each baseline against limited-strength Stockfish.

Usage: uv run python scripts/baseline_match.py [games_per_pairing]
"""

from __future__ import annotations

import sys
import time

from chessbot.baselines import AlphaBetaBot, Bot, GreedyBot, RandomBot
from chessbot.eval import StockfishBot, play_match


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    bots: list[Bot] = [RandomBot(0), GreedyBot(0), AlphaBetaBot(2, 0), AlphaBetaBot(3, 0)]

    print(f"--- baseline round robin ({games} games each) ---")
    for i, a in enumerate(bots):
        for b in bots[i + 1 :]:
            start = time.time()
            result = play_match(b, a, games)
            print(f"{b.name:>14} vs {a.name:<14} {result}   [{time.time() - start:.0f}s]")

    print(f"--- vs Stockfish (50 ms/move, {games} games each) ---")
    for elo in (1320, 1500):
        with StockfishBot(elo) as sf:
            for bot in bots[2:]:
                start = time.time()
                result = play_match(bot, sf, games)
                print(f"{bot.name:>14} vs {sf.name:<14} {result}   [{time.time() - start:.0f}s]")


if __name__ == "__main__":
    main()
