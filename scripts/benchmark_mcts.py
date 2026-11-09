"""Search speed: simulations/second for the trained network at several batch sizes.

Usage: uv run python scripts/benchmark_mcts.py checkpoints/supervised/best.pt
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np

from chessbot.config import MCTSConfig, load_config
from chessbot.mcts import MCTS, NetEvaluator
from chessbot.model import load_checkpoint
from chessbot.smoke import pick_device

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
]


def main() -> None:
    device = pick_device()
    evaluator = NetEvaluator(load_checkpoint(Path(sys.argv[1]), device), device)
    base = load_config(MCTSConfig, "mcts")
    sims = 200
    print(f"device={device}, {sims} simulations per search")
    print(f"{'batch':>6} {'sims/s':>8} {'net evals':>10} {'cache hits':>11}")
    for batch_size in (1, 8, 16, 32):
        mcts = MCTS(evaluator, replace(base, batch_size=batch_size), np.random.default_rng(0))
        start = time.perf_counter()
        for fen in POSITIONS:
            mcts.search(chess.Board(fen), sims)
        elapsed = time.perf_counter() - start
        print(
            f"{batch_size:>6} {sims * len(POSITIONS) / elapsed:>8.0f} "
            f"{mcts.evaluations:>10} {mcts.cache_hits:>11}"
        )


if __name__ == "__main__":
    main()
