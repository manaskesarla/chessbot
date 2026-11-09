"""Turn a Lichess .pgn.zst file into training shards.

Usage:
    uv run python scripts/prepare_data.py data/raw/lichess-2016-06/games.pgn.zst \
        data/processed/lichess-2016-06 --max-games 150000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessbot.data import GameFilter, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-games", type=int, default=150_000)
    parser.add_argument("--min-elo", type=int, default=2000)
    args = parser.parse_args()

    stats = prepare_dataset(
        args.source, args.out_dir, args.max_games, GameFilter(min_elo=args.min_elo)
    )
    print(stats)


if __name__ == "__main__":
    main()
