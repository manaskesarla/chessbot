"""Stream Lichess .pgn.zst files, filter games by header, and turn them into Records."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

from chessbot.data.records import FIELDS, Records, board_to_record
from chessbot.encoding.moves import move_to_index

_TAG = re.compile(r'\[(\w+) "(.*)"\]')
_RESULT_FOR_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


@dataclass(frozen=True)
class GameFilter:
    min_elo: int = 2000
    min_plies: int = 10
    exclude_bullet: bool = True


def iter_game_texts(path: Path) -> Iterator[str]:
    """Yield the raw PGN text of each game, streaming (never loads the whole file)."""
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        lines: list[str] = []
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
            if line.startswith("[Event ") and lines:
                yield "".join(lines)
                lines = []
            lines.append(line)
        if lines:
            yield "".join(lines)


def read_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.split("\n", 30):
        if not line.startswith("["):
            break
        match = _TAG.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
    return headers


def accept(headers: dict[str, str], flt: GameFilter) -> bool:
    if headers.get("Result") not in _RESULT_FOR_WHITE:
        return False
    if headers.get("Termination", "Normal") != "Normal":
        return False
    try:
        if min(int(headers["WhiteElo"]), int(headers["BlackElo"])) < flt.min_elo:
            return False
    except (KeyError, ValueError):
        return False
    return not (flt.exclude_bullet and "Bullet" in headers.get("Event", ""))


def game_to_records(text: str, game_id: int, min_plies: int = 10) -> Records | None:
    """One Records entry per position reached in the game (position -> human move, result)."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        return None
    white_result = _RESULT_FOR_WHITE[game.headers["Result"]]
    moves = list(game.mainline_moves())
    if len(moves) < min_plies:
        return None

    n = len(moves)
    squares = np.zeros((n, 64), dtype=np.uint8)
    flags = np.zeros(n, dtype=np.uint8)
    ep = np.zeros(n, dtype=np.int8)
    halfmove = np.zeros(n, dtype=np.uint8)
    move_idx = np.zeros(n, dtype=np.int16)
    result = np.zeros(n, dtype=np.int8)

    board = game.board()
    for i, move in enumerate(moves):
        squares[i], flags[i], ep[i], halfmove[i] = board_to_record(board)
        move_idx[i] = move_to_index(move, board)
        result[i] = white_result if board.turn == chess.WHITE else -white_result
        board.push(move)
    return Records(squares, flags, ep, halfmove, move_idx, result, np.full(n, game_id, np.int32))


def save_records(rec: Records, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        np.save(directory / f"{field}.npy", getattr(rec, field))


def load_records(directory: Path) -> Records:
    return Records(*(np.load(directory / f"{f}.npy") for f in FIELDS))


def load_dataset(root: Path) -> Records:
    """Load and concatenate every ``shard_*`` directory under ``root``."""
    shards = sorted(root.glob("shard_*"))
    if not shards:
        raise FileNotFoundError(f"no shard_* directories in {root}")
    return Records.concat([load_records(s) for s in shards])


def prepare_dataset(
    source: Path,
    out_dir: Path,
    max_games: int,
    flt: GameFilter = GameFilter(),
    shard_positions: int = 2_000_000,
    progress: bool = True,
) -> dict[str, int]:
    """Stream ``source``, keep up to ``max_games`` accepted games, write shards to ``out_dir``."""
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[Records] = []
    buffered = shard_count = games = positions = scanned = 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_count
        if buffer:
            save_records(Records.concat(buffer), out_dir / f"shard_{shard_count:04d}")
            shard_count += 1
            buffer, buffered = [], 0

    bar = tqdm(total=max_games, unit="game", disable=not progress)
    for text in iter_game_texts(source):
        scanned += 1
        if not accept(read_headers(text), flt):
            continue
        rec = game_to_records(text, game_id=games, min_plies=flt.min_plies)
        if rec is None:
            continue
        buffer.append(rec)
        buffered += len(rec)
        positions += len(rec)
        games += 1
        bar.update(1)
        if buffered >= shard_positions:
            flush()
        if games >= max_games:
            break
    flush()
    bar.close()

    stats = {"games": games, "positions": positions, "scanned": scanned, "shards": shard_count}
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
    return stats
