import io
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

from chessbot.data import (
    GameFilter,
    Records,
    accept,
    board_to_record,
    game_to_records,
    load_dataset,
    prepare_dataset,
    read_headers,
    records_to_planes,
)
from chessbot.data.pgn import iter_game_texts
from chessbot.encoding import encode_board, index_to_move, move_to_index
from tests.test_encoding import POSITIONS


def records_from_boards(boards: list[chess.Board]) -> Records:
    parts = [board_to_record(b) for b in boards]
    n = len(boards)
    return Records(
        np.stack([p[0] for p in parts]),
        np.array([p[1] for p in parts], dtype=np.uint8),
        np.array([p[2] for p in parts], dtype=np.int8),
        np.array([p[3] for p in parts], dtype=np.uint8),
        np.zeros(n, np.int16),
        np.zeros(n, np.int8),
        np.zeros(n, np.int32),
    )


def test_records_to_planes_matches_encode_board() -> None:
    boards = POSITIONS[::3]
    planes = records_to_planes(records_from_boards(boards))
    expected = np.stack([encode_board(b) for b in boards])
    assert planes.shape == expected.shape
    assert np.array_equal(planes, expected)


def test_records_to_planes_covers_en_passant_and_black() -> None:
    boards = [
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ]
    planes = records_to_planes(records_from_boards(boards))
    assert np.array_equal(planes, np.stack([encode_board(b) for b in boards]))
    assert planes[:, 16].sum() == 2


PGN = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "2050"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0

[Event "Rated Bullet game"]
[Result "0-1"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Termination "Normal"]

1. e4 e5 0-1

[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "2300"]
[Termination "Normal"]

1. e4 e5 1-0
"""


def test_header_filter() -> None:
    texts = [t for t in PGN.split("\n\n[Event ")]
    texts = [texts[0]] + ["[Event " + t for t in texts[1:]]
    flt = GameFilter()
    verdicts = [accept(read_headers(t), flt) for t in texts]
    assert verdicts == [True, False, False]  # ok / bullet / one player under 2000


def test_game_to_records_labels() -> None:
    rec = game_to_records(PGN.split("\n\n[Event ")[0], game_id=7, min_plies=4)
    assert rec is not None
    assert len(rec) == 14
    assert (rec.game == 7).all()
    # Move 0 is White's e2e4; White won, so the value from White's view is +1, Black's -1.
    start = chess.Board()
    assert rec.move[0] == move_to_index(chess.Move.from_uci("e2e4"), start)
    assert rec.result[0] == 1 and rec.result[1] == -1
    board = chess.Board()
    board.push_uci("e2e4")
    assert index_to_move(int(rec.move[1]), board) == chess.Move.from_uci("e7e5")


def test_prepare_dataset_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "games.pgn.zst"
    source.write_bytes(zstandard.ZstdCompressor().compress(PGN.encode()))
    assert len(list(iter_game_texts(source))) == 3
    out = tmp_path / "out"
    stats = prepare_dataset(source, out, max_games=10, flt=GameFilter(min_plies=4), progress=False)
    assert stats["games"] == 1 and stats["positions"] == 14
    data = load_dataset(out)
    assert len(data) == 14
    assert records_to_planes(data).shape == (14, 19, 8, 8)


def test_pgn_text_io_roundtrip() -> None:
    game = chess.pgn.read_game(io.StringIO(PGN))
    assert game is not None
    assert game.headers["WhiteElo"] == "2100"
