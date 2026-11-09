import random

import chess
import numpy as np
import pytest

from chessbot.encoding import (
    NUM_PLANES,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_mask,
    legal_move_indices,
    move_to_index,
)


def random_positions(n_games: int, seed: int = 0, max_plies: int = 120) -> list[chess.Board]:
    """Positions from random playouts (covers castling, en passant, promotions often enough)."""
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            positions.append(board.copy())
            board.push(rng.choice(list(board.legal_moves)))
    return positions


POSITIONS = random_positions(150)


def test_enough_positions() -> None:
    assert len(POSITIONS) > 10_000


def test_round_trip_every_legal_move() -> None:
    for board in POSITIONS:
        for move in board.legal_moves:
            index = move_to_index(move, board)
            assert 0 <= index < POLICY_SIZE
            assert index_to_move(index, board) == move, (board.fen(), move)


def test_legal_moves_get_unique_indices() -> None:
    for board in POSITIONS[::5]:
        indices = legal_move_indices(board)
        assert len(indices) == len(set(indices)), board.fen()


def test_policy_size() -> None:
    assert POLICY_SIZE == 4672


def test_known_index_e2e4() -> None:
    # e2 = square 12; double push north = direction 0, distance 2 -> type 1.
    assert move_to_index(chess.Move.from_uci("e2e4"), chess.Board()) == 12 * 73 + 1


def test_black_moves_encoded_from_own_perspective() -> None:
    # Black's e7e5 must encode identically to White's e2e4.
    board = chess.Board()
    board.push_uci("a2a3")
    assert move_to_index(chess.Move.from_uci("e7e5"), board) == 12 * 73 + 1


def test_castling_en_passant_and_promotions_round_trip() -> None:
    cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", {"e1g1", "e1c1"}),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", {"e8g8", "e8c8"}),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", {"e5d6"}),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}),
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", {"a7b8q", "a7b8n", "a7b8b", "a7b8r"}),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", {"a2a1q", "a2a1n", "a2a1b", "a2a1r"}),
    ]
    for fen, uci_moves in cases:
        board = chess.Board(fen)
        for uci in uci_moves:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, (fen, uci)
            assert index_to_move(move_to_index(move, board), board) == move, (fen, uci)


def test_underpromotions_are_distinct_from_queen_promotion() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    indices = {move_to_index(chess.Move.from_uci(f"a7a8{p}"), board) for p in "qrbn"}
    assert len(indices) == 4


def test_legal_mask_matches_legal_moves() -> None:
    board = chess.Board()
    mask = legal_mask(board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum() == 20


def test_encode_start_position() -> None:
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32
    assert planes[0].sum() == 8 and planes[0, 1].sum() == 8  # own pawns on rank 2
    assert planes[3, 0, 0] == 1 and planes[3, 0, 7] == 1  # own rooks a1, h1
    assert planes[5, 0, 4] == 1  # own king e1
    assert planes[6].sum() == 8 and planes[6, 6].sum() == 8  # opponent pawns on rank 7
    assert planes[11, 7, 4] == 1  # opponent king e8
    assert planes[12:16].sum() == 4 * 64  # all castling rights
    assert planes[16].sum() == 0
    assert planes[18, 0, 0] == 1


def test_black_to_move_is_encoded_like_mirrored_white() -> None:
    for board in POSITIONS[::7]:
        if board.turn != chess.BLACK:
            continue
        a = encode_board(board)
        b = encode_board(board.mirror())
        # Identical except the side-to-move plane.
        assert np.array_equal(np.delete(a, 18, axis=0), np.delete(b, 18, axis=0)), board.fen()
        assert a[18].sum() == 0 and b[18].sum() == 64


def test_en_passant_plane() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    planes = encode_board(board)
    assert planes[16].sum() == 1 and planes[16, 5, 3] == 1


@pytest.mark.parametrize("fen", [p.fen() for p in POSITIONS[:50]])
def test_encoding_is_deterministic(fen: str) -> None:
    board = chess.Board(fen)
    assert np.array_equal(encode_board(board), encode_board(board))
