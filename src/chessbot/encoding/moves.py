"""AlphaZero-style move encoding: 8x8x73 = 4672 move slots.

A move is encoded as ``from_square * 73 + move_type`` where everything is expressed
from the side to move's point of view (squares are flipped vertically for Black), so the
network only ever has to learn one perspective.

Move types (0-72):
    0-55   "queen-like" moves: 8 directions x 7 distances (also covers castling, en passant,
           pawn pushes and queen promotions)
    56-63  knight moves
    64-72  underpromotions (knight, bishop, rook) x (capture-left, straight, capture-right)
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

NUM_MOVE_TYPES = 73
POLICY_SIZE = 64 * NUM_MOVE_TYPES

# (file delta, rank delta) seen from the mover: N, NE, E, SE, S, SW, W, NW.
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_MOVES = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_KNIGHT_OFFSET = 56
_UNDERPROMO_OFFSET = 64


def _flip(square: int) -> int:
    """Mirror a square vertically (a1 <-> a8)."""
    return square ^ 56


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode ``move`` (played by ``board.turn``) as an index in [0, 4672)."""
    from_sq, to_sq = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        from_sq, to_sq = _flip(from_sq), _flip(to_sq)
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece = _UNDERPROMOTIONS.index(move.promotion)
        move_type = _UNDERPROMO_OFFSET + (df + 1) * 3 + piece
    elif (df, dr) in _KNIGHT_MOVES:
        move_type = _KNIGHT_OFFSET + _KNIGHT_MOVES.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        direction = _DIRECTIONS.index((df // distance, dr // distance))
        move_type = direction * 7 + (distance - 1)
    return from_sq * NUM_MOVE_TYPES + move_type


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode an index back to a move for ``board.turn``. Does not check legality."""
    from_sq, move_type = divmod(index, NUM_MOVE_TYPES)
    promotion: int | None = None

    if move_type >= _UNDERPROMO_OFFSET:
        df_index, piece = divmod(move_type - _UNDERPROMO_OFFSET, 3)
        df, dr = df_index - 1, 1
        promotion = _UNDERPROMOTIONS[piece]
    elif move_type >= _KNIGHT_OFFSET:
        df, dr = _KNIGHT_MOVES[move_type - _KNIGHT_OFFSET]
    else:
        direction, dist_minus_one = divmod(move_type, 7)
        distance = dist_minus_one + 1
        df, dr = _DIRECTIONS[direction][0] * distance, _DIRECTIONS[direction][1] * distance

    to_sq = chess.square(chess.square_file(from_sq) + df, chess.square_rank(from_sq) + dr)
    if promotion is None and chess.square_rank(to_sq) == 7:
        # A pawn reaching the last rank with a queen-like move is a queen promotion.
        mover_from = _flip(from_sq) if board.turn == chess.BLACK else from_sq
        if board.piece_type_at(mover_from) == chess.PAWN:
            promotion = chess.QUEEN

    if board.turn == chess.BLACK:
        from_sq, to_sq = _flip(from_sq), _flip(to_sq)
    return chess.Move(from_sq, to_sq, promotion)


def legal_move_indices(board: chess.Board) -> list[int]:
    return [move_to_index(m, board) for m in board.legal_moves]


def legal_mask(board: chess.Board) -> npt.NDArray[np.bool_]:
    """Boolean vector of length 4672, True for legal moves."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    mask[legal_move_indices(board)] = True
    return mask
