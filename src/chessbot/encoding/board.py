"""Board-to-tensor encoding (19 planes of 8x8), from the side to move's perspective.

Plane layout:
    0-5    side-to-move pieces: pawn, knight, bishop, rook, queen, king
    6-11   opponent pieces, same order
    12-13  side-to-move castling rights: kingside, queenside
    14-15  opponent castling rights: kingside, queenside
    16     en passant target square
    17     halfmove clock / 100 (constant plane)
    18     1.0 if White is to move, else 0.0 (constant plane)

For Black to move the board is flipped vertically so "forward" always points up the planes.
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

NUM_PLANES = 19


def encode_board(board: chess.Board) -> npt.NDArray[np.float32]:
    """Return a float32 array of shape (19, 8, 8)."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    me = board.turn
    flip = me == chess.BLACK

    for square, piece in board.piece_map().items():
        sq = square ^ 56 if flip else square
        plane = piece.piece_type - 1 + (0 if piece.color == me else 6)
        planes[plane, sq >> 3, sq & 7] = 1.0

    planes[12] = board.has_kingside_castling_rights(me)
    planes[13] = board.has_queenside_castling_rights(me)
    planes[14] = board.has_kingside_castling_rights(not me)
    planes[15] = board.has_queenside_castling_rights(not me)

    if board.ep_square is not None:
        sq = board.ep_square ^ 56 if flip else board.ep_square
        planes[16, sq >> 3, sq & 7] = 1.0

    planes[17] = board.halfmove_clock / 100.0
    planes[18] = 1.0 if me == chess.WHITE else 0.0
    return planes
