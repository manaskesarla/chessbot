"""Compact position records for training data, and fast vectorised conversion to planes.

Storing 19 float planes per position would need ~5 KB each (tens of GB for millions of
positions). Instead each position is stored in ~70 bytes and expanded to planes per batch.
``records_to_planes`` must produce exactly the same tensor as ``encoding.encode_board``;
tests enforce that.

Piece code per square: 0 = empty, 1-6 = white P N B R Q K, 7-12 = black P N B R Q K.
Flags bits: 1 = White to move, 2/4 = White kingside/queenside castling,
8/16 = Black kingside/queenside castling.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import numpy.typing as npt

from chessbot.encoding.board import NUM_PLANES

FLIP = np.arange(64) ^ 56  # vertical mirror of square indices


@dataclass
class Records:
    """Struct-of-arrays for N positions."""

    squares: npt.NDArray[np.uint8]  # (N, 64)
    flags: npt.NDArray[np.uint8]  # (N,)
    ep: npt.NDArray[np.int8]  # (N,) en passant square or -1
    halfmove: npt.NDArray[np.uint8]  # (N,) clamped to 255
    move: npt.NDArray[np.int16]  # (N,) move index taken by the human
    result: npt.NDArray[np.int8]  # (N,) +1 / 0 / -1 from the side to move's view
    game: npt.NDArray[np.int32]  # (N,) game id, used to split train/validation by game

    def __len__(self) -> int:
        return len(self.flags)

    def take(self, index: npt.NDArray[np.intp]) -> Records:
        return Records(
            self.squares[index],
            self.flags[index],
            self.ep[index],
            self.halfmove[index],
            self.move[index],
            self.result[index],
            self.game[index],
        )

    @staticmethod
    def concat(parts: list[Records]) -> Records:
        return Records(*(np.concatenate([getattr(p, f) for p in parts]) for f in FIELDS))


FIELDS = ("squares", "flags", "ep", "halfmove", "move", "result", "game")


def board_to_record(board: chess.Board) -> tuple[npt.NDArray[np.uint8], int, int, int]:
    """(squares, flags, ep_square_or_-1, halfmove) for one position."""
    squares = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        squares[square] = piece.piece_type + (6 if piece.color == chess.BLACK else 0)
    flags = (
        int(board.turn == chess.WHITE)
        | (2 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        | (4 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        | (8 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        | (16 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep = board.ep_square if board.ep_square is not None else -1
    return squares, flags, ep, min(board.halfmove_clock, 255)


def records_to_planes(rec: Records) -> npt.NDArray[np.float32]:
    """Expand records to a float32 array of shape (N, 19, 8, 8), matching ``encode_board``."""
    n = len(rec)
    me_white = (rec.flags & 1).astype(bool)
    squares = np.where(me_white[:, None], rec.squares, rec.squares[:, FLIP])

    planes = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    rows, cols = np.nonzero(squares)
    code = squares[rows, cols].astype(np.int64) - 1
    piece_type, piece_black = code % 6, code // 6 == 1
    own = piece_black != me_white[rows]  # black piece & black to move, or white & white
    planes[rows, piece_type + np.where(own, 0, 6), cols] = 1.0

    flags = rec.flags
    white = me_white
    own_k = np.where(white, flags & 2, flags & 8) > 0
    own_q = np.where(white, flags & 4, flags & 16) > 0
    opp_k = np.where(white, flags & 8, flags & 2) > 0
    opp_q = np.where(white, flags & 16, flags & 4) > 0
    planes[:, 12] = own_k[:, None]
    planes[:, 13] = own_q[:, None]
    planes[:, 14] = opp_k[:, None]
    planes[:, 15] = opp_q[:, None]

    has_ep = rec.ep >= 0
    ep_sq = np.where(white, rec.ep, rec.ep ^ 56)[has_ep].astype(np.int64)
    planes[np.nonzero(has_ep)[0], 16, ep_sq] = 1.0

    planes[:, 17] = (rec.halfmove.astype(np.float32) / 100.0)[:, None]
    planes[:, 18] = white[:, None]
    return planes.reshape(n, NUM_PLANES, 8, 8)
