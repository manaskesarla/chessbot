"""A small opening book: short, varied, balanced lines (SAN) so matches don't repeat one game."""

from __future__ import annotations

import chess

_LINES = [
    "e4 e5 Nf3 Nc6 Bb5 a6",
    "e4 e5 Nf3 Nc6 Bc4 Bc5",
    "e4 e5 Nf3 Nf6 Nxe5 d6",
    "e4 c5 Nf3 d6 d4 cxd4",
    "e4 c5 Nf3 Nc6 d4 cxd4",
    "e4 e6 d4 d5 Nc3 Nf6",
    "e4 c6 d4 d5 Nc3 dxe4",
    "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "e4 d6 d4 Nf6 Nc3 g6",
    "d4 d5 c4 e6 Nc3 Nf6",
    "d4 d5 c4 c6 Nf3 Nf6",
    "d4 Nf6 c4 e6 Nc3 Bb4",
    "d4 Nf6 c4 g6 Nc3 Bg7",
    "d4 Nf6 Nf3 g6 g3 Bg7",
    "c4 e5 Nc3 Nf6 Nf3 Nc6",
    "c4 c5 Nf3 Nf6 Nc3 Nc6",
    "Nf3 d5 g3 Nf6 Bg2 e6",
    "Nf3 Nf6 c4 e6 Nc3 d5",
    "e4 e5 Nf3 Nc6 d4 exd4",
    "d4 d5 Nf3 Nf6 Bf4 e6",
]


def opening_boards() -> list[chess.Board]:
    boards = []
    for line in _LINES:
        board = chess.Board()
        for san in line.split():
            board.push_san(san)
        boards.append(board)
    return boards
