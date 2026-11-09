from __future__ import annotations

import random

import chess

from chessbot.baselines.evaluation import PIECE_VALUES, evaluate

MATE_SCORE = 100_000
INF = 1_000_000


def _move_order_key(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched first: promotions, then captures by MVV-LVA."""
    score = 0
    if move.promotion:
        score += PIECE_VALUES[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        score += 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker] + 1000
    return score


class AlphaBetaBot:
    """Negamax alpha-beta with capture ordering and quiescence search."""

    def __init__(self, depth: int = 3, seed: int | None = None) -> None:
        self.depth = depth
        self.name = f"alphabeta-d{depth}"
        self._rng = random.Random(seed)
        self.nodes = 0

    def choose_move(self, board: chess.Board) -> chess.Move:
        self.nodes = 0
        moves = sorted(board.legal_moves, key=lambda m: -_move_order_key(board, m))
        best_score = -INF
        best_moves: list[chess.Move] = []
        for move in moves:
            # Window is one below the best score so far: moves tying the best are scored
            # exactly (for random tie-breaking) while clearly worse ones still get pruned.
            alpha = best_score - 1
            board.push(move)
            score = -self._negamax(board, self.depth - 1, -INF, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self._rng.choice(best_moves)

    def _terminal_score(self, board: chess.Board, ply: int) -> int | None:
        if board.is_checkmate():
            return -MATE_SCORE + ply  # prefer faster mates
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0
        return None

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)
        for move in sorted(board.legal_moves, key=lambda m: -_move_order_key(board, m)):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Search captures only until the position is quiet, to avoid horizon blunders."""
        self.nodes += 1
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        for move in sorted(captures, key=lambda m: -_move_order_key(board, m)):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
