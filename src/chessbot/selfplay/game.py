"""Play one self-play game with MCTS and turn it into training records."""

from __future__ import annotations

import chess
import numpy as np

from chessbot.config import MCTSConfig
from chessbot.data.records import Records, board_to_record
from chessbot.encoding.moves import move_to_index
from chessbot.mcts import MCTS
from chessbot.selfplay.buffer import POLICY_WIDTH, SelfPlayRecords


def play_selfplay_game(
    mcts: MCTS,
    simulations: int,
    mcts_cfg: MCTSConfig,
    max_plies: int,
    rng: np.random.Generator,
    game_id: int = 0,
    start: chess.Board | None = None,
) -> tuple[SelfPlayRecords, float, int]:
    """Returns (positions, White's score 1/0.5/0, number of plies).

    Moves are sampled in proportion to visit counts for the first ``temperature_moves`` plies
    (variety), then the most-visited move is played. Dirichlet noise is added at every root.
    """
    board = start.copy() if start is not None else chess.Board()
    squares: list[np.ndarray] = []
    flags: list[int] = []
    eps: list[int] = []
    halfmoves: list[int] = []
    played: list[int] = []
    movers: list[bool] = []
    pol_idx: list[np.ndarray] = []
    pol_prob: list[np.ndarray] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        result = mcts.search(board, simulations, add_noise=True)
        temperature = 1.0 if board.ply() < mcts_cfg.temperature_moves else 0.0
        move = result.sample_move(temperature, rng)

        sq, fl, ep, hm = board_to_record(board)
        squares.append(sq)
        flags.append(fl)
        eps.append(ep)
        halfmoves.append(hm)
        played.append(move_to_index(move, board))
        movers.append(board.turn == chess.WHITE)

        top = np.argsort(-result.visits)[:POLICY_WIDTH]
        idx = np.zeros(POLICY_WIDTH, np.int16)
        prob = np.zeros(POLICY_WIDTH, np.float16)
        kept = result.visits[top].astype(np.float64)
        idx[: len(top)] = [move_to_index(result.moves[i], board) for i in top]
        prob[: len(top)] = kept / kept.sum()
        pol_idx.append(idx)
        pol_prob.append(prob)

        board.push(move)

    outcome = board.outcome(claim_draw=True)
    white_value = 0 if outcome is None or outcome.winner is None else (
        1 if outcome.winner == chess.WHITE else -1
    )
    n = len(played)
    if n == 0:
        return SelfPlayRecords.empty(), 0.5, 0
    result_arr = np.array([white_value if w else -white_value for w in movers], np.int8)
    rec = Records(
        np.stack(squares),
        np.array(flags, np.uint8),
        np.array(eps, np.int8),
        np.array(halfmoves, np.uint8),
        np.array(played, np.int16),
        result_arr,
        np.full(n, game_id, np.int32),
    )
    records = SelfPlayRecords(rec, np.stack(pol_idx), np.stack(pol_prob))
    return records, (white_value + 1) / 2, n
