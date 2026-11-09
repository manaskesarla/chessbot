from dataclasses import replace

import chess
import numpy as np
import pytest
import torch

from chessbot.config import MCTSConfig, ModelConfig
from chessbot.mcts import MCTS, HeuristicEvaluator, MCTSBot, NetEvaluator
from chessbot.model import ChessNet

CFG = MCTSConfig()


def make(batch_size: int = 8, seed: int = 0) -> MCTS:
    cfg = replace(CFG, batch_size=batch_size)
    return MCTS(HeuristicEvaluator(), cfg, np.random.default_rng(seed))


def test_visits_sum_to_simulations_and_board_restored() -> None:
    board = chess.Board()
    fen = board.fen()
    result = make().search(board, 64)
    assert result.visits.sum() == 64
    assert board.fen() == fen and len(board.move_stack) == 0
    assert all(m in board.legal_moves for m in result.moves)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_finds_mate_in_one(batch_size: int) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")  # Ra8#
    result = make(batch_size).search(board, 80)
    assert result.best_move() == chess.Move.from_uci("a1a8")


def test_sees_forced_mate_two_plies_ahead() -> None:
    # After 1.Ra7 both Black king moves lose to Rb8#. With uniform priors a full search from the
    # start converges slowly (a trained policy fixes that), so test the two halves.
    black_to_move = chess.Board("4k3/R7/8/8/8/8/1R6/4K3 b - - 1 1")
    result = make().search(black_to_move, 300)
    assert (result.q < -0.8).all()  # every reply is judged losing for Black

    after_kd8 = chess.Board("3k4/R7/8/8/8/8/1R6/4K3 w - - 2 2")
    assert make().search(after_kd8, 300).best_move() == chess.Move.from_uci("b2b8")


def test_avoids_hanging_queen() -> None:
    board = chess.Board("3r2k1/8/8/3p4/8/8/8/3QK3 w - - 0 1")
    result = make().search(board, 200)
    assert result.best_move() != chess.Move.from_uci("d1d5")


def test_does_not_stalemate_when_winning() -> None:
    # Qc7?? would stalemate Black. A strong search must not choose it.
    board = chess.Board("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1")
    result = make().search(board, 200)
    assert result.best_move() != chess.Move.from_uci("c1c7")


def test_transposition_cache_is_used() -> None:
    mcts = make()
    mcts.search(chess.Board(), 200)
    assert mcts.cache_hits > 0
    assert mcts.evaluations < 200


def test_single_legal_move_terminates() -> None:
    board = chess.Board("7k/8/8/8/8/8/r7/7K w - - 0 1")
    assert board.legal_moves.count() == 1
    result = make().search(board, 20)
    assert result.visits.sum() == 20


def test_finished_game_rejected() -> None:
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1")
    with pytest.raises(ValueError):
        make().search(mated, 10)


def test_dirichlet_noise_changes_priors_but_keeps_search_valid() -> None:
    board = chess.Board()
    result = make().search(board, 40, add_noise=True)
    assert result.visits.sum() == 40


def test_sampling_follows_visits() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    result = make().search(board, 100)
    rng = np.random.default_rng(0)
    assert result.sample_move(0.0, rng) == result.best_move()
    assert result.sample_move(1.0, rng) in result.moves
    assert result.policy().sum() == pytest.approx(1.0)


def test_net_evaluator_search_and_bot() -> None:
    net = ChessNet(ModelConfig(blocks=1, filters=8))
    evaluator = NetEvaluator(net, torch.device("cpu"))
    board = chess.Board()
    bot = MCTSBot(evaluator, simulations=30, seed=0)
    move = bot.choose_move(board)
    assert move in board.legal_moves
    assert len(board.move_stack) == 0
    assert bot.last_result is not None and bot.last_result.visits.sum() == 30
