import chess
import pytest

from chessbot.baselines import AlphaBetaBot, GreedyBot, RandomBot
from chessbot.baselines.evaluation import evaluate, material
from chessbot.eval import elo_difference, elo_from_score, expected_score, play_game, play_match


def test_material_and_eval_symmetric_at_start() -> None:
    board = chess.Board()
    assert material(board) == 0
    assert evaluate(board) == 0  # tables are mirror-symmetric for the two sides


def test_eval_is_from_side_to_move() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert evaluate(board) > 0
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1")
    assert evaluate(board) < 0


@pytest.mark.parametrize("bot", [RandomBot(1), GreedyBot(1), AlphaBetaBot(2, 1)])
def test_bots_return_legal_moves(bot: object) -> None:
    board = chess.Board()
    for _ in range(12):
        if board.is_game_over():
            break
        move = bot.choose_move(board)  # type: ignore[attr-defined]
        assert move in board.legal_moves
        board.push(move)


def test_greedy_captures_free_queen() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert GreedyBot(0).choose_move(board) == chess.Move.from_uci("d1d5")


def test_greedy_and_alphabeta_find_mate_in_one() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")  # Ra8#
    assert GreedyBot(0).choose_move(board) == chess.Move.from_uci("a1a8")
    assert AlphaBetaBot(2, 0).choose_move(board) == chess.Move.from_uci("a1a8")


def test_alphabeta_avoids_hanging_queen() -> None:
    # Qxb7?? loses the queen to the rook on b8's defender... here Qxd5 is a poisoned grab.
    board = chess.Board("3r2k1/8/8/3p4/8/8/8/3QK3 w - - 0 1")
    move = AlphaBetaBot(3, 0).choose_move(board)
    assert move != chess.Move.from_uci("d1d5")


def test_play_game_finishes() -> None:
    score, board = play_game(RandomBot(1), RandomBot(2))
    assert score in (0.0, 0.5, 1.0)
    assert board.is_game_over(claim_draw=True) or board.ply() >= 400


def test_alphabeta_beats_random() -> None:
    result = play_match(AlphaBetaBot(2, 0), RandomBot(0), games=4)
    assert result.wins == 4


def test_odd_game_count_rejected() -> None:
    with pytest.raises(ValueError):
        play_match(RandomBot(), RandomBot(), games=3)


def test_elo_helpers() -> None:
    assert elo_from_score(0.5) == pytest.approx(0.0)
    assert expected_score(elo_from_score(0.75)) == pytest.approx(0.75)
    even = elo_difference(40, 20, 40)
    assert even.diff == pytest.approx(0.0)
    assert even.low < 0 < even.high
    strong = elo_difference(90, 5, 5)
    assert strong.diff > 300
    assert elo_difference(10, 0, 0).high < float("inf")  # clamped, not infinite
