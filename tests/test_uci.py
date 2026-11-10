import chess

from chessbot.config import MCTSConfig
from chessbot.mcts import HeuristicEvaluator, MCTSBot
from chessbot.uci import TimeManager, UciEngine


def engine() -> UciEngine:
    bot = MCTSBot(HeuristicEvaluator(), 16, MCTSConfig(batch_size=4), seed=0)
    return UciEngine(bot, "test", TimeManager(min_sims=8, max_sims=32))


def test_handshake() -> None:
    e = engine()
    assert e.handle("uci")[0][-1] == "uciok"
    assert e.handle("isready")[0] == ["readyok"]
    assert e.handle("quit")[1] is False


def test_position_and_go_returns_legal_move() -> None:
    e = engine()
    e.handle("position startpos moves e2e4 e7e5")
    out, _ = e.handle("go wtime 60000 btime 60000 winc 1000 binc 1000")
    move = chess.Move.from_uci(out[0].split()[1])
    assert move in e.board.legal_moves


def test_fen_position() -> None:
    e = engine()
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    e.handle(f"position fen {fen} moves a1a8")
    assert e.board.is_checkmate()


def test_finds_back_rank_mate_on_movetime() -> None:
    e = engine()
    e.handle("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    out, _ = e.handle("go movetime 1000")
    assert out[0] == "bestmove a1a8"


def test_time_manager_scales_with_clock() -> None:
    tm = TimeManager(min_sims=8, max_sims=1000, sims_per_second=500)
    b = chess.Board()
    low = tm.simulations(b, {"wtime": 5000, "btime": 5000})
    high = tm.simulations(b, {"wtime": 300000, "btime": 300000})
    assert 8 <= low < high <= 1000
    assert tm.simulations(b, {}) == tm.default_sims


def test_garbage_input_is_ignored() -> None:
    e = engine()
    assert e.handle("")[0] == [] and e.handle("nonsense foo")[0] == []
