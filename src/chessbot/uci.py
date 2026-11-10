"""A UCI engine around the network + MCTS bot, for use with the lichess-bot bridge.

    uv run uci [--checkpoint PATH] [--max-sims N]

Only protocol messages go to stdout; anything else goes to stderr. Time management turns the
clock into a simulation count using the measured search speed, so the bot does not flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess

from chessbot.mcts import MCTSBot

DEFAULT_CHECKPOINT = "checkpoints/selfplay/best.pt"


@dataclass
class TimeManager:
    """Choose a simulation count for the next move from the clock and the search speed."""

    min_sims: int = 16
    max_sims: int = 800
    default_sims: int = 100
    sims_per_second: float = 300.0  # conservative until the first real measurement
    expected_moves_left: int = 30
    safety: float = 0.85
    history: list[float] = field(default_factory=list)

    def budget_seconds(self, board: chess.Board, params: dict[str, int]) -> float | None:
        """Seconds to spend on this move, or None when the GUI gave no clock."""
        if "movetime" in params:
            return params["movetime"] / 1000 * self.safety
        mine = "wtime" if board.turn == chess.WHITE else "btime"
        if mine not in params:
            return None
        remaining = params[mine] / 1000
        inc = params.get("winc" if board.turn == chess.WHITE else "binc", 0) / 1000
        budget = remaining / self.expected_moves_left + 0.7 * inc
        budget = min(budget, remaining * 0.2)  # never spend more than a fifth of what is left
        return max(0.05, budget) * self.safety

    def simulations(self, board: chess.Board, params: dict[str, int]) -> int:
        seconds = self.budget_seconds(board, params)
        if seconds is None:
            return self.default_sims
        return max(self.min_sims, min(self.max_sims, int(seconds * self.sims_per_second)))

    def record(self, simulations: int, seconds: float) -> None:
        if seconds > 0.02 and simulations >= self.min_sims:
            measured = simulations / seconds
            self.sims_per_second = 0.6 * self.sims_per_second + 0.4 * measured


class UciEngine:
    def __init__(
        self, bot: MCTSBot, name: str = "chessbot", time_manager: TimeManager | None = None
    ) -> None:
        self.bot = bot
        self.name = name
        self.tm = time_manager or TimeManager()
        self.board = chess.Board()

    def handle(self, line: str) -> tuple[list[str], bool]:
        """Process one command. Returns (lines to print, keep running)."""
        parts = line.strip().split()
        if not parts:
            return [], True
        cmd = parts[0]
        if cmd == "uci":
            return [f"id name {self.name}", "id author chessbot", "uciok"], True
        if cmd == "isready":
            return ["readyok"], True
        if cmd == "ucinewgame":
            self.board = chess.Board()
            return [], True
        if cmd == "position":
            self._position(parts[1:])
            return [], True
        if cmd == "go":
            return [self._go(parts[1:])], True
        if cmd == "quit":
            return [], False
        return [], True  # setoption, stop, debug, ...: ignored

    def _position(self, args: list[str]) -> None:
        if args and args[0] == "startpos":
            board, rest = chess.Board(), args[1:]
        elif args and args[0] == "fen":
            split = args.index("moves") if "moves" in args else len(args)
            board, rest = chess.Board(" ".join(args[1:split])), args[split:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                board.push_uci(uci)
        self.board = board

    def _go(self, args: list[str]) -> str:
        params: dict[str, int] = {}
        keys = {"wtime", "btime", "winc", "binc", "movetime", "movestogo"}
        for i, tok in enumerate(args[:-1]):
            if tok in keys and args[i + 1].lstrip("-").isdigit():
                params[tok] = int(args[i + 1])
        if "movestogo" in params:
            self.tm.expected_moves_left = max(2, params["movestogo"])
        sims = self.tm.simulations(self.board, params)
        self.bot.simulations = sims
        start = time.perf_counter()
        try:
            move = self.bot.choose_move(self.board)
        except Exception as exc:  # a legal move is always better than a lost-on-time game
            print(f"search failed ({exc!r}); playing first legal move", file=sys.stderr, flush=True)
            move = next(iter(self.board.legal_moves))
        self.tm.record(sims, time.perf_counter() - start)
        return f"bestmove {move.uci()}"


def main() -> None:
    from chessbot.config import REPO_ROOT, MCTSConfig, load_config

    parser = argparse.ArgumentParser(prog="uci")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-sims", type=int, default=800)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    bot = MCTSBot.from_checkpoint(ckpt, 100, load_config(MCTSConfig, "mcts"))
    engine = UciEngine(bot, "chessbot", TimeManager(max_sims=args.max_sims))
    print(f"loaded {ckpt}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            out, keep_going = engine.handle(line)
        except Exception as exc:  # never die mid-game: report and carry on
            print(f"error handling {line.strip()!r}: {exc!r}", file=sys.stderr, flush=True)
            continue
        for text in out:
            print(text, flush=True)
        if not keep_going:
            break


if __name__ == "__main__":
    main()
