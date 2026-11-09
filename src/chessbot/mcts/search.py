"""Monte Carlo Tree Search with PUCT selection, batched leaf evaluation and a position cache.

One simulation walks from the root choosing, at each node, the move maximising
    Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
(Q: average result so far, P: network prior, N: visit count), until it reaches an unexpanded
position. That position is scored by the network (or exactly, if the game is over), and the
result is backed up the path with the sign flipped at every level.

Speed-ups:
  * Leaves are collected into batches and evaluated in one network call. "Virtual loss"
    temporarily marks each in-flight path as a loss so concurrent simulations spread out.
  * Evaluations are cached by position hash, so transpositions are evaluated once.

All statistics live on the *edges* of a node, from the point of view of the side to move there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import chess.polyglot
import numpy as np
import numpy.typing as npt

from chessbot.config import MCTSConfig
from chessbot.encoding.moves import move_to_index
from chessbot.mcts.evaluator import Evaluator

CACHE_LIMIT = 200_000

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]


class Node:
    __slots__ = (
        "N",
        "P",
        "W",
        "children",
        "indices",
        "moves",
        "pending",
        "terminal",
        "total_n",
        "value",
    )

    def __init__(self) -> None:
        self.moves: list[chess.Move] | None = None
        self.indices: I64 | None = None
        self.P: F32 | None = None  # priors over legal moves
        self.N: npt.NDArray[np.int32] | None = None  # visits per move
        self.W: F32 | None = None  # summed value per move (side to move's view)
        self.children: list[Node | None] = []
        self.total_n = 0
        self.value = 0.0  # network value of this position, side to move's view
        self.terminal: float | None = None  # exact value if the game is over here
        self.pending = False  # waiting for a network evaluation in the current batch

    def expand(self, moves: list[chess.Move], indices: I64, priors: F32, value: float) -> None:
        n = len(moves)
        self.moves, self.indices, self.P = moves, indices, priors
        self.N = np.zeros(n, dtype=np.int32)
        self.W = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.value = value
        self.total_n = 0
        self.pending = False


@dataclass
class SearchResult:
    moves: list[chess.Move]
    visits: npt.NDArray[np.int32]
    q: F32  # mean value of each root move (root side to move's view)
    value: float  # root network value

    def best_move(self) -> chess.Move:
        return self.moves[int(np.argmax(self.visits))]

    def sample_move(self, temperature: float, rng: np.random.Generator) -> chess.Move:
        """Pick a move with probability proportional to visits^(1/temperature)."""
        if temperature <= 1e-3:
            return self.best_move()
        weights = self.visits.astype(np.float64) ** (1.0 / temperature)
        return self.moves[int(rng.choice(len(self.moves), p=weights / weights.sum()))]

    def policy(self) -> F32:
        """Visit distribution over root moves (the self-play training target)."""
        return (self.visits / self.visits.sum()).astype(np.float32)


@dataclass
class _Pending:
    node: Node
    path: list[tuple[Node, int]]
    board: chess.Board
    moves: list[chess.Move]
    indices: I64


_TERMINAL, _CACHED, _NEEDS_EVAL = 0, 1, 2


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.cfg = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()
        self.cache: dict[int, tuple[list[chess.Move], I64, F32, float]] = {}
        self.cache_hits = 0
        self.evaluations = 0

    # ----------------------------------------------------------------- public API

    def search(
        self, board: chess.Board, simulations: int, add_noise: bool = False
    ) -> SearchResult:
        """Run ``simulations`` simulations from ``board`` (restored before returning)."""
        root = Node()
        batch: list[_Pending] = []
        status = self._resolve(board, root, is_root=True, batch=batch)
        if status == _TERMINAL:
            raise ValueError("cannot search a finished game")
        if status == _NEEDS_EVAL:
            self._evaluate(batch)
            batch.clear()
        assert root.P is not None and root.moves is not None
        if add_noise:
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.moves))
            f = self.cfg.dirichlet_fraction
            root.P = ((1 - f) * root.P + f * noise).astype(np.float32)

        done = 0
        while done < simulations:
            limit = min(self.cfg.batch_size, simulations - done)
            counted = 0
            while counted < limit:
                if self._simulate(board, root, batch) == "collision":
                    break  # an in-flight leaf was reached twice: evaluate the batch first
                counted += 1
            # A collision needs a pending leaf, so counted >= 1 and every batch makes progress.
            self._evaluate(batch)
            batch.clear()
            done += counted
        assert root.N is not None and root.W is not None
        q = np.where(root.N > 0, root.W / np.maximum(root.N, 1), 0.0).astype(np.float32)
        return SearchResult(root.moves, root.N.copy(), q, root.value)

    # ------------------------------------------------------------------ internals

    def _simulate(self, board: chess.Board, root: Node, batch: list[_Pending]) -> str:
        path: list[tuple[Node, int]] = []
        node = root
        outcome = "backed"
        while True:
            if node.terminal is not None:
                self._backup(path, node.terminal)
                break
            if node.moves is None:
                if node.pending:
                    self._undo_virtual_loss(path)
                    outcome = "collision"
                    break
                status = self._resolve(board, node, is_root=False, batch=batch, path=path)
                if status == _TERMINAL:
                    assert node.terminal is not None
                    self._backup(path, node.terminal)
                elif status == _CACHED:
                    self._backup(path, node.value)
                else:
                    outcome = "pending"
                break
            a = self._select(node)
            assert node.N is not None and node.W is not None and node.moves is not None
            node.N[a] += 1
            node.W[a] -= 1.0
            node.total_n += 1
            path.append((node, a))
            board.push(node.moves[a])
            child = node.children[a]
            if child is None:
                child = Node()
                node.children[a] = child
            node = child
        for _ in path:
            board.pop()
        return outcome

    def _select(self, node: Node) -> int:
        assert node.N is not None and node.W is not None and node.P is not None
        fpu = node.value - self.cfg.fpu_reduction
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), fpu)
        u = self.cfg.c_puct * node.P * math.sqrt(max(node.total_n, 1)) / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _resolve(
        self,
        board: chess.Board,
        node: Node,
        is_root: bool,
        batch: list[_Pending],
        path: list[tuple[Node, int]] | None = None,
    ) -> int:
        """Classify an unexpanded position: game over, cached, or needing the network."""
        moves = list(board.legal_moves)
        if not moves:
            node.terminal = -1.0 if board.is_checkmate() else 0.0
            return _TERMINAL
        drawn = (
            board.is_insufficient_material()
            or board.halfmove_clock >= 100
            or board.is_repetition(2)
        )
        if drawn and not is_root:
            node.terminal = 0.0
            return _TERMINAL

        key = chess.polyglot.zobrist_hash(board)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            node.expand(*hit)
            return _CACHED

        indices = np.array([move_to_index(m, board) for m in moves], dtype=np.int64)
        node.pending = True
        batch.append(_Pending(node, list(path or []), board.copy(stack=False), moves, indices))
        return _NEEDS_EVAL

    def _evaluate(self, batch: list[_Pending]) -> None:
        if not batch:
            return
        priors, values = self.evaluator.evaluate(
            [p.board for p in batch], [p.indices for p in batch]
        )
        self.evaluations += len(batch)
        if len(self.cache) + len(batch) > CACHE_LIMIT:
            self.cache.clear()
        for item, prior, value in zip(batch, priors, values, strict=True):
            v = float(value)
            item.node.expand(item.moves, item.indices, prior, v)
            self.cache[chess.polyglot.zobrist_hash(item.board)] = (
                item.moves,
                item.indices,
                prior,
                v,
            )
            self._backup(item.path, v)

    @staticmethod
    def _backup(path: list[tuple[Node, int]], leaf_value: float) -> None:
        """Propagate the leaf's value (for its side to move) up the path, flipping sign.

        Each edge already carries a virtual loss (N+1, W-1); adding ``value + 1`` to W turns it
        into a real visit without touching N.
        """
        v = leaf_value
        for node, a in reversed(path):
            v = -v
            assert node.W is not None
            node.W[a] += v + 1.0

    @staticmethod
    def _undo_virtual_loss(path: list[tuple[Node, int]]) -> None:
        for node, a in path:
            assert node.N is not None and node.W is not None
            node.N[a] -= 1
            node.W[a] += 1.0
            node.total_n -= 1
