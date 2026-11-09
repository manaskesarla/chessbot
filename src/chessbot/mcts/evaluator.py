"""Position evaluators used by MCTS.

An evaluator takes a batch of positions plus, for each, the indices of its legal moves, and
returns (priors over those legal moves, value for the side to move in [-1, 1]).
"""

from __future__ import annotations

from typing import Protocol

import chess
import numpy as np
import numpy.typing as npt
import torch

from chessbot.baselines.evaluation import evaluate as material_pst_eval
from chessbot.encoding import encode_board
from chessbot.model import ChessNet

F32 = npt.NDArray[np.float32]


class Evaluator(Protocol):
    def evaluate(
        self, boards: list[chess.Board], legal_indices: list[npt.NDArray[np.int64]]
    ) -> tuple[list[F32], F32]: ...


class NetEvaluator:
    """Neural-network evaluator: softmax of policy logits over legal moves, plus value head."""

    def __init__(self, net: ChessNet, device: torch.device) -> None:
        self.device = device
        self.net = net.to(device).eval()

    @torch.no_grad()
    def evaluate(
        self, boards: list[chess.Board], legal_indices: list[npt.NDArray[np.int64]]
    ) -> tuple[list[F32], F32]:
        planes = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(self.device)
        logits, values = self.net(planes)
        logits_np = logits.float().cpu().numpy()
        priors: list[F32] = []
        for row, idx in zip(logits_np, legal_indices, strict=True):
            legal = row[idx]
            exp = np.exp(legal - legal.max())
            priors.append((exp / exp.sum()).astype(np.float32))
        return priors, values.float().cpu().numpy().astype(np.float32)


class HeuristicEvaluator:
    """Uniform priors and a squashed material/piece-square value. For tests and debugging
    the search logic independently of any trained network."""

    def evaluate(
        self, boards: list[chess.Board], legal_indices: list[npt.NDArray[np.int64]]
    ) -> tuple[list[F32], F32]:
        priors = [np.full(len(i), 1.0 / len(i), dtype=np.float32) for i in legal_indices]
        values = np.array([np.tanh(material_pst_eval(b) / 1000.0) for b in boards], np.float32)
        return priors, values
