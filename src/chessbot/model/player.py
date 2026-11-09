"""Play with the raw network (no search): highest-probability legal move."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from chessbot.encoding import encode_board, index_to_move, legal_mask
from chessbot.model.network import ChessNet, load_checkpoint
from chessbot.smoke import pick_device


class NetBot:
    def __init__(
        self, net: ChessNet, device: torch.device | None = None, name: str = "net"
    ) -> None:
        self.device = device or pick_device()
        self.net = net.to(self.device).eval()
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: Path, name: str | None = None) -> NetBot:
        device = pick_device()
        return cls(load_checkpoint(path, device), device, name or path.parent.name)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Return (logits over 4672 slots, value for the side to move)."""
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        return logits[0].cpu().numpy(), float(value.item())

    def choose_move(self, board: chess.Board) -> chess.Move:
        logits, _ = self.evaluate(board)
        logits = np.where(legal_mask(board), logits, -np.inf)
        return index_to_move(int(np.argmax(logits)), board)
