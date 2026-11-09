"""Replay-buffer storage for self-play positions.

Each position keeps the compact board record (see ``data.records``), the MCTS visit
distribution as a sparse (move index, probability) list of the ``POLICY_WIDTH`` most-visited
moves, and the final game result from the side to move's point of view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from chessbot.data.records import FIELDS, Records
from chessbot.encoding.moves import POLICY_SIZE

POLICY_WIDTH = 128  # >= simulations per move, so every visited move fits


@dataclass
class SelfPlayRecords:
    rec: Records
    pol_idx: npt.NDArray[np.int16]  # (N, POLICY_WIDTH) move indices (padding has prob 0)
    pol_prob: npt.NDArray[np.float16]  # (N, POLICY_WIDTH)

    def __len__(self) -> int:
        return len(self.rec)

    def take(self, index: npt.NDArray[np.intp]) -> SelfPlayRecords:
        return SelfPlayRecords(self.rec.take(index), self.pol_idx[index], self.pol_prob[index])

    @staticmethod
    def empty() -> SelfPlayRecords:
        rec = Records(
            np.zeros((0, 64), np.uint8),
            np.zeros(0, np.uint8),
            np.zeros(0, np.int8),
            np.zeros(0, np.uint8),
            np.zeros(0, np.int16),
            np.zeros(0, np.int8),
            np.zeros(0, np.int32),
        )
        return SelfPlayRecords(
            rec, np.zeros((0, POLICY_WIDTH), np.int16), np.zeros((0, POLICY_WIDTH), np.float16)
        )

    @staticmethod
    def concat(parts: list[SelfPlayRecords]) -> SelfPlayRecords:
        parts = [p for p in parts if len(p)]
        if not parts:
            return SelfPlayRecords.empty()
        return SelfPlayRecords(
            Records.concat([p.rec for p in parts]),
            np.concatenate([p.pol_idx for p in parts]),
            np.concatenate([p.pol_prob for p in parts]),
        )

    def last(self, n: int) -> SelfPlayRecords:
        """The most recent ``n`` positions (the rolling replay window)."""
        if len(self) <= n:
            return self
        return self.take(np.arange(len(self) - n, len(self)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f: getattr(self.rec, f) for f in FIELDS}
        np.savez(path, pol_idx=self.pol_idx, pol_prob=self.pol_prob, **arrays)

    @staticmethod
    def load(path: Path) -> SelfPlayRecords:
        with np.load(path) as data:
            rec = Records(*(data[f] for f in FIELDS))
            return SelfPlayRecords(rec, data["pol_idx"], data["pol_prob"])


def human_to_policy(rec: Records) -> SelfPlayRecords:
    """View human positions as one-hot policy targets (the move the human played)."""
    n = len(rec)
    idx = np.zeros((n, POLICY_WIDTH), np.int16)
    prob = np.zeros((n, POLICY_WIDTH), np.float16)
    idx[:, 0] = rec.move
    prob[:, 0] = 1.0
    return SelfPlayRecords(rec, idx, prob)


def dense_policy(
    pol_idx: npt.NDArray[np.int16], pol_prob: npt.NDArray[np.float16], device: torch.device
) -> Tensor:
    """Sparse (idx, prob) rows -> dense (N, 4672) float target distribution."""
    idx = torch.from_numpy(pol_idx.astype(np.int64)).to(device)
    prob = torch.from_numpy(pol_prob.astype(np.float32)).to(device)
    target = torch.zeros(len(idx), POLICY_SIZE, device=device)
    target.scatter_add_(1, idx, prob)  # padding has prob 0, so it adds nothing
    return target
