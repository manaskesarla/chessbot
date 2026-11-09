"""Two-headed residual CNN: policy (4672 move logits) and value (-1..1)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from chessbot.config import ModelConfig
from chessbot.encoding.moves import NUM_MOVE_TYPES, POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class ChessNet(nn.Module):
    """Input (B, input_planes, 8, 8) -> (policy_logits (B, 4672), value (B,)).

    Policy logits are indexed exactly like ``chessbot.encoding.move_to_index``:
    ``from_square * 73 + move_type``. They are unmasked; apply the legal mask outside.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.policy_size != POLICY_SIZE:
            raise ValueError(f"policy_size must be {POLICY_SIZE}, got {cfg.policy_size}")

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_planes, cfg.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.filters),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(cfg.filters) for _ in range(cfg.blocks)])

        # Policy head: one 73-channel map; channel = move type, position = from-square.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(cfg.filters, cfg.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.filters),
            nn.ReLU(),
            nn.Conv2d(cfg.filters, NUM_MOVE_TYPES, 1),
        )

        # Value head: squeeze to one plane, then a small MLP ending in tanh.
        self.value_conv = nn.Sequential(
            nn.Conv2d(cfg.filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        body = self.tower(self.stem(x))
        policy = self.policy_conv(body)  # (B, 73, 8, 8) -> (B, rank, file, 73)
        policy = policy.permute(0, 2, 3, 1).reshape(x.shape[0], POLICY_SIZE)
        value = self.value_fc(self.value_conv(body).flatten(1)).squeeze(1)
        return policy, value


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log-probabilities over legal moves; illegal moves get -inf."""
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: ChessNet, path: Path, extra: dict[str, Any] | None = None) -> None:
    """Save weights together with the config, so the model can be rebuilt without guesswork."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> ChessNet:
    data = torch.load(path, map_location=device, weights_only=True)
    model = ChessNet(ModelConfig(**data["config"]))
    model.load_state_dict(data["state_dict"])
    return model.to(device)
