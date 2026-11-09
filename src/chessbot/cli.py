"""Command-line entry points (``uv run train|play|evaluate``)."""

from __future__ import annotations

from chessbot.config import ModelConfig, TrainConfig, load_config


def train() -> None:
    """Supervised pretraining using configs/train.yaml and configs/model.yaml."""
    from chessbot.training import train as run_training

    best = run_training(load_config(TrainConfig, "train"), load_config(ModelConfig, "model"))
    print(f"best checkpoint: {best}")


def play() -> None:
    raise SystemExit("play: not implemented yet (Step 6)")


def evaluate() -> None:
    raise SystemExit("evaluate: not implemented yet (Step 8)")
