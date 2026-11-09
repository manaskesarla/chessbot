from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import torch

from chessbot.config import ModelConfig, TrainConfig
from chessbot.data import Records, load_dataset, save_records
from chessbot.model import ChessNet
from chessbot.model.player import NetBot
from chessbot.training import evaluate, split_by_game, train
from chessbot.training.supervised import lr_factor
from tests.test_data import records_from_boards
from tests.test_encoding import POSITIONS


def synthetic_records(n_games: int = 60) -> Records:
    boards = POSITIONS[: n_games * 20]
    rec = records_from_boards(boards)
    rec.game[:] = np.repeat(np.arange(n_games), 20).astype(np.int32)
    rec.move[:] = np.arange(len(boards)) % 4672
    rec.result[:] = np.arange(len(boards)) % 3 - 1
    return rec


def test_split_is_by_game() -> None:
    train_set, val_set = split_by_game(synthetic_records())
    assert set(train_set.game.tolist()).isdisjoint(val_set.game.tolist())
    assert len(val_set) > 0 and len(train_set) > len(val_set)


def test_lr_schedule_shape() -> None:
    assert lr_factor(0, 10, 100) < lr_factor(9, 10, 100) <= 1.0
    assert lr_factor(10, 10, 100) > lr_factor(99, 10, 100) >= 0.05


def test_training_runs_and_checkpoints(tmp_path: Path) -> None:
    data = tmp_path / "data"
    save_records(synthetic_records(), data / "shard_0000")
    assert len(load_dataset(data)) == 1200
    cfg = replace(
        TrainConfig(),
        data_dir=str(data),
        checkpoint_dir=str(tmp_path / "ckpt"),
        batch_size=32,
        max_steps=6,
        eval_every=3,
        warmup_steps=2,
    )
    best = train(cfg, ModelConfig(blocks=1, filters=8), device=torch.device("cpu"))
    assert best.exists() and (tmp_path / "ckpt" / "last.pt").exists()
    assert (tmp_path / "ckpt" / "metrics.jsonl").read_text().count("\n") == 2

    # Resuming a finished run does nothing but must not crash.
    train(cfg, ModelConfig(blocks=1, filters=8), device=torch.device("cpu"))


def test_evaluate_metrics_in_range() -> None:
    _, val = split_by_game(synthetic_records())
    metrics = evaluate(ChessNet(ModelConfig(blocks=1, filters=8)), val, torch.device("cpu"))
    assert 0.0 <= metrics["top1"] <= metrics["top5"] <= 1.0
    assert metrics["policy_loss"] > 0


def test_net_bot_plays_legal_moves() -> None:
    bot = NetBot(ChessNet(ModelConfig(blocks=1, filters=8)), torch.device("cpu"))
    board = chess.Board()
    for _ in range(10):
        move = bot.choose_move(board)
        assert move in board.legal_moves
        board.push(move)
