"""Typed YAML config loading.

Hyperparameters live in ``configs/*.yaml`` and are loaded into dataclasses, so an
experiment is changed by editing a file rather than code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    blocks: int = 6
    filters: int = 64
    input_planes: int = 19
    policy_size: int = 4672


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2
    value_loss_weight: float = 1.0
    seed: int = 0
    data_dir: str = "data/processed/lichess-2016-06"
    checkpoint_dir: str = "checkpoints/supervised"
    eval_every: int = 2000
    max_steps: int = 0  # 0 = run `epochs` full passes
    warmup_steps: int = 200


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature_moves: int = 30
    batch_size: int = 16  # leaves evaluated per network call
    fpu_reduction: float = 0.2  # value assumed for unvisited moves = parent value - this


@dataclass(frozen=True)
class SelfPlayConfig:
    iterations: int = 8
    games_per_iteration: int = 64
    simulations: int = 64  # MCTS simulations per self-play move
    workers: int = 4  # parallel game-generating processes (0 = in this process)
    max_plies: int = 200  # longer games are adjudicated as draws
    buffer_positions: int = 100_000  # rolling replay window
    human_fraction: float = 0.5  # share of each training batch drawn from human games
    human_subset: int = 1_000_000  # human positions kept in memory for mixing
    train_steps: int = 300
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    gate_games: int = 32
    gate_simulations: int = 64
    gate_threshold: float = 0.55
    seed: int = 0
    init_checkpoint: str = "checkpoints/supervised/best.pt"
    human_data_dir: str = "data/processed/lichess-2016-06"
    output_dir: str = "checkpoints/selfplay"


@dataclass(frozen=True)
class EvalConfig:
    stockfish_path: str = "stockfish"
    stockfish_elos: tuple[int, ...] = (1320, 1500, 1700, 1900)
    games_per_level: int = 100
    simulations: int = 100
    stockfish_movetime: float = 0.05  # seconds per Stockfish move
    results_dir: str = "results"


def load_config(cls: type[T], name: str, config_dir: Path = CONFIG_DIR) -> T:
    """Load ``configs/<name>.yaml`` into dataclass ``cls``.

    Unknown keys raise ``ValueError`` so typos don't silently fall back to defaults.
    """
    path = config_dir / f"{name}.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path.name}: {sorted(unknown)}")
    if "stockfish_elos" in raw:
        raw["stockfish_elos"] = tuple(raw["stockfish_elos"])
    return cls(**raw)
