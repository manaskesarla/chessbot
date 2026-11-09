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


@dataclass(frozen=True)
class EvalConfig:
    stockfish_path: str = "stockfish"
    stockfish_elos: tuple[int, ...] = (1320, 1500, 1700, 1900)
    games_per_level: int = 100
    simulations: int = 100


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
