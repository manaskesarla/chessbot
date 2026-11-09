"""Convert match results into Elo estimates with confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass


def elo_from_score(score: float) -> float:
    """Elo difference implied by an expected score in (0, 1)."""
    return float(-400.0 * math.log10(1.0 / score - 1.0))


def expected_score(elo_diff: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)))


@dataclass(frozen=True)
class EloEstimate:
    diff: float
    low: float
    high: float


def elo_difference(wins: int, draws: int, losses: int, z: float = 1.96) -> EloEstimate:
    """Elo difference (player minus opponent) with a normal-approximation 95% interval.

    Scores of exactly 0 or 1 are clamped to half a game from the extreme, so results stay
    finite; such intervals are very wide and mean "need more games".
    """
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    clamp = 0.5 / n
    mean = (wins + 0.5 * draws) / n
    variance = (wins * (1 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * mean**2) / n
    se = math.sqrt(variance / n)

    def to_elo(score: float) -> float:
        return elo_from_score(min(max(score, clamp), 1 - clamp))

    return EloEstimate(to_elo(mean), to_elo(mean - z * se), to_elo(mean + z * se))
