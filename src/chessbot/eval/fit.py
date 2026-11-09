"""Fit one rating to results against several Stockfish levels, with a bootstrap interval.

Model: the expected score against an opponent rated ``e`` is ``1 / (1 + 10^((e - R) / 400))``.
We choose the rating ``R`` that makes total expected score equal total actual score across
all levels (the maximum-likelihood solution), then resample games to get a 95% interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LevelResult:
    opponent_elo: int
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games


@dataclass(frozen=True)
class RatingFit:
    rating: float
    low: float
    high: float


def _solve(
    elos: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    games: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Rating(s) R with sum_l n_l * (s_l - expected_l(R)) = 0. ``scores`` is (B, L)."""
    lo = np.full(scores.shape[0], float(np.min(elos)) - 2000.0)
    hi = np.full(scores.shape[0], float(np.max(elos)) + 2000.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = 1.0 / (1.0 + 10.0 ** ((elos[None, :] - mid[:, None]) / 400.0))
        surplus = (games[None, :] * (scores - expected)).sum(axis=1)
        lo = np.where(surplus > 0, mid, lo)  # scored above expectation: true rating is higher
        hi = np.where(surplus > 0, hi, mid)
    return np.asarray((lo + hi) / 2, dtype=np.float64)


def fit_rating(levels: list[LevelResult], n_boot: int = 2000, seed: int = 0) -> RatingFit:
    levels = [lv for lv in levels if lv.games > 0]
    if not levels:
        raise ValueError("no games to fit")
    elos = np.array([lv.opponent_elo for lv in levels], dtype=np.float64)
    games = np.array([lv.games for lv in levels], dtype=np.float64)
    clamp = 0.5 / games  # keep 0% / 100% scores finite

    def clamped(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.clip(scores, clamp, 1.0 - clamp)

    observed = clamped(np.array([[lv.score for lv in levels]]))
    rating = float(_solve(elos, observed, games)[0])

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(levels)))
    for j, lv in enumerate(levels):
        probs = np.array([lv.wins, lv.draws, lv.losses], dtype=np.float64) / lv.games
        counts = rng.multinomial(lv.games, probs, size=n_boot)
        boot[:, j] = (counts[:, 0] + 0.5 * counts[:, 1]) / lv.games
    ratings = _solve(elos, clamped(boot), games)
    low, high = np.percentile(ratings, [2.5, 97.5])
    return RatingFit(rating, float(low), float(high))
