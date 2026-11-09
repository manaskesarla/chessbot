"""Generate self-play games, optionally across several worker processes.

Each worker loads the checkpoint itself (own GPU context, own position cache) and appends one
line per finished game to ``log_path`` so progress can be followed live with ``tail -f``.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessbot.config import MCTSConfig
from chessbot.mcts import MCTS, NetEvaluator
from chessbot.model import load_checkpoint
from chessbot.selfplay.buffer import SelfPlayRecords
from chessbot.selfplay.game import play_selfplay_game
from chessbot.smoke import pick_device


@dataclass(frozen=True)
class GenerationStats:
    games: int
    positions: int
    white_wins: int
    black_wins: int
    draws: int
    mean_plies: float
    seconds: float


@dataclass(frozen=True)
class _Job:
    checkpoint: Path
    games: int
    simulations: int
    mcts_cfg: MCTSConfig
    max_plies: int
    seed: int
    first_game_id: int
    worker: int
    iteration: int
    log_path: Path | None


def _run_job(job: _Job) -> tuple[SelfPlayRecords, list[float], list[int]]:
    device = pick_device()
    evaluator = NetEvaluator(load_checkpoint(job.checkpoint, device), device)
    rng = np.random.default_rng(job.seed)
    mcts = MCTS(evaluator, job.mcts_cfg, rng)
    parts: list[SelfPlayRecords] = []
    scores: list[float] = []
    plies: list[int] = []
    for g in range(job.games):
        start = time.time()
        rec, white_score, n_plies = play_selfplay_game(
            mcts, job.simulations, job.mcts_cfg, job.max_plies, rng, job.first_game_id + g
        )
        parts.append(rec)
        scores.append(white_score)
        plies.append(n_plies)
        if job.log_path is not None:
            outcome = {1.0: "1-0", 0.0: "0-1"}.get(white_score, "1/2")
            with job.log_path.open("a") as f:
                f.write(
                    f"iter {job.iteration:>3}  worker {job.worker}  game {g + 1:>3}/{job.games}  "
                    f"{n_plies:>3} plies  {outcome:>4}  {time.time() - start:5.1f}s\n"
                )
    return SelfPlayRecords.concat(parts), scores, plies


def generate_games(
    checkpoint: Path,
    games: int,
    simulations: int,
    mcts_cfg: MCTSConfig,
    max_plies: int,
    workers: int,
    seed: int,
    iteration: int = 0,
    first_game_id: int = 0,
    log_path: Path | None = None,
) -> tuple[SelfPlayRecords, GenerationStats]:
    start = time.time()
    n_jobs = max(1, workers)
    shares = [games // n_jobs + (1 if i < games % n_jobs else 0) for i in range(n_jobs)]
    jobs: list[_Job] = []
    offset = first_game_id
    for i, share in enumerate(shares):
        if share:
            jobs.append(
                _Job(checkpoint, share, simulations, mcts_cfg, max_plies, seed * 1000 + i,
                     offset, i, iteration, log_path)
            )
            offset += share

    if workers <= 1:
        outputs = [_run_job(job) for job in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")  # fresh process per worker: safe with MPS
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            outputs = list(pool.map(_run_job, jobs))

    records = SelfPlayRecords.concat([o[0] for o in outputs])
    scores = [s for o in outputs for s in o[1]]
    plies = [p for o in outputs for p in o[2]]
    stats = GenerationStats(
        games=len(scores),
        positions=len(records),
        white_wins=sum(s == 1.0 for s in scores),
        black_wins=sum(s == 0.0 for s in scores),
        draws=sum(s == 0.5 for s in scores),
        mean_plies=float(np.mean(plies)) if plies else 0.0,
        seconds=time.time() - start,
    )
    return records, stats
