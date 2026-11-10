# chessbot

An AlphaZero-style chess engine: a ResNet policy/value network, PUCT Monte Carlo Tree Search, supervised pretraining on Lichess games, then self-play reinforcement learning. Target: ~1750 Lichess rapid, measured on a live bot account. Runs on an Apple Silicon Mac (PyTorch MPS).

The full step-by-step plan is in [docs/plan/](docs/plan/README.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Stockfish (`brew install stockfish`).

```bash
uv sync                # create .venv and install dependencies
uv run smoke           # check PyTorch device, python-chess and Stockfish
uv run pytest          # run the tests
uv run ruff check .    # lint
uv run mypy src        # type check
```

## Layout

```
src/chessbot/   encoding, baselines, model, data, mcts, selfplay, eval, config
configs/        YAML hyperparameters (model, train, mcts, selfplay, eval)
results/        Stockfish evaluation results and the Elo-by-stage chart
scripts/        standalone entry scripts
tests/          pytest suite
docs/plan/      project plan, one file per step
```

## Commands

| Command | Status |
|---|---|
| `uv run smoke` | working |
| `uv run train` | working: supervised pretraining |
| `uv run selfplay` | working: gated self-play RL loop (resumable) |
| `uv run evaluate mcts:CKPT:SIMS --name NAME` | working: rate a player against Stockfish, writes `results/` |
| `uv run python scripts/dashboard.py --watch` | live progress page at `reports/progress.html` |
| `uv run play` | planned (Step 9) |

## Progress

- [x] Steps 0-5: measurement protocol, tooling, encoding, baselines, network, supervised pretraining
- [x] Step 6: MCTS search
- [x] Step 7: self-play RL loop, run and evaluated
- [~] Step 8: evaluation harness built; full-protocol runs done for the MCTS stages only
- [ ] Steps 9-10: Lichess deployment, packaging

Ratings so far (Stockfish scale, 400 games each, 95% interval; not Lichess ratings):

| Player | Rating |
|---|---|
| Supervised network + MCTS (100 sims) | 1630 (1591 to 1667) |
| After self-play RL + MCTS (100 sims) | 1647 (1611 to 1687) |

Self-play's +17 Elo is within noise, so no real gain is claimed. Details are in [docs/progress-report.md](docs/progress-report.md) and the plan in [docs/plan/](docs/plan/README.md).
