"""Command-line entry points (``uv run train|play|evaluate``)."""

from __future__ import annotations

from chessbot.config import MCTSConfig, ModelConfig, SelfPlayConfig, TrainConfig, load_config


def train() -> None:
    """Supervised pretraining using configs/train.yaml and configs/model.yaml."""
    from chessbot.training import train as run_training

    best = run_training(load_config(TrainConfig, "train"), load_config(ModelConfig, "model"))
    print(f"best checkpoint: {best}")


def selfplay() -> None:
    """Self-play RL loop using configs/selfplay.yaml and configs/mcts.yaml."""
    from chessbot.selfplay import run

    best = run(load_config(SelfPlayConfig, "selfplay"), load_config(MCTSConfig, "mcts"))
    print(f"best checkpoint: {best}")


def play() -> None:
    raise SystemExit("play: not implemented yet (Step 6)")


def evaluate() -> None:
    """Evaluate a player against limited-strength Stockfish and record the rating.

    uv run evaluate mcts:checkpoints/selfplay/best.pt:100 --name selfplay-iter3
    uv run evaluate --plot-only      # redraw results/elo_by_stage.png from summary.csv
    """
    import argparse
    from pathlib import Path

    from chessbot.config import REPO_ROOT, EvalConfig
    from chessbot.eval.harness import evaluate_stage, plot_summary
    from chessbot.eval.players import make_bot

    cfg = load_config(EvalConfig, "eval")
    parser = argparse.ArgumentParser(prog="evaluate", description=evaluate.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("player", nargs="?", help="random | greedy | alphabeta:N | net:CKPT | "
                        "mcts:CKPT:SIMS")
    parser.add_argument("--name", help="stage label used in results and the chart")
    parser.add_argument("--games", type=int, default=cfg.games_per_level, help="per level")
    parser.add_argument("--levels", type=int, nargs="+", default=list(cfg.stockfish_elos))
    parser.add_argument("--movetime", type=float, default=cfg.stockfish_movetime)
    parser.add_argument("--results-dir", help="override results folder (e.g. for experiments)")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or cfg.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    if not args.plot_only:
        if not args.player:
            parser.error("a player spec is required unless --plot-only")
        name = args.name or args.player.replace(":", "_").replace("/", "_")
        result = evaluate_stage(name, make_bot(args.player), tuple(args.levels), args.games,
                                args.movetime, results_dir, cfg.stockfish_path)
        fit = result.fit
        print(f"\n{name}: {fit.rating:.0f} Elo on the Stockfish scale "
              f"(95% interval {fit.low:.0f} to {fit.high:.0f}, {result.games} games)")
    print(f"chart: {plot_summary(results_dir)}")
