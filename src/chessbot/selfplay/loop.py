"""The reinforcement-learning loop: generate -> store -> train -> gate -> repeat."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chessbot.config import REPO_ROOT, MCTSConfig, SelfPlayConfig
from chessbot.data import Records, load_dataset, records_to_planes
from chessbot.eval import play_match
from chessbot.mcts import MCTSBot, NetEvaluator
from chessbot.model import ChessNet, load_checkpoint, save_checkpoint
from chessbot.selfplay.buffer import SelfPlayRecords, dense_policy, human_to_policy
from chessbot.selfplay.worker import generate_games
from chessbot.smoke import pick_device
from chessbot.training.supervised import split_by_game


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def train_candidate(
    net: ChessNet,
    selfplay: SelfPlayRecords,
    human: SelfPlayRecords,
    cfg: SelfPlayConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Fit ``net`` to MCTS visit distributions + game results, mixed with human games."""
    n_human = int(cfg.batch_size * cfg.human_fraction) if len(human) else 0
    n_self = cfg.batch_size - n_human
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    net.train()
    totals = {"policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(cfg.train_steps):
        parts = [selfplay.take(rng.integers(0, len(selfplay), n_self))]
        if n_human:
            parts.append(human.take(rng.integers(0, len(human), n_human)))
        batch = SelfPlayRecords.concat(parts)
        planes = torch.from_numpy(records_to_planes(batch.rec)).to(device)
        target = dense_policy(batch.pol_idx, batch.pol_prob, device)
        result = torch.from_numpy(batch.rec.result.astype(np.float32)).to(device)

        logits, value = net(planes)
        policy_loss = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, result)
        loss = policy_loss + cfg.value_loss_weight * value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        totals["policy_loss"] += policy_loss.item()
        totals["value_loss"] += value_loss.item()
    return {k: v / cfg.train_steps for k, v in totals.items()}


def gate(
    candidate: ChessNet,
    best: ChessNet,
    cfg: SelfPlayConfig,
    mcts_cfg: MCTSConfig,
    device: torch.device,
) -> tuple[int, int, int]:
    """Candidate vs current best, equal search budget, colours alternated."""
    a = MCTSBot(NetEvaluator(candidate, device), cfg.gate_simulations, mcts_cfg, "candidate")
    b = MCTSBot(NetEvaluator(best, device), cfg.gate_simulations, mcts_cfg, "best")
    r = play_match(a, b, cfg.gate_games)
    return r.wins, r.draws, r.losses


def run(cfg: SelfPlayConfig, mcts_cfg: MCTSConfig, device: torch.device | None = None) -> Path:
    device = device or pick_device()
    out = _resolve(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    games_log = reports / "selfplay_games.log"
    iter_log = out / "iterations.jsonl"
    state_path = out / "state.json"
    buffer_path = out / "buffer.npz"
    best_path = out / "best.pt"

    if not best_path.exists():
        shutil.copy(_resolve(cfg.init_checkpoint), best_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"iteration": 0}
    buffer = SelfPlayRecords.load(buffer_path) if buffer_path.exists() else SelfPlayRecords.empty()

    human = SelfPlayRecords.empty()
    if cfg.human_fraction > 0:
        train_set, _ = split_by_game(load_dataset(_resolve(cfg.human_data_dir)))
        pick = np.random.default_rng(cfg.seed).permutation(len(train_set))[: cfg.human_subset]
        human = human_to_policy(Records.concat([train_set.take(pick)]))
        del train_set
    print(f"human positions for mixing: {len(human):,}; resuming at iteration {state['iteration']}")

    for iteration in range(state["iteration"], cfg.iterations):
        t0 = time.time()
        print(f"\n=== iteration {iteration + 1}/{cfg.iterations}: self-play with best.pt ===")
        new, gen = generate_games(
            best_path, cfg.games_per_iteration, cfg.simulations, mcts_cfg, cfg.max_plies,
            cfg.workers, cfg.seed + iteration, iteration + 1,
            first_game_id=iteration * cfg.games_per_iteration, log_path=games_log,
        )
        print(
            f"  {gen.games} games, {gen.positions:,} positions, mean {gen.mean_plies:.0f} plies, "
            f"W/D/B {gen.white_wins}/{gen.draws}/{gen.black_wins}, {gen.seconds:.0f}s"
        )
        buffer = SelfPlayRecords.concat([buffer, new]).last(cfg.buffer_positions)
        buffer.save(buffer_path)

        print(f"=== training candidate on {len(buffer):,} buffered positions ===")
        rng = np.random.default_rng(cfg.seed * 7919 + iteration)
        candidate = load_checkpoint(best_path, device)
        losses = train_candidate(candidate, buffer, human, cfg, device, rng)
        cand_path = out / f"iter_{iteration + 1:04d}.pt"
        save_checkpoint(candidate, cand_path, {"iteration": iteration + 1, **losses})
        print(f"  policy loss {losses['policy_loss']:.3f}  value loss {losses['value_loss']:.3f}")

        print(f"=== gating: candidate vs best, {cfg.gate_games} games ===")
        w, d, lo = gate(candidate.eval(), load_checkpoint(best_path, device).eval(), cfg,
                        mcts_cfg, device)
        score = (w + 0.5 * d) / max(1, w + d + lo)
        promoted = score >= cfg.gate_threshold
        if promoted:
            shutil.copy(cand_path, best_path)
        print(f"  +{w} ={d} -{lo}  score {score:.1%}  -> {'PROMOTED' if promoted else 'rejected'}")

        record = {
            "iteration": iteration + 1,
            "games": gen.games,
            "positions": gen.positions,
            "mean_plies": gen.mean_plies,
            "white_wins": gen.white_wins,
            "black_wins": gen.black_wins,
            "draws": gen.draws,
            "buffer": len(buffer),
            **losses,
            "gate_wins": w,
            "gate_draws": d,
            "gate_losses": lo,
            "gate_score": score,
            "promoted": promoted,
            "seconds": time.time() - t0,
        }
        with iter_log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        state["iteration"] = iteration + 1
        state_path.write_text(json.dumps(state))
    return best_path
