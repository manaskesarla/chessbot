"""Supervised pretraining: predict the human move (policy) and the game result (value)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from chessbot.config import REPO_ROOT, ModelConfig, TrainConfig
from chessbot.data import Records, load_dataset, records_to_planes
from chessbot.model import ChessNet, count_parameters, save_checkpoint
from chessbot.smoke import pick_device

VAL_MODULUS = 50  # games with id % 50 == 0 (2%) are held out; split by game, never by position


def split_by_game(rec: Records) -> tuple[Records, Records]:
    is_val = rec.game % VAL_MODULUS == 0
    return rec.take(np.nonzero(~is_val)[0]), rec.take(np.nonzero(is_val)[0])


def to_tensors(rec: Records, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    planes = torch.from_numpy(records_to_planes(rec)).to(device)
    move = torch.from_numpy(rec.move.astype(np.int64)).to(device)
    result = torch.from_numpy(rec.result.astype(np.float32)).to(device)
    return planes, move, result


def iter_batches(
    rec: Records, batch_size: int, rng: np.random.Generator
) -> Iterator[Records]:
    """Endless shuffled batches (reshuffles every pass)."""
    while True:
        order = rng.permutation(len(rec))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield rec.take(order[start : start + batch_size])


@torch.no_grad()
def evaluate(
    net: ChessNet, val: Records, device: torch.device, max_positions: int = 40_000
) -> dict[str, float]:
    """Held-out metrics: move top-1/top-5 accuracy, policy loss, value MSE."""
    net.eval()
    n = min(len(val), max_positions)
    index = np.linspace(0, len(val) - 1, n).astype(np.int64)
    totals = {"top1": 0.0, "top5": 0.0, "policy_loss": 0.0, "value_mse": 0.0}
    for start in range(0, n, 1024):
        planes, move, result = to_tensors(val.take(index[start : start + 1024]), device)
        logits, value = net(planes)
        top5 = logits.topk(5, dim=1).indices
        totals["top1"] += (top5[:, 0] == move).sum().item()
        totals["top5"] += (top5 == move[:, None]).any(dim=1).sum().item()
        totals["policy_loss"] += F.cross_entropy(logits, move, reduction="sum").item()
        totals["value_mse"] += F.mse_loss(value, result, reduction="sum").item()
    net.train()
    return {k: v / n for k, v in totals.items()}


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up then cosine decay to 5% of the peak."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(
    cfg: TrainConfig, model_cfg: ModelConfig, device: torch.device | None = None
) -> Path:
    device = device or pick_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    data_dir = Path(cfg.data_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = split_by_game(load_dataset(data_dir))
    steps_per_epoch = len(train_set) // cfg.batch_size
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    print(
        f"train positions: {len(train_set):,}  val positions: {len(val_set):,}  "
        f"steps: {total_steps:,}  device: {device}"
    )

    net = ChessNet(model_cfg).to(device)
    print(f"model: {model_cfg.blocks} blocks x {model_cfg.filters} filters, "
          f"{count_parameters(net):,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, cfg.warmup_steps, total_steps)
    )

    state_path = ckpt_dir / "train_state.pt"
    step = 0
    best_top1 = -1.0
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=True)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, best_top1 = int(state["step"]), float(state["best_top1"])
        print(f"resumed from step {step}")

    log_path = ckpt_dir / "metrics.jsonl"
    batches = iter_batches(train_set, cfg.batch_size, rng)
    net.train()
    start_time, running = time.time(), {"policy": 0.0, "value": 0.0, "n": 0}

    while step < total_steps:
        planes, move, result = to_tensors(next(batches), device)
        logits, value = net(planes)
        policy_loss = F.cross_entropy(logits, move)
        value_loss = F.mse_loss(value, result)
        loss = policy_loss + cfg.value_loss_weight * value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1

        running["policy"] += policy_loss.item()
        running["value"] += value_loss.item()
        running["n"] += 1

        if step % 100 == 0:
            n = running["n"]
            rate = 100 * cfg.batch_size / (time.time() - start_time)
            print(
                f"step {step:>7}/{total_steps}  policy {running['policy'] / n:.3f}  "
                f"value {running['value'] / n:.3f}  {rate:,.0f} pos/s"
            )
            running, start_time = {"policy": 0.0, "value": 0.0, "n": 0}, time.time()

        if step % cfg.eval_every == 0 or step == total_steps:
            metrics = evaluate(net, val_set, device)
            print(f"  val @ {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")
            save_checkpoint(net, ckpt_dir / "last.pt", {"step": step, **metrics})
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                save_checkpoint(net, ckpt_dir / "best.pt", {"step": step, **metrics})
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "step": step,
                    "best_top1": best_top1,
                },
                state_path,
            )
    return ckpt_dir / "best.pt"
