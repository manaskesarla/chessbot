"""Inference throughput (positions/second) at several batch sizes.

Usage: uv run python scripts/benchmark_model.py [blocks filters]
"""

from __future__ import annotations

import sys
import time

import torch

from chessbot.config import ModelConfig, load_config
from chessbot.model import ChessNet, count_parameters
from chessbot.smoke import pick_device


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    cfg = load_config(ModelConfig, "model")
    if len(sys.argv) == 3:
        cfg = ModelConfig(blocks=int(sys.argv[1]), filters=int(sys.argv[2]))
    device = pick_device()
    net = ChessNet(cfg).to(device).eval()
    print(
        f"blocks={cfg.blocks} filters={cfg.filters} "
        f"params={count_parameters(net):,} device={device}"
    )
    print(f"{'batch':>6} {'ms/batch':>10} {'positions/s':>13}")
    for batch_size in (1, 16, 64, 256):
        x = torch.randn(batch_size, cfg.input_planes, 8, 8, device=device)
        with torch.no_grad():
            for _ in range(5):  # warm-up
                net(x)
            sync(device)
            runs = 30
            start = time.perf_counter()
            for _ in range(runs):
                net(x)
            sync(device)
        elapsed = (time.perf_counter() - start) / runs
        print(f"{batch_size:>6} {elapsed * 1000:>10.2f} {batch_size / elapsed:>13,.0f}")


if __name__ == "__main__":
    main()
