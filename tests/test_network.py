from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chessbot.config import ModelConfig
from chessbot.encoding import NUM_PLANES, POLICY_SIZE, encode_board, legal_mask, move_to_index
from chessbot.model import (
    ChessNet,
    count_parameters,
    load_checkpoint,
    masked_log_softmax,
    save_checkpoint,
)
from chessbot.smoke import pick_device

SMALL = ModelConfig(blocks=2, filters=16)


def batch(boards: list[chess.Board]) -> torch.Tensor:
    return torch.from_numpy(np.stack([encode_board(b) for b in boards]))


def test_shapes_and_value_range() -> None:
    net = ChessNet(SMALL).eval()
    policy, value = net(batch([chess.Board(), chess.Board()]))
    assert policy.shape == (2, POLICY_SIZE)
    assert value.shape == (2,)
    assert ((value >= -1) & (value <= 1)).all()


def test_input_planes_match_encoder() -> None:
    assert ModelConfig().input_planes == NUM_PLANES


def test_wrong_policy_size_rejected() -> None:
    with pytest.raises(ValueError):
        ChessNet(ModelConfig(policy_size=100))


def test_masked_softmax_only_legal_moves() -> None:
    board = chess.Board()
    net = ChessNet(SMALL).eval()
    policy, _ = net(batch([board]))
    mask = torch.from_numpy(legal_mask(board)).unsqueeze(0)
    probs = masked_log_softmax(policy, mask).exp()
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (probs[~mask] == 0).all()
    assert (probs[mask] > 0).all()


def test_policy_layout_matches_move_index() -> None:
    # Perturbing the policy conv's output channel for move type 1 at square e2 must change
    # exactly logit index e2*73+1 (the e2e4 slot) and nothing else.
    net = ChessNet(SMALL).eval()
    x = batch([chess.Board()])
    with torch.no_grad():
        base, _ = net(x)
        last = net.policy_conv[-1]
        assert isinstance(last, torch.nn.Conv2d)
        assert last.bias is not None
        last.bias[1] += 5.0  # bump the bias of move type 1 only
        bumped, _ = net(x)
    changed = (bumped - base).abs().gt(1e-6).nonzero()[:, 1]
    assert len(changed) == 64
    assert all(int(i) % 73 == 1 for i in changed)
    e2e4 = move_to_index(chess.Move.from_uci("e2e4"), chess.Board())
    assert e2e4 in changed.tolist()


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    net = ChessNet(SMALL).eval()
    path = tmp_path / "net.pt"
    save_checkpoint(net, path)
    loaded = load_checkpoint(path).eval()
    x = batch([chess.Board()])
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = loaded(x)
    assert torch.allclose(p1, p2) and torch.allclose(v1, v2)
    assert loaded.config == SMALL


def test_default_model_size_is_modest() -> None:
    assert count_parameters(ChessNet(ModelConfig())) < 1_000_000


def test_device_matches_cpu() -> None:
    device = pick_device()
    net = ChessNet(SMALL).eval()
    x = batch([chess.Board()])
    with torch.no_grad():
        cpu_policy, cpu_value = net(x)
        policy, value = net.to(device)(x.to(device))
    assert torch.allclose(cpu_policy, policy.cpu(), atol=1e-4)
    assert torch.allclose(cpu_value, value.cpu(), atol=1e-4)
