from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import torch

from chessbot.config import MCTSConfig, ModelConfig, SelfPlayConfig
from chessbot.mcts import MCTS, HeuristicEvaluator, NetEvaluator
from chessbot.model import ChessNet, save_checkpoint
from chessbot.selfplay import (
    POLICY_WIDTH,
    SelfPlayRecords,
    dense_policy,
    generate_games,
    human_to_policy,
    play_selfplay_game,
    train_candidate,
)
from tests.test_training import synthetic_records

CPU = torch.device("cpu")
MCFG = replace(MCTSConfig(), batch_size=4, temperature_moves=6)


def tiny_net() -> ChessNet:
    torch.manual_seed(0)
    return ChessNet(ModelConfig(blocks=1, filters=8))


def one_game(max_plies: int = 24) -> tuple[SelfPlayRecords, float, int]:
    rng = np.random.default_rng(0)
    mcts = MCTS(NetEvaluator(tiny_net(), CPU), MCFG, rng)
    return play_selfplay_game(mcts, 8, MCFG, max_plies, rng)


def test_game_records_are_consistent() -> None:
    rec, white_score, plies = one_game()
    assert plies == len(rec) <= 24
    assert white_score in (0.0, 0.5, 1.0)
    expected = {1.0: 1, 0.5: 0, 0.0: -1}[white_score]
    assert abs(int(rec.rec.result[0])) == abs(expected)  # result labels match the outcome
    assert np.allclose(rec.pol_prob.astype(np.float32).sum(axis=1), 1.0, atol=1e-2)
    assert (rec.pol_idx >= 0).all() and (rec.pol_idx < 4672).all()
    # first position is the start position: White to move
    assert rec.rec.flags[0] & 1 == 1 and rec.rec.flags[1] & 1 == 0


def test_result_sign_follows_side_to_move() -> None:
    rng = np.random.default_rng(0)
    cfg = replace(MCFG, temperature_moves=0)  # always play the most-visited move
    mcts = MCTS(HeuristicEvaluator(), cfg, rng)

    # White to move, mate in one (Ra8#): White wins, so the single position scores +1.
    white_mates = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    rec, white_score, plies = play_selfplay_game(mcts, 60, cfg, 50, rng, start=white_mates)
    assert (white_score, plies, rec.rec.result.tolist()) == (1.0, 1, [1])

    # Black to move, mate in one (...Ra1#): Black wins, so that position also scores +1 for
    # its side to move, and White's score is 0.
    black_mates = chess.Board("r3k3/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    rec, white_score, plies = play_selfplay_game(mcts, 60, cfg, 50, rng, start=black_mates)
    assert (white_score, plies, rec.rec.result.tolist()) == (0.0, 1, [1])


def test_dense_policy_scatter() -> None:
    idx = np.zeros((2, POLICY_WIDTH), np.int16)
    prob = np.zeros((2, POLICY_WIDTH), np.float16)
    idx[0, :2], prob[0, :2] = [5, 9], [0.75, 0.25]
    idx[1, 0], prob[1, 0] = 100, 1.0
    dense = dense_policy(idx, prob, CPU)
    assert dense.shape == (2, 4672)
    assert dense[0, 5].item() == 0.75 and dense[0, 9].item() == 0.25
    assert dense[1, 100].item() == 1.0 and dense.sum().item() == 2.0


def test_buffer_window_and_roundtrip(tmp_path: Path) -> None:
    rec, _, _ = one_game()
    buf = SelfPlayRecords.concat([rec, rec, rec])
    assert len(buf) == 72 and len(buf.last(30)) == 30
    buf.save(tmp_path / "b.npz")
    loaded = SelfPlayRecords.load(tmp_path / "b.npz")
    assert len(loaded) == 72
    assert np.array_equal(loaded.pol_idx, buf.pol_idx)
    assert np.array_equal(loaded.rec.squares, buf.rec.squares)


def test_train_candidate_changes_weights_and_reports_losses() -> None:
    selfplay, _, _ = one_game()
    human = human_to_policy(synthetic_records().take(np.arange(200)))
    net = tiny_net()
    first = next(net.parameters())
    before = first.detach().clone()
    cfg = replace(SelfPlayConfig(), train_steps=3, batch_size=16, human_fraction=0.5)
    losses = train_candidate(net, selfplay, human, cfg, CPU, np.random.default_rng(0))
    assert losses["policy_loss"] > 0 and losses["value_loss"] >= 0
    assert not torch.equal(before, first.detach())


def test_generate_games_inline_and_logging(tmp_path: Path) -> None:
    ckpt = tmp_path / "net.pt"
    save_checkpoint(tiny_net(), ckpt)
    log = tmp_path / "games.log"
    records, stats = generate_games(
        ckpt, games=2, simulations=4, mcts_cfg=MCFG, max_plies=10, workers=0, seed=0,
        log_path=log,
    )
    assert stats.games == 2 and stats.positions == len(records) == 20
    assert stats.draws == 2
    assert len(log.read_text().splitlines()) == 2
    assert set(records.rec.game.tolist()) == {0, 1}


def test_human_policy_is_one_hot() -> None:
    rec = human_to_policy(synthetic_records().take(np.arange(10)))
    assert (rec.pol_prob[:, 0] == 1).all() and rec.pol_prob[:, 1:].sum() == 0
