"""Duplicate self-play PPO (plan Task 36, :mod:`big2.rl.selfplay`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from big2.rl import selfplay  # noqa: E402
from big2.rl.model import NetConfigV2, QNetV2  # noqa: E402


def test_duplicate_baseline_pairs_the_same_hand_against_the_same_lineup() -> None:
    lu_self, lu_frozen = [0, 0, 0, 0], [0, 1, 0, 1]
    games = []
    for rotation in range(4):
        # Hand h scores 10 * h + rotation wherever it sits: the hand's strength is 10 * h.
        scores = [10.0 * ((s + rotation) % 4) + rotation for s in range(4)]
        games.append({"group": 0, "rotation": rotation, "lineup": lu_self, "scores": scores})
        games.append({"group": 1, "rotation": rotation, "lineup": lu_frozen, "scores": scores})
    base = selfplay.duplicate_baselines(games)
    # Self-play: hand 2 in rotation 0 (seat 2) against the mean of rotations 1-3.
    assert base[(0, 2)] == pytest.approx(20 + (1 + 2 + 3) / 3)
    # ABAB: rotations 0 and 2 keep the lineup around each hand, so they pair with each other.
    g0 = 1  # group 1, rotation 0
    assert base[(g0, 0)] == pytest.approx(0 + 2)  # hand 0 at seat 0; rotation 2 scored 2
    g1 = 3  # group 1, rotation 1: hand 1 at seat 0 (learner); rotation 3 puts it at seat 2
    assert base[(g1, 0)] == pytest.approx(10 + 3)


def test_selfplay_trains_snapshots_and_resumes(tmp_path: Path) -> None:
    cfg_net = NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    init = tmp_path / "init.pt"
    torch.save(
        {
            "arch": "qnet_v2",
            "model": QNetV2(cfg_net).state_dict(),
            "net_config": cfg_net.__dict__,
        },
        init,
    )
    cfg = selfplay.SelfPlayConfig(
        run_dir=str(tmp_path / "run"),
        init=str(init),
        deals_per_iter=3,
        copies=1,
        frozen_share=1.0,
        minibatch=256,
        max_minutes=0.0,
        until_minutes=0.02,
        snapshot_minutes=0.01,
        device="cpu",
    )
    run = selfplay.train(cfg, log=lambda r: None)
    recs = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    assert recs and all(r["decisions"] > 0 for r in recs)
    assert {"resid_duplicate", "resid_oracle", "kl", "critic"} <= set(recs[0])
    (snap, *_) = sorted(run.glob("hour_*.pt"))
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["arch"] == "qnet_v2" and ck["snapshots"]
    # The snapshot plays as a network v2 agent.
    from big2.agents import make_agent
    from big2.games.big2 import Rules

    agent = make_agent(f"pvnet:checkpoint={snap},mode=prior")
    state = Rules().new_deal(1)
    assert agent.act(state) in state.legal_actions()
    # Resuming past the limit trains no further.
    n = len(recs)
    selfplay.train(cfg, log=lambda r: None)
    assert len((run / "log.jsonl").read_text().splitlines()) == n
