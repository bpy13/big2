"""The evaluation harness (big2.eval.harness), the ``big2 match`` CLI and the
``qnet`` agent spec (big2.agents.qnet)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from big2.agents import make_agent
from big2.agents.qnet import DEFAULT_QNET, QNetAgent, load_numpy_qnet
from big2.cli.main import app
from big2.eval.harness import (
    bootstrap_ci,
    lineup_seats,
    match,
    parse_lineups,
    play_match,
    summarise_match,
)
from big2.games.big2 import Rules
from typer.testing import CliRunner

REPORT_KEYS = {
    "candidate",
    "opponent",
    "lineup",
    "deals",
    "mean",
    "ci95",
    "win_rate",
    "win_ci95",
    "seconds_per_move",
}


def synthetic(n: int, seed: int = 0) -> list[dict[str, list[float]]]:
    """``n`` deals of two agents in A B A B: A gets 1 + noise, B −1 + noise per game."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        luck = rng.normal(0, 10, size=2)  # per-deal card luck, shared by both sides
        a = [float(1 + luck[0] + rng.normal(0, 3)) for _ in range(8)]
        b = [float(-1 + luck[1] + rng.normal(0, 3)) for _ in range(8)]
        out.append(
            {
                "score|A": a,
                "score|B": b,
                "win|A": [1.0, 0.0] * 4,
                "win|B": [0.0, 0.0, 0.0, 1.0] * 2,
                "time|A": [0.5] * 8,
                "time|B": [1.0] * 8,
                "moves|A": [10] * 8,
                "moves|B": [10] * 8,
            }
        )
    return out


def test_summary_of_a_two_seat_synthetic_result() -> None:
    recs = synthetic(400)
    r = summarise_match("A", "B", "ABAB", recs)
    a = np.array([np.mean(x["score|A"]) for x in recs])
    b = np.array([np.mean(x["score|B"]) for x in recs])
    assert r.mean == pytest.approx(a.mean())
    assert r.opponent_mean == pytest.approx(b.mean())
    assert r.paired_diff == pytest.approx((a - b).mean())
    assert r.paired_ci95[0] < r.paired_diff < r.paired_ci95[1]
    assert r.ci95[0] < r.mean < r.ci95[1]
    assert r.win_rate == pytest.approx(0.5) and r.opponent_win_rate == pytest.approx(0.25)
    assert r.seconds_per_move == pytest.approx(0.05)
    assert r.opponent_seconds_per_move == pytest.approx(0.1)
    assert set(r.to_dict()) >= REPORT_KEYS
    json.dumps(r.to_dict())  # the report is plain JSON


def test_interval_shrinks_with_root_n() -> None:
    def width(n: int) -> float:
        lo, hi = summarise_match("A", "B", "ABAB", synthetic(n, seed=n)).paired_ci95
        return hi - lo

    ratio = width(400) / width(1600)  # 4× the deals: half the width
    assert 1.6 < ratio < 2.5, ratio
    # The bootstrap agrees with the normal interval 1.96 σ / √N.
    x = np.random.default_rng(5).normal(0, 1, 2000)
    lo, hi = bootstrap_ci(x)
    assert (hi - lo) / 2 == pytest.approx(1.96 * x.std(ddof=1) / np.sqrt(len(x)), rel=0.1)


def test_lineups() -> None:
    assert lineup_seats("c", "o", "1+3") == ["c", "o", "o", "o"]
    assert lineup_seats("c", "o", "ABAB") == ["c", "o", "c", "o"]
    assert parse_lineups("both") == ["1+3", "ABAB"]
    assert parse_lineups("abab") == ["ABAB"]
    with pytest.raises(ValueError):
        parse_lineups("2+2")


def test_play_match_resumes_from_its_blocks(tmp_path: Path) -> None:
    first = play_match("greedy", "random", "ABAB", 4, cache_dir=tmp_path, block=2)
    assert first.deals == 4 and first.mean > 0  # greedy beats random
    blocks = sorted(p.name for p in tmp_path.glob("block_*.json"))
    assert blocks == ["block_s0_d0-2.json", "block_s0_d2-4.json"]
    again = play_match("greedy", "random", "ABAB", 4, cache_dir=tmp_path, block=2)
    assert again.mean == first.mean and again.seconds == first.seconds  # read back
    with pytest.raises(ValueError, match="another match"):
        play_match("greedy", "random", "1+3", 4, cache_dir=tmp_path, block=2)


def test_both_lineups_are_zero_sum() -> None:
    reports = match("greedy", "random", 3, rules=Rules())
    assert [r.lineup for r in reports] == ["1+3", "ABAB"]
    one, abab = reports
    # Tiered scores sum to zero over the four seats of every game.
    assert one.mean + 3 * one.opponent_mean == pytest.approx(0, abs=1e-9)
    assert abab.mean + abab.opponent_mean == pytest.approx(0, abs=1e-9)


def test_match_cli(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    r = CliRunner().invoke(
        app, ["match", "greedy", "random", "--deals", "2", "--lineups", "ABAB", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    saved = json.loads(out.read_text())
    assert len(saved) == 1 and set(saved[0]) >= REPORT_KEYS and saved[0]["lineup"] == "ABAB"


# qnet -------------------------------------------------------------------------


@pytest.mark.skipif(not DEFAULT_QNET.exists(), reason="models/big2-qnet-duty.pknn missing")
def test_qnet_spec_plays_legal_moves() -> None:
    agent = make_agent("qnet:temperature=1.0", seed=3)
    assert isinstance(agent, QNetAgent) and agent.temperature == 1.0
    state = Rules().new_deal(7)
    while not state.is_terminal:
        a = agent.act(state)
        assert a in state.legal_actions()
        state.apply(a)


def test_numpy_qnet_matches_pytorch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from big2._core import encode
    from big2.rl.export import export_qnet_pknn, load_qnet, stored_weight
    from big2.rl.model import ACTION_LEN, SCORE_SCALE, NetConfig, QNet

    torch.manual_seed(0)
    cfg = NetConfig(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    ckpt = tmp_path / "small.pt"
    torch.save({"net_config": asdict(cfg), "model": QNet(cfg).state_dict()}, ckpt)
    net, _ = load_qnet(ckpt)
    exact = load_numpy_qnet(export_qnet_pknn(ckpt, tmp_path / "f32.pknn", precision="f32"))
    mixed = load_numpy_qnet(export_qnet_pknn(ckpt, tmp_path / "mixed.pknn"))
    # The mixed file's first layer is the int8 rounding the Rust reader sees.
    w = net.state_tower[0].weight.detach().numpy()
    np.testing.assert_allclose(mixed.state[0][0].T, stored_weight(w, 1), atol=1e-6)

    state = Rules().new_deal(11)
    for _ in range(12):
        s, a, legal = encode(state)
        st = torch.from_numpy(np.frombuffer(s, np.uint8).astype(np.float32)[None])
        ac = torch.from_numpy(np.frombuffer(a, np.uint8).reshape(-1, ACTION_LEN).astype(np.float32))
        with torch.no_grad():
            ref = net(st, ac, torch.zeros(len(legal), dtype=torch.long))[0].numpy() * SCORE_SCALE
        moves, got = exact.score_moves(state)
        assert moves == legal
        np.testing.assert_allclose(got, ref, atol=1e-3)
        state.apply(legal[0])


def test_qnet_temperature_zero_plays_the_best_score(tmp_path: Path) -> None:
    if not DEFAULT_QNET.exists():
        pytest.skip("models/big2-qnet-duty.pknn missing")
    agent = QNetAgent(temperature=0.0)
    # The first deal whose opening offers several moves (3♦ alone is sometimes the only one).
    state = next(s for s in map(Rules().new_deal, range(50)) if len(s.legal_actions()) > 3)
    legal, scores = agent.net.score_moves(state)
    assert agent.act(state) == legal[int(np.argmax(scores))]
    hot = QNetAgent(temperature=1e6, seed=1)  # nearly uniform: several moves show up
    assert len({hot.act(state) for _ in range(60)}) > 1
