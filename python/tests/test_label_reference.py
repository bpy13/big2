"""Task 42's reference pieces in ``analyze``: ``root_move`` (one PUCT search per graded
move) and the exact endgame weighted by the belief network (decision 0011)."""

from __future__ import annotations

from pathlib import Path

import pytest
from big2._core import Big2State, analyze, parse_cards

ROOT = Path(__file__).resolve().parents[2]
NET = ROOT / "runs" / "p6_net" / "selfplay-h24.pknn"


def _belief_pknn(tmp_path: Path, zero: bool) -> str:
    torch = pytest.importorskip("torch")
    from big2.rl.belief import BeliefConfig, BeliefNet, export_belief_pknn

    torch.manual_seed(3)
    net = BeliefNet(BeliefConfig(hidden1=16, hidden2=16))
    if zero:  # the hand-size baseline: every deal of the hidden cards equally likely
        for p in net.net[-1].parameters():
            torch.nn.init.zeros_(p)
    else:  # a network with strong opinions
        with torch.no_grad():
            for p in net.net[-1].parameters():
                p.mul_(200.0)
    name = "zero" if zero else "loud"
    torch.save(
        {"arch": "belief", "net_config": vars(net.cfg), "model": net.state_dict()},
        tmp_path / f"{name}.pt",
    )
    return str(export_belief_pknn(tmp_path / f"{name}.pt", tmp_path / f"{name}.pknn"))


def _endgame() -> Big2State:
    # Seat 0 leads with 4 cards; 6 hidden cards, two each: 90 deals, solved exactly.
    hands = [parse_cards(h) for h in ("3d 7h 9s Kc", "4c 8d", "5h Js", "6s Qd")]
    return Big2State.from_position(hands, 0)


def _means(r: dict) -> dict[int, float]:
    return {m["cards"]: m["mean_score"] for m in r["moves"]}


def test_exact_endgame_weighs_deals_by_the_belief_net(tmp_path: Path) -> None:
    s = _endgame()
    uniform = analyze(s, iterations=10, infoset_endgame=12)
    assert uniform["exact"] is True
    zero = analyze(
        s, iterations=10, infoset_endgame=12, belief="net", belief_net=_belief_pknn(tmp_path, True)
    )
    assert zero["exact"] is True
    for a, v in _means(uniform).items():
        assert _means(zero)[a] == pytest.approx(v, abs=1e-9)
    loud = analyze(
        s, iterations=10, infoset_endgame=12, belief="net", belief_net=_belief_pknn(tmp_path, False)
    )
    assert loud["exact"] is True
    assert any(abs(_means(loud)[a] - v) > 1e-3 for a, v in _means(uniform).items())


@pytest.mark.skipif(not NET.exists(), reason="network file missing")
def test_root_move_spends_the_whole_budget_on_one_move() -> None:
    s = Big2State.new_deal(seed=11)
    for _ in range(6):
        s.apply(s.legal_actions()[0])
    legal = s.legal_actions()
    free = analyze(s, algorithm="puct", net=str(NET), iterations=200, seed=3)
    rare = free["moves"][-1]["cards"]
    r = analyze(s, algorithm="puct", net=str(NET), iterations=200, seed=3, root_move=rare)
    assert [m["cards"] for m in r["moves"]] == [rare]
    assert r["moves"][0]["visits"] == 200
    again = analyze(s, algorithm="puct", net=str(NET), iterations=200, seed=3, root_move=rare)
    assert again["moves"][0]["mean_score"] == r["moves"][0]["mean_score"]
    assert len(legal) > 1
    with pytest.raises(ValueError, match="root_move"):
        analyze(s, iterations=10, root_move=rare)
    held = s.hands[s.current_player]
    other = next(1 << c for c in range(52) if not held >> c & 1)
    with pytest.raises(ValueError):  # a card the seat does not hold
        analyze(s, algorithm="puct", net=str(NET), iterations=10, root_move=other)
