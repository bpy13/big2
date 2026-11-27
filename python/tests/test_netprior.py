"""The network's prior at a temperature (plan Task 40, :mod:`big2.agents.netprior`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from big2._core import net_search
from big2.agents import make_agent
from big2.agents.netprior import NetPriorAgent
from big2.games.big2 import Rules

NET = Path(__file__).resolve().parents[2] / "runs" / "p6_net" / "selfplay-h24.pknn"
pytestmark = pytest.mark.skipif(not NET.exists(), reason="Task 37's exported network is not here")


def positions(n: int = 12) -> list:
    out = []
    s = Rules().new_deal(3)
    while not s.is_terminal and len(out) < n:
        if len(s.legal_actions()) > 1:
            out.append(s.copy())
        s.apply(s.greedy_action(0.0, 1, True, 8))
    return out


def test_temperature_zero_plays_the_top_prior_move() -> None:
    agent = make_agent(f"netprior:net={NET},temperature=0", seed=1)
    assert isinstance(agent, NetPriorAgent)
    for s in positions():
        top = net_search(s, net=str(NET), k=1, n=1, depth=0)["candidates"][0]["cards"]
        assert agent.act(s) == top


def test_prior_covers_every_legal_move_and_sums_to_one() -> None:
    agent = NetPriorAgent(net=str(NET))
    for s in positions(4):
        moves, p = agent.prior(s)
        assert sorted(moves) == sorted(s.legal_actions())
        assert abs(float(p.sum()) - 1.0) < 1e-4
        assert list(p) == sorted(p, reverse=True)


def test_sampling_is_reproducible_and_flattens_with_temperature() -> None:
    s = positions(1)[0]
    a = make_agent(f"netprior:net={NET},temperature=1", seed=5)
    b = make_agent(f"netprior:net={NET},temperature=1", seed=5)
    assert [a.act(s) for _ in range(20)] == [b.act(s) for _ in range(20)]
    moves, _ = NetPriorAgent(net=str(NET)).prior(s)
    hot = make_agent(f"netprior:net={NET},temperature=50", seed=5)
    cold = make_agent(f"netprior:net={NET},temperature=0.05", seed=5)
    cold_top = sum(cold.act(s) == moves[0] for _ in range(60))
    hot_top = sum(hot.act(s) == moves[0] for _ in range(60))
    assert cold_top >= hot_top
    assert cold_top >= 55
