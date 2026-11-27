"""Network play-out search (crates/search/src/netsearch.rs) through ``big2._core``,
the ``netsearch:`` agent spec and the ``big2 search`` command (plan Task 38)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from big2._core import Big2State, net_search  # noqa: E402
from big2.agents import NetSearchAgent, make_agent, parse_depth  # noqa: E402
from big2.cli.main import app  # noqa: E402
from big2.rl.export import export_pknn  # noqa: E402
from big2.rl.model import NetConfigV2, QNetV2  # noqa: E402
from typer.testing import CliRunner  # noqa: E402


@pytest.fixture(scope="module")
def pknn(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small untrained policy-value network, exported."""
    d = tmp_path_factory.mktemp("netsearch")
    torch.manual_seed(0)
    cfg = NetConfigV2(state_hidden=32, state_out=16, action_out=16, head_hidden=32)
    ckpt = d / "small-v2.pt"
    torch.save(
        {"arch": "qnet_v2", "net_config": asdict(cfg), "model": QNetV2(cfg).state_dict()}, ckpt
    )
    return export_pknn(ckpt, d / "small-v2.pknn", precision="f32")


def test_net_search_reports_paired_candidates(pknn: Path) -> None:
    state = Big2State.new_deal(3)
    r = net_search(state, net=str(pknn), k=3, n=16, seed=4)
    assert r["player"] == state.current_player and r["deals"] == 16
    cands = r["candidates"]
    assert len(cands) == 3
    assert cands[0]["loss"] == 0.0 and all(c["loss"] >= 0 for c in cands)
    assert all(c["cards"] in state.legal_actions() for c in cands)
    assert [m["cards"] for m in r["moves"]] == [c["cards"] for c in cands]
    # Reproducible, and the same with more threads (deals are seeded by index).
    again = net_search(state, net=str(pknn), k=3, n=16, seed=4, threads=3)
    assert again["candidates"] == cands
    # k = 1: only the network's top move.
    assert len(net_search(state, net=str(pknn), k=1, n=4)["candidates"]) == 1
    # A forced-in move is searched besides the top k.
    others = [c for c in state.legal_actions() if c not in {x["cards"] for x in cands}]
    r = net_search(state, net=str(pknn), k=3, n=4, extra=[others[0]])
    assert len(r["candidates"]) == 4
    # Depth in plies, value_mix, uniform deals.
    r = net_search(state, net=str(pknn), k=2, n=4, depth=0, value_mix=0.5, belief=None)
    assert r["deals"] == 4
    with pytest.raises(ValueError, match="between 0 and 1"):
        net_search(state, net=str(pknn), value_mix=2.0)
    with pytest.raises(ValueError, match="belief"):
        net_search(state, net=str(pknn), belief="greedy")


def test_netsearch_agent_spec(pknn: Path) -> None:
    agent = make_agent(f"netsearch:net={pknn},k=2,n=8,depth=6,value_mix=0.5", seed=1)
    assert isinstance(agent, NetSearchAgent)
    assert (agent.k, agent.n, agent.depth, agent.options) == (2, 8, 6, {"value_mix": 0.5})
    state = Big2State.new_deal(5)
    assert agent.act(state) in state.legal_actions()
    full = make_agent(f"netsearch:net={pknn},depth=full")
    assert isinstance(full, NetSearchAgent) and full.depth is None
    assert parse_depth("FULL") is None and parse_depth(0) == 0
    with pytest.raises(ValueError):
        parse_depth(-1)
    with pytest.raises(ValueError, match="net="):
        make_agent("netsearch:k=2")


def test_big2_search_command(pknn: Path) -> None:
    state = Big2State.new_deal(2)
    seat = state.current_player
    from big2.games.big2 import fmt

    hand = fmt(state.hand(seat))
    out = CliRunner().invoke(
        app,
        ["search", "--net", str(pknn), "--seat", str(seat), "--hand", hand, "--k", "2", "--n", "8"],
    )
    assert out.exit_code == 0, out.output
    assert "Network search" in out.output and "Loss vs best" in out.output
