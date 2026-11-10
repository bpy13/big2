"""The learned belief (plan Task 19): labels, training, and the Rust reader against PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from big2._core import Big2State, belief_log_probs, encode  # noqa: E402
from big2.agents import make_agent  # noqa: E402
from big2.rl.belief import (  # noqa: E402
    BeliefConfig,
    BeliefNet,
    evaluate,
    export_belief_pknn,
    holder_labels,
)
from big2.rl.model import STATE_LEN  # noqa: E402


def _random_positions(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """``n`` mid-game positions: state features, true hands, seats to act, the states."""
    rng = np.random.default_rng(seed)
    states, hands, players, objs = [], [], [], []
    for k in range(n):
        s = Big2State.new_deal(seed=seed * 1000 + k)
        for _ in range(int(rng.integers(0, 20))):
            if s.is_terminal:
                break
            legal = s.legal_actions()
            s.apply(legal[int(rng.integers(len(legal)))])
        if s.is_terminal:
            continue
        st, _, _ = encode(s)
        states.append(np.frombuffer(st, dtype=np.uint8))
        hands.append(s.hands)
        players.append(s.current_player)
        objs.append(s)
    return (
        np.stack(states),
        np.asarray(hands, dtype=np.uint64),
        np.asarray(players, dtype=np.int64),
        objs,
    )


def test_holder_labels_name_the_true_holder_of_each_unseen_card() -> None:
    states, hands, players, _ = _random_positions(30, 1)
    labels = holder_labels(states, hands, players)
    for i in range(len(states)):
        unseen = np.flatnonzero(states[i, 52:104])
        assert (labels[i] >= 0).sum() == len(unseen)
        for c in unseen:
            holder = (players[i] + labels[i, c] + 1) % 4
            assert (int(hands[i, holder]) >> int(c)) & 1 == 1
        assert (labels[i, states[i, 52:104] == 0] == -1).all()


def test_untrained_network_is_the_hand_size_baseline() -> None:
    states, hands, players, _ = _random_positions(20, 2)
    net = BeliefNet(BeliefConfig(hidden1=16, hidden2=16))
    for p in net.net[-1].parameters():
        torch.nn.init.zeros_(p)
    labels = holder_labels(states, hands, players)
    packed = np.packbits(states, axis=1)
    r = evaluate(net, packed, labels, "cpu")
    assert r["loglik_per_card"] == pytest.approx(r["uniform_loglik_per_card"], abs=1e-4)
    # Each row of the output is a distribution over the three opponents.
    lp = torch.log_softmax(net(torch.from_numpy(states).float()), dim=2)
    assert torch.allclose(lp.exp().sum(dim=2), torch.ones(lp.shape[:2]), atol=1e-5)


def test_rust_reader_matches_pytorch(tmp_path: Path) -> None:
    states, _, _, objs = _random_positions(12, 3)
    net = BeliefNet(BeliefConfig(hidden1=32, hidden2=24))
    torch.save(
        {"arch": "belief", "net_config": vars(net.cfg), "model": net.state_dict()},
        tmp_path / "b.pt",
    )
    out = export_belief_pknn(tmp_path / "b.pt", tmp_path / "b.pknn", precision="f32")
    with torch.no_grad():
        want = torch.log_softmax(net(torch.from_numpy(states).float()), dim=2).numpy()
    for i, s in enumerate(objs):
        got = np.asarray(belief_log_probs(s, str(out)))
        unseen = states[i, 52:104].astype(bool)
        assert got.shape == (52, 3)
        np.testing.assert_allclose(got[unseen], want[i][unseen], atol=1e-4)


def test_net_belief_plays_a_legal_move(tmp_path: Path) -> None:
    net = BeliefNet(BeliefConfig(hidden1=16, hidden2=16))
    torch.save(
        {"arch": "belief", "net_config": vars(net.cfg), "model": net.state_dict()},
        tmp_path / "b.pt",
    )
    path = export_belief_pknn(tmp_path / "b.pt", tmp_path / "b.pknn")
    s = Big2State.new_deal(seed=5)
    for _ in range(8):
        s.apply(s.legal_actions()[0])
    agent = make_agent(f"ismcts:iterations=200,belief=net,belief_net={path}", seed=1)
    assert agent.act(s) in s.legal_actions()
    with pytest.raises(ValueError, match="belief_net"):
        make_agent("ismcts:iterations=50,belief=net", seed=1).act(s)
    assert net.net[0].in_features == STATE_LEN
