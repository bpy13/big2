"""Task 21: the restricted pickle loader, and Charlesworth's network as an agent.

The agent tests need his weights (``runs/external/charlesworth/``, gitignored, never
committed) and skip when they are absent.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest
from big2._core import Big2State
from big2.agents import make_agent
from big2.agents.external import safe_pickle
from big2.agents.external.charlesworth import (
    N_ACTIONS,
    N_FEATURES,
    PASS_ACTION,
    WEIGHTS_DIR,
    CharlesworthAgent,
)

HAVE_WEIGHTS = (WEIGHTS_DIR / "modelParameters136500").exists() and (
    WEIGHTS_DIR / "actionIndices.pkl"
).exists()
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="Charlesworth's weights not downloaded")


class _CallsSystem:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ("echo should never run",))


class _CallsEval:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("1 + 1",))


@pytest.mark.parametrize("payload", [_CallsSystem(), _CallsEval(), {"k": [_CallsSystem()]}])
def test_restricted_unpickler_refuses_code(payload: object) -> None:
    data = pickle.dumps(payload)
    with pytest.raises(safe_pickle.UnsafePickle):
        safe_pickle.loads(data)


def test_restricted_unpickler_refuses_other_modules() -> None:
    import collections

    with pytest.raises(safe_pickle.UnsafePickle):
        safe_pickle.loads(pickle.dumps(collections.Counter(a=1)))


def test_restricted_unpickler_reads_numpy_arrays() -> None:
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    out = safe_pickle.loads(pickle.dumps([a, a.T.copy(), np.zeros((2, 2), np.float32)]))
    assert np.array_equal(out[0], a)
    assert np.array_equal(out[1], a.T)
    assert out[2].dtype == np.float32


def test_restricted_unpickler_refuses_object_arrays() -> None:
    obj = np.array([_CallsSystem()], dtype=object)
    with pytest.raises(safe_pickle.UnsafePickle):
        safe_pickle.loads(pickle.dumps(obj))


@needs_weights
def test_weights_and_tables_have_the_published_shapes() -> None:
    params, tables = (
        safe_pickle.load(str(WEIGHTS_DIR / "modelParameters136500")),
        safe_pickle.load(str(WEIGHTS_DIR / "actionIndices.pkl")),
    )
    assert [p.shape for p in params[:6]] == [
        (412, 512),
        (512,),
        (512, 256),
        (256,),
        (256, N_ACTIONS),
        (N_ACTIONS,),
    ]
    assert [t.shape[0] for t in tables[4:]] == [33, 31, 330, 1287]
    assert all(np.isfinite(p).all() for p in params)


@needs_weights
def test_agent_plays_legal_four_card_games_and_maps_every_action() -> None:
    agent = CharlesworthAgent(temperature=1.0, seed=3)
    seen_four = 0
    for seed in range(12):
        s = Big2State.new_deal(seed=seed, four_card_plays=True)
        while not s.is_terminal:
            x = agent.features(s)
            assert x.shape == (N_FEATURES,) and set(np.unique(x)) <= {0.0, 1.0}
            legal = list(s.legal_actions())
            nums = agent.action_numbers(s, legal)
            assert all(n >= 0 for n in nums), "a legal action has no network action number"
            assert (PASS_ACTION in nums) == (0 in legal)
            seen_four += sum(1 for a in legal if a.bit_count() == 4)
            a = agent.act(s)
            assert a in legal
            s.apply(a)
    assert agent.unmapped == 0
    assert seen_four > 0, "no deal offered a four-card play"


@needs_weights
def test_charlesworth_spec_and_his_network_beats_random() -> None:
    from big2.eval.tournament import run_tournament
    from big2.games.big2 import Rules

    assert isinstance(make_agent("charlesworth"), CharlesworthAgent)
    r = run_tournament(
        ["charlesworth", "random", "random", "random"],
        8,
        rules=Rules(four_card_plays=True),
        score_as="sum_of_others",
    )
    by = {s.agent: s for s in r.summaries}
    assert by["charlesworth"].mean_score > 0 > by["random"].mean_score


def test_rules_four_card_plays_stays_hidden_unless_on() -> None:
    from big2.games.big2 import Rules

    assert "four_card_plays" not in Rules().kwargs()
    assert Rules(four_card_plays=True).kwargs()["four_card_plays"] is True
    assert "four_card_plays" not in Big2State.new_deal(seed=1).rules()
    assert Big2State.new_deal(seed=1, four_card_plays=True).rules()["four_card_plays"] is True


class _Table:
    """A hand-built table: just the attributes `CharlesworthAgent.features` reads."""

    current_player = 1
    hand_sizes = (13, 7, 8, 13)

    def __init__(self, history, current_play, played) -> None:
        self.history = history
        self.current_play = current_play
        self.played = played

    def rules(self) -> dict[str, str]:
        return {"straights": "no_twos"}

    def hand(self, player: int) -> int:
        # 3D 3C 3H 4D 5D 6D 7D: a triple, a diamond flush and a 3-7 straight.
        return sum(1 << c for c in (0, 1, 2, 4, 8, 12, 16))


def test_feature_layout_is_pinned_on_one_hand_built_state() -> None:
    """Indices follow the author's published encoding (his source, read-only; Task 21)."""
    agent = object.__new__(CharlesworthAgent)  # features() needs no weights
    full_house = sum(1 << c for c in (36, 37, 38, 24, 25))  # Q Q Q 9 9
    # Player 2 (the next seat) played the full house; player 3 passed since.
    table = _Table([(2, full_house), (3, 0)], (2, full_house, 0), full_house)
    expected = {
        # hand, 22 per card: value, suit, pair, triple, four, straight, flush
        *(0, 13, 17, 18, 20, 21),  # 3D
        *(22, 36, 39, 40, 42),  # 3C
        *(44, 59, 61, 62, 64),  # 3H
        *(67, 79, 86, 87),  # 4D
        *(90, 101, 108, 109),  # 5D
        *(113, 123, 130, 131),  # 6D
        *(136, 145, 152, 153),  # 7D
        # opponents, 27 each from 286: cards left, then (next seat) played a full house
        293,  # next seat holds 8 cards
        286 + 26,  # ... and has played a full house
        286 + 27 + 12,  # second seat: 13 cards
        286 + 54 + 12,  # third seat: 13 cards
        # Q of D, C, H played (before the previous-hand group)
        367,
        368,
        369,
        # previous hand from 383: value Q (the three's), no suit, full house, one pass
        383 + 9,
        383 + 25,
        383 + 27,
    }
    x = agent.features(table)  # type: ignore[arg-type]
    assert x.shape == (N_FEATURES,)
    assert set(np.flatnonzero(x).tolist()) == expected

    # Leading a new trick: control (group offset 17) is the only previous-hand bit.
    lead = agent.features(_Table([], None, 0))  # type: ignore[arg-type]
    assert np.flatnonzero(lead[383:]).tolist() == [17]


def test_restricted_unpickler_refuses_extension_codes() -> None:
    # PROTO 2, EXT1 code 1, STOP: copyreg's extension cache would skip find_class.
    with pytest.raises(safe_pickle.UnsafePickle):
        safe_pickle.loads(b"\x80\x02\x82\x01.")
