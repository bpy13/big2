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
