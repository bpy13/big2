"""Differential test: the Rust engine vs the independent Python reference.

Random games are played under random rule variants. At every step both engines
must agree on the exact set of legal actions; at the end, on the scores.

Default: 150 games (~30 s). For a long run: ``BIG2_DIFF_GAMES=20000 pytest -k differential``.
"""

from __future__ import annotations

import os
import random

import pytest
from poker._core import Big2State

from tests.reference.big2_reference import RefGame, cards_from_mask

GAMES = int(os.environ.get("BIG2_DIFF_GAMES", "150"))

VARIANTS = {
    "straights": ["with_low_twos", "no_twos"],
    "flush_order": ["rank_first", "suit_first"],
    "pass_rule": ["may_reenter", "locked_out"],
    "scoring": ["tiered", "original", "win_loss"],
    "last_card_duty": [False, True],
}


@pytest.mark.parametrize("seed", range(GAMES))
def test_engines_agree(seed: int) -> None:
    rng = random.Random(seed)
    rules = {k: rng.choice(v) for k, v in VARIANTS.items()}
    state = Big2State.new_deal(seed=seed, **rules)
    ref = RefGame(hands=[set(cards_from_mask(h)) for h in state.hands], **rules)  # type: ignore[arg-type]
    assert ref.current == state.current_player

    while not state.is_terminal:
        rust_legal = set(state.legal_actions())
        assert rust_legal == ref.legal_actions(), (
            f"legal actions differ (seed {seed}, rules {rules})"
        )
        # Bias toward playing (not passing) so games are not all passes.
        plays = sorted(a for a in rust_legal if a)
        action = (
            rng.choice(plays) if plays and rng.random() < 0.8 else rng.choice(sorted(rust_legal))
        )
        state.apply(action)
        ref.apply(action)
        assert state.current_player == (None if ref.winner is not None else ref.current)

    assert state.winner == ref.winner
    assert state.scores == ref.scores()
