"""Tests for the Python helpers in big2.games.big2."""

import pytest
from big2.games.big2 import (
    PASS,
    Big2State,
    Rules,
    card_list,
    cards,
    fmt,
    kind_of,
    pretty,
    state_from_observation,
)


def test_card_text_round_trip() -> None:
    m = cards("2s 3d Tc")
    assert fmt(m) == "3d Tc 2s"
    assert pretty(m) == "3♦ 10♣ 2♠"
    assert card_list(m) == ["3d", "Tc", "2s"]
    assert cards("pass") == PASS
    assert fmt(PASS) == "pass"
    with pytest.raises(ValueError):
        cards("3d 3d")


def test_kind_of_respects_rules() -> None:
    assert kind_of(cards("Ad 2c 3h 4s 5d")) == "straight"
    assert kind_of(cards("Ad 2c 3h 4s 5d"), Rules(straights="no_twos")) is None
    assert kind_of(cards("3d 4d")) is None


def test_rules_reach_the_engine() -> None:
    s = Rules(scoring="win_loss", pass_rule="locked_out").new_deal(seed=1)
    assert s.rules()["scoring"] == "win_loss"
    assert s.rules()["pass_rule"] == "locked_out"
    with pytest.raises(ValueError):
        Big2State.new_deal(1, scoring="bogus")


def test_illegal_moves_raise_with_a_reason() -> None:
    s = Rules().new_deal(seed=3)
    p = s.current_player
    assert p is not None
    non_opening = next(1 << i for i in range(1, 52) if s.hand(p) >> i & 1)
    with pytest.raises(ValueError, match="3d"):
        s.apply(non_opening)
    with pytest.raises(ValueError, match="leader cannot pass"):
        s.apply(PASS)


def test_state_from_observation_replays_history() -> None:
    full = Rules().new_deal(seed=5)
    # Play 10 greedy moves on the true state, then rebuild from one seat's view.
    for _ in range(10):
        full.apply(full.greedy_action())
    seat = full.current_player
    assert seat is not None
    rebuilt = state_from_observation(
        seat, full.hand(seat), [(p, c) for p, c in full.history], seed=1
    )
    assert rebuilt.current_player == seat
    assert rebuilt.hand(seat) == full.hand(seat)
    assert rebuilt.hand_sizes == full.hand_sizes
    assert rebuilt.played == full.played
    assert rebuilt.current_play == full.current_play
    assert sorted(rebuilt.legal_actions()) == sorted(full.legal_actions())


def test_state_from_observation_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="13"):
        state_from_observation(0, cards("3d 4d"), [])
    with pytest.raises(ValueError, match="3d"):
        state_from_observation(0, cards("4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad 2d 4c"), [])
