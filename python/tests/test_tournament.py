"""Tournament runner tests."""

import pytest
from big2.eval import run_tournament
from big2.eval.tournament import _seatings, sum_of_others_scores
from big2.games.big2 import Big2State, Rules, cards


def test_duplicate_tournament_is_zero_sum_and_symmetric() -> None:
    r = run_tournament(["greedy", "random", "random", "random"], deals=6, seed=1)
    by = {s.agent: s for s in r.summaries}
    # One greedy seat and three random seats; scores are zero-sum per game.
    total = by["greedy"].mean_score + 3 * by["random"].mean_score
    assert total == pytest.approx(0.0, abs=1e-9)
    assert by["greedy"].games == 6 * 4
    assert by["random"].games == 6 * 4 * 3


def test_identical_agents_score_exactly_zero() -> None:
    # With four copies of the same deterministic agent, each spec's per-deal mean is 0.
    r = run_tournament(["greedy"] * 4, deals=3, seed=2)
    assert r.summaries[0].mean_score == pytest.approx(0.0, abs=1e-9)


def test_parallel_matches_serial() -> None:
    a = run_tournament(["greedy", "random", "greedy", "random"], deals=4, seed=3, workers=1)
    b = run_tournament(["greedy", "random", "greedy", "random"], deals=4, seed=3, workers=2)
    assert sorted(a.per_deal, key=str) == sorted(b.per_deal, key=str)


def test_lineup_must_have_four_agents() -> None:
    with pytest.raises(ValueError):
        run_tournament(["greedy"], deals=1)


def test_sum_of_others_disagrees_with_tiered_on_a_hand_made_deal() -> None:
    # Seat 0 leads its last card and goes out; the others hold 10, 1 and 2 cards.
    hands = [
        cards("2s"),
        cards("3d 4d 5d 6d 7d 8d 9d Td Jd Qd"),
        cards("3c"),
        cards("4c 5c"),
    ]
    state = Big2State.from_position(hands, 0, None, **Rules().kwargs())
    state.apply(cards("2s"))
    assert state.winner == 0
    # Tiered: penalties 0, 10×3, 1, 2 → score_i = 33 − 4·p_i; seats 2 and 3 gain.
    assert state.scores == [33.0, -87.0, 29.0, 25.0]
    # Charlesworth/Patwa: the winner takes the 13 cards left, each loser pays its own.
    assert sum_of_others_scores(state) == [13.0, -10.0, -1.0, -2.0]


def test_sum_of_others_needs_a_finished_game() -> None:
    with pytest.raises(ValueError):
        sum_of_others_scores(Rules().new_deal(seed=1))


def test_score_as_sum_of_others_is_zero_sum() -> None:
    r = run_tournament(
        ["greedy", "random", "random", "random"], deals=6, seed=1, score_as="sum_of_others"
    )
    by = {s.agent: s for s in r.summaries}
    assert by["greedy"].mean_score + 3 * by["random"].mean_score == pytest.approx(0.0, abs=1e-9)
    assert r.score_as == "sum_of_others"
    # Every game's winner gets at least 3 points (one card from each loser) under this rule.
    assert by["greedy"].mean_score > 0


def test_score_as_rejects_unknown_rules() -> None:
    with pytest.raises(ValueError):
        run_tournament(["greedy"] * 4, deals=1, score_as="charlesworth")


def test_seat_random_plays_each_deal_once_with_shuffled_seats() -> None:
    lineup = ["greedy", "random", "random", "random"]
    r = run_tournament(lineup, deals=20, seed=4, seat_random=True)
    by = {s.agent: s for s in r.summaries}
    assert by["greedy"].games == 20
    assert by["random"].games == 60
    assert r.seat_random
    # The greedy seat moves around: it is not always seat 0.
    seats = {_seatings(lineup, 4_000_000 + d, True)[0].index("greedy") for d in range(20)}
    assert len(seats) > 1
    # Serial and parallel runs agree.
    b = run_tournament(lineup, deals=20, seed=4, seat_random=True, workers=2)
    assert sorted(r.per_deal, key=str) == sorted(b.per_deal, key=str)
