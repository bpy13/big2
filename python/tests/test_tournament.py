"""Tournament runner tests."""

import pytest
from big2.eval import run_tournament


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
