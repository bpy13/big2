"""The game-agnostic Elo fit (plan Task 40, :mod:`big2.teach.grading.elo`)."""

from __future__ import annotations

import math

import numpy as np
from big2.teach.grading import elo

TRUE = {"r600": 600.0, "r900": 900.0, "medium": 1200.0, "r1450": 1450.0, "r1700": 1700.0}


def synthetic_games(
    true_elo: dict[str, float], deals: int, seed: int, duplicate: bool = True
) -> list[dict]:
    """Four-seat games whose pairwise outcomes follow Bradley-Terry exactly: each seat's
    score is its strength on the natural-log scale plus Gumbel noise (the difference of two
    Gumbels is logistic). Every deal plays the four rotations of a random ABAB lineup of two
    adjacent rungs, so every rung meets the one above it, as the ladder does."""
    rng = np.random.default_rng(seed)
    names = sorted(true_elo, key=true_elo.get)
    games = []
    for d in range(deals):
        i = int(rng.integers(0, len(names) - 1))
        lineup = [names[i], names[i + 1], names[i], names[i + 1]]
        for rot in range(4 if duplicate else 1):
            players = [lineup[(s - rot) % 4] for s in range(4)]
            strength = np.array([true_elo[p] * math.log(10) / 400 for p in players])
            scores = strength + rng.gumbel(size=4)
            games.append({"deal": d, "players": players, "scores": [float(x) for x in scores]})
    return games


def test_pairwise_results_follow_the_apps_rule() -> None:
    pairs = elo.pairwise_results(["a", "b", "a", "c"], [3.0, -1.0, 3.0, -1.0])
    # a beats b and c from both seats; b and c draw; a does not play itself.
    wins = {}
    for x, y, w in pairs:
        wins[(x, y)] = wins.get((x, y), 0) + w
        wins[(y, x)] = wins.get((y, x), 0) + 1 - w
    assert len(pairs) == 5
    assert wins[("a", "b")] == 2 and wins[("b", "a")] == 0
    assert wins[("a", "c")] == 2 and wins[("c", "a")] == 0
    assert wins[("b", "c")] == 0.5 and wins[("c", "b")] == 0.5


def test_two_players_elo_matches_the_closed_form() -> None:
    # 75 % wins for a over b: Elo gap 400 log10(3) = 190.8.
    games = []
    for d in range(400):
        w = d % 4 != 0  # 3 of 4
        games.append({"deal": d, "players": ["a", "b"], "scores": [1.0, 0.0] if w else [0.0, 1.0]})
    fit = elo.fit_elo(games, anchor=("b", 1000.0), prior_games=0.0)
    assert fit["b"] == 1000.0
    assert abs(fit["a"] - 1000.0 - 400 * math.log10(3)) < 1e-6


def test_draws_count_as_half() -> None:
    games = [{"deal": d, "players": ["a", "b"], "scores": [0.0, 0.0]} for d in range(50)]
    fit = elo.fit_elo(games, anchor=("a", 1500.0))
    assert abs(fit["b"] - 1500.0) < 1e-6


def test_synthetic_ladder_is_recovered_within_its_range() -> None:
    games = synthetic_games(TRUE, deals=3000, seed=1)
    report = elo.ladder_report(games, anchor=("medium", 1200.0), resamples=300, seed=2)
    assert report.order == sorted(TRUE, key=TRUE.get)
    assert report.elo["medium"] == 1200.0
    covered = 0
    for name, true in TRUE.items():
        lo, hi = report.range95[name]
        covered += lo <= true <= hi
        assert abs(report.elo[name] - true) < 40, (name, true, report.elo[name])
    # A 95 % range misses its true value 1 time in 20; 4 of 5 covered is the floor here.
    assert covered >= 4, report.range95
    # Adjacent gaps: the right size, with a finite range from the direct pairings.
    gaps_covered = 0
    for g, true_gap in zip(report.gaps, (300, 300, 250, 250), strict=True):
        gaps_covered += g["range95"][0] <= true_gap <= g["range95"][1]
        assert abs(g["gap"] - true_gap) < 3 * g["half_width"], (g, true_gap)
        assert g["pair_games"] > 0
        assert 0 < g["half_width"] < 100
    assert gaps_covered >= 3, report.gaps


def test_disconnected_players_are_refused() -> None:
    games = [
        {"deal": 0, "players": ["a", "b"], "scores": [1.0, 0.0]},
        {"deal": 1, "players": ["c", "d"], "scores": [1.0, 0.0]},
    ]
    try:
        elo.fit_elo(games)
    except ValueError as e:
        assert "connected" in str(e)
    else:
        raise AssertionError("two unlinked pairs must not fit")


def test_deals_for_half_width_scales_by_the_square() -> None:
    assert elo.deals_for_half_width(200, 50.0, 25.0) == 800
    assert elo.deals_for_half_width(200, 20.0, 25.0) == 200
