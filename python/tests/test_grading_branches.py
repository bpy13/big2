"""Branch exploration (plan Task 50): coverage, best-first order, cap prefix, forced nodes."""

from __future__ import annotations

import math

import pytest
from big2.teach.grading.branches import END, Game, coverage_at, explore

P = {0: 0.7, 1: 0.3}


def tree_game(depth: int = 3, forced_at: frozenset[int] = frozenset()):
    """Own decisions are binary (0 with 0.7, 1 with 0.3); the deal ends after ``depth`` own
    moves. A decision at a depth in ``forced_at`` has one move (0)."""
    calls: list[tuple] = []

    def describe(h: tuple) -> tuple[str, dict, bool]:
        if len(h) in forced_at:
            return "k" + "".join(map(str, h)), {0: 1.0}, True
        return "k" + "".join(map(str, h)), dict(P), False

    def step(h: tuple, m: int):
        h2 = (*h, m)
        return None if len(h2) >= depth else h2

    def grade(h: tuple):
        calls.append(h)
        return {"graded": h}

    return Game(describe, step, grade), calls


def test_main_line_only():
    g, calls = tree_game()
    r = explore(g, (), [0, 0, 0], {"k": 1, "k0": 1, "k00": 1}, cap=3)
    assert r.graded == 3 and r.new_grades == 0 and calls == []
    assert r.coverage == pytest.approx(0.7**3)
    assert all(n.main_line for n in r.nodes.values())


def test_full_tree_covers_everything():
    g, calls = tree_game()
    r = explore(g, (), [0, 0, 0], {}, cap=100, target=1.0)
    assert r.graded == 7 and len(calls) == 7
    assert r.coverage == pytest.approx(1.0)
    assert r.nodes["k"].children[1] == "k1"
    assert r.nodes["k11"].children[0] == END
    assert r.nodes["k11"].path_p == pytest.approx(0.09)


def test_best_first_order():
    g, calls = tree_game()
    explore(g, (), [0, 0, 0], {"k": 1, "k0": 1, "k00": 1}, cap=4)
    # off-line edges: root->1 (0.3), k0->1 (0.21), k00->1 (end); the most likely is first
    assert calls == [(1,)]


def test_cap_prefix_matches_smaller_cap():
    g, _ = tree_game(depth=5)
    big = explore(g, (), [0] * 5, {}, cap=80, target=1.0)
    assert math.isclose(big.coverage, 1.0)
    for cap in (5, 8, 12, 20):
        g2, _ = tree_game(depth=5)
        small = explore(g2, (), [0] * 5, {}, cap=cap, target=1.0)
        assert coverage_at(big.curve, cap) == pytest.approx(small.coverage)
    # coverage never falls as the cap grows
    covs = [c for _, c in big.curve]
    assert covs == sorted(covs)


def test_target_stops_early():
    g, _ = tree_game(depth=5)
    r = explore(g, (), [0] * 5, {}, cap=80, target=0.5)
    assert 0.5 <= r.coverage < 1.0
    assert r.graded < 31


def test_forced_decisions_are_free():
    g, calls = tree_game(depth=3, forced_at=frozenset({1}))
    r = explore(g, (), [0, 0, 0], {}, cap=100, target=1.0)
    # depth 1 is forced: nodes k, k0, k1 (forced, free), k00, k10
    assert r.nodes["k1"].forced and r.nodes["k1"].grade is None
    assert r.graded == 3 and len(calls) == 3
    assert r.coverage == pytest.approx(1.0)


def test_main_line_must_fit_the_deal():
    g, _ = tree_game()
    with pytest.raises(ValueError):
        explore(g, (), [0, 0], {})
    with pytest.raises(ValueError):
        explore(g, (), [0, 0, 0, 0], {})
