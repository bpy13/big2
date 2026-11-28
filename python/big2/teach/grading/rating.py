"""Luck-removed rating from points won (plan Task 43, decision 0008 amended), game-agnostic.

A *game* is one finished table, ``{"deal": <id>, "players": [name, ...], "scores": [float,
...]}`` (the shape of :mod:`big2.teach.grading.elo`). *Luck* is, per deal, the score each
absolute seat gets when the expert's network plays all four seats of that very deal
(``luck[deal] = [float, ...]``): the part of the result that the cards decide. A seat's
**margin** is its score minus its luck. Because the luck term depends on the deal alone, every
player at the table is corrected by the same rule: equal players come out equal, and a player
who beats the network's own play on the same cards gets a positive margin, so nothing caps a
rating at the expert's.

**Strength model** (:func:`fit_strengths`): a player ``p`` has a strength ``s_p`` in points a
deal; the expected margin of seat ``i`` at a table is ``s_i - mean_j s_j`` over the four seats
(zero-sum, additive). Strengths are the least-squares fit of the observed margins.

**Scale** (:func:`to_rating`): rating = ``top_rating * (s - s_zero) / (s_top - s_zero)``, with
the zero player (random play) at 0 and the top player (the expert's network playing its top
move) at ``top_rating`` (3000). ``c`` = rating points per point of margin a deal.

**A player's rating after each deal** (:func:`update`): with ``n`` deals so far,
``R <- R + K * c * (margin - expected_margin)``, ``K = max(1 / (n + 1), k_min)``: the running
mean of the margins while few deals are known, a fixed step afterwards so the rating follows a
player who improves. :func:`deals_for_half_width` gives the deals a stable rating needs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Game = Mapping[str, Any]
Luck = Mapping[Any, Sequence[float]]

TOP_RATING = 3000.0
"""The expert's network (top move) on the scale; random play sits at 0."""


def margins(games: Iterable[Game], luck: Luck) -> list[tuple[str, Any, int, float]]:
    """``(name, deal, seat, score - luck)`` for every seat of every game whose deal has a luck
    estimate."""
    out = []
    for g in games:
        lk = luck.get(g["deal"])
        if lk is None:
            continue
        for i, (name, score) in enumerate(zip(g["players"], g["scores"], strict=True)):
            out.append((name, g["deal"], i, float(score) - float(lk[i])))
    return out


def fit_strengths(
    games: Iterable[Game], luck: Luck, anchor: tuple[str, float] | None = None
) -> dict[str, float]:
    """Least-squares strengths in points a deal: margin of seat ``i`` = ``s_i - mean(s at the
    table)``. Adding a constant to every strength changes no margin, so the solution is pinned
    with ``anchor = (name, value)`` (default: the strengths sum to 0)."""
    games = [g for g in games if g["deal"] in luck]
    if not games:
        raise ValueError("no games with a luck estimate")
    names = sorted({n for g in games for n in g["players"]})
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    y = []
    for g in games:
        lk = luck[g["deal"]]
        k = len(g["players"])
        for i, name in enumerate(g["players"]):
            row = np.zeros(len(names))
            row[idx[name]] += 1.0
            for other in g["players"]:
                row[idx[other]] -= 1.0 / k
            rows.append(row)
            y.append(float(g["scores"][i]) - float(lk[i]))
    a = np.array(rows)
    s, *_ = np.linalg.lstsq(a, np.array(y), rcond=None)
    s = s - s.mean()
    if anchor is not None:
        name, value = anchor
        if name not in idx:
            raise KeyError(f"anchor {name!r} played no game")
        s = s - s[idx[name]] + value
    return {n: float(v) for n, v in zip(names, s, strict=True)}


def expected_margin(strength: float, opponents: Sequence[float]) -> float:
    """The additive model's expected margin of a player against these opponents' strengths."""
    table = [strength, *opponents]
    return strength - sum(table) / len(table)


def to_rating(
    strengths: Mapping[str, float], zero: str, top: str, top_rating: float = TOP_RATING
) -> tuple[dict[str, float], float]:
    """Ratings on the 0 to ``top_rating`` scale (``zero`` at 0, ``top`` at ``top_rating``) and
    ``c``, the rating points per point of margin a deal."""
    span = strengths[top] - strengths[zero]
    if span <= 0:
        raise ValueError(f"{top!r} must be stronger than {zero!r}")
    c = top_rating / span
    return {n: c * (s - strengths[zero]) for n, s in strengths.items()}, c


def update(
    rating: float,
    deals: int,
    margin: float,
    opponent_ratings: Sequence[float],
    c: float,
    k_min: float = 0.02,
) -> float:
    """The rating after one more deal. ``deals`` is the count before this deal; the step is the
    running-mean weight ``1 / (deals + 1)`` until it falls to ``k_min``, then ``k_min``."""
    exp = expected_margin(rating / c, [r / c for r in opponent_ratings])
    k = max(1.0 / (deals + 1), k_min)
    return rating + k * c * (margin - exp)


def deals_for_half_width(c: float, margin_sd: float, half_width: float = 100.0) -> int:
    """Deals whose mean margin puts the rating inside ``± half_width`` (95 %)."""
    return int(math.ceil((1.96 * c * margin_sd / half_width) ** 2))


def margin_sd(games: Iterable[Game], luck: Luck, strengths: Mapping[str, float]) -> float:
    """SD of a seat's margin around the model's expectation, over every seat of every game."""
    res = []
    for g in games:
        lk = luck.get(g["deal"])
        if lk is None:
            continue
        s = [strengths[n] for n in g["players"]]
        for i in range(len(g["players"])):
            res.append(float(g["scores"][i]) - float(lk[i]) - (s[i] - sum(s) / len(s)))
    return float(np.std(res, ddof=1))


def variance_cut(games: Iterable[Game], luck: Luck) -> dict[str, Any]:
    """How much of a seat's score variance the luck term removes: ``1 - Var(score - luck) /
    Var(score)`` over every seat of every game with a luck estimate, and the same for the
    duplicate leave-one-out baseline (the mean score of the same seat's hand in the deal's other
    games) where a deal has more than one game."""
    raw, res, dup_raw, dup_res = [], [], [], []
    by_deal: dict[Any, list[Game]] = {}
    for g in games:
        if g["deal"] in luck:
            by_deal.setdefault(g["deal"], []).append(g)
    for deal, gs in by_deal.items():
        lk = luck[deal]
        for g in gs:
            for i, score in enumerate(g["scores"]):
                raw.append(float(score))
                res.append(float(score) - float(lk[i]))
        if len(gs) > 1:
            for j, g in enumerate(gs):
                for i, score in enumerate(g["scores"]):
                    others = [float(h["scores"][i]) for k, h in enumerate(gs) if k != j]
                    dup_raw.append(float(score))
                    dup_res.append(float(score) - sum(others) / len(others))
    out = {
        "seats": len(raw),
        "score_sd": float(np.std(raw)),
        "margin_sd": float(np.std(res)),
        "cut": 1 - float(np.var(res)) / float(np.var(raw)) if raw else None,
    }
    if dup_raw:
        out["duplicate_cut"] = 1 - float(np.var(dup_res)) / float(np.var(dup_raw))
        out["duplicate_margin_sd"] = float(np.std(dup_res))
    return out


def bootstrap_ratings(
    games: Sequence[Game],
    luck: Luck,
    zero: str,
    top: str,
    *,
    resamples: int = 500,
    seed: int = 0,
    top_rating: float = TOP_RATING,
) -> dict[str, tuple[float, float]]:
    """95 % percentile ranges of every player's rating by a bootstrap over deals (the games
    of one deal move together)."""
    by_deal: dict[Any, list[Game]] = {}
    for g in games:
        if g["deal"] in luck:
            by_deal.setdefault(g["deal"], []).append(g)
    deals = list(by_deal)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(resamples):
        pick = rng.integers(0, len(deals), len(deals))
        gs = [g for k in pick for g in by_deal[deals[k]]]
        try:
            r, _ = to_rating(fit_strengths(gs, luck), zero, top, top_rating)
        except (ValueError, KeyError):
            continue
        for n, v in r.items():
            samples.setdefault(n, []).append(v)
    return {
        n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in samples.items()
    }


def split_by_rotation(games: Iterable[Game], parity_of: Any) -> list[dict[str, Any]]:
    """Equal-players check: every name is split into ``name/A`` and ``name/B`` by
    ``parity_of(game_index_within_deal)``; the two halves of one player should rate the same."""
    by_deal: dict[Any, list[Game]] = {}
    for g in games:
        by_deal.setdefault(g["deal"], []).append(g)
    out = []
    for gs in by_deal.values():
        for k, g in enumerate(gs):
            tag = "A" if parity_of(k) == 0 else "B"
            out.append({**g, "players": [f"{n}/{tag}" for n in g["players"]]})
    return out


__all__ = [
    "TOP_RATING",
    "bootstrap_ratings",
    "deals_for_half_width",
    "expected_margin",
    "fit_strengths",
    "margin_sd",
    "margins",
    "split_by_rotation",
    "to_rating",
    "update",
    "variance_cut",
]
