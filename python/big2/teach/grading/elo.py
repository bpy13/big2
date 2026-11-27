"""Elo ratings from multi-player games, by maximum-likelihood Bradley-Terry.

Game-agnostic: a *game* is one finished table, ``{"deal": <id>, "players": [name, ...],
"scores": [float, ...]}`` (one name and one final score per seat; names may repeat when a
rung holds several seats). Every pair of seats with different names gives one pairwise
result, by the app's rule (hk-games ``server/hkgames/elo.py`` ``pairwise_results``): the
higher final score wins (1), equal scores are a draw (0.5). Draws count as half a win each
way in the likelihood. Ratings are on the Elo scale (400 points per factor of ten in
strength), anchored at one named player.

Uncertainty comes from a bootstrap over *deals*: the games of one deal (its duplicate
rotations) move together, so card luck shared by them is not counted twice.

Usage::

    games = read_games("runs/p6_ladder/games.jsonl")
    fit = fit_elo(games, anchor=("medium", 1200.0))
    report = ladder_report(games, anchor=("medium", 1200.0), resamples=1000)
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCALE = 400.0
"""Elo points per factor of ten in Bradley-Terry strength."""

Game = Mapping[str, Any]


def pairwise_results(
    players: Sequence[str], scores: Sequence[float]
) -> list[tuple[str, str, float]]:
    """Every unordered pair of seats with different names, as ``(a, b, w)``: ``w`` is 1 if
    ``a`` scored more than ``b``, 0.5 if equal, 0 otherwise. Seats of the same name give no
    result (a rung does not play itself)."""
    if len(players) != len(scores):
        raise ValueError("players and scores must have one entry per seat")
    out = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            if players[i] == players[j]:
                continue
            w = 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
            out.append((players[i], players[j], w))
    return out


def read_games(path: str | Path) -> list[dict[str, Any]]:
    """Reads games from a JSON-lines file (one game per line)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class PairCounts:
    """Pairwise results aggregated per deal: ``wins[d, i, j]`` is the wins (draws half) of
    player ``i`` over ``j`` in deal ``d``, ``games[d, i, j]`` their pairwise results there.
    """

    names: list[str]
    deals: list[Any]
    wins: np.ndarray  # (D, K, K)
    games: np.ndarray  # (D, K, K), symmetric

    @classmethod
    def from_games(cls, games: Iterable[Game]) -> PairCounts:
        games = list(games)
        names: list[str] = []
        index: dict[str, int] = {}
        deals: list[Any] = []
        deal_index: dict[Any, int] = {}
        for g in games:
            for p in g["players"]:
                if p not in index:
                    index[p] = len(names)
                    names.append(p)
            if g["deal"] not in deal_index:
                deal_index[g["deal"]] = len(deals)
                deals.append(g["deal"])
        k, d = len(names), len(deals)
        wins = np.zeros((d, k, k))
        count = np.zeros((d, k, k))
        for g in games:
            di = deal_index[g["deal"]]
            for a, b, w in pairwise_results(g["players"], g["scores"]):
                i, j = index[a], index[b]
                wins[di, i, j] += w
                wins[di, j, i] += 1.0 - w
                count[di, i, j] += 1.0
                count[di, j, i] += 1.0
        return cls(names, deals, wins, count)

    def total(self, deal_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Wins and games summed over all deals (or over a resample of them)."""
        if deal_indices is None:
            return self.wins.sum(axis=0), self.games.sum(axis=0)
        return self.wins[deal_indices].sum(axis=0), self.games[deal_indices].sum(axis=0)


def connected(games_matrix: np.ndarray) -> bool:
    """True if every player is linked to every other through played pairs."""
    k = games_matrix.shape[0]
    if k == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(games_matrix[i] > 0)[0]:
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == k


def fit_strengths(
    wins: np.ndarray,
    games: np.ndarray,
    *,
    prior_games: float = 0.5,
    iterations: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Maximum-likelihood Bradley-Terry strengths (geometric mean 1) by Hunter's (2004)
    minorization-maximization: ``gamma_i = W_i / sum_j n_ij / (gamma_i + gamma_j)``.

    ``prior_games`` adds that many drawn games between every pair that played at all, so
    a player who won or lost everything still gets a finite rating (a 0.5-game prior moves
    a 1,000-game pairing by under one Elo point). Draws in the data already count as half a
    win each way in ``wins``.
    """
    k = wins.shape[0]
    if k == 0:
        return np.zeros(0)
    played = games > 0
    n = games + prior_games * played
    w = wins + 0.5 * prior_games * played
    total_wins = w.sum(axis=1)
    gamma = np.ones(k)
    for _ in range(iterations):
        denom = (n / (gamma[:, None] + gamma[None, :])).sum(axis=1)
        new = np.where(denom > 0, total_wins / np.where(denom > 0, denom, 1.0), gamma)
        new = new / np.exp(np.log(new).mean())  # geometric mean 1: fixes the scale
        done = np.max(np.abs(np.log(new) - np.log(gamma))) < tol
        gamma = new
        if done:
            break
    return gamma


def strengths_to_elo(
    gamma: np.ndarray, names: Sequence[str], anchor: tuple[str, float] | None
) -> dict[str, float]:
    """Elo per name: ``anchor_value + 400 log10(gamma / gamma_anchor)``; with no anchor the
    ratings average 1,500."""
    log_elo = SCALE * np.log10(gamma)
    if anchor is None:
        offset = 1500.0 - log_elo.mean()
    else:
        name, value = anchor
        if name not in names:
            raise ValueError(f"anchor {name!r} played no game")
        offset = value - log_elo[list(names).index(name)]
    return {nm: float(e + offset) for nm, e in zip(names, log_elo, strict=True)}


def fit_elo(
    games: Iterable[Game], anchor: tuple[str, float] | None = None, **kw: Any
) -> dict[str, float]:
    """Elo per player from the games (see the module doc). Raises if the comparison graph
    is not connected (a rating would then be arbitrary)."""
    counts = PairCounts.from_games(games)
    wins, n = counts.total()
    if not connected(n):
        raise ValueError("the players do not form one connected group of pairings")
    return strengths_to_elo(fit_strengths(wins, n, **kw), counts.names, anchor)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Bradley-Terry win chance of ``a`` over ``b`` on the Elo scale."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / SCALE))


@dataclass
class LadderReport:
    """Fitted ratings with 95 % bootstrap ranges and the adjacent gaps of the fitted order."""

    anchor: tuple[str, float] | None
    elo: dict[str, float]
    range95: dict[str, list[float]]
    order: list[str] = field(default_factory=list)
    """Players from the lowest fitted Elo to the highest."""
    gaps: list[dict[str, Any]] = field(default_factory=list)
    """Per adjacent pair of the order: ``{lower, upper, gap, range95, half_width, pair_games}``."""
    deals: int = 0
    games: int = 0
    resamples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": list(self.anchor) if self.anchor else None,
            "elo": self.elo,
            "range95": self.range95,
            "order": self.order,
            "gaps": self.gaps,
            "deals": self.deals,
            "games": self.games,
            "resamples": self.resamples,
        }


def ladder_report(
    games: Iterable[Game],
    anchor: tuple[str, float] | None = None,
    *,
    resamples: int = 1000,
    seed: int = 0,
    order: Sequence[str] | None = None,
    **fit_kw: Any,
) -> LadderReport:
    """Fits the ratings, then bootstraps over deals (``resamples`` refits on deals drawn
    with replacement) for the 95 % percentile range of every rating and of every adjacent
    gap in ``order`` (default: the fitted order, weakest first). Gap ranges come from the
    same resamples as the ratings, so a gap's range is narrower than its two endpoints'
    ranges suggest when the pair played each other directly."""
    games = list(games)
    counts = PairCounts.from_games(games)
    wins, n = counts.total()
    if not connected(n):
        raise ValueError("the players do not form one connected group of pairings")
    point = strengths_to_elo(fit_strengths(wins, n, **fit_kw), counts.names, anchor)
    names = counts.names
    order = list(order) if order is not None else sorted(names, key=lambda x: point[x])
    rng = np.random.default_rng(seed)
    d = len(counts.deals)
    samples = np.zeros((resamples, len(names)))
    for r in range(resamples):
        idx = rng.integers(0, d, size=d)
        w, g = counts.total(idx)
        if not connected(g):
            samples[r] = [point[x] for x in names]  # a resample that broke the graph
            continue
        e = strengths_to_elo(fit_strengths(w, g, **fit_kw), names, anchor)
        samples[r] = [e[x] for x in names]
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    range95 = {x: [float(lo[i]), float(hi[i])] for i, x in enumerate(names)}
    gaps = []
    col = {x: i for i, x in enumerate(names)}
    for a, b in zip(order[:-1], order[1:], strict=False):
        diff = samples[:, col[b]] - samples[:, col[a]]
        glo, ghi = np.percentile(diff, [2.5, 97.5])
        gaps.append(
            {
                "lower": a,
                "upper": b,
                "gap": point[b] - point[a],
                "range95": [float(glo), float(ghi)],
                "half_width": float(ghi - glo) / 2,
                "pair_games": int(n[col[a], col[b]]),
            }
        )
    return LadderReport(
        anchor=anchor,
        elo=point,
        range95=range95,
        order=order,
        gaps=gaps,
        deals=d,
        games=len(games),
        resamples=resamples,
    )


def deals_for_half_width(deals: int, half_width: float, target: float) -> int:
    """Deals needed for a gap's 95 % half-width to shrink from ``half_width`` (measured on
    ``deals`` deals) to ``target``, by the 1/sqrt(n) rule."""
    if half_width <= target:
        return deals
    return int(math.ceil(deals * (half_width / target) ** 2))


__all__ = [
    "SCALE",
    "Game",
    "LadderReport",
    "PairCounts",
    "connected",
    "deals_for_half_width",
    "expected_score",
    "fit_elo",
    "fit_strengths",
    "ladder_report",
    "pairwise_results",
    "read_games",
    "strengths_to_elo",
]
