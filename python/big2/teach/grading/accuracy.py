"""Performance measures from labelled decisions (plan Task 43), game-agnostic.

* **APL**, average points lost per non-trivial decision (chess's average centipawn loss).
* **Accuracy**, 0-100 per game: the mean over non-trivial decisions of
  ``100 * exp(-loss / s)``, ``s`` set so a loss at the inaccuracy line scores 50
  (:func:`accuracy_scale`). A best move scores 100, a blunder near 0.
* **Rating from APL**: a least-squares fit of a rung's rating against ``log(APL)`` across the
  rungs (:func:`fit_log_apl`), with the per-game spread of APL turned into the games an
  estimate needs for a given half-width (:func:`games_for_half_width`).
* **Ordering**: Spearman rank correlation (:func:`spearman`) of a measure with the rungs'
  ratings, over rungs and over single games.

Decisions are the shape of :mod:`big2.teach.grading.losses`; a *game* is one seat of one deal
(``band``, ``deal_id``, ``seat``). Nothing here imports the rest of ``big2``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from big2.teach.grading.lines import trivial


def apl(losses: Sequence[float]) -> float | None:
    """Average points lost (``None`` for no decisions)."""
    return float(np.mean(losses)) if len(losses) else None


def accuracy_scale(inaccuracy_line: float) -> float:
    """The ``s`` of ``100 * exp(-loss / s)`` at which a loss on the inaccuracy line scores 50."""
    if inaccuracy_line <= 0:
        raise ValueError("the inaccuracy line must be positive")
    return inaccuracy_line / math.log(2)


def move_accuracy(loss: float, s: float) -> float:
    """One move's accuracy, 0-100."""
    return 100.0 * math.exp(-max(loss, 0.0) / s)


def accuracy(losses: Sequence[float], s: float) -> float | None:
    """A game's accuracy: the mean move accuracy over its non-trivial decisions."""
    if not len(losses):
        return None
    return float(np.mean([move_accuracy(x, s) for x in losses]))


def per_game(
    decisions: Iterable[Mapping[str, Any]], s: float, good_line: float
) -> list[dict[str, Any]]:
    """One row per (band, deal, seat) with its non-trivial losses, APL, accuracy and the share
    of best-or-good moves (loss at most ``good_line``, today's measure). Seats with no
    non-trivial decision are left out."""
    groups: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for d in decisions:
        if trivial(d):
            continue
        key = (d["band"], d.get("deal_id", d.get("deal")), d["seat"])
        g = groups.setdefault(
            key,
            {
                "band": key[0],
                "deal": key[1],
                "seat": key[2],
                "rung_elo": d.get("rung_elo"),
                "losses": [],
            },
        )
        g["losses"].append(float(d["loss"]))
    for g in groups.values():
        ls = g["losses"]
        g["decisions"] = len(ls)
        g["apl"] = apl(ls)
        g["accuracy"] = accuracy(ls, s)
        g["best_good_share"] = float(np.mean([x <= good_line for x in ls]))
    return list(groups.values())


def per_band(games: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Band -> games, decisions, pooled APL, mean and SD of per-game APL and accuracy, pooled
    best-or-good share."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for g in games:
        by.setdefault(g["band"], []).append(g)
    out = {}
    for band, gs in by.items():
        losses = [x for g in gs for x in g["losses"]]
        game_apl = np.array([g["apl"] for g in gs])
        game_acc = np.array([g["accuracy"] for g in gs])
        out[band] = {
            "rung_elo": gs[0].get("rung_elo"),
            "games": len(gs),
            "decisions": len(losses),
            "apl": float(np.mean(losses)),
            "apl_game_sd": float(game_apl.std(ddof=1)) if len(gs) > 1 else None,
            "accuracy": float(game_acc.mean()),
            "accuracy_game_sd": float(game_acc.std(ddof=1)) if len(gs) > 1 else None,
            "best_good_share": float(
                np.mean([g["best_good_share"] * g["decisions"] for g in gs]) * len(gs) / len(losses)
            ),
        }
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation (ties get the mean rank); ``None`` under 3 points."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        return None

    def ranks(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        order = a.argsort(kind="mergesort")
        r = np.empty(len(a))
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_log_apl(rating: Mapping[str, float], apl_by: Mapping[str, float]) -> dict[str, Any]:
    """Least squares ``rating = a + b * log(APL)`` over the bands in both maps. Returns
    ``a``, ``b``, the RMS residual and each band's fitted rating."""
    bands = [b for b in rating if b in apl_by]
    if len(bands) < 2:
        raise ValueError("need at least two bands")
    x = np.log([apl_by[b] for b in bands])
    y = np.array([rating[b] for b in bands], dtype=float)
    b, a = np.polyfit(x, y, 1)
    fitted = a + b * x
    return {
        "a": float(a),
        "b": float(b),
        "rms_residual": float(np.sqrt(np.mean((y - fitted) ** 2))),
        "fitted": {band: float(v) for band, v in zip(bands, fitted, strict=True)},
        "bands": bands,
    }


def rating_from_apl(apl_value: float, a: float, b: float) -> float:
    """The fitted rating of an APL (``apl_value`` must be positive)."""
    return a + b * math.log(apl_value)


def games_for_half_width(
    b: float, apl_mean: float, apl_game_sd: float, half_width: float = 100.0
) -> int:
    """Games whose mean APL puts the fitted rating inside ``± half_width`` (95 %): the delta
    method gives ``SE(rating) = |b| * sd / (APL * sqrt(n))``."""
    if apl_mean <= 0 or apl_game_sd < 0:
        raise ValueError("APL must be positive, its spread non-negative")
    return int(math.ceil((1.96 * abs(b) * apl_game_sd / (apl_mean * half_width)) ** 2))


__all__ = [
    "accuracy",
    "accuracy_scale",
    "apl",
    "fit_log_apl",
    "games_for_half_width",
    "move_accuracy",
    "per_band",
    "per_game",
    "rating_from_apl",
    "spearman",
]
