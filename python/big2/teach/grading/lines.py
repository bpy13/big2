"""Grade lines and the trivial flag from labelled decisions (plan Task 43), game-agnostic.

A *decision* is the shape every game's ``decisions.jsonl`` writes (see
:mod:`big2.teach.grading.losses`): ``band``, ``forced``, ``flat``, ``loss`` (points the move
played lost against the reference's best move), ``loss_se`` (its standard error) and ``moves``
(``[{action, loss, loss_se, ...}]``). Nothing here imports the rest of ``big2``.

* **Trivial** (:func:`trivial`): a decision is ``forced`` (one legal move) or ``flat`` (every
  evaluated move's loss is inside the reference's noise or under 0.2 points). Trivial decisions
  get no grade word and count in no score: about half of all Big Two decisions.
* **Lines** (:func:`derive_lines`): upper limits of great / good / inaccuracy / mistake in
  points lost; beyond the last line a move is a blunder. The rule of Task 43: great = the
  reference's own noise (median 95 % half-width of a paired loss, at least 0.2); good = the loss
  the top rung exceeds on ``good_share`` of its non-trivial decisions; inaccuracy = the loss the
  top rung exceeds on ``inaccuracy_share``; mistake = the loss the app's ``medium`` rung exceeds
  on ``mistake_share``. Lines must rise, each at least ``ratio`` times the one below; a line
  that would break this is lifted to ``ratio`` times the one below. Optional rounding to the
  nearest step (half points) so players see round numbers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from big2.teach.grading.losses import D102_LINES, WORDS, percentile, word

LINE_NAMES: tuple[str, ...] = ("great", "good", "inaccuracy", "mistake")
"""The four lines, each the upper limit of that word; above the last: ``blunder``."""

FLAT_LOSS = 0.2
"""Task 42's flat rule: every move's loss under this, or inside the noise band."""


def trivial(decision: Mapping[str, Any]) -> bool:
    """``True`` when the decision gets no grade: forced (one legal move) or flat (nothing at
    stake: every evaluated move's loss is inside the noise band or under ``FLAT_LOSS``)."""
    return bool(decision.get("forced")) or bool(decision.get("flat"))


def is_flat(moves: Sequence[Mapping[str, Any]], flat_loss: float = FLAT_LOSS) -> bool:
    """Task 42's flat rule on a decision's evaluated moves: every move's loss is under
    ``flat_loss`` or its 95 % range (``loss ± 1.96 · loss_se``) covers 0."""
    return all(m["loss"] < flat_loss or m["loss"] - 1.96 * m["loss_se"] <= 0 for m in moves)


def graded(decisions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The non-trivial decisions."""
    return [dict(d) for d in decisions if not trivial(d)]


def losses_by_band(decisions: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Band -> ascending losses of the move played, over non-trivial decisions."""
    out: dict[str, list[float]] = {}
    for d in decisions:
        if not trivial(d):
            out.setdefault(d["band"], []).append(float(d["loss"]))
    return {b: sorted(v) for b, v in out.items()}


def exceeded_on(sorted_losses: Sequence[float], share: float) -> float:
    """The loss exceeded on ``share`` of the decisions (the ``1 - share`` percentile)."""
    if not 0 < share < 1:
        raise ValueError("share must be between 0 and 1")
    v = percentile(sorted_losses, 100 * (1 - share))
    if v is None:
        raise ValueError("no losses")
    return v


def noise_half_width(decisions: Iterable[Mapping[str, Any]]) -> float:
    """The reference's noise: median over non-trivial decisions of the 95 % half-width
    (``1.96 · loss_se``) of the played move's loss, taken where the move was not the best
    (the best's own loss has no standard error)."""
    hw = sorted(1.96 * float(d["loss_se"]) for d in decisions if not trivial(d) and d["loss"] > 0)
    v = percentile(hw, 50)
    if v is None:
        raise ValueError("no non-best decisions to measure the noise on")
    return v


def round_to(x: float, step: float) -> float:
    """``x`` rounded to the nearest multiple of ``step`` (``step`` 0 = no rounding)."""
    if step <= 0:
        return x
    return round(x / step) * step


def ceil_to(x: float, step: float) -> float:
    """``x`` rounded up to a multiple of ``step`` (``step`` 0 = no rounding)."""
    if step <= 0:
        return x
    return math.ceil(x / step - 1e-9) * step


def enforce_monotone(lines: Sequence[float], ratio: float) -> list[float]:
    """Lifts each line to at least ``ratio`` times the one below (lines must rise)."""
    out: list[float] = []
    for x in lines:
        if out and x < ratio * out[-1]:
            x = ratio * out[-1]
        out.append(x)
    return out


def derive_lines(
    decisions: Iterable[Mapping[str, Any]],
    top_band: str,
    medium_band: str,
    *,
    good_share: float = 0.10,
    inaccuracy_share: float = 0.03,
    mistake_share: float = 0.03,
    min_great: float = 0.2,
    ratio: float = 1.5,
    step: float = 0.0,
) -> dict[str, Any]:
    """Task 43's line rule. Returns ``{"lines": [great, good, inaccuracy, mistake], "raw":
    the four values before rounding and the monotone lift, "noise": the reference's noise
    half-width, "inputs": the shares used}``."""
    decisions = list(decisions)
    by_band = losses_by_band(decisions)
    for b in (top_band, medium_band):
        if b not in by_band:
            raise KeyError(f"band {b!r} has no non-trivial decisions")
    noise = noise_half_width(decisions)
    raw = [
        max(min_great, noise),
        exceeded_on(by_band[top_band], good_share),
        exceeded_on(by_band[top_band], inaccuracy_share),
        exceeded_on(by_band[medium_band], mistake_share),
    ]
    lines = [round_to(x, step) for x in raw]
    # a lifted line is rounded up, so the ratio still holds after rounding
    lines = [ceil_to(x, step) for x in enforce_monotone(lines, ratio)]
    return {
        "lines": lines,
        "raw": raw,
        "noise": noise,
        "inputs": {
            "top_band": top_band,
            "medium_band": medium_band,
            "good_share": good_share,
            "inaccuracy_share": inaccuracy_share,
            "mistake_share": mistake_share,
            "min_great": min_great,
            "ratio": ratio,
            "step": step,
        },
    }


def check_lines(lines: Sequence[float], ratio: float = 1.5) -> None:
    """Raises ``ValueError`` unless the lines rise, each at least ``ratio`` times the one below."""
    if len(lines) != len(LINE_NAMES):
        raise ValueError(f"expected {len(LINE_NAMES)} lines, got {len(lines)}")
    if lines[0] <= 0:
        raise ValueError("the great line must be positive")
    for lo, hi in zip(lines, lines[1:], strict=False):
        if hi < ratio * lo - 1e-9:
            raise ValueError(f"line {hi} is under {ratio} x {lo}")


def word_shares(
    decisions: Iterable[Mapping[str, Any]], lines: Sequence[float] = D102_LINES
) -> dict[str, dict[str, float]]:
    """Band -> share of each grade word over its non-trivial decisions under ``lines``."""
    counts: dict[str, dict[str, int]] = {}
    for d in decisions:
        if trivial(d):
            continue
        c = counts.setdefault(d["band"], dict.fromkeys(WORDS, 0))
        c[word(float(d["loss"]), lines)] += 1
    return {b: {w: n / max(sum(c.values()), 1) for w, n in c.items()} for b, c in counts.items()}


def same_word_share(
    decisions: Iterable[Mapping[str, Any]], lines: Sequence[float], key: str = "repeat"
) -> tuple[float | None, int]:
    """Share of relabelled non-trivial decisions (``key`` = ``repeat`` or ``rules``) whose
    word under ``lines`` is unchanged, and how many were checked."""
    n = same = 0
    for d in decisions:
        if key in d and not trivial(d):
            n += 1
            same += word(float(d[key]["loss"]), lines) == word(float(d["loss"]), lines)
    return (same / n if n else None), n


__all__ = [
    "FLAT_LOSS",
    "LINE_NAMES",
    "ceil_to",
    "check_lines",
    "derive_lines",
    "enforce_monotone",
    "exceeded_on",
    "graded",
    "is_flat",
    "losses_by_band",
    "noise_half_width",
    "round_to",
    "same_word_share",
    "trivial",
    "word_shares",
]
