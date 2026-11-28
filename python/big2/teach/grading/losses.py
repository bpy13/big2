"""Loss distributions per Elo band from ``decisions.jsonl`` (plan Task 42), game-agnostic.

Reads the decision shape every game writes (one JSON object a line): ``band``,
``rung_elo``, ``forced``, ``flat``, ``loss`` (points the move played lost to the reference's
best), optionally ``repeat`` and ``rules`` (the same decision labelled again with a new seed
or another belief: ``{"loss": ...}``). Nothing here imports the rest of ``big2``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

#: D102 lines in points: upper limits of great, good, inaccuracy, mistake (beyond: blunder).
D102_LINES: tuple[float, ...] = (0.2, 0.69, 2.75, 6.86)
WORDS: tuple[str, ...] = ("great", "good", "inaccuracy", "mistake", "blunder")
PERCENTILES: tuple[int, ...] = (50, 75, 90, 97, 99)


def read_decisions(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every decision of the given ``.jsonl`` files, in order."""
    out: list[dict[str, Any]] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def word(loss: float, lines: Sequence[float] = D102_LINES) -> str:
    """The grade word of ``loss`` under ``lines`` (upper limit of each word)."""
    for limit, w in zip(lines, WORDS, strict=False):
        if loss <= limit:
            return w
    return WORDS[len(lines)]


def percentile(sorted_xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolated ``p``-th percentile of an ascending list (``None`` if empty)."""
    if not sorted_xs:
        return None
    k = (len(sorted_xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def band_table(
    decisions: Iterable[dict[str, Any]], lines: Sequence[float] = D102_LINES
) -> dict[str, dict[str, Any]]:
    """Per band: decision counts, the share forced and flat, and the distribution of the
    played move's loss over non-forced, non-flat decisions (mean, percentiles, share above
    each line); also how often a second labelling keeps the word (``repeat``, ``rules``)."""
    acc: dict[str, dict[str, Any]] = {}
    for r in decisions:
        a = acc.setdefault(
            r["band"],
            {"rung_elo": r.get("rung_elo"), "deals": set(), "decisions": 0, "forced": 0,
             "flat": 0, "losses": [], "repeat": [0, 0], "rules": [0, 0]},
        )
        a["decisions"] += 1
        a["deals"].add(r.get("deal_id"))
        if r["forced"]:
            a["forced"] += 1
            continue
        if r["flat"]:
            a["flat"] += 1
        else:
            a["losses"].append(float(r["loss"]))
        for key in ("repeat", "rules"):
            if key in r:
                a[key][0] += 1
                a[key][1] += word(r[key]["loss"], lines) == word(r["loss"], lines)
    out = {}
    for band, a in acc.items():
        ls = sorted(a["losses"])
        n = a["decisions"]
        out[band] = {
            "rung_elo": a["rung_elo"],
            "deals": len(a["deals"]),
            "decisions": n,
            "forced_share": a["forced"] / n,
            "flat_share": a["flat"] / max(n - a["forced"], 1),
            "graded": len(ls),
            "mean": sum(ls) / len(ls) if ls else None,
            "percentiles": {str(p): percentile(ls, p) for p in PERCENTILES},
            "share_above": {str(x): (sum(v > x for v in ls) / len(ls) if ls else None)
                            for x in lines},
            "repeat_checked": a["repeat"][0],
            "repeat_same_word": a["repeat"][1] / a["repeat"][0] if a["repeat"][0] else None,
            "rules_checked": a["rules"][0],
            "rules_same_word": a["rules"][1] / a["rules"][0] if a["rules"][0] else None,
        }
    return out


def overall_share(table: dict[str, dict[str, Any]], key: str) -> float | None:
    """A ``*_same_word`` share pooled over bands, weighted by decisions checked."""
    checked = key.replace("_same_word", "_checked")
    n = sum(t[checked] for t in table.values())
    if not n:
        return None
    return sum(t[key] * t[checked] for t in table.values() if t[key] is not None) / n


def histogram(losses: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Counts of ``losses`` in ``[edges[i], edges[i+1])``; the last bin is open above."""
    counts = [0] * len(edges)
    for x in losses:
        i = max((j for j, e in enumerate(edges) if x >= e), default=0)
        counts[i] += 1
    return counts


__all__ = [
    "D102_LINES", "PERCENTILES", "WORDS", "band_table", "histogram", "overall_share",
    "percentile", "read_decisions", "word",
]
