"""Convention-gap check (plan Task 30): where do human players disagree with the solver for
reasons that are not lack of skill?

A game log in the hk-games review shape is a list of reviewed decisions, each
``{"position", "played", "best", "loss", "rating", "reasons": [{"key", "args"}]}``
(``engine-interface.md`` §5 ``ReviewItem`` plus the player's ``rating``). A decision
*disagrees* when the player did not play the solver's best move (``played != best``, or
``loss`` above ``loss_eps`` when ``best`` is missing).

Decisions are grouped by the set of reason keys attached to them, then by rating band. For a
key set, a disagreement rate that falls as rating rises looks like skill: stronger players
learn it. A rate that stays flat (or rises) across bands looks like a **convention**: Hong
Kong tables play it another way at every level, so the solver's answer may be "right" for
its own model but not how the table plays (hard-problems §1). Those sets are tagged
``convention`` for a human to look at; they are candidates for a different opponent model or
for a reason that explains the local habit.

Real logs exist only after release, with consent (B13); until then the check runs on
synthetic logs (``python/tests/test_convention_gap.py``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Keys of a decision without reasons.
NO_REASONS = "(none)"


@dataclass(frozen=True)
class GapConfig:
    #: Width of a rating band (Elo points).
    band_width: int = 200
    #: A key set needs at least this many decisions to be judged.
    min_items: int = 30
    #: ... spread over at least this many bands with ``min_band_items`` each.
    min_bands: int = 3
    min_band_items: int = 5
    #: A set is skill-like when its disagreement rate falls by at least this much per 100
    #: rating points (0.01 = one percentage point).
    fall_per_100: float = 0.01
    #: A flat set is only a convention when players disagree this often overall.
    min_rate: float = 0.15
    #: ``loss`` above this counts as a disagreement when ``best`` is missing.
    loss_eps: float = 1e-9


@dataclass
class Band:
    low: int
    items: int
    disagreements: int

    @property
    def rate(self) -> float:
        return self.disagreements / self.items if self.items else 0.0


@dataclass
class KeySetReport:
    keys: tuple[str, ...]
    items: int
    rate: float
    #: Change of the disagreement rate per 100 rating points (weighted least squares).
    slope_per_100: float
    #: ``convention``, ``skill``, ``rare`` (flat but seldom disagreed) or ``too_few``.
    tag: str
    bands: list[Band] = field(default_factory=list)


def load_log(path: str | Path) -> list[dict[str, Any]]:
    """Reads a log: a JSON list of decisions, or an object with an ``items`` list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of reviewed decisions")
    return items


def key_set(item: dict[str, Any]) -> tuple[str, ...]:
    """The sorted, distinct reason keys of a decision."""
    keys = sorted({r["key"] for r in item.get("reasons") or []})
    return tuple(keys) if keys else (NO_REASONS,)


def disagrees(item: dict[str, Any], loss_eps: float = 1e-9) -> bool:
    if "best" in item and item["best"] is not None:
        return bool(item["played"] != item["best"])
    return float(item.get("loss") or 0.0) > loss_eps


def _slope(bands: list[Band], band_width: int) -> float:
    """Weighted least-squares slope of the rate on the band centre, per 100 points."""
    w = [b.items for b in bands]
    x = [(b.low + band_width / 2) / 100 for b in bands]
    y = [b.rate for b in bands]
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, x, strict=True)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y, strict=True)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x, strict=True))
    if sxx == 0:
        return 0.0
    return sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y, strict=True)) / sxx


def analyse(items: list[dict[str, Any]], cfg: GapConfig | None = None) -> list[KeySetReport]:
    """One report per key set, conventions first, then by number of decisions."""
    cfg = cfg or GapConfig()
    groups: dict[tuple[str, ...], dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        band = int(float(it["rating"]) // cfg.band_width) * cfg.band_width
        groups[key_set(it)][band].append(disagrees(it, cfg.loss_eps))
    reports = []
    for keys, by_band in groups.items():
        bands = [
            Band(low=lo, items=len(v), disagreements=sum(v)) for lo, v in sorted(by_band.items())
        ]
        n = sum(b.items for b in bands)
        rate = sum(b.disagreements for b in bands) / n
        usable = [b for b in bands if b.items >= cfg.min_band_items]
        if n < cfg.min_items or len(usable) < cfg.min_bands:
            reports.append(KeySetReport(keys, n, rate, 0.0, "too_few", bands))
            continue
        slope = _slope(usable, cfg.band_width)
        if slope <= -cfg.fall_per_100:
            tag = "skill"
        elif rate >= cfg.min_rate:
            tag = "convention"
        else:
            tag = "rare"
        reports.append(KeySetReport(keys, n, rate, slope, tag, bands))
    order = {"convention": 0, "skill": 1, "rare": 2, "too_few": 3}
    reports.sort(key=lambda r: (order[r.tag], -r.items))
    return reports


def format_report(reports: list[KeySetReport]) -> str:
    """A plain-text table, conventions first."""
    lines = ["tag         decisions  rate   slope/100  keys"]
    for r in reports:
        lines.append(
            f"{r.tag:<11} {r.items:>9}  {r.rate:5.2f}  {r.slope_per_100:+9.3f}  {', '.join(r.keys)}"
        )
    return "\n".join(lines)
