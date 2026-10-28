"""Head-to-head matches between two agents: the evaluation harness.

One candidate against one opponent on duplicate deals (each deal 4×, seats
rotated), in one or both lineups:

* ``"1+3"``: the candidate in one seat, the opponent in the other three;
* ``"ABAB"``: candidate and opponent in alternate seats.

Every match gives one report of the same shape (:class:`MatchReport`):
``{"candidate", "opponent", "lineup", "deals", "mean", "ci95", "win_rate",
"win_ci95", "seconds_per_move"}`` plus the paired difference. ``mean`` is the
candidate's score per game; the paired difference is, per deal, the candidate's
mean score minus the opponent's, so each deal's card luck cancels. Intervals are
95 % percentile bootstrap intervals over deals (the four games of a deal count as
one sample), given as ``[low, high]``.

This is the ``eval-harness`` seed of the strategy proposal (§8). The CLI is
``big2 match <candidate> <opponent> --deals N --lineups both``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from big2.agents import make_agent
from big2.eval.tournament import _play_deals
from big2.games.big2 import Rules

LINEUPS = ("1+3", "ABAB")
BOOTSTRAP_RESAMPLES = 2000

Records = list[dict[str, list[float]]]
"""Per deal, the raw results of :func:`big2.eval.tournament._play_deals`: for each
agent spec ``score|<spec>``, ``win|<spec>``, ``time|<spec>`` and ``moves|<spec>``,
one entry per seat the agent held in each of the deal's games."""


def lineup_seats(candidate: str, opponent: str, lineup: str) -> list[str]:
    """The four seat specs of a lineup (rotated by the duplicate format)."""
    if lineup == "1+3":
        return [candidate, opponent, opponent, opponent]
    if lineup == "ABAB":
        return [candidate, opponent, candidate, opponent]
    raise ValueError(f"lineup must be one of {LINEUPS}, not {lineup!r}")


def parse_lineups(text: str) -> list[str]:
    """``"both"`` → both lineups; otherwise a comma list of ``1+3`` and ``ABAB``."""
    if text == "both":
        return list(LINEUPS)
    names = [x.strip().upper().replace(" ", "") for x in text.split(",") if x.strip()]
    for n in names:
        if n not in LINEUPS:
            raise ValueError(f"unknown lineup {n!r} (1+3, ABAB or both)")
    return names


def bootstrap_ci(
    values: Sequence[float] | np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> list[float]:
    """95 % percentile bootstrap interval ``[low, high]`` of the mean of ``values``."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return [-float("inf"), float("inf")]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


@dataclass
class MatchReport:
    """One match in one lineup. Scores are points per game for the candidate."""

    candidate: str
    opponent: str
    lineup: str
    deals: int
    mean: float
    ci95: list[float]
    win_rate: float
    win_ci95: list[float]
    seconds_per_move: float
    opponent_mean: float = 0.0
    opponent_win_rate: float = 0.0
    paired_diff: float = 0.0
    """Per deal: candidate's mean score minus the opponent's; averaged over deals."""
    paired_ci95: list[float] = field(default_factory=lambda: [0.0, 0.0])
    opponent_seconds_per_move: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deal_means(records: Records, key: str) -> np.ndarray:
    return np.array([sum(r[key]) / len(r[key]) for r in records], dtype=np.float64)


def summarise_match(
    candidate: str, opponent: str, lineup: str, records: Records, seconds: float = 0.0
) -> MatchReport:
    """Turns per-deal records into a :class:`MatchReport` (bootstrap over deals)."""
    if not records:
        raise ValueError("no deals to summarise")
    cand = _deal_means(records, f"score|{candidate}")
    opp = _deal_means(records, f"score|{opponent}")
    cwin = _deal_means(records, f"win|{candidate}")
    owin = _deal_means(records, f"win|{opponent}")
    diff = cand - opp

    def per_move(spec: str) -> float:
        secs = sum(sum(r[f"time|{spec}"]) for r in records)
        moves = sum(sum(r[f"moves|{spec}"]) for r in records)
        return secs / max(moves, 1)

    return MatchReport(
        candidate=candidate,
        opponent=opponent,
        lineup=lineup,
        deals=len(records),
        mean=float(cand.mean()),
        ci95=bootstrap_ci(cand, seed=1),
        win_rate=float(cwin.mean()),
        win_ci95=bootstrap_ci(cwin, seed=2),
        seconds_per_move=per_move(candidate),
        opponent_mean=float(opp.mean()),
        opponent_win_rate=float(owin.mean()),
        paired_diff=float(diff.mean()),
        paired_ci95=bootstrap_ci(diff, seed=3),
        opponent_seconds_per_move=per_move(opponent),
        seconds=seconds,
    )


def play_records(
    seats: list[str], deal_seeds: list[int], workers: int = 1, rules: Rules | None = None
) -> Records:
    """Plays the deals in duplicate; returns their records in ``deal_seeds`` order."""
    rules = rules or Rules()
    if workers <= 1 or len(deal_seeds) <= 1:
        return _play_deals(seats, deal_seeds, rules)
    workers = min(workers, len(deal_seeds))
    chunks = [deal_seeds[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_play_deals, [seats] * workers, chunks, [rules] * workers))
    by_seed = {
        s: r
        for chunk, part in zip(chunks, parts, strict=True)
        for s, r in zip(chunk, part, strict=True)
    }
    return [by_seed[s] for s in deal_seeds]


def play_match(
    candidate: str,
    opponent: str,
    lineup: str,
    deals: int,
    *,
    seed: int = 0,
    workers: int = 1,
    rules: Rules | None = None,
    cache_dir: str | Path | None = None,
    block: int = 250,
) -> MatchReport:
    """Plays one match in one lineup and summarises it.

    Deal ``d`` uses the seed ``seed * 1_000_000 + d`` (as
    :func:`big2.eval.run_tournament`). With ``cache_dir``, deals are played in
    blocks of ``block``; each finished block is saved there as JSON and read back
    on a rerun, so an interrupted match resumes.
    """
    if candidate == opponent:
        raise ValueError("candidate and opponent must be different specs")
    seats = lineup_seats(candidate, opponent, lineup)
    for spec in (candidate, opponent):
        make_agent(spec)  # fail fast on bad specs
    seeds = [seed * 1_000_000 + d for d in range(deals)]
    t0 = time.perf_counter()
    if cache_dir is None:
        records = play_records(seats, seeds, workers, rules)
        return summarise_match(candidate, opponent, lineup, records, time.perf_counter() - t0)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    seconds = 0.0
    for a in range(0, deals, block):
        b = min(a + block, deals)
        path = cache / f"block_s{seed}_d{a}-{b}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["seats"] != seats:
                raise ValueError(f"{path} holds another match ({saved['seats']})")
        else:
            tb = time.perf_counter()
            recs = play_records(seats, seeds[a:b], workers, rules)
            saved = {"seats": seats, "deals": [a, b], "seconds": time.perf_counter() - tb}
            saved["records"] = recs
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved))
            tmp.replace(path)
        records += saved["records"]
        seconds += saved["seconds"]
    return summarise_match(candidate, opponent, lineup, records, seconds)


def match(
    candidate: str,
    opponent: str,
    deals: int,
    lineups: Sequence[str] = LINEUPS,
    **kwargs: Any,
) -> list[MatchReport]:
    """Plays :func:`play_match` in each lineup; keyword options pass through."""
    out = []
    for lineup in lineups:
        sub = dict(kwargs)
        if sub.get("cache_dir") is not None:
            sub["cache_dir"] = Path(sub["cache_dir"]) / lineup.replace("+", "p")
        out.append(play_match(candidate, opponent, lineup, deals, **sub))
    return out
