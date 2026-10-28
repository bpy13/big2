"""Duplicate tournaments between agents.

Card luck dominates single Big Two games, so comparisons use **duplicate deals**:
every deal is played four times with the lineup rotated one seat each time, so
every agent plays every seat — and therefore every hand — of every deal.

Statistics are computed over *deals* (each deal's four games averaged per agent),
which treats the four correlated games of a deal as one sample, so the
confidence intervals are honest.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from big2.agents import Agent, make_agent
from big2.games.big2 import Big2State, Rules


@dataclass
class AgentSummary:
    """Results for one agent spec across the tournament."""

    agent: str
    games: int
    mean_score: float
    score_ci95: float
    win_rate: float
    win_ci95: float
    seconds_per_move: float


@dataclass
class TournamentResult:
    lineup: list[str]
    deals: int
    rules: dict[str, Any]
    seed: int
    summaries: list[AgentSummary]
    # One entry per deal: {agent: mean score over that deal's games}
    per_deal: list[dict[str, float]] = field(repr=False, default_factory=list)
    seconds: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def play_game(
    state: Big2State, agents: list[Agent], move_time: list[float] | None = None
) -> Big2State:
    """Plays ``state`` to the end, seat ``i`` controlled by ``agents[i]``."""
    while not state.is_terminal:
        p = state.current_player
        assert p is not None
        t0 = time.perf_counter()
        action = agents[p].act(state)
        if move_time is not None:
            move_time[p] += time.perf_counter() - t0
        state.apply(action)
    return state


def _play_deals(
    lineup: list[str], deal_seeds: list[int], rules: Rules
) -> list[dict[str, list[float]]]:
    """Worker: plays the given deals; returns per-deal raw results."""
    out = []
    for ds in deal_seeds:
        deal = rules.new_deal(ds)
        rec: dict[str, list[float]] = defaultdict(list)
        for rot in range(4):
            seat_specs = [lineup[(s - rot) % 4] for s in range(4)]
            agents = [make_agent(spec, seed=ds * 4 + rot) for spec in seat_specs]
            move_time = [0.0] * 4
            final = play_game(deal.copy(), agents, move_time)
            moves = [0] * 4
            for p, _ in final.history:
                moves[p] += 1
            scores = final.scores
            assert scores is not None
            for s, spec in enumerate(seat_specs):
                rec[f"score|{spec}"].append(scores[s])
                rec[f"win|{spec}"].append(1.0 if final.winner == s else 0.0)
                rec[f"time|{spec}"].append(move_time[s])
                rec[f"moves|{spec}"].append(moves[s])
        out.append(dict(rec))
    return out


def _mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, math.inf
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def run_tournament(
    lineup: list[str],
    deals: int,
    *,
    seed: int = 0,
    workers: int = 1,
    rules: Rules | None = None,
) -> TournamentResult:
    """Runs a duplicate tournament.

    Args:
        lineup: four agent specs (see :func:`big2.agents.make_agent`), e.g.
            ``["ismcts:iterations=1000", "greedy", "greedy", "greedy"]``.
        deals: number of deals; each is played 4 times (rotated).
        seed: base seed for deals and agents.
        workers: parallel processes.
        rules: rule variants.
    """
    if len(lineup) != 4:
        raise ValueError("a Big Two lineup has exactly 4 agents")
    for spec in set(lineup):
        make_agent(spec)  # fail fast on bad specs
    rules = rules or Rules()
    seeds = [seed * 1_000_000 + d for d in range(deals)]
    t0 = time.perf_counter()
    if workers <= 1:
        raw = _play_deals(lineup, seeds, rules)
    else:
        chunks = [seeds[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_play_deals, [lineup] * workers, chunks, [rules] * workers))
        raw = [r for part in parts for r in part]

    summaries = []
    per_deal: list[dict[str, float]] = [{} for _ in raw]
    for spec in dict.fromkeys(lineup):  # unique, in lineup order
        deal_scores = [sum(r[f"score|{spec}"]) / len(r[f"score|{spec}"]) for r in raw]
        deal_wins = [sum(r[f"win|{spec}"]) / len(r[f"win|{spec}"]) for r in raw]
        for i, x in enumerate(deal_scores):
            per_deal[i][spec] = x
        total_time = sum(sum(r[f"time|{spec}"]) for r in raw)
        total_moves = sum(sum(r[f"moves|{spec}"]) for r in raw)
        m, ci = _mean_ci(deal_scores)
        w, wci = _mean_ci(deal_wins)
        summaries.append(
            AgentSummary(
                agent=spec,
                games=sum(len(r[f"score|{spec}"]) for r in raw),
                mean_score=m,
                score_ci95=ci,
                win_rate=w,
                win_ci95=wci,
                seconds_per_move=total_time / max(total_moves, 1),
            )
        )
    return TournamentResult(
        lineup=lineup,
        deals=deals,
        rules=rules.kwargs(),
        seed=seed,
        summaries=summaries,
        per_deal=per_deal,
        seconds=time.perf_counter() - t0,
    )
