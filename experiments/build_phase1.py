"""Build-run phase 1 experiments. Results: docs/study-results/build-phase1.md.

Usage::

    python experiments/build_phase1.py b1 --games 10000 --seeds 1,2,3 --workers 14
    python experiments/build_phase1.py structured-rollout --deals 300 --workers 14

One subcommand per experiment (plan ``docs/plans/2026-10-29-big2-bot-build.md``):

* ``structured-rollout`` (Task 5, §1): ISMCTS-1000 with the planner-aware
  ``StructuredGreedy`` rollout against ISMCTS-1000 with today's greedy rollout,
  A B A B on duplicate deals; verdict against the +1.0 bar (A4).
* ``b1`` (Task 8, §3): ISMCTS-1000 against three Patwa-greedy and three
  Patwa-smart bots, under Patwa's protocol (seat-random games, ``sum_of_others``
  scoring), to set beside his PPO agent's 64.8 % and 43.5 % win rates.

Later tasks add ``rebaseline`` (Task 6, §2) and ``ladder`` (Task 9, §4) here.

Every subcommand writes to ``runs/<job>/`` (git-ignored): one JSON per finished
cell, ``progress.json`` after each cell, ``results.json`` and ``table.md`` at the
end. Rerunning the same command resumes: finished cells are read back, not replayed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from big2.eval import TournamentResult, run_tournament

ROOT = Path(__file__).resolve().parent.parent
ISMCTS_1000 = "ismcts:iterations=1000"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _load_result(path: Path) -> TournamentResult:
    from big2.eval.tournament import AgentSummary

    d = json.loads(path.read_text())
    d["summaries"] = [AgentSummary(**s) for s in d["summaries"]]
    return TournamentResult(**d)


# b1 ---------------------------------------------------------------------------

#: Patwa (2026)'s PPO agent (entropy 0.05): win rate against three copies of each bot.
B1_PATWA_PPO = {"patwa-greedy": 0.648, "patwa-smart": 0.435}


def _b1_cell(
    candidate: str, opponent: str, games: int, seed: int, workers: int
) -> TournamentResult:
    return run_tournament(
        [candidate, opponent, opponent, opponent],
        games,
        seed=seed,
        workers=workers,
        score_as="sum_of_others",
        seat_random=True,
    )


def _pooled(results: list[TournamentResult], candidate: str) -> dict[str, float]:
    """Pools a candidate's games over seeds. In seat-random mode the candidate plays
    one game per deal, so the win interval is a binomial one over games."""
    scores = [d[candidate] for r in results for d in r.per_deal]
    n = len(scores)
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    wins = sum(s.win_rate * s.games for r in results for s in r.summaries if s.agent == candidate)
    p = wins / n
    return {
        "games": n,
        "win_rate": p,
        "win_ci95": 1.96 * math.sqrt(p * (1 - p) / n),
        "mean_score": mean,
        "score_ci95": 1.96 * math.sqrt(var / n),
    }


def b1(args: argparse.Namespace) -> None:
    """ISMCTS-1000 vs 3 Patwa-greedy and vs 3 Patwa-smart (build-phase1.md §3)."""
    out = ROOT / "runs" / "p1_b1"
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    opponents = list(B1_PATWA_PPO)
    cells = [(opp, seed) for opp in opponents for seed in seeds]
    done: dict[tuple[str, int], TournamentResult] = {}
    t0 = time.time()
    for opp, seed in cells:
        path = out / f"{opp}_s{seed}_g{args.games}.json"
        if path.exists():
            done[(opp, seed)] = _load_result(path)
        else:
            r = _b1_cell(args.candidate, opp, args.games, seed, args.workers)
            r.save(path)
            done[(opp, seed)] = r
        _write_json(
            out / "progress.json",
            {
                "done": [f"{o}_s{s}" for o, s in done],
                "next": [f"{o}_s{s}" for o, s in cells if (o, s) not in done],
                "minutes_this_run": round((time.time() - t0) / 60, 1),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print(f"done {opp} seed {seed}", flush=True)

    rows: list[dict[str, Any]] = []
    lines = [
        f"Candidate `{args.candidate}`, {args.games:,} seat-random games per seed, seeds "
        f"{args.seeds}, `sum_of_others` scoring, last-card duty on, no four-card plays.",
        "",
        "| Opponents (×3) | Seed | Games | Win rate (95% CI) | Mean score (95% CI) "
        "| Patwa's PPO win rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for opp in opponents:
        per_seed = [done[(opp, s)] for s in seeds]
        for s, r in zip(seeds, per_seed, strict=True):
            c = _pooled([r], args.candidate)
            rows.append({"opponent": opp, "seed": s, **c})
            lines.append(
                f"| {opp} | {s} | {c['games']:,} | {c['win_rate']:.1%} ± {c['win_ci95']:.1%} "
                f"| {c['mean_score']:+.2f} ± {c['score_ci95']:.2f} | |"
            )
        c = _pooled(per_seed, args.candidate)
        rows.append({"opponent": opp, "seed": "pooled", **c, "patwa_ppo": B1_PATWA_PPO[opp]})
        lines.append(
            f"| **{opp}** | **pooled** | {c['games']:,} "
            f"| **{c['win_rate']:.1%} ± {c['win_ci95']:.1%}** "
            f"| {c['mean_score']:+.2f} ± {c['score_ci95']:.2f} | {B1_PATWA_PPO[opp]:.1%} |"
        )
    table = "\n".join(lines) + "\n"
    _write_json(out / "results.json", {"candidate": args.candidate, "rows": rows})
    (out / "table.md").write_text(table, encoding="utf-8")
    print(table, flush=True)


# structured-rollout -----------------------------------------------------------

#: A4 bar: the structured rollout must gain at least this many points per game.
A4_BAR = 1.0


def structured_rollout(args: argparse.Namespace) -> None:
    """ISMCTS with the structured rollout vs today's greedy rollout, A B A B
    (build-phase1.md §1). Equivalent to ``big2 tournament "<S>" "<B>" "<S>" "<B>"
    --deals N --out runs/p1_structured/results.json``."""
    out = ROOT / "runs" / "p1_structured"
    out.mkdir(parents=True, exist_ok=True)
    structured = f"ismcts:iterations={args.iterations},rollout=structured"
    baseline = f"ismcts:iterations={args.iterations}"
    path = out / "results.json"
    t0 = time.time()
    if path.exists():
        r = _load_result(path)
    else:
        _write_json(
            out / "progress.json",
            {
                "done": [],
                "next": [f"{args.deals} deals A B A B: {structured} vs {baseline}"],
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "one tournament call; it writes results.json when it ends",
            },
        )
        r = run_tournament(
            [structured, baseline, structured, baseline],
            args.deals,
            seed=args.seed,
            workers=args.workers,
        )
        r.save(path)
    _write_json(
        out / "progress.json",
        {
            "done": [f"{args.deals} deals A B A B"],
            "next": [],
            "minutes_this_run": round((time.time() - t0) / 60, 1),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    s = {x.agent: x for x in r.summaries}[structured]
    b = {x.agent: x for x in r.summaries}[baseline]
    # Paired difference per deal (structured seat minus baseline seat): the two sides
    # sum to zero in A B A B, so it is twice the structured mean; reported for clarity.
    diffs = [d[structured] - d[baseline] for d in r.per_deal]
    n = len(diffs)
    dm = sum(diffs) / n
    dci = 1.96 * math.sqrt(sum((x - dm) ** 2 for x in diffs) / (n - 1) / n)
    passed = s.mean_score >= A4_BAR
    verdict = (
        f"passes the +{A4_BAR:.1f} bar"
        if passed
        else f"fails the +{A4_BAR:.1f} bar (shortfall {A4_BAR - s.mean_score:.2f})"
    )
    lines = [
        f"{r.deals} duplicate deals, lineup A B A B, seed {r.seed}, "
        f"{r.seconds / 60:.0f} min wall time.",
        "",
        "| Agent | Games | Mean score per game (95% CI) | Win rate (95% CI) | s per move |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in (s, b):
        lines.append(
            f"| `{x.agent}` | {x.games:,} | {x.mean_score:+.2f} ± {x.score_ci95:.2f} "
            f"| {x.win_rate:.1%} ± {x.win_ci95:.1%} | {x.seconds_per_move:.3f} |"
        )
    lines += [
        "",
        f"Paired difference per deal (structured seat − greedy-rollout seat): "
        f"{dm:+.2f} ± {dci:.2f}.",
        f"Verdict (A4): the structured side's mean {s.mean_score:+.2f} {verdict}.",
    ]
    table = "\n".join(lines) + "\n"
    _write_json(
        out / "summary.json",
        {
            "structured": structured,
            "baseline": baseline,
            "deals": r.deals,
            "mean": s.mean_score,
            "ci95": s.score_ci95,
            "win_rate": s.win_rate,
            "win_ci95": s.win_ci95,
            "paired_diff": dm,
            "paired_diff_ci95": dci,
            "bar": A4_BAR,
            "passed": passed,
        },
    )
    (out / "table.md").write_text(table, encoding="utf-8")
    print(table, flush=True)


# main -------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "b1": b1,
    "structured-rollout": structured_rollout,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("b1", help="ISMCTS vs Patwa's bots under his protocol (Task 8)")
    p.add_argument("--candidate", default=ISMCTS_1000)
    p.add_argument("--games", type=int, default=10_000, help="games per opponent and seed")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--workers", type=int, default=14)

    p = sub.add_parser(
        "structured-rollout",
        help="ISMCTS with the structured rollout vs the greedy rollout, A B A B (Task 5)",
    )
    p.add_argument("--deals", type=int, default=300)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=14)

    args = ap.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
