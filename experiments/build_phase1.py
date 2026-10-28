"""Build-run phase 1 experiments. Results: docs/study-results/build-phase1.md.

Usage::

    python experiments/build_phase1.py b1 --games 10000 --seeds 1,2,3 --workers 14
    python experiments/build_phase1.py structured-rollout --deals 300 --workers 14
    python experiments/build_phase1.py rebaseline --deals 1000 --workers 14

One subcommand per experiment (plan ``docs/plans/2026-10-29-big2-bot-build.md``):

* ``structured-rollout`` (Task 5, §1): ISMCTS-1000 with the planner-aware
  ``StructuredGreedy`` rollout against ISMCTS-1000 with today's greedy rollout,
  A B A B on duplicate deals; verdict against the +1.0 bar (A4).
* ``b1`` (Task 8, §3): ISMCTS-1000 against three Patwa-greedy and three
  Patwa-smart bots, under Patwa's protocol (seat-random games, ``sum_of_others``
  scoring), to set beside his PPO agent's 64.8 % and 43.5 % win rates.
* ``rebaseline`` (Task 6, §2): ISMCTS-1000 vs 3 greedy, vs DMC-10h (A B A B) and
  in self-play under duty-on, 1,000 duplicate deals each; self-play gives σ, the
  spread of one seat's score in one game, which scales the grade thresholds.

A later task adds ``ladder`` (Task 9, §4) here.

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


# rebaseline -------------------------------------------------------------------

#: DMC-10h, trained with last-card duty off (Task 6 notes it beside its row).
DMC_10H = "dmc:checkpoint=models/big2-dmc-selfplay-10h.pt"

#: The three tournaments of Task 6: (name, label, lineup with ``C`` = candidate).
REBASELINE_CELLS: list[tuple[str, str, list[str]]] = [
    ("vs_greedy", "vs 3 greedy (1 + 3)", ["C", "greedy", "greedy", "greedy"]),
    ("vs_dmc10h", "vs DMC-10h (A B A B)", ["C", DMC_10H, "C", DMC_10H]),
    ("selfplay", "self-play (all four seats)", ["C", "C", "C", "C"]),
]


def _play_block(
    lineup: list[str], deal_seeds: list[int], workers: int
) -> list[dict[str, list[float]]]:
    """Plays deals in duplicate (four rotations each) and returns the raw per-deal
    records of :func:`big2.eval.tournament._play_deals`, in ``deal_seeds`` order.

    The raw records keep every seat's score in every game, which a
    ``TournamentResult`` does not (it averages each agent over the deal; in
    self-play that average is always zero). Deal and agent seeds are the ones
    ``run_tournament(lineup, deals, seed=0)`` uses, so the games are the same.
    """
    from concurrent.futures import ProcessPoolExecutor

    from big2.eval.tournament import _play_deals
    from big2.games.big2 import Rules

    rules = Rules()  # duty on, tiered, no four-card plays: the defaults (B22)
    if workers <= 1:
        return _play_deals(lineup, deal_seeds, rules)
    chunks = [deal_seeds[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_play_deals, [lineup] * workers, chunks, [rules] * workers))
    by_seed = {
        s: r
        for chunk, part in zip(chunks, parts, strict=True)
        for s, r in zip(chunk, part, strict=True)
    }
    return [by_seed[s] for s in deal_seeds]


def _std(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ci95(xs: list[float]) -> float:
    return 1.96 * _std(xs) / math.sqrt(len(xs))


def _summarise_cell(lineup: list[str], raw: list[dict[str, list[float]]]) -> dict[str, Any]:
    """Mean ± 95 % (over deals) and win rate per agent, plus σ for self-play."""
    agents = []
    for spec in dict.fromkeys(lineup):
        deal_scores = [sum(r[f"score|{spec}"]) / len(r[f"score|{spec}"]) for r in raw]
        deal_wins = [sum(r[f"win|{spec}"]) / len(r[f"win|{spec}"]) for r in raw]
        moves = sum(sum(r[f"moves|{spec}"]) for r in raw)
        secs = sum(sum(r[f"time|{spec}"]) for r in raw)
        agents.append(
            {
                "agent": spec,
                "games": sum(len(r[f"score|{spec}"]) for r in raw),
                "mean_score": sum(deal_scores) / len(raw),
                "score_ci95": _ci95(deal_scores),
                "win_rate": sum(deal_wins) / len(raw),
                "win_ci95": _ci95(deal_wins),
                "seconds_per_move": secs / max(moves, 1),
            }
        )
    out: dict[str, Any] = {"deals": len(raw), "agents": agents}
    if len(set(lineup)) == 1:
        # Self-play: each record holds 16 scores, rotation-major (rotation r, seat s
        # at index 4r + s). Seat s in rotation r is the same cards in every rotation.
        spec = lineup[0]
        games = [r[f"score|{spec}"] for r in raw]
        wins = [r[f"win|{spec}"] for r in raw]
        one_seat = [x for g in games for x in g]
        # The cards dealt to seat s are fixed within a deal; only the agents' seeds
        # change between rotations. The spread across rotations is the play noise.
        within = [
            x - sum(g[4 * r + s] for r in range(4)) / 4
            for g in games
            for s in range(4)
            for x in (g[4 * r + s] for r in range(4))
        ]
        sigma_within = math.sqrt(sum(d * d for d in within) / (len(raw) * 4 * 3))
        seat0 = [sum(g[4 * r] for r in range(4)) / 4 for g in games]
        seat0_wins = [sum(w[4 * r] for r in range(4)) / 4 for w in wins]
        out["sigma"] = {
            "one_seat_one_game": _std(one_seat),
            "within_deal_and_seat": sigma_within,
            "samples": len(one_seat),
        }
        out["seat0"] = {
            "mean_score": sum(seat0) / len(seat0),
            "score_ci95": _ci95(seat0),
            "win_rate": sum(seat0_wins) / len(seat0_wins),
            "win_ci95": _ci95(seat0_wins),
        }
    return out


def rebaseline(args: argparse.Namespace) -> None:
    """ISMCTS-1000 vs 3 greedy, vs DMC-10h (A B A B), and in self-play with σ
    (build-phase1.md §2). Each tournament runs in blocks of ``--block`` deals; a
    finished block is saved and read back on rerun, so a rerun resumes."""
    import os

    os.chdir(ROOT)  # the DMC checkpoint path is relative to the repo root
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else [c[0] for c in REBASELINE_CELLS]
    cells = [c for c in REBASELINE_CELLS if c[0] in names]
    blocks = [(i, min(i + args.block, args.deals)) for i in range(0, args.deals, args.block)]
    todo = [f"{n}_d{a}-{b}" for n, _, _ in cells for a, b in blocks]
    finished: list[str] = []
    t0 = time.time()

    def progress(running: str | None) -> None:
        _write_json(
            out / "progress.json",
            {
                "done": finished,
                "running": running,
                "next": [t for t in todo if t not in finished and t != running],
                "minutes_this_run": round((time.time() - t0) / 60, 1),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    results: dict[str, dict[str, Any]] = {}
    for name, label, pattern in cells:
        lineup = [args.candidate if x == "C" else x for x in pattern]
        raw: list[dict[str, list[float]]] = []
        cell_seconds = 0.0
        for a, b in blocks:
            tag = f"{name}_d{a}-{b}"
            path = out / f"{tag}.json"
            if path.exists():
                block = json.loads(path.read_text())
            else:
                progress(tag)
                tb = time.time()
                recs = _play_block(lineup, list(range(a, b)), args.workers)
                block = {
                    "lineup": lineup,
                    "deal_seeds": [a, b],
                    "seconds": time.time() - tb,
                    "raw": recs,
                }
                _write_json(path, block)
            raw += block["raw"]
            cell_seconds += block["seconds"]
            finished.append(tag)
            progress(None)
            print(f"done {tag}", flush=True)
        summary = _summarise_cell(lineup, raw)
        results[name] = {"label": label, "lineup": lineup, "seconds": cell_seconds, **summary}
        _write_json(out / f"{name}.json", results[name])

    lines = [
        f"Candidate `{args.candidate}` (greedy rollout: Task 5's structured rollout did not pay). "
        f"{args.deals:,} duplicate deals per tournament (each deal 4×, seats rotated), deal "
        "seeds 0…N−1, last-card duty on, tiered scoring, no four-card plays. Mean score per "
        "game, 95 % interval over deals. Seconds per move measured under load (other CPU jobs "
        "ran beside it); results do not depend on load (fixed iterations).",
        "",
        "| Tournament | Agent | Games | Mean score (95% CI) | Win rate (95% CI) "
        "| s per move (under load) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, label, _ in cells:
        r = results[name]
        rows = r["agents"]
        if "seat0" in r:
            s0 = r["seat0"]
            # Every seat is the same agent, so the mean is zero by construction;
            # report seat 0 (one fixed hand of each deal) instead.
            rows = [
                {
                    **rows[0],
                    **s0,
                    "agent": f"{rows[0]['agent']} (seat 0 only)",
                    "games": 4 * r["deals"],
                }
            ]
        for x in rows:
            note = " (trained with duty off)" if x["agent"] == DMC_10H else ""
            lines.append(
                f"| {label} | `{x['agent']}`{note} | {x['games']:,} "
                f"| {x['mean_score']:+.2f} ± {x['score_ci95']:.2f} "
                f"| {x['win_rate']:.1%} ± {x['win_ci95']:.1%} | {x['seconds_per_move']:.3f} |"
            )
    if "selfplay" in results:
        sg = results["selfplay"]["sigma"]
        lines += [
            "",
            f"σ of one seat's score in one game (ISMCTS self-play, {sg['samples']:,} seat-games): "
            f"**{sg['one_seat_one_game']:.2f} points**. Of that, the spread between rotations "
            f"of the same deal and seat (same cards, new search seeds) is "
            f"{sg['within_deal_and_seat']:.2f} points.",
        ]
    table = "\n".join(lines) + "\n"
    _write_json(out / "results.json", {"candidate": args.candidate, "deals": args.deals, **results})
    (out / "table.md").write_text(table, encoding="utf-8")
    progress(None)
    print(table, flush=True)


# main -------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "b1": b1,
    "rebaseline": rebaseline,
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

    p = sub.add_parser("rebaseline", help="ISMCTS vs greedy, vs DMC-10h, and self-play σ (Task 6)")
    p.add_argument("--candidate", default=ISMCTS_1000)
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--block", type=int, default=250, help="deals per saved block")
    p.add_argument("--only", default="", help="comma list of vs_greedy,vs_dmc10h,selfplay")
    p.add_argument("--out", default="p1_rebaseline", help="folder under runs/")
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
