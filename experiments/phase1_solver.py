"""Phase 1 solver experiments. Results: docs/study-results/phase1-solver.md.

Usage::

    python experiments/phase1_solver.py --deals 100 --workers 7 [--only ladder,budget,...]

Each experiment is a duplicate tournament (see poker.eval.tournament). Raw results
are written as JSON to runs/phase1/ (git-ignored); this script prints a Markdown
table per experiment that is pasted into the results document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poker.eval import TournamentResult, run_tournament

BASE = "ismcts:iterations=1000"

EXPERIMENTS: dict[str, list[list[str]]] = {
    # One candidate against three greedy bots.
    "ladder": [
        ["random", "greedy", "greedy", "greedy"],
        ["greedy:epsilon=0.1", "greedy", "greedy", "greedy"],
        ["pimc:iterations=40", "greedy", "greedy", "greedy"],
        [BASE, "greedy", "greedy", "greedy"],
    ],
    # Head-to-head, two copies of each agent in alternating seats.
    "budget": [["ismcts:iterations=4000", BASE, "ismcts:iterations=4000", BASE]],
    "pimc_vs_ismcts": [[BASE, "pimc:iterations=40", BASE, "pimc:iterations=40"]],
    "rollout_noise": [
        [BASE + ",rollout_epsilon=0.0", BASE, BASE + ",rollout_epsilon=0.0", BASE],
        [BASE + ",rollout_epsilon=0.3", BASE, BASE + ",rollout_epsilon=0.3", BASE],
    ],
    "objective": [[BASE + ",objective=win", BASE, BASE + ",objective=win", BASE]],
}


def markdown(name: str, r: TournamentResult) -> str:
    lines = [
        f"**{name}** — lineup `{' | '.join(r.lineup)}`, "
        f"{r.deals} deals × 4 rotations, {r.seconds:.0f}s",
        "",
        "| Agent | Games | Mean score (95% CI) | Win rate (95% CI) | ms / move |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in sorted(r.summaries, key=lambda s: -s.mean_score):
        lines.append(
            f"| `{s.agent}` | {s.games} | {s.mean_score:+.2f} ± {s.score_ci95:.2f} "
            f"| {s.win_rate:.1%} ± {s.win_ci95:.1%} | {1000 * s.seconds_per_move:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    args = ap.parse_args()
    out = Path("runs/phase1")
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in EXPERIMENTS if not args.only or n in args.only.split(",")]
    for name in names:
        for i, lineup in enumerate(EXPERIMENTS[name]):
            r = run_tournament(lineup, args.deals, seed=args.seed, workers=args.workers)
            r.save(out / f"{name}_{i}.json")
            print(markdown(name, r), flush=True)


if __name__ == "__main__":
    main()
