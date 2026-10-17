"""New search methods vs the Phase 1 solver (ISMCTS, 1,000 simulations).

Usage::

    python experiments/search_methods.py --variants belief belief-soft --deals 200 --workers 2
    python experiments/search_methods.py --list

Every variant plays a head-to-head match against the baseline: duplicate deals,
the two agents in alternating seats (A B A B), each deal played four times with
seats rotated. The result is the variant's mean score per game against the
baseline, with a 95% CI over deals, and both agents' time per move measured in
the same match (so they share the same machine load).

Variants (see docs/solver.md §6):

* ``belief`` — opponent-aware determinization: sampled deals weighted by how well
  the greedy-bot model (ε = 0.2) explains the opponents' actions.
* ``belief-soft`` — the same with a looser model (ε = 0.5).
* ``value`` — leaves judged by the learned value network instead of rollouts.
* ``value-mix`` — half value network, half greedy rollout.
* ``belief+value`` — both.

``--vs-greedy`` instead seats each agent (variant, then baseline) against three
greedy bots on the same deals. That is where the belief model is exactly right
(the opponents *are* greedy bots), so it checks the mechanism itself.

Raw JSON goes to runs/search_methods/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker.eval import run_tournament

BASELINE = "ismcts:iterations=1000"
VALUE = "models/big2-value.pknn"

VARIANTS: dict[str, str] = {
    "belief": f"{BASELINE},belief=greedy,belief_epsilon=0.2",
    "belief-soft": f"{BASELINE},belief=greedy,belief_epsilon=0.5",
    "value": f"{BASELINE},value={VALUE}",
    "value-mix": f"{BASELINE},value={VALUE},value_rollout_weight=0.5",
    "belief+value": f"{BASELINE},belief=greedy,belief_epsilon=0.2,value={VALUE}",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default="runs/search_methods")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--vs-greedy", action="store_true", help="each agent vs 3 greedy bots")
    args = ap.parse_args()
    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:<14} {v}")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.vs_greedy:
        vs_greedy(args, out)
        return
    path = out / "vs_baseline.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.variants:
        spec = VARIANTS.get(name, name)
        lineup = [spec, args.baseline, spec, args.baseline]
        r = run_tournament(lineup, args.deals, seed=args.seed, workers=args.workers)
        mine = next(s for s in r.summaries if s.agent == spec)
        base = next(s for s in r.summaries if s.agent == args.baseline)
        results[name] = {
            "spec": spec,
            "baseline": args.baseline,
            "deals": args.deals,
            "seed": args.seed,
            "score": mine.mean_score,
            "ci95": mine.score_ci95,
            "win_rate": mine.win_rate,
            "baseline_win_rate": base.win_rate,
            "ms_per_move": 1000 * mine.seconds_per_move,
            "baseline_ms_per_move": 1000 * base.seconds_per_move,
            "seconds": r.seconds,
        }
        print(
            f"{name:<14} {mine.mean_score:+6.2f} ± {mine.score_ci95:.2f}"
            f"   win {mine.win_rate:.1%} vs {base.win_rate:.1%}"
            f"   ms/move {1000 * mine.seconds_per_move:.1f} vs {1000 * base.seconds_per_move:.1f}",
            flush=True,
        )
        path.write_text(json.dumps(results, indent=2))


def vs_greedy(args: argparse.Namespace, out: Path) -> None:
    path = out / "vs_greedy.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in ["baseline", *args.variants]:
        spec = args.baseline if name == "baseline" else VARIANTS.get(name, name)
        r = run_tournament(
            [spec, "greedy", "greedy", "greedy"], args.deals, seed=args.seed, workers=args.workers
        )
        s = next(x for x in r.summaries if x.agent == spec)
        results[name] = {
            "spec": spec,
            "deals": args.deals,
            "seed": args.seed,
            "score": s.mean_score,
            "ci95": s.score_ci95,
            "win_rate": s.win_rate,
            "ms_per_move": 1000 * s.seconds_per_move,
        }
        print(f"vs greedy {name:<14} {s.mean_score:+6.2f} ± {s.score_ci95:.2f}", flush=True)
        path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
