"""Distillation retrained for last-card duty (the default rule since 2026-10-18).

Usage::

    python experiments/duty_training.py --deals 200 --workers 7

The network after 4 hours of distillation under duty-on rules
(``runs/distill/duty_v1/latest.pt``, started from ``models/big2-distilled-10h.pt``)
plays head-to-head (duplicate deals, A B A B, seats rotated), under duty-on rules,
against

* the search solver (ISMCTS, 1,000 simulations), and
* the network it started from, trained with duty off.

Raw JSON goes to runs/duty_training/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker.eval import run_tournament
from poker.games.big2 import Rules

SOLVER = "ismcts:iterations=1000"
NEW = "runs/distill/duty_v1/latest.pt"
BEFORE = "models/big2-distilled-10h.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=NEW)
    ap.add_argument("--out", default="runs/duty_training")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rules = Rules(last_card_duty=True)
    new = f"dmc:checkpoint={args.model}"
    results: dict[str, dict[str, float]] = {}
    opponents = {"vs solver": SOLVER, "vs duty-off network": f"dmc:checkpoint={BEFORE}"}
    for label, opponent in opponents.items():
        lineup = [new, opponent, new, opponent]
        r = run_tournament(lineup, args.deals, seed=args.seed, workers=args.workers, rules=rules)
        s = next(x for x in r.summaries if x.agent == new)
        results[label] = {"score": s.mean_score, "ci95": s.score_ci95, "win_rate": s.win_rate}
        print(f"{label:<20} {s.mean_score:+6.2f} ± {s.score_ci95:.2f}", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
