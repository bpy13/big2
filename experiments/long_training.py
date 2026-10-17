"""Did 10 more hours of training help? (continuation of Phase 3)

Usage::

    python experiments/long_training.py --deals 200 --workers 7

Each network after the extra training (``runs/*/v1_10h/latest.pt``) plays
head-to-head (duplicate deals, A B A B, seats rotated) against

* the search solver (ISMCTS, 1,000 simulations), and
* its own version from before the extra training (the Phase 3 model in ``models/``).

Raw JSON goes to runs/long_training/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker.eval import run_tournament

SOLVER = "ismcts:iterations=1000"

#: name → (after 10 more hours, before)
MODELS: dict[str, tuple[str, str]] = {
    "DMC": ("runs/dmc/v1_10h/latest.pt", "models/big2-dmc-selfplay.pt"),
    "PPO": ("runs/ppo/v1_10h/latest.pt", "models/big2-ppo.pt"),
    "Distilled": ("runs/distill/v1_10h/latest.pt", "models/big2-distilled.pt"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/long_training")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for name, (after, before) in MODELS.items():
        new = f"dmc:checkpoint={after}"
        for label, opponent in [("vs solver", SOLVER), ("vs before", f"dmc:checkpoint={before}")]:
            r = run_tournament(
                [new, opponent, new, opponent], args.deals, seed=args.seed, workers=args.workers
            )
            s = next(x for x in r.summaries if x.agent == new)
            results[f"{name} {label}"] = {
                "score": s.mean_score,
                "ci95": s.score_ci95,
                "win_rate": s.win_rate,
            }
            print(f"{name:<10} {label:<10} {s.mean_score:+6.2f} ± {s.score_ci95:.2f}", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
