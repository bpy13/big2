"""Rollout best response against the shipped ``expert`` (plan Task 32).

One seat (the attacker) plays a best response found by rollouts, no training. At each decision
it tries every legal move; for each of ``--rollouts`` deals of the hidden cards (drawn as the
attacker's information allows, ``Big2State.determinize``) it plays the move and then the rest
of the game with all four seats played by ``expert``'s own code (ISMCTS + exact endgame), and
keeps the move with the best mean score. The same hidden deals are used for every move, so
the comparison between moves is paired. The three other seats in the *real* game play the full
``expert`` (ISMCTS-10,000); the rollouts inside the attacker use ``--rollout-iterations``
instead of 10,000 (a full-strength rollout costs about a day per deal; see build-phase3.md §7).

Lineups, as in the other matches: ``1+3`` (attacker, 3 expert; four rotations so the attacker
plays every hand) and ``ABAB`` (attackers in seats 0 and 2, ``expert`` in 1 and 3; each
attacker models its partner as ``expert`` in its rollouts). Expert vs expert scores 0 by
symmetry, so the attacker's mean score per deal is its gain; intervals are over deals.

Usage::

    python experiments/rollout_br.py --deals 40 --out rollout_br --workers 8
    python experiments/rollout_br.py --deals 2 --rollouts 4 --rollout-iterations 20
        --out rollout_br_smoke

Resumes from ``runs/<out>/deals.jsonl`` (one line per finished deal and lineup);
``progress.json`` there moves after every deal; ``results.json`` is written when done.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

#: The bot that ships by default (docs/open-items.md), one thread: parallelism is across games.
EXPERT = "ismcts:iterations=10000,infoset_endgame=12,infoset_max_cards=10,threads=1"
LINEUPS = {"1+3": [0], "ABAB": [0, 2]}  # attacker seats at rotation 0


def expert_spec(iterations: int) -> str:
    return f"ismcts:iterations={iterations},infoset_endgame=12,infoset_max_cards=10,threads=1"


class RolloutBR:
    """Best response by rollouts: the legal move with the best mean score over shared deals."""

    def __init__(self, rollouts: int, rollout_iterations: int, seed: int) -> None:
        self.rollouts = rollouts
        self.spec = expert_spec(rollout_iterations)
        self.seed = seed
        self.calls = 0
        self.moves = 0
        self.seconds = 0.0

    def act(self, state: Any) -> int:
        from big2.agents import make_agent

        legal = [int(a) for a in state.legal_actions()]
        if len(legal) == 1:
            return legal[0]
        seat = state.current_player
        rng = random.Random(self.seed * 1_000_003 + self.calls)
        self.calls += 1
        t0 = time.perf_counter()
        totals = [0.0] * len(legal)
        for _ in range(self.rollouts):
            d = state.determinize(seat, seed=rng.getrandbits(63))
            for j, m in enumerate(legal):
                s = d.child(m)
                agents = [make_agent(self.spec, seed=rng.getrandbits(32)) for _ in range(4)]
                while not s.is_terminal:
                    s.apply(agents[s.current_player].act(s))
                totals[j] += s.scores[seat]
        self.moves += 1
        self.seconds += time.perf_counter() - t0
        best = max(range(len(legal)), key=lambda j: (totals[j], -j))
        return legal[best]


def play_deal(
    deal_seed: int, lineup: str, rollouts: int, rollout_iterations: int
) -> dict[str, Any]:
    """Four rotations of one deal; the attacker's seats' mean score over them."""
    from big2.agents import make_agent
    from big2.eval.tournament import play_game
    from big2.games.big2 import Rules

    deal = Rules().new_deal(deal_seed)
    scores: list[float] = []
    per_rot: list[list[float]] = []
    decisions = 0
    seconds = 0.0
    for rot in range(4):
        attackers = {(a + rot) % 4 for a in LINEUPS[lineup]}
        brs = {
            s: RolloutBR(rollouts, rollout_iterations, deal_seed * 16 + rot * 4 + s)
            for s in attackers
        }
        agents = [
            brs[s] if s in attackers else make_agent(EXPERT, seed=deal_seed * 4 + rot)
            for s in range(4)
        ]
        final = play_game(deal.copy(), agents)  # type: ignore[arg-type]
        assert final.scores is not None
        mine = [final.scores[s] for s in sorted(attackers)]
        per_rot.append(mine)
        scores.append(sum(mine) / len(mine))
        decisions += sum(b.moves for b in brs.values())
        seconds += sum(b.seconds for b in brs.values())
    return {
        "deal": deal_seed,
        "lineup": lineup,
        "score": sum(scores) / 4,
        "per_rotation": per_rot,
        "searched_decisions": decisions,
        "br_seconds": seconds,
    }


def mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return math.nan, math.inf
    m = sum(xs) / n
    if n < 2:
        return m, math.inf
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lu in LINEUPS:
        xs = [r["score"] for r in rows if r["lineup"] == lu]
        m, ci = mean_ci(xs)
        out[lu] = {"deals": len(xs), "mean": m, "ci95": ci, "excludes_zero": abs(m) > ci}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deals", type=int, default=40, help="deals per lineup")
    ap.add_argument("--rollouts", type=int, default=16, help="hidden deals per decision")
    ap.add_argument(
        "--rollout-iterations",
        type=int,
        default=200,
        help="ISMCTS iterations of the seats inside a rollout",
    )
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    ap.add_argument("--seed", type=int, default=32, help="deal seed block")
    ap.add_argument("--out", default="rollout_br")
    args = ap.parse_args(argv)

    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "deals.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    have = {(r["deal"], r["lineup"]) for r in rows}
    todo = [
        (args.seed * 1_000_000 + d, lu)
        for d in range(args.deals)
        for lu in LINEUPS
        if (args.seed * 1_000_000 + d, lu) not in have
    ]
    t0 = time.time()

    def progress(running: int) -> None:
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "finished": len(rows),
                    "total": args.deals * len(LINEUPS),
                    "running": running,
                    "minutes_this_run": round((time.time() - t0) / 60, 1),
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "so_far": summarize(rows),
                },
                indent=2,
            )
        )

    progress(0)
    if todo:
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [
                pool.submit(play_deal, d, lu, args.rollouts, args.rollout_iterations)
                for d, lu in todo
            ]
            for f in as_completed(futs):
                row = f.result()
                rows.append(row)
                with log.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                progress(len(futs) - (len(rows) - len(have)))
    result = {
        "expert": EXPERT,
        "rollouts": args.rollouts,
        "rollout_iterations": args.rollout_iterations,
        "summary": summarize(rows),
        "br_seconds_per_searched_decision": (
            sum(r["br_seconds"] for r in rows) / max(1, sum(r["searched_decisions"] for r in rows))
        ),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
