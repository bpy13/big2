"""Plan Task 52: does PUCT's exploration weight ``c_puct`` let the search use its scores?

The position is Brian's hk-games deal 23 (``data/dev.db`` deal_logs id 23, copied read-only to
``experiments/data/hkgames_deal23.json``), seat 0's first decision: after the straights
3d-7h and 5h-9s, seat 0 holds ``3h 3s 4c 5c 6s 7s 8s 9h Td Tc Js As 2d``. The other hands are
the cards those seats played in the deal, filled up from the rest of the deck (the search only
sees seat 0's view, so the fill does not matter).

Stages:

* ``root``: one PUCT search at ``--simulations`` (32,000) and ``c_puct`` 2.5, one thread; per top
  move: visits, mean score (points), Q in utility units (score / the game's largest possible
  score, as ``puct.rs`` backs up), the network prior P and the final prior term
  ``c · P · sqrt(N + 1) / (1 + n)``.
* ``hand``: ``c_puct`` in {0.1, 0.25, 0.5, 1, 2.5} x {2,000, 8,000} simulations, 10 seeds each:
  picked x/10, visit share and mean score of the top moves.

Writes ``runs/p6_budget/cpuct/{root,hand}.json`` and prints markdown tables.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from big2 import _core as c

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p6_budget" / "cpuct"
NET = str(ROOT / "runs" / "p6_net" / "selfplay-h24.pknn")
DEAL = ROOT / "experiments" / "data" / "hkgames_deal23.json"
MINE = "3h 3s 4c 5c 6s 7s 8s 9h Td Tc Js As 2d"
WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.5)
HAND_SIMS = (2000, 8000)
RUNS = 10


def position() -> Any:
    """Seat 0 to move at its first decision of deal 23."""
    d = json.loads(DEAL.read_text(encoding="utf-8"))
    v = d["variants"]
    kw = dict(
        straights=v["straights"],
        flush_order=v["flush_order"],
        pass_rule=v["pass_rule"],
        scoring=v["scoring"],
        last_card_duty=v["last_card_duty"] == "on",
    )
    acts = d["actions"]
    hands: list[set[str]] = [set() for _ in range(4)]
    hands[0] = set(MINE.split())
    for i, a in enumerate(acts):  # seat 2 opened; seats then follow in order
        if a["type"] == "action.play":
            hands[(2 + i) % 4].update(a["cards"])
    deck = [r + s for r in "3456789TJQKA2" for s in "dcsh"]
    used = set().union(*hands)
    rest = [x for x in deck if x not in used]
    for p in (1, 2, 3):
        while len(hands[p]) < 13:
            hands[p].add(rest.pop())
    s = c.Big2State.from_hands([c.parse_cards(" ".join(h)) for h in hands], **kw)
    for a in acts[:2]:
        s.apply(c.parse_cards(" ".join(a["cards"])))
    assert s.current_player == 0
    return s


def max_abs_score() -> float:
    """Tiered scoring: 13 cards left pay 52 to each of 3 opponents."""
    return 52.0 * 3


def priors(s: Any) -> dict[str, float]:
    """The network's prior over every legal move (softmax over legal moves)."""
    r = c.net_search(s, net=NET, k=len(s.legal_actions()), n=1, depth=0)
    return {c.format_cards(m["cards"]): m["prior"] for m in r["candidates"]}


def search(s: Any, sims: int, c_puct: float, seed: int) -> dict[str, Any]:
    return c.analyze(
        s,
        algorithm="puct",
        iterations=sims,
        seed=seed,
        net=NET,
        c_puct=c_puct,
        belief="rules",
        threads=1,
    )


def root_stage(args: argparse.Namespace) -> None:
    s = position()
    pr = priors(s)
    r = search(s, args.simulations, 2.5, args.seed)
    total = sum(m["visits"] for m in r["moves"])
    scale = max_abs_score()
    rows = []
    for m in r["moves"][:4]:
        name = c.format_cards(m["cards"])
        p = pr[name] / sum(pr.values())
        q = m["mean_score"] / scale
        term = 2.5 * p * math.sqrt(total + 1) / (1 + m["visits"])
        rows.append(
            {
                "move": name,
                "visits": m["visits"],
                "share": m["visits"] / total,
                "mean_score": m["mean_score"],
                "q": q,
                "prior": p,
                "prior_term": term,
                "q_plus_term": q + term,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "root.json").write_text(
        json.dumps(
            {"simulations": total, "seed": args.seed, "scale": scale, "rows": rows}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"{total} simulations, utility = score / {scale:g}")
    print("| Move | Visits | Share | Mean score | Q | Prior P | c·P·√N/(1+n) | Q + term |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        print(
            f"| {x['move']} | {x['visits']:,} | {x['share']:.1%} | {x['mean_score']:+.2f} | "
            f"{x['q']:+.4f} | {x['prior']:.3f} | {x['prior_term']:.4f} | {x['q_plus_term']:+.4f} |"
        )


def hand_stage(args: argparse.Namespace) -> None:
    s = position()
    path = OUT / "hand.json"
    done: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    for sims in HAND_SIMS:
        for w in WEIGHTS:
            key = f"{w:g}@{sims}"
            if key in done:
                continue
            runs = []
            for seed in range(RUNS):
                r = search(s, sims, w, 1000 + seed)
                total = sum(m["visits"] for m in r["moves"])
                runs.append(
                    [
                        (c.format_cards(m["cards"]), m["visits"] / total, m["mean_score"])
                        for m in r["moves"]
                    ]
                )
            done[key] = runs
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(done), encoding="utf-8")
            print(key, collections.Counter(run[0][0] for run in runs), flush=True)
    print("| c_puct | Simulations | Picked (x/10) | Visit share | Mean score |")
    print("|---:|---:|---|---|---|")
    for sims in HAND_SIMS:
        for w in WEIGHTS:
            runs = done[f"{w:g}@{sims}"]
            picks = collections.Counter(run[0][0] for run in runs)
            agg: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
            for run in runs:
                for name, share, mean in run:
                    agg[name].append((share, mean))
            top = sorted(agg, key=lambda k: -sum(x[0] for x in agg[k]))[:3]
            pk = ", ".join(f"{k} {n}" for k, n in picks.most_common())
            sh = ", ".join(f"{k} {st.mean(x[0] for x in agg[k]):.0%}" for k in top)
            mn = ", ".join(f"{k} {st.mean(x[1] for x in agg[k]):+.2f}" for k in top)
            print(f"| {w:g} | {sims:,} | {pk} | {sh} | {mn} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("root")
    p.add_argument("--simulations", type=int, default=32000)
    p.add_argument("--seed", type=int, default=301)
    p.set_defaults(fn=root_stage)
    p = sub.add_parser("hand")
    p.set_defaults(fn=hand_stage)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
