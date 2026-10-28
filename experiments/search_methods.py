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
* ``oracle`` — **cheats**: searches the true deal (hidden cards visible). Not a
  playable method; it bounds what any better model of the hidden cards could gain.

``--vs-greedy`` instead seats each agent (variant, then baseline) against three
greedy bots on the same deals. That is where the belief model is exactly right
(the opponents *are* greedy bots), so it checks the mechanism itself.

``--belief-accuracy`` measures the samplers directly: at positions from real
games, how much probability each one gives to the true holder of each hidden
card (uniform sampling gives about the holder's share of the hidden cards).

Raw JSON goes to runs/search_methods/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from big2._core import sample_deals
from big2.agents import make_agent
from big2.eval import run_tournament
from big2.games.big2 import Big2State, Rules

BASELINE = "ismcts:iterations=1000"
VALUE = "models/big2-value.pknn"

VARIANTS: dict[str, str] = {
    "belief": f"{BASELINE},belief=greedy,belief_epsilon=0.2",
    "belief-soft": f"{BASELINE},belief=greedy,belief_epsilon=0.5",
    "value": f"{BASELINE},value={VALUE}",
    "value-mix": f"{BASELINE},value={VALUE},value_rollout_weight=0.5",
    "belief+value": f"{BASELINE},belief=greedy,belief_epsilon=0.2,value={VALUE}",
    "oracle": f"{BASELINE},belief=oracle",
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
    ap.add_argument("--belief-accuracy", action="store_true", help="score the samplers")
    ap.add_argument("--positions", type=int, default=100)
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
    if args.belief_accuracy:
        belief_accuracy(args, out)
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


def _positions(players: str, n: int, seed: int) -> list[Big2State]:
    """Mid-game decision points (≥ 8 actions in, ≥ 2 legal moves) from games
    between four copies of the agent ``players``."""
    rng = random.Random(seed)
    agents = [make_agent(players, seed=i) for i in range(4)]
    out: list[Big2State] = []
    while len(out) < n:
        s = Rules().new_deal(seed=rng.getrandbits(32))
        stops = 0
        while not s.is_terminal:
            if len(s.history) >= 8 and len(s.legal_actions()) > 1 and rng.random() < 0.15:
                out.append(s.copy())
                stops += 1
                if stops == 3:
                    break
            s.apply(agents[s.current_player].act(s))
    return out[:n]


def belief_accuracy(args: argparse.Namespace, out: Path) -> None:
    samplers = {
        "uniform": {},
        "greedy ε=0.2": {"belief": "greedy", "epsilon": 0.2},
        "greedy ε=0.5": {"belief": "greedy", "epsilon": 0.5},
        "greedy ε=0.2, untempered": {"belief": "greedy", "epsilon": 0.2, "min_ess": 0.0},
    }
    results: dict[str, dict[str, float]] = {}
    for label, players in [("greedy games", "greedy"), ("solver games", "ismcts:iterations=300")]:
        positions = _positions(players, args.positions, args.seed)
        for name, kw in samplers.items():
            probs = []
            for k, s in enumerate(positions):
                me = s.current_player
                assert me is not None
                truth = s.hands
                deals = sample_deals(s, 200, seed=k, **kw)  # type: ignore[arg-type]
                for p in range(4):
                    if p == me:
                        continue
                    for c in range(52):
                        if truth[p] >> c & 1:
                            probs.append(sum(d[p] >> c & 1 for d in deals) / len(deals))
            acc = sum(probs) / len(probs)
            results.setdefault(label, {})[name] = acc
            print(f"{label:<13} {name:<26} P(true holder) = {acc:.3f}", flush=True)
    (out / "belief_accuracy.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
