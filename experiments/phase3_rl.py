"""Phase 3 experiments: learned agents vs the search solver.

Usage::

    python experiments/phase3_rl.py round-robin --deals 100 --workers 6
    python experiments/phase3_rl.py agreement --positions 300

**round-robin** plays every pair of agents head-to-head: duplicate deals with the
two agents in alternating seats (A B A B), each deal played four times with seats
rotated. Scores are zero-sum, so each match gives one number: A's mean score per
game against B, with a 95% CI over deals. "vs greedy" results are reported
separately, because beating the greedy bot is not the same as being strong (the
solver's rollouts model opponents as greedy bots, which inflates its score there).

**agreement** samples decision points from greedy-bot games and checks how often
each agent plays the same move as a high-effort solver (ISMCTS, 20,000
simulations), and what its choices cost in expected score according to an
equal-effort evaluation of every legal move (PIMC, 400 sampled deals).

Raw JSON goes to runs/phase3/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

from big2._core import analyze
from big2.agents import make_agent
from big2.eval import run_tournament
from big2.games.big2 import Big2State, Rules

#: Agents compared; checkpoints are the ones shipped in models/ (see docs/rl.md).
AGENTS: dict[str, str] = {
    "ISMCTS-1000": "ismcts:iterations=1000",
    "PIMC-40": "pimc:iterations=40",
    "DMC": "dmc:checkpoint=models/big2-dmc-selfplay.pt",
    "PPO": "dmc:checkpoint=models/big2-ppo.pt",
    "Distilled": "dmc:checkpoint=models/big2-distilled.pt",
    "Distilled+DMC": "dmc:checkpoint=models/big2-distilled-finetuned.pt",
    "Distilled+search": "dmcsearch:checkpoint=models/big2-distilled.pt,determinizations=32,top=6",
}


def round_robin(
    agents: dict[str, str], deals: int, workers: int, seed: int, out: Path
) -> dict[str, Any]:
    results: dict[str, Any] = {"vs_greedy": {}, "pairs": {}}
    for name, spec in agents.items():
        r = run_tournament([spec, "greedy", "greedy", "greedy"], deals, seed=seed, workers=workers)
        s = next(x for x in r.summaries if x.agent == spec)
        results["vs_greedy"][name] = {
            "score": s.mean_score,
            "ci95": s.score_ci95,
            "win_rate": s.win_rate,
            "ms_per_move": 1000 * s.seconds_per_move,
        }
        print(f"vs greedy  {name:<18} {s.mean_score:+6.2f} ± {s.score_ci95:.2f}", flush=True)
    for (a, sa), (b, sb) in itertools.combinations(agents.items(), 2):
        r = run_tournament([sa, sb, sa, sb], deals, seed=seed, workers=workers)
        s = next(x for x in r.summaries if x.agent == sa)
        results["pairs"][f"{a} vs {b}"] = {
            "a": a,
            "b": b,
            "score": s.mean_score,
            "ci95": s.score_ci95,
            "win_a": s.win_rate,
        }
        print(f"{a:<18} vs {b:<18} {s.mean_score:+6.2f} ± {s.score_ci95:.2f}", flush=True)
        (out / "round_robin.json").write_text(json.dumps(results, indent=2))
    return results


def sample_positions(n: int, seed: int) -> list[Big2State]:
    """Decision points with at least two legal moves, from greedy self-play games."""
    rng = random.Random(seed)
    out: list[Big2State] = []
    while len(out) < n:
        s = Rules().new_deal(rng.getrandbits(32))
        stops = set(rng.sample(range(60), 3))
        k = 0
        while not s.is_terminal:
            if k in stops and len(s.legal_actions()) > 1:
                out.append(s.copy())
            s.apply(s.greedy_action(0.1, rng.getrandbits(32)))
            k += 1
    return out[:n]


def agreement(agents: dict[str, str], positions: int, seed: int) -> dict[str, Any]:
    states = sample_positions(positions, seed)
    players = {name: make_agent(spec, seed=1) for name, spec in agents.items()}
    players["greedy"] = make_agent("greedy")
    stats = {name: {"agree": 0, "loss": 0.0, "seconds": 0.0} for name in players}
    for i, s in enumerate(states):
        ref = analyze(s, iterations=20_000, threads=4, seed=i)["moves"][0]["cards"]
        pimc = analyze(s, algorithm="pimc", iterations=400, threads=4, seed=i)["moves"]
        values = {m["cards"]: m["mean_score"] for m in pimc}
        best = max(values.values())
        for name, agent in players.items():
            t = time.perf_counter()
            a = agent.act(s)
            stats[name]["seconds"] += time.perf_counter() - t
            stats[name]["agree"] += a == ref
            stats[name]["loss"] += best - values[a]
    n = len(states)
    return {
        "positions": n,
        "agents": {
            k: {
                "agreement": v["agree"] / n,
                "mean_loss": v["loss"] / n,
                "ms_per_move": 1000 * v["seconds"] / n,
            }
            for k, v in stats.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["round-robin", "agreement"])
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma-separated agent names (default: all)")
    args = ap.parse_args()
    out = Path("runs/phase3")
    out.mkdir(parents=True, exist_ok=True)
    agents = {k: v for k, v in AGENTS.items() if not args.only or k in args.only.split(",")}
    if args.experiment == "round-robin":
        round_robin(agents, args.deals, args.workers, args.seed, out)
    else:
        res = agreement(agents, args.positions, args.seed)
        (out / "agreement.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
