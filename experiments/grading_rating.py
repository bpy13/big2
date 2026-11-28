"""Luck estimates and the new expert's games for the luck-removed rating (plan Task 43).

    python experiments/grading_rating.py luck --m 0    # one top-move self-play of every deal
    python experiments/grading_rating.py luck --m 16   # 16 sampled self-plays (tau 1) of every deal
    python experiments/grading_rating.py luck --m 16   # 16 sampled self-plays (tau 1) a deal
    python experiments/grading_rating.py report        # runs/p6_grading/report.md + proposal.json

``luck`` plays every deal of Task 40's ladder games (``runs/p6_ladder/pairs``, copied from the
big2-3 worktree) and of Task 42's same-rung files (``runs/p6_mistakes/decisions-<band>.jsonl``)
with the hour-24 network in all four seats and writes each seat's mean final score to
``runs/p6_grading/luck-m<m>.jsonl``: the part of a deal's result that the cards decide when the
expert's network plays them (decision 0008 amended). ``play`` adds the network-plus-search
expert's games in the ladder's block format (``runs/p6_grading/pairs``). ``report`` fits the
lines, the accuracy and the luck-removed rating (:mod:`big2.teach.grading`). Resumable: a deal
already written is not replayed; a block already played is not replayed. Runs from this checkout
(``PYTHONPATH=python``); worker processes come from ``HKQ_THREADS`` (default 8).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

NET = "runs/p6_net/selfplay-h24.pknn"
BELIEF_NET = "runs/p6_mistakes/belief-r1.pknn"
LADDER_PAIRS = ROOT / "runs" / "p6_ladder" / "pairs"
MISTAKES = ROOT / "runs" / "p6_mistakes"
OUT = ROOT / "runs" / "p6_grading"
#: The network-plus-search expert: Task 42's reference (decision 0010; Task 39 has not picked
#: the in-game budget, so PUCT-2000 stands in for it).
EXPERT_NAME = "puct2000"
EXPERT_SPEC = (
    f"puct:net={NET},simulations=2000,c_puct=2.5,belief=net,belief_net={BELIEF_NET},"
    "belief_candidates=2048,infoset_endgame=12"
)
#: The new expert plays the two strongest ladder rungs (Task 41's method), ABAB, 500 deals each.
EXPERT_OPPONENTS = ("net", "expert")
EXPERT_DEALS = 500
BLOCK = 50


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def threads() -> int:
    return int(os.environ.get("HKQ_THREADS", "8"))


# -- deals --------------------------------------------------------------------------------


def ladder_games(pairs: Path = LADDER_PAIRS) -> list[dict[str, Any]]:
    """Every game of the ladder's blocks (``{"deal", "players", "scores", "seconds"}``)."""
    out: list[dict[str, Any]] = []
    if not pairs.exists():
        return out
    for d in sorted(pairs.glob("*__*")):
        for p in sorted(d.glob("block_d*.json")):
            out.extend(json.loads(p.read_text(encoding="utf-8"))["games"])
    return out


def mistakes_deals() -> dict[str, list[int]]:
    """Band -> the deal seeds of Task 42's same-rung files (a deal is one line with ``history``)."""
    out: dict[str, list[int]] = {}
    for p in sorted(MISTAKES.glob("decisions-*.jsonl")):
        band = p.stem[len("decisions-") :]
        seen: dict[int, None] = {}
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.setdefault(json.loads(line)["deal_id"], None)
        out[band] = list(seen)
    return out


def all_deals() -> list[int]:
    seeds: dict[int, None] = {}
    for g in ladder_games():
        seeds.setdefault(g["deal"], None)
    for g in ladder_games(OUT / "pairs"):
        seeds.setdefault(g["deal"], None)
    for ds in mistakes_deals().values():
        for s in ds:
            seeds.setdefault(s, None)
    return list(seeds)


# -- luck -----------------------------------------------------------------------------------


def _luck_block(seeds: list[int], m: int) -> list[dict[str, Any]]:
    """Worker: the hour-24 network plays each deal in all four seats; ``m`` sampled games at
    temperature 1 (plus one top-move game when ``m`` is 0). Returns the per-seat means."""
    import numpy as np
    from big2.agents import make_agent
    from big2.eval.tournament import play_game
    from big2.games.big2 import Rules

    rules = Rules()
    out = []
    for ds in seeds:
        deal = rules.new_deal(ds)
        rec: dict[str, Any] = {"deal": ds, "m": m}
        if m == 0:
            agents = [make_agent(f"netprior:net={NET},temperature=0", seed=s) for s in range(4)]
            final = play_game(deal.copy(), agents)
            rec["mean"] = [float(x) for x in final.scores]
            rec["sd"] = [0.0] * 4
        else:
            scores = np.zeros((m, 4))
            for i in range(m):
                agents = [
                    make_agent(f"netprior:net={NET},temperature=1", seed=ds * 64 + i * 4 + s)
                    for s in range(4)
                ]
                scores[i] = play_game(deal.copy(), agents).scores
            rec["mean"] = [round(float(x), 4) for x in scores.mean(0)]
            rec["sd"] = [round(float(x), 4) for x in scores.std(0, ddof=1)]
        out.append(rec)
    return out


def read_luck(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    return {r["deal"]: r for r in recs}


def stage_luck(m: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"luck-m{m}.jsonl"
    have = read_luck(path)
    todo = [s for s in all_deals() if s not in have]
    log(f"luck m={m}: {len(have)} deals done, {len(todo)} to play ({threads()} workers)")
    t0 = time.perf_counter()
    chunk = 200
    with path.open("a", encoding="utf-8") as f:
        for lo in range(0, len(todo), chunk * threads()):
            part = todo[lo : lo + chunk * threads()]
            pieces = [part[i :: threads()] for i in range(threads())]
            pieces = [p for p in pieces if p]
            with ProcessPoolExecutor(len(pieces)) as pool:
                for recs in pool.map(_luck_block, pieces, [m] * len(pieces)):
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
            f.flush()
            done = min(lo + len(part), len(todo))
            log(f"  {done}/{len(todo)} deals, {time.perf_counter() - t0:.0f} s")
    log(f"luck done in {time.perf_counter() - t0:.0f} s")


# -- the new expert's games -----------------------------------------------------------------


def stage_play() -> None:
    """``puct2000`` against ``net`` and ``expert`` (ladder specs), ABAB with four rotations a
    deal, blocks of 50 deals in the ladder's format under ``runs/p6_grading/pairs``."""
    import zlib

    from elo_ladder import play_deals

    ladder = json.loads((ROOT / "runs/p6_ladder/ladder.json").read_text(encoding="utf-8"))
    specs = {r["name"]: r["spec"] for r in ladder["rungs"]}
    specs[EXPERT_NAME] = EXPERT_SPEC
    for opp in EXPERT_OPPONENTS:
        a, b = sorted((EXPERT_NAME, opp))
        pair = OUT / "pairs" / f"{a}__{b}"
        pair.mkdir(parents=True, exist_ok=True)
        base = (zlib.crc32(f"{a}|{b}".encode()) % 100_000) * 100_000
        for lo in range(0, EXPERT_DEALS, BLOCK):
            hi = min(lo + BLOCK, EXPERT_DEALS)
            p = pair / f"block_d{lo}-{hi}.json"
            if p.exists():
                continue
            t0 = time.perf_counter()
            games = play_deals(
                [a, b, a, b],
                {a: specs[a], b: specs[b]},
                [base + d for d in range(lo, hi)],
                threads(),
            )
            p.write_text(json.dumps({"games": games}), encoding="utf-8")
            log(f"play {a} vs {b}: deals {lo}-{hi} in {time.perf_counter() - t0:.0f} s")


# -- report ---------------------------------------------------------------------------------


def stage_report(m: int) -> None:
    from grading_report import build_report

    build_report(OUT, m)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["luck", "play", "report", "all"])
    ap.add_argument(
        "--m", type=int, default=16, help="sampled self-plays a deal (0 = one top-move game)"
    )
    args = ap.parse_args()
    if args.stage in ("play", "all"):
        stage_play()
    if args.stage in ("luck", "all"):
        stage_luck(args.m)  # after play, so the new games' deals are covered
    if args.stage in ("report", "all"):
        stage_report(args.m)


if __name__ == "__main__":
    main()
