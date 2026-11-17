"""The win-chance bot against `expert` (plan Task 34).

Results: docs/research/results/win-chance-bot.md.

Usage::

    python experiments/win_chance.py final --workers 8     # 10,000 duplicate deals, both lineups
    python experiments/win_chance.py habits --workers 8    # 1,000 logged deals, ABAB, both seatings
    python experiments/win_chance.py table                 # results.json and table.md from both

The win-chance bot is PUCT-2000 with the round-3 network of the win pipeline
(``experiments/expert_iteration.py --objective win``, ``runs/net_r3_win``), ``c_puct`` 2.5,
leaf weight λ = 0.5 (r3's settings; orchestrator's call 2026-11-17), searching with
``objective=win``: every simulation counts going out first as 1 and anything else as 0, and the
network's leaf value is its probability of going out. `expert` is today's shipped bot,
ISMCTS-10,000 + exact endgame (expected points). The match itself is scored as always: tiered,
last-card duty on, no four-card plays (B22).

* **final** — duplicate deals (each deal 4×, seats rotated), lineups ``1+3`` and ``ABAB``, the
  same deal seeds as every other match. For each bot: share of deals won (went out first) and
  points per deal, each with a 95 % bootstrap interval over deals, plus the paired differences.
* **habits** — Brian's expected play (2026-11-11: "It should pass greedily to optimize late game
  dominance"): 1,000 deals in ``ABAB``, each played twice with the bots' seats swapped, and for
  each bot (a) how often it passes when it holds a legal play (following a trick), (b) at the
  first moment an opponent is down to 3 cards or fewer, how many 2s it still holds and how many
  five-card hands its best way to shed the rest uses (``decompositions(hand)[0]``).

Every stage resumes: the match reuses its saved blocks; habits saves one JSON per block.
``runs/win_chance/progress.json`` moves with both.
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

import numpy as np
from build_phase2 import (  # also opts this process out of power throttling
    TABLE_HEAD,
    Progress,
    _row,
    _write_json,
    play_cell,
)
from build_phase3 import EXPERT

ROOT = Path(__file__).resolve().parent.parent
OUT = "win_chance"
NET = "runs/net_r3_win/pvnet-r3.pknn"
WIN_BOT = f"puct:net={NET},simulations=2000,c_puct=2.5,leaf_network_weight=0.5,objective=win"
#: Every 2: rank 12 (``rank * 4 + suit``).
TWOS = 0xF << 48
LOW_CARDS = 3


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def out_dir() -> Path:
    return ROOT / "runs" / OUT


# final --------------------------------------------------------------------------


def final(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    margs = argparse.Namespace(out=OUT, block=args.block, seed=0, workers=args.workers)
    reports = play_cell(margs, progress, "final", args.bot, EXPERT, args.deals, folder="final")
    for lu, r in reports.items():
        r.update(win_shares(out_dir() / "final" / lu.replace("+", "p"), args.bot, EXPERT))
    return reports


def win_shares(cache: Path, bot: str, opp: str) -> dict[str, Any]:
    """`expert`'s share of deals won with its interval, and the paired difference of the
    two bots' shares per deal, from the match's saved blocks."""
    from big2.eval.harness import bootstrap_ci

    recs = [
        r for p in sorted(cache.glob("block_*.json")) for r in json.loads(p.read_text())["records"]
    ]
    mean = lambda r, k: sum(r[k]) / len(r[k])  # noqa: E731
    b = np.array([mean(r, f"win|{bot}") for r in recs])
    o = np.array([mean(r, f"win|{opp}") for r in recs])
    return {
        "opponent_win_ci95": bootstrap_ci(o, seed=4),
        "win_share_diff": float((b - o).mean()),
        "win_share_diff_ci95": bootstrap_ci(b - o, seed=5),
    }


# habits -------------------------------------------------------------------------


def _habits_deal(seats: list[str], seed: int) -> dict[str, dict[str, list[float]]]:
    """Plays one deal; per spec: pass decisions, held 2s and five-card hands."""
    from big2._core import Big2State, decompositions
    from big2.agents import make_agent

    agents = [make_agent(s, seed=seed * 4 + i) for i, s in enumerate(seats)]
    st = Big2State.new_deal(seed)
    out: dict[str, dict[str, list[float]]] = {
        s: {"could_play": [], "passed": [], "twos": [], "fives": []} for s in set(seats)
    }
    noted = [False] * 4

    def note_low() -> None:
        sizes = st.hand_sizes
        for p in range(4):
            if noted[p] or not any(sizes[q] <= LOW_CARDS for q in range(4) if q != p):
                continue
            noted[p] = True
            hand = st.hand(p)
            best = decompositions(hand, 1)
            out[seats[p]]["twos"].append(float(bin(hand & TWOS).count("1")))
            out[seats[p]]["fives"].append(
                float(sum(bin(c).count("1") == 5 for c in best[0]) if best else 0)
            )

    while not st.is_terminal:
        p = st.current_player
        assert p is not None
        legal = st.legal_actions()
        action = agents[p].act(st)
        if st.current_play is not None and len(legal) > 1:  # a pass and at least one play
            out[seats[p]]["could_play"].append(1.0)
            out[seats[p]]["passed"].append(float(action == 0))
        st.apply(action)
        note_low()
    return out


def habits(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    folder = out_dir() / "habits"
    folder.mkdir(parents=True, exist_ok=True)
    bot, opp = args.bot, EXPERT
    seatings = [[bot, opp, bot, opp], [opp, bot, opp, bot]]
    totals: dict[str, dict[str, list[float]]] = {
        s: {"passed": [], "twos": [], "fives": []} for s in (bot, opp)
    }
    for a in range(0, args.habit_deals, args.block):
        b = min(a + args.block, args.habit_deals)
        path = folder / f"block_d{a}-{b}.json"
        if not path.exists():
            progress.update(f"habits, deals up to {b} of {args.habit_deals}")
            jobs = [(seats, 7_000_000 + d) for d in range(a, b) for seats in seatings]
            with ProcessPoolExecutor(args.workers) as pool:
                res = list(pool.map(_habits_deal, *zip(*jobs, strict=True), chunksize=1))
            block: dict[str, dict[str, list[float]]] = {
                s: {"passed": [], "twos": [], "fives": []} for s in (bot, opp)
            }
            for r in res:
                for s, v in r.items():
                    for k in block[s]:
                        block[s][k] += v[k]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"seats": seatings, "stats": block}))
            tmp.replace(path)
        saved = json.loads(path.read_text())
        if saved["seats"] != seatings:
            raise ValueError(f"{path} holds another match ({saved['seats']})")
        for s, v in saved["stats"].items():
            for k in totals[s]:
                totals[s][k] += v[k]
    progress.update(None, done="habits")
    from big2.eval.harness import bootstrap_ci

    summary = {}
    for s, v in totals.items():
        summary[s] = {
            "pass_rate": float(np.mean(v["passed"])),
            "pass_rate_ci95": bootstrap_ci(v["passed"], seed=6),
            "pass_decisions": len(v["passed"]),
            "twos_held": float(np.mean(v["twos"])),
            "twos_held_ci95": bootstrap_ci(v["twos"], seed=7),
            "fives_held": float(np.mean(v["fives"])),
            "fives_held_ci95": bootstrap_ci(v["fives"], seed=8),
            "low_moments": len(v["twos"]),
        }
    return {"deals": args.habit_deals, "seatings": seatings, "by_bot": summary}


# table --------------------------------------------------------------------------


def table(final_reports: dict[str, Any] | None, habit: dict[str, Any] | None, bot: str) -> str:
    lines = [
        f"Win-chance bot `{bot}` vs `expert` `{EXPERT}`; duplicate deals, tiered scoring, "
        "last-card duty on, no four-card plays.",
        "",
    ]
    if final_reports:
        lines += [
            *TABLE_HEAD,
            *(_row("win-chance vs expert", lu, r) for lu, r in final_reports.items()),
            "",
            "| Lineup | win-chance deals won | expert deals won | difference (95 %) |",
            "|---|---:|---:|---:|",
        ]
        for lu, r in final_reports.items():
            oc = r["opponent_win_ci95"]
            dc = r["win_share_diff_ci95"]
            lines.append(
                f"| {lu} | {r['win_rate']:.1%} [{r['win_ci95'][0]:.1%}, {r['win_ci95'][1]:.1%}] "
                f"| {r['opponent_win_rate']:.1%} [{oc[0]:.1%}, {oc[1]:.1%}] "
                f"| {r['win_share_diff']:+.1%} [{dc[0]:+.1%}, {dc[1]:+.1%}] |"
            )
        lines.append("")
    if habit:
        lines += [
            f"Habits ({habit['deals']:,} deals, ABAB, both seatings):",
            "",
            "| Bot | passes when it could play | 2s held when an opponent has ≤ 3 cards "
            "| five-card hands held then |",
            "|---|---:|---:|---:|",
        ]
        for s, v in habit["by_bot"].items():
            name = "win-chance" if s == bot else "expert"
            lines.append(
                f"| {name} | {v['pass_rate']:.1%} [{v['pass_rate_ci95'][0]:.1%}, "
                f"{v['pass_rate_ci95'][1]:.1%}] ({v['pass_decisions']:,} decisions) "
                f"| {v['twos_held']:.2f} [{v['twos_held_ci95'][0]:.2f}, "
                f"{v['twos_held_ci95'][1]:.2f}] "
                f"| {v['fives_held']:.2f} [{v['fives_held_ci95'][0]:.2f}, "
                f"{v['fives_held_ci95'][1]:.2f}] ({v['low_moments']:,} moments) |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("stages", nargs="+", choices=["final", "habits", "table"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deals", type=int, default=10_000)
    ap.add_argument("--habit-deals", type=int, default=1_000)
    ap.add_argument("--block", type=int, default=250)
    ap.add_argument("--out", default=OUT, help="folder under runs/ (tests: a scratch one)")
    ap.add_argument(
        "--bot", default=WIN_BOT, help="the win-chance bot's spec (tests: a cheaper one)"
    )
    args = ap.parse_args()
    globals()["OUT"] = args.out  # out_dir() and final() read it
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    progress = Progress(out, ["final 1+3", "final ABAB", "habits"])
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for stage in args.stages:
        if stage == "final":
            results["final"] = final(args, progress)
        elif stage == "habits":
            results["habits"] = habits(args, progress)
        results.update(bot=args.bot, expert=EXPERT, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        _write_json(results_path, results)
    text = table(results.get("final"), results.get("habits"), args.bot)
    (out / "table.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    progress.update(None)


if __name__ == "__main__":
    main()
