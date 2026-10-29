"""Build-run phase 3 experiments (Tasks 18 and 20). Results: docs/study-results/build-phase3.md.

Usage::

    python experiments/build_phase3.py rules-mask --workers 8
    python experiments/build_phase3.py endgame --workers 8
    python experiments/build_phase3.py endgame --positions 6 --deals 4 --out p3_tiny  # smoke run

* ``rules-mask`` (Task 18, bar A13): the share of drawn deals the rules mask rejects per
  decision (on positions from ISMCTS self-play), then PUCT-2000 with ``belief=rules``
  against the same PUCT without it, ``--deals`` duplicate deals, both lineups. A13: the
  interval includes or exceeds 0 (the mask loses nothing).
* ``endgame`` (Task 20, bar A15): ``--positions`` late positions (at most 12 cards hidden
  and at most 14 in all hands) from PUCT self-play; on each, the information-set endgame
  (every deal solved, cap ``--exact-max-nodes``) against ISMCTS-100,000: agreement on the
  move (A15: ≥ 95 %), native time per call by hidden-card count (the plan's thresholds 8,
  10, 12) and by cards in all hands (8–14, what the cost follows; the largest limit with
  every position solved and 95 % of calls under 1 s is picked), then PUCT-2000 with
  ``infoset_endgame=12,infoset_max_cards=<that limit>`` against PUCT-2000 without it,
  ``--deals`` duplicate deals, both lineups. Node WebAssembly timing is Task 16's harness
  (not here).

PUCT everywhere is the strongest setting measured so far (build-phase2.md §6.2): c_puct 2.5,
leaf ``0.5·network + 0.5·rollout``. Matches resume from saved blocks; a subcommand whose
``results.json`` exists is skipped. ``progress.json`` in ``runs/<out>/`` moves every block.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from build_phase2 import (  # also opts this process out of power throttling
    TABLE_HEAD,
    Progress,
    _done,
    _finish,
    _row,
    _write_json,
    cpu_load,
    no_power_throttling,
    play_cell,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NET = "runs/net_r0/pvnet-r0.pknn"
#: PUCT as in build-phase2.md §6.2 (the best PUCT so far).
PUCT_BEST = "c_puct=2.5,leaf_network_weight=0.5"
THRESHOLDS = [8, 10, 12]
#: Cards in all hands together (``infoset_max_cards``): what the solve's cost follows
#: (found in the smoke run: 8 hidden cards beside 13 of the player's own took > 1 min).
MAX_CARDS = [8, 10, 12, 14]
#: The phone budget per call (plan Task 20: the largest threshold under 1 s).
SECONDS_BAR = 1.0
#: A15: agreement with ISMCTS-100,000.
A15_BAR = 0.95


def puct2000(net: str, extra: str = "") -> str:
    spec = f"puct:net={net},simulations=2000,{PUCT_BEST}"
    return f"{spec},{extra}" if extra else spec


# rules-mask (Task 18) -------------------------------------------------------------


def _rejection_worker(game_seed: int, iterations: int) -> list[dict[str, Any]]:
    """One ISMCTS self-play game; every searched decision's rejection share."""
    no_power_throttling()
    from big2._core import Big2State, analyze

    s = Big2State.new_deal(game_seed)
    rng = random.Random(game_seed)
    rows = []
    while not s.is_terminal:
        legal = s.legal_actions()
        if len(legal) == 1:
            s.apply(legal[0])
            continue
        r = analyze(s, iterations=iterations, belief="rules", seed=rng.getrandbits(32))
        rows.append(
            {
                "share": r["belief_rejection_share"],
                "fallbacks": r["belief_fallbacks"],
                "hidden": sum(s.hand_sizes) - s.hand_sizes[s.current_player or 0],
            }
        )
        s.apply(r["moves"][0]["cards"])
    return rows


def rules_mask(args: argparse.Namespace, progress: Progress) -> None:
    name = "a_rules_mask"
    if _done(args, name):
        return
    folder = ROOT / "runs" / args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    share_path = folder / "rejections.json"
    if not share_path.exists():
        progress.update(f"{name} rejection shares")
        with ProcessPoolExecutor(args.workers) as pool:
            parts = list(
                pool.map(
                    _rejection_worker,
                    range(10_000, 10_000 + args.share_games),
                    [args.share_iterations] * args.share_games,
                )
            )
        rows = [r for p in parts for r in p]
        shares = [r["share"] for r in rows]
        _write_json(
            share_path,
            {
                "games": args.share_games,
                "decisions": len(rows),
                "mean_share": statistics.fmean(shares),
                "decisions_with_rejections": sum(1 for x in shares if x > 0) / len(rows),
                "mean_share_when_any": statistics.fmean([x for x in shares if x > 0] or [0.0]),
                "max_share": max(shares),
                "fallbacks": sum(r["fallbacks"] for r in rows),
                "search": f"ismcts:iterations={args.share_iterations},belief=rules (self-play)",
            },
        )
    shares = json.loads(share_path.read_text())
    cand = puct2000(args.net, "belief=rules")
    opp = puct2000(args.net)
    reports = play_cell(args, progress, name, cand, opp, args.deals)
    ok = {lu: r["paired_ci95"][1] >= 0 for lu, r in reports.items()}
    lines = [
        f"(a) Rules mask (Task 18). Rejected draws per decision, over {shares['decisions']:,} "
        f"decisions of {shares['games']} self-play games ({shares['search']}): mean "
        f"{shares['mean_share']:.1%}; {shares['decisions_with_rejections']:.1%} of decisions "
        f"reject at least one draw, and those reject {shares['mean_share_when_any']:.1%} on "
        f"average (largest {shares['max_share']:.1%}); {shares['fallbacks']} samples gave up "
        "after 1,000 rejections.",
        "",
        f"`{cand}` vs `{opp}`, {args.deals:,} duplicate deals, duty on, tiered.",
        "",
        *TABLE_HEAD,
        *(_row("PUCT-2000 rules mask vs uniform", lu, r) for lu, r in reports.items()),
        "",
        "A13 (the mask loses nothing: interval includes or exceeds 0): "
        + ", ".join(f"{lu} {'yes' if v else 'no'}" for lu, v in ok.items())
        + ".",
    ]
    _finish(
        args,
        name,
        {"candidate": cand, "opponent": opp, "rejections": shares, "reports": reports, "a13": ok},
        lines,
    )
    progress.update(None, done=name)


# endgame (Task 20) ----------------------------------------------------------------


def _late_positions_worker(game_seed: int, net: str, simulations: int) -> list[int] | None:
    """One PUCT self-play game; returns the actions up to one random late decision
    (at most 12 cards hidden from the player to move, two or more legal moves)."""
    no_power_throttling()
    from big2._core import Big2State, analyze

    s = Big2State.new_deal(game_seed)
    rng = random.Random(game_seed)
    actions: list[int] = []
    late: list[int] = []  # lengths of `actions` at late decisions
    while not s.is_terminal:
        legal = s.legal_actions()
        me = s.current_player
        assert me is not None
        hidden = sum(s.hand_sizes) - s.hand_sizes[me]
        if len(legal) > 1 and hidden <= max(THRESHOLDS) and sum(s.hand_sizes) <= max(MAX_CARDS):
            late.append(len(actions))
        if len(legal) == 1:
            a = legal[0]
        else:
            r = analyze(
                s,
                algorithm="puct",
                iterations=simulations,
                net=net,
                c_puct=2.5,
                seed=rng.getrandbits(32),
            )
            a = r["moves"][0]["cards"]
        s.apply(a)
        actions.append(a)
    if not late:
        return None
    return actions[: rng.choice(late)]


def _replay(game_seed: int, actions: list[int]) -> Any:
    from big2._core import Big2State

    s = Big2State.new_deal(game_seed)
    for a in actions:
        s.apply(a)
    return s


def _endgame_worker(
    game_seed: int, actions: list[int], ismcts_iterations: int, max_nodes: int
) -> dict[str, Any]:
    """The exact information-set solve and ISMCTS on one position."""
    no_power_throttling()
    from big2._core import analyze

    s = _replay(game_seed, actions)
    me = s.current_player
    assert me is not None
    hidden = sum(s.hand_sizes) - s.hand_sizes[me]
    total = sum(s.hand_sizes)
    t = time.perf_counter()
    ex = analyze(
        s, iterations=1, infoset_endgame=12, infoset_max_cards=52, infoset_max_nodes=max_nodes
    )
    exact_seconds = time.perf_counter() - t
    if ex["exact"] is not True:
        # Over the node cap: not solved (the search would fall back); counted apart.
        return {
            "game_seed": game_seed,
            "hidden": hidden,
            "total": total,
            "exact": False,
            "exact_seconds": exact_seconds,
        }
    t = time.perf_counter()
    mc = analyze(s, iterations=ismcts_iterations, seed=game_seed)
    mc_seconds = time.perf_counter() - t
    value = {m["cards"]: m["mean_score"] for m in ex["moves"]}
    best = ex["moves"][0]["cards"]
    pick = mc["moves"][0]["cards"]
    return {
        "game_seed": game_seed,
        "exact": True,
        "moves_played": len(actions),
        "hidden": hidden,
        "total": total,
        "legal": len(ex["moves"]),
        "deals": ex["simulations"],
        "exact_seconds": exact_seconds,
        "ismcts_seconds": mc_seconds,
        "exact_best": best,
        "ismcts_best": pick,
        "same_move": best == pick,
        # ISMCTS's move is worth as much as the exact best (ties between moves).
        "same_value": abs(value[pick] - value[best]) < 1e-9,
        "loss_points": value[best] - value[pick],
        "forced": ex["forced"],
    }


def endgame(args: argparse.Namespace, progress: Progress) -> None:
    name = "b_endgame"
    if _done(args, name):
        return
    folder = ROOT / "runs" / args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    pos_path = folder / "positions.json"
    if not pos_path.exists():
        progress.update(f"{name} positions from PUCT-{args.position_simulations} self-play")
        found: list[dict[str, Any]] = []
        seed = 20_000
        with ProcessPoolExecutor(args.workers) as pool:
            while len(found) < args.positions:
                seeds = list(range(seed, seed + args.workers * 4))
                seed += len(seeds)
                for gs, acts in zip(
                    seeds,
                    pool.map(
                        _late_positions_worker,
                        seeds,
                        [args.net] * len(seeds),
                        [args.position_simulations] * len(seeds),
                    ),
                    strict=True,
                ):
                    if acts is not None and len(found) < args.positions:
                        found.append({"game_seed": gs, "actions": acts})
        _write_json(pos_path, {"positions": found})
    positions = json.loads(pos_path.read_text())["positions"]

    solved_path = folder / "solved.json"
    if not solved_path.exists():
        progress.update(f"{name} exact vs ISMCTS-{args.ismcts_iterations} on {len(positions)}")
        load = {**cpu_load(), "workers": args.workers}
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(
                pool.map(
                    _endgame_worker,
                    [p["game_seed"] for p in positions],
                    [p["actions"] for p in positions],
                    [args.ismcts_iterations] * len(positions),
                    [args.exact_max_nodes] * len(positions),
                )
            )
        _write_json(solved_path, {"load": load, "rows": rows})
    solved = json.loads(solved_path.read_text())
    all_rows, load = solved["rows"], solved["load"]
    rows = [r for r in all_rows if r["exact"]]
    unsolved = len(all_rows) - len(rows)
    n = max(1, len(rows))
    same_move = sum(r["same_move"] for r in rows) / n
    same_value = sum(r["same_value"] for r in rows) / n

    def time_rows(key: str, limits: list[int]) -> list[dict[str, Any]]:
        out = []
        for lim in limits:
            # Unsolved positions count with the time it took to hit the cap (a lower bound).
            sel = [r for r in all_rows if r[key] <= lim]
            if not sel:
                continue
            secs = sorted(r["exact_seconds"] for r in sel)
            out.append(
                {
                    "threshold": lim,
                    "positions": len(sel),
                    "unsolved": sum(1 for r in sel if not r["exact"]),
                    "mean": statistics.fmean(secs),
                    "p95": secs[min(len(secs) - 1, int(0.95 * len(secs)))],
                    "max": secs[-1],
                }
            )
        return out

    timing = time_rows("hidden", THRESHOLDS)  # the plan's thresholds (hidden cards)
    timing_cards = time_rows("total", MAX_CARDS)  # what the cost follows
    under = [t["threshold"] for t in timing_cards if t["p95"] < SECONDS_BAR and t["unsolved"] == 0]
    picked = max(under) if under else min(MAX_CARDS)
    mean_loss = statistics.fmean(r["loss_points"] for r in rows) if rows else float("nan")

    def table(rows_: list[dict[str, Any]], head: str) -> list[str]:
        return [
            f"| {head} | Positions | Not solved | Mean s | 95th percentile s | Max s |",
            "|---:|---:|---:|---:|---:|---:|",
            *(
                f"| {t['threshold']} | {t['positions']} | {t['unsolved']} | {t['mean']:.3f} | "
                f"{t['p95']:.3f} | {t['max']:.3f} |"
                for t in rows_
            ),
        ]

    cand = puct2000(args.net, f"infoset_endgame=12,infoset_max_cards={picked}")
    opp = puct2000(args.net)
    reports = play_cell(args, progress, name, cand, opp, args.deals)
    lines = [
        f"(b) Information-set exact endgame (Task 20) on {len(all_rows)} late positions (≤ 12 "
        f"cards hidden from the player to move, two or more legal moves) from "
        f"PUCT-{args.position_simulations} self-play; {unsolved} not solved within "
        f"{args.exact_max_nodes:,} positions. On the {len(rows)} solved, every deal solved, "
        f"against `ismcts:iterations="
        f"{args.ismcts_iterations}`: same move {same_move:.1%}, a move of the same exact value "
        f"{same_value:.1%} (A15 bar ≥ {A15_BAR:.0%}: "
        f"{'passes' if same_value >= A15_BAR else 'fails'} on the value-equal count); ISMCTS's "
        f"move loses {mean_loss:.2f} points on average "
        f"by the exact values. Forced results: {sum(1 for r in rows if r['forced'])} of {n}.",
        "",
        f"Native time per exact call (one thread; {load.get('workers')} positions solved in "
        f"parallel, {load.get('cpu_percent_before_start')} % CPU busy before the start):",
        "",
        *table(timing, "Hidden cards ≤"),
        "",
        "By cards in all hands together (the observer's own included), which the cost follows:",
        "",
        *table(timing_cards, "Cards in play ≤"),
        "",
        f"Largest limit on cards in play with every position solved and 95 % of calls under "
        f"{SECONDS_BAR:.0f} s natively: {picked} (with at most 12 hidden). Node WebAssembly "
        "timing waits for Task 16's harness.",
        "",
        f"`{cand}` vs `{opp}`, {args.deals:,} duplicate deals.",
        "",
        *TABLE_HEAD,
        *(
            _row(f"PUCT-2000 + exact endgame (≤ {picked} cards in play)", lu, r)
            for lu, r in reports.items()
        ),
    ]
    _finish(
        args,
        name,
        {
            "positions": n,
            "same_move": same_move,
            "same_value": same_value,
            "a15_passed": same_value >= A15_BAR,
            "unsolved": unsolved,
            "timing_by_hidden": timing,
            "timing_by_cards_in_play": timing_cards,
            "picked_max_cards": picked,
            "candidate": cand,
            "opponent": opp,
            "reports": reports,
        },
        lines,
    )
    progress.update(None, done=name)


COMMANDS = {"rules-mask": rules_mask, "endgame": endgame}
CELLS = {
    "rules-mask": ["a_rules_mask 1+3", "a_rules_mask ABAB"],
    "endgame": ["b_endgame 1+3", "b_endgame ABAB"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--net", default=DEFAULT_NET, help=".pknn network v2 for PUCT")
    ap.add_argument("--deals", type=int, default=300, help="deals per match")
    ap.add_argument("--positions", type=int, default=300, help="endgame: late positions")
    ap.add_argument("--position-simulations", type=int, default=300)
    ap.add_argument("--ismcts-iterations", type=int, default=100_000)
    ap.add_argument(
        "--exact-max-nodes", type=int, default=20_000_000, help="endgame: cap per exact solve"
    )
    ap.add_argument("--share-games", type=int, default=40, help="rules-mask: self-play games")
    ap.add_argument("--share-iterations", type=int, default=300)
    ap.add_argument("--block", type=int, default=100, help="deals per saved block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="p3_matches", help="folder under runs/")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.block = min(args.block, args.deals)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)  # network paths in agent specs are relative to the repo root
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    progress = Progress(out, [c for n in names for c in CELLS[n]])
    for n in names:
        COMMANDS[n](args, progress)
    progress.update(None)


if __name__ == "__main__":
    main()
