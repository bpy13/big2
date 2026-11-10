"""Build-run phase 3 experiments (Tasks 18, 20-24). Results: docs/study-results/build-phase3.md.

Usage::

    python experiments/build_phase3.py rules-mask --workers 8
    python experiments/build_phase3.py belief --net N.pknn --belief-net B.pknn --out belief_r1
    python experiments/build_phase3.py endgame --workers 8
    python experiments/build_phase3.py endgame --positions 6 --deals 4 --out p3_tiny  # smoke run
    python experiments/build_phase3.py final --deals 10000 --block 200 --out final_matches
    python experiments/build_phase3.py b2 --deals 1000 --out b2_charlesworth
    python experiments/build_phase3.py seeds-train --out seed_crossplay
    python experiments/build_phase3.py seeds-match --deals 1000 --out seed_crossplay
    python experiments/build_phase3.py probe --probes validate_greedy validate_ismcts
    python experiments/build_phase3.py probe-eval --deals 1000 --out probe_eval

* ``rules-mask`` (Task 18, bar A13): the share of drawn deals the rules mask rejects per
  decision (on positions from ISMCTS self-play), then PUCT-2000 with ``belief=rules``
  against the same PUCT without it, ``--deals`` duplicate deals, both lineups. A13: the
  interval includes or exceeds 0 (the mask loses nothing).
* ``belief`` (Task 19, bar A14): PUCT-2000 with ``belief=net`` (uniform candidates weighted by
  the learned belief network, ``--belief-net``) against PUCT-2000 with ``belief=rules``,
  ``--deals`` duplicate deals, both lineups. A14: the net stays in the ladder only if the
  paired gain is at least +0.5 with the interval excluding 0.
* ``endgame`` (Task 20, bar A15): ``--positions`` late positions (at most 12 cards hidden
  and at most 14 in all hands) from PUCT self-play; on each, the information-set endgame
  (every deal solved, cap ``--exact-max-nodes``) against ISMCTS-100,000: agreement on the
  move (A15: ≥ 95 %), native time per call by hidden-card count (the plan's thresholds 8,
  10, 12) and by cards in all hands (8–14, what the cost follows; the largest limit with
  every position solved and 95 % of calls under 1 s is picked), then PUCT-2000 with
  ``infoset_endgame=12,infoset_max_cards=<that limit>`` against PUCT-2000 without it,
  ``--deals`` duplicate deals, both lineups. Node WebAssembly timing is Task 16's harness
  (not here).

* ``final`` (Task 24, bar A19): the bot that ships by default against ISMCTS-1000. The plan
  named PUCT-2000; no network passed A9, so the default in docs/open-items.md ships ISMCTS as
  ``expert``, and this measures that bot.
* ``b2`` (Task 21, bar A16): ``expert`` against Charlesworth's released network, under his
  rules (the engine's hidden ``four_card_plays`` flag) and his scoring (``sum_of_others``, the
  tournament's ``--score-as``), ``--deals`` duplicate deals, both lineups. The weights are
  local only (``runs/external/charlesworth/``, never committed; B14). The match is the plan's
  PUCT-2000 slot filled by ``expert``, the bot that ships (see ``final``).
* ``seeds-train``, ``seeds-match`` (Task 23, bar A18): a second network from fresh weights
  (seed 1) on the labels of rounds 1-3, then PUCT-2000 r3 against PUCT-2000 of that network.
* ``probe-scratch-eval`` (Task 33): the from-scratch probe (``probe --probes scratch_expert``,
  fresh weights, three `expert` seats) played against `expert`, 1 + 3.
* ``probe``, ``probe-eval`` (Task 22, bar A17): best-response probes. A learner in one seat
  trains against three frozen seats of the candidate. Every probe starts from the same
  distilled network: a search opponent allows too few games for a learner from scratch.
  ``probe-eval`` plays that start and each probe's end against the candidate, 1 + 3; the gain
  is the difference. A search candidate gives far fewer games per hour, so the evaluation
  also plays the other probes at the snapshot with as many games as the slowest probe had.

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
from expert_iteration import WEIGHTS, gpu_busy_percent, puct

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
#: The bot that ships by default (docs/open-items.md): the ladder's `expert`.
EXPERT = "ismcts:iterations=10000,infoset_endgame=12,infoset_max_cards=10"
ISMCTS_1000 = "ismcts:iterations=1000"
DMC_10H = "dmc:checkpoint=models/big2-dmc-selfplay-10h.pt"
#: Every probe's start: the distilled duty-on network (`medium` plays its export).
PROBE_INIT = "models/big2-distilled-duty.pt"
#: Probe name -> frozen seats, minutes of training (plan Task 22: 10 h each, validation 2 h).
PROBES = {
    "validate_greedy": ("greedy", 120),
    "validate_ismcts": (ISMCTS_1000, 120),
    "ismcts1000": (ISMCTS_1000, 600),
    "dmc10h": (DMC_10H, 600),
    "expert": (EXPERT, 600),
}
#: Task 33: probes from fresh weights (no ``init``), same net shape as ``PROBE_INIT``.
SCRATCH_PROBES = {"scratch_expert": (EXPERT, 600)}
#: Round 3's PUCT setting (build-phase3.md §1).
R3_PKNN = ROOT / "runs" / "net_r3" / "pvnet-r3.pknn"
R3_LAMBDA = 0.5
SEED_NET = ROOT / "runs" / "net_seedB"


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


# belief (Task 19) -----------------------------------------------------------------

#: A14: the learned belief stays in the ladder only with this gain, interval above 0.
A14_GAIN = 0.5


def belief(args: argparse.Namespace, progress: Progress) -> None:
    name = "a_belief_net"
    if _done(args, name):
        return
    if not args.belief_net:
        raise SystemExit("belief needs --belief-net <belief .pknn>")
    cand = puct2000(args.net, f"belief=net,belief_net={args.belief_net}")
    opp = puct2000(args.net, "belief=rules")
    reports = play_cell(args, progress, name, cand, opp, args.deals)
    verdict = {
        lu: bool(r["paired_diff"] >= A14_GAIN and r["paired_ci95"][0] > 0)
        for lu, r in reports.items()
    }
    keep = all(verdict.values())
    lines = [
        f"`{cand}` vs `{opp}`, {args.deals:,} duplicate deals, duty on, tiered.",
        "",
        *TABLE_HEAD,
        *(_row("PUCT-2000 belief net vs rules mask", lu, r) for lu, r in reports.items()),
        "",
        f"A14 (paired gain ≥ +{A14_GAIN} with the interval above 0): "
        + ", ".join(f"{lu} {'yes' if v else 'no'}" for lu, v in verdict.items())
        + (". Keep the sampler in the ladder." if keep else ". Teaching only."),
    ]
    _finish(
        args,
        name,
        {"candidate": cand, "opponent": opp, "reports": reports, "a14": verdict, "keep": keep},
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


# final (Task 24) ------------------------------------------------------------------


def final(args: argparse.Namespace, progress: Progress) -> None:
    name = "a_expert_vs_ismcts1000"
    if _done(args, name):
        return
    reports = play_cell(args, progress, name, EXPERT, ISMCTS_1000, args.deals)
    lines = [
        f"`{EXPERT}` vs `{ISMCTS_1000}`, {args.deals:,} duplicate deals.",
        "",
        *TABLE_HEAD,
        *(_row("`expert` vs ISMCTS-1000", lu, r) for lu, r in reports.items()),
    ]
    results = {"candidate": EXPERT, "opponent": ISMCTS_1000, "reports": reports}
    _finish(args, name, results, lines)
    progress.update(None, done=name)


# b2 (Task 21) ---------------------------------------------------------------------

#: His agent plays the network's best legal action (``temperature`` 0).
CHARLESWORTH = "charlesworth"
B2_LINEUPS = {
    "1+3": [EXPERT, CHARLESWORTH, CHARLESWORTH, CHARLESWORTH],
    "ABAB": [EXPERT, CHARLESWORTH, EXPERT, CHARLESWORTH],
}


def _b2_block(lineup: list[str], block_seed: int, deals: int, workers: int) -> list[dict]:
    """Raw per-deal records of ``deals`` duplicate deals (four rotations each) under his
    rules and scoring; the deals are those of ``seed = block_seed`` in the tournament."""
    from big2.eval.tournament import _play_deals
    from big2.games.big2 import Rules

    seeds = [block_seed * 1_000_000 + d for d in range(deals)]
    chunks = [seeds[i::workers] for i in range(workers)]
    rules = Rules(four_card_plays=True)
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(
            _play_deals,
            [lineup] * workers,
            chunks,
            [rules] * workers,
            ["sum_of_others"] * workers,
        )
        return [r for part in parts for r in part]


def _b2_stats(raw: list[dict], spec: str) -> dict[str, Any]:
    """Mean score and win rate of ``spec`` over deals (each deal's four games averaged)."""
    from big2.eval.tournament import _mean_ci

    scores = [sum(r[f"score|{spec}"]) / len(r[f"score|{spec}"]) for r in raw]
    wins = [sum(r[f"win|{spec}"]) / len(r[f"win|{spec}"]) for r in raw]
    time_s = sum(sum(r[f"time|{spec}"]) for r in raw)
    moves = sum(sum(r[f"moves|{spec}"]) for r in raw)
    m, ci = _mean_ci(scores)
    w, wci = _mean_ci(wins)
    return {
        "scores": scores,
        "mean": m,
        "ci95": ci,
        "win": w,
        "win_ci95": wci,
        "seconds_per_move": time_s / max(moves, 1),
    }


def b2(args: argparse.Namespace, progress: Progress) -> None:
    name = "b2_charlesworth"
    if _done(args, name):
        return
    from big2.eval.tournament import _mean_ci

    folder = ROOT / "runs" / args.out / name
    reports: dict[str, Any] = {}
    for lu, lineup in B2_LINEUPS.items():
        raw: list[dict] = []
        for start in range(0, args.deals, args.block):
            n = min(args.block, args.deals - start)
            path = folder / lu.replace("+", "p") / f"block_s{args.seed}_d{start}-{start + n}.json"
            if path.exists():
                raw += json.loads(path.read_text(encoding="utf-8"))
                continue
            progress.update(f"{name} {lu}, deals up to {start + n} of {args.deals}")
            # Seed 1_000 * seed + block index: blocks never share a deal.
            block = _b2_block(
                lineup, 1_000 * (args.seed + 1) + start // args.block, n, args.workers
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(block), encoding="utf-8")
            tmp.replace(path)
            raw += block
        e, c = _b2_stats(raw, EXPERT), _b2_stats(raw, CHARLESWORTH)
        diff = [a - b for a, b in zip(e["scores"], c["scores"], strict=True)]
        d_mean, d_ci = _mean_ci(diff)
        reports[lu] = {
            "deals": len(raw),
            "expert": {k: v for k, v in e.items() if k != "scores"},
            "charlesworth": {k: v for k, v in c.items() if k != "scores"},
            "paired_diff": d_mean,
            "paired_ci95": d_ci,
        }
        print(
            f"done {name} {lu}: expert {e['mean']:+.2f} +- {e['ci95']:.2f}, win {e['win']:.1%}; "
            f"Charlesworth {c['mean']:+.2f} +- {c['ci95']:.2f}",
            flush=True,
        )
        progress.update(None, done=f"{name} {lu}")
    lines = [
        f"`expert` (`{EXPERT}`) vs Charlesworth's network (`{CHARLESWORTH}`, best legal action), "
        f"his rules (`four_card_plays`) and his scoring (`sum_of_others`), "
        f"{args.deals:,} duplicate deals per lineup, each deal played in four rotations; "
        "mean score per game with the 95 % interval over deals.",
        "",
        "| Lineup | Deals | `expert` mean | `expert` win rate | Charlesworth mean "
        "| Charlesworth win rate | Paired diff expert - Charlesworth | s per move expert / his |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {lu} | {r['deals']:,} | {r['expert']['mean']:+.2f} ± {r['expert']['ci95']:.2f} "
            f"| {r['expert']['win']:.1%} ± {r['expert']['win_ci95']:.1%} "
            f"| {r['charlesworth']['mean']:+.2f} ± {r['charlesworth']['ci95']:.2f} "
            f"| {r['charlesworth']['win']:.1%} ± {r['charlesworth']['win_ci95']:.1%} "
            f"| {r['paired_diff']:+.2f} ± {r['paired_ci95']:.2f} "
            f"| {r['expert']['seconds_per_move']:.3f} "
            f"/ {r['charlesworth']['seconds_per_move']:.4f} |"
            for lu, r in reports.items()
        ),
    ]
    _finish(args, name, {"candidate": EXPERT, "opponent": CHARLESWORTH, "reports": reports}, lines)
    progress.update(None, done=name)


# seeds (Task 23) ------------------------------------------------------------------


def wait_for_gpu(args: argparse.Namespace, progress: Progress, what: str) -> None:
    """Plan rule: wait while another job keeps the GPU over 20 % busy (at most 2 h)."""
    waited = 0
    while (busy := gpu_busy_percent()) is not None and busy > 20 and waited < args.gpu_wait:
        progress.update(f"{what}: waiting for the GPU ({busy:.0f} % busy)")
        time.sleep(300)
        waited += 5


def seeds_train(args: argparse.Namespace, progress: Progress) -> None:
    """A network from fresh weights, seed 1, on the labels r3 was trained on."""
    from big2.rl.distill import DistillV2Config, train_v2
    from big2.rl.export import export_pknn

    pknn = SEED_NET / "pvnet-seedB.pknn"
    if not (SEED_NET / "results.json").exists():
        wait_for_gpu(args, progress, "seed B")
        progress.update("seed B training, see runs/net_seedB/progress.json")
        latest = SEED_NET / "latest.pt"
        data = [str(ROOT / "runs" / f"round{n}" / "data") for n in (3, 1, 2)]
        train_v2(
            DistillV2Config(
                data_dir=data[0],
                extra_data_dirs=data[1:],
                run_dir=str(SEED_NET),
                device=args.device,
                max_minutes=args.train_minutes,
                resume=str(latest) if latest.exists() else None,
                seed=1,
                **WEIGHTS,
            ),
            log=lambda r: print(r, flush=True),
        )
    if not pknn.exists():
        export_pknn(SEED_NET / "best.pt", pknn)
    progress.update(None, done="seed B training")


def seeds_match(args: argparse.Namespace, progress: Progress) -> None:
    name = "a_r3_vs_seedB"
    if _done(args, name):
        return
    cand = puct(R3_PKNN, R3_LAMBDA)
    opp = puct(SEED_NET / "pvnet-seedB.pknn", R3_LAMBDA)
    reports = play_cell(args, progress, name, cand, opp, args.deals)
    covers = {lu: r["paired_ci95"][0] <= 0 <= r["paired_ci95"][1] for lu, r in reports.items()}
    lines = [
        f"`{cand}` vs `{opp}`, {args.deals:,} duplicate deals. A18 (the interval covers 0): "
        + ", ".join(f"{lu} {'passes' if ok else 'fails'}" for lu, ok in covers.items())
        + ".",
        "",
        *TABLE_HEAD,
        *(_row("r3 vs seed B, PUCT-2000 λ = 0.5", lu, r) for lu, r in reports.items()),
    ]
    results = {"candidate": cand, "opponent": opp, "reports": reports, "a18": covers}
    _finish(args, name, results, lines)
    progress.update(None, done=name)


# probe (Task 22) ------------------------------------------------------------------


def probe(args: argparse.Namespace, progress: Progress) -> None:
    """Runs the named probes one at a time; a finished one is skipped, a stopped one resumes
    from its ``latest.pt``, and the time budget counts over resumes."""
    import torch
    from big2.games.big2 import Rules
    from big2.rl.dmc import TrainConfig, train
    from big2.rl.model import NetConfig

    net = NetConfig(**torch.load(PROBE_INIT, map_location="cpu", weights_only=False)["net_config"])
    for name in args.probes:
        opponents, minutes = (PROBES | SCRATCH_PROBES)[name]
        init = None if name in SCRATCH_PROBES else PROBE_INIT
        minutes = args.probe_minutes or minutes
        run = ROOT / "runs" / f"probe_{name}{args.tag}"
        if (run / "results.json").exists():
            print(f"skip {name}: results.json exists", flush=True)
            continue
        wait_for_gpu(args, progress, name)
        run.mkdir(parents=True, exist_ok=True)
        latest = run / "latest.pt"
        search = opponents.startswith("ismcts")

        def log(
            r: dict[str, Any], run: Path = run, name: str = name, total: float = minutes
        ) -> None:
            state = {k: r.get(k) for k in ("iter", "games", "minutes_total", "learner_score")}
            state |= {"probe": name, "budget_minutes": total}
            state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json(run / "progress.json", state)

        progress.update(f"probe {name}: see runs/probe_{name}/progress.json")
        cfg = TrainConfig(
            run_dir=str(run),
            # A search takes one call per decision: few games at a time, on the workers.
            envs=4 * args.workers if search else 256,
            games_per_iter=256,
            eval_every=50,
            snapshot_every=100,
            max_minutes=minutes,
            total_minutes=minutes,
            opponents=opponents,
            opponent_workers=args.workers,
            device=args.device,
            rules=Rules().kwargs(),
            net=net,
        )
        train(cfg, resume=str(latest) if latest.exists() else None, init=init, log=log)
        info = torch.load(latest, map_location="cpu", weights_only=False)["info"]
        _write_json(
            run / "results.json",
            {
                "probe": name,
                "opponents": opponents,
                "init": init,
                "iter": info["iter"],
                "games": info["games"],
                "samples": info["samples"],
                "minutes": info["minutes_total"],
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        progress.update(None, done=name)


def probe_scratch_eval(args: argparse.Namespace, progress: Progress) -> None:
    """Task 33: plays the from-scratch probe's end against three `expert`, 1 + 3. The mean
    score of that match is the probe's gain (zero-sum, so parity is 0)."""
    from big2.eval.harness import play_match

    name = "a_probe_scratch"
    if _done(args, name):
        return
    run = ROOT / "runs" / f"probe_scratch_expert{args.tag}"
    games = json.loads((run / "results.json").read_text())["games"]
    rel = (run / "latest.pt").as_posix().removeprefix(ROOT.as_posix() + "/")
    progress.update(f"scratch_expert_end: {rel} vs expert")
    r = play_match(
        f"dmc:checkpoint={rel}",
        EXPERT,
        "1+3",
        args.deals,
        seed=args.seed,
        workers=args.workers,
        cache_dir=ROOT / "runs" / args.out / name / "end",
        block=args.block,
    ).to_dict()
    lo, hi = r["ci95"]
    lines = [
        f"From-scratch probe (fresh weights, three frozen `expert`), {games:,} training games; "
        f"{args.deals:,} deals, 1 + 3.",
        "",
        "| Frozen seats | Games | Gain (mean score) | 95 % interval |",
        "|---|---:|---:|---|",
        f"| expert (`{EXPERT}`) | {games:,} | {r['mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |",
        "",
        f"Interval excludes 0: {lo > 0 or hi < 0}",
    ]
    _finish(args, name, {"games": games, "end": r}, lines)
    progress.update(None, done=name)


def _snapshot_at(run: Path, games: int) -> Path:
    """The first snapshot of a probe with at least ``games`` games, else its last state."""
    for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["games"] >= games and rec["iter"] % 100 == 0:
            return run / f"iter_{rec['iter']:05d}.pt"
    return run / "latest.pt"


def probe_eval(args: argparse.Namespace, progress: Progress) -> None:
    from big2.eval.harness import play_match

    name = "a_probes"
    if _done(args, name):
        return

    def score(label: str, learner: Path | str, opponent: str) -> dict[str, Any]:
        rel = Path(learner).as_posix().removeprefix(ROOT.as_posix() + "/")
        progress.update(f"{label}: {rel} vs {opponent}")
        r = play_match(
            f"dmc:checkpoint={rel}",
            opponent,
            "1+3",
            args.deals,
            seed=args.seed,
            workers=args.workers,
            cache_dir=ROOT / "runs" / args.out / name / label,
            block=args.block,
        ).to_dict()
        progress.update(None, done=label)
        return r

    runs = {n: ROOT / "runs" / f"probe_{n}{args.tag}" for n in PROBES}
    done = {n: json.loads((run / "results.json").read_text()) for n, run in runs.items()}
    fewest = min(d["games"] for d in done.values())
    rows: dict[str, dict[str, Any]] = {}
    for n, (opp, _) in PROBES.items():
        start = score(f"{n}_start", PROBE_INIT, opp)
        end = score(f"{n}_end", runs[n] / "latest.pt", opp)
        same = _snapshot_at(runs[n], fewest)
        early = score(f"{n}_at_{fewest}", same, opp) if same.name != "latest.pt" else end
        half_width = (start["ci95"][1] - start["ci95"][0] + end["ci95"][1] - end["ci95"][0]) / 2
        rows[n] = {
            "opponents": opp,
            "games": done[n]["games"],
            "start": start,
            "end": end,
            "early_snapshot": same.name,
            "early": early,
            "gain": end["mean"] - start["mean"],
            # Adds the two half widths: wider than the true interval of the difference.
            "gain_half_width": half_width,
            "gain_at_equal_games": early["mean"] - start["mean"],
        }
    gain = {n: r["gain"] for n, r in rows.items()}
    early = {n: r["gain_at_equal_games"] for n, r in rows.items()}
    a17 = {
        "smallest_gain_vs_expert": gain["expert"] < min(gain["ismcts1000"], gain["dmc10h"]),
        "smallest_at_equal_games": early["expert"] < min(early["ismcts1000"], early["dmc10h"]),
        "validation_greedy_above_ismcts": gain["validate_greedy"] > gain["validate_ismcts"],
    }
    lines = [
        f"Probes in one seat vs three frozen seats, {args.deals:,} deals each, 1 + 3. Start: "
        f"`{PROBE_INIT}`. The ± adds the two half widths, so it is wider than the true interval.",
        "",
        f"| Frozen seats | Games | Start | End | Gain | Gain at {fewest:,} games |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {n} (`{r['opponents']}`) | {r['games']:,} | {r['start']['mean']:+.2f} | "
            f"{r['end']['mean']:+.2f} | {r['gain']:+.2f} ± {r['gain_half_width']:.2f} | "
            f"{r['gain_at_equal_games']:+.2f} |"
            for n, r in rows.items()
        ),
        "",
        f"A17: {a17}",
    ]
    _finish(args, name, {"rows": rows, "fewest_games": fewest, "a17": a17}, lines)
    progress.update(None, done=name)


COMMANDS = {
    "rules-mask": rules_mask,
    "belief": belief,
    "endgame": endgame,
    "final": final,
    "b2": b2,
    "seeds-train": seeds_train,
    "seeds-match": seeds_match,
    "probe": probe,
    "probe-eval": probe_eval,
    "probe-scratch-eval": probe_scratch_eval,
}
#: What ``all`` runs (the later commands have their own job).
ALL = ["rules-mask", "endgame"]
CELLS = {
    "rules-mask": ["a_rules_mask 1+3", "a_rules_mask ABAB"],
    "belief": ["a_belief_net 1+3", "a_belief_net ABAB"],
    "endgame": ["b_endgame 1+3", "b_endgame ABAB"],
    "final": ["a_expert_vs_ismcts1000 1+3", "a_expert_vs_ismcts1000 ABAB"],
    "b2": ["b2_charlesworth 1+3", "b2_charlesworth ABAB"],
    "seeds-train": ["seed B training"],
    "seeds-match": ["a_r3_vs_seedB 1+3", "a_r3_vs_seedB ABAB"],
    "probe": list(PROBES),
    "probe-scratch-eval": ["scratch_expert_end"],
    "probe-eval": [f"{n}_{part}" for n in PROBES for part in ("start", "end")],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--net", default=DEFAULT_NET, help=".pknn network v2 for PUCT")
    ap.add_argument("--belief-net", default="", help="belief: the learned belief .pknn")
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-minutes", type=float, default=180.0, help="seeds-train: budget")
    ap.add_argument("--gpu-wait", type=int, default=120, help="minutes to wait for a free GPU")
    ap.add_argument("--probes", nargs="+", choices=[*PROBES, *SCRATCH_PROBES], default=list(PROBES))
    ap.add_argument("--probe-minutes", type=float, default=0, help="0: the plan's budgets")
    ap.add_argument("--tag", default="", help="suffix for the probe folders (a test run)")
    args = ap.parse_args()
    args.block = min(args.block, args.deals)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)  # network paths in agent specs are relative to the repo root
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = ALL if args.command == "all" else [args.command]
    progress = Progress(out, [c for n in names for c in CELLS[n]])
    for n in names:
        COMMANDS[n](args, progress)
    progress.update(None)


if __name__ == "__main__":
    main()
