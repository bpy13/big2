"""Mistake mining: same-rung games, every decision labelled by network + search (plan Tasks 42a, 42).

    python experiments/mine_mistakes.py label  --out runs/p6_mistakes_pilot --deals 100
    python experiments/mine_mistakes.py puzzles --out runs/p6_mistakes_pilot --count 60
    python experiments/mine_mistakes.py all    --out runs/p6_mistakes_pilot --deals 100

``label`` plays, for every band, ``--deals`` deals with all four seats held by that band's
rung (so the positions are the ones the band's games produce) and writes one line per
decision to ``<out>/decisions.jsonl`` (shape: :func:`validate_decision`). Each non-forced
decision is searched twice with :func:`big2._core.net_search`: the **reference** (offline,
no time limit) and the **knee** (the in-game setting), both from the same network. The
reference covers the top ``k`` moves by prior plus every move with prior >= 1 % plus the
move played; the knee's pick is scored by the reference's paired loss. ``puzzles`` turns the
strongest decisions into ``<out>/candidates.json``.

Bands are rungs of ``runs/p6_ladder/ladder.json`` (Task 40). Resumable: a deal already in
``decisions.jsonl`` is not replayed. Runs from this checkout (``PYTHONPATH=python``);
worker processes come from ``HKQ_THREADS`` (default 8), one search thread each.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

NET = "runs/p6_net/selfplay-h24.pknn"
LADDER = "runs/p6_ladder/ladder.json"
#: Pilot bands (Task 42a): lowest non-random rung, two middle ones, the raw network. Rung name
#: -> ladder rung of runs/p6_ladder/ladder.json.
BANDS = ["net-t4.34", "easy", "net-t1", "net"]
#: Provisional settings (Task 39 has not fixed the real ones): reference and in-game knee.
REF = {"k": 8, "n": 256, "depth": None}
KNEE = {"k": 3, "n": 64, "depth": None}
PRIOR_FLOOR = 0.01
FLAT_LOSS = 0.2
#: D102 lines (points): great, good, inaccuracy, mistake (upper limits of each word).
LINES = (0.2, 0.69, 2.75, 6.86)
WORDS = ("great", "good", "inaccuracy", "mistake", "blunder")
ALL_LEGAL = 1_000_000


def word(loss: float, lines: tuple[float, ...] = LINES) -> str:
    """The grade word of a loss under ``lines`` (upper limit of each word)."""
    for limit, w in zip(lines, WORDS):
        if loss <= limit:
            return w
    return WORDS[-1]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def band_specs() -> dict[str, dict[str, Any]]:
    d = json.loads((ROOT / LADDER).read_text(encoding="utf-8"))
    rungs = {r["name"]: r for r in d["rungs"]}
    return {b: {"spec": rungs[b]["spec"], "elo": rungs[b]["elo"]} for b in BANDS}


def deal_seed(band: str, i: int) -> int:
    return (zlib.crc32(f"mistakes|{band}".encode()) % 100_000) * 100_000 + i


def _cards(mask: int) -> str:
    from big2._core import format_cards

    return format_cards(mask) if mask else "pass"


def position(state: Any) -> dict[str, Any]:
    """The decision's position: hands, who acts, the play to beat."""
    from big2._core import format_cards

    cp = state.current_play
    return {
        "hands": [format_cards(h).split() for h in state.hands],
        "current_player": state.current_player,
        "current_play": None
        if cp is None
        else {"player": cp[0], "cards": format_cards(cp[1]).split(), "combo": "combo." + cp[2]},
    }


def search(state: Any, cfg: dict[str, Any], seed: int, extra: list[int]) -> dict[str, Any]:
    from big2._core import net_search

    return net_search(
        state, net=NET, k=cfg["k"], n=cfg["n"], depth=cfg["depth"], seed=seed, extra=extra
    )


def label(state: Any, played: int, seed: int) -> dict[str, Any]:
    """Reference and knee labels for one decision (the state before ``played``)."""
    from big2._core import net_search

    legal = state.legal_actions()
    rec: dict[str, Any] = {"legal": len(legal), "forced": len(legal) == 1}
    if rec["forced"]:
        return rec
    pri = net_search(state, net=NET, k=ALL_LEGAL, n=1, depth=0, seed=0)
    n_prior = sum(1 for c in pri["candidates"] if c["prior"] >= PRIOR_FLOOR)
    ref_cfg = {**REF, "k": min(len(legal), max(REF["k"], n_prior))}
    t0 = time.perf_counter()
    ref = search(state, ref_cfg, seed, [played])
    ref_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    knee = search(state, KNEE, seed + 1, [])
    knee_s = time.perf_counter() - t0
    moves = [
        {
            "cards": _cards(c["cards"]),
            "prior": round(c["prior"], 5),
            "mean": round(c["mean"], 4),
            "se": round(c["se"], 4),
            "loss": round(c["loss"], 4),
            "loss_se": round(c["loss_se"], 4),
        }
        for c in ref["candidates"]
    ]
    by_cards = {c["cards"]: c for c in ref["candidates"]}
    best = min(ref["candidates"], key=lambda c: (c["loss"], -c["prior"]))
    knee_pick = knee["candidates"][0]["cards"]
    flat = all(
        c["loss"] < FLAT_LOSS or c["loss"] - 1.96 * c["loss_se"] <= 0 for c in ref["candidates"]
    )
    rec.update(
        flat=flat,
        ref=moves,
        best=_cards(best["cards"]),
        played_loss=round(by_cards[played]["loss"], 4),
        played_loss_se=round(by_cards[played]["loss_se"], 4),
        knee_best=_cards(knee_pick),
        knee_loss=round(by_cards[knee_pick]["loss"], 4) if knee_pick in by_cards else None,
        ref_seconds=round(ref_s, 4),
        knee_seconds=round(knee_s, 4),
    )
    return rec


def play_deal(band: str, spec: str, ds: int) -> list[dict[str, Any]]:
    """One same-rung deal: all four seats are ``spec``; every decision labelled."""
    from big2.agents import make_agent
    from big2.games.big2 import Rules

    state = Rules().new_deal(ds)
    agents = [make_agent(spec, seed=ds * 4 + s) for s in range(4)]
    out = []
    turn = 0
    while not state.is_terminal:
        seat = state.current_player
        action = int(agents[seat].act(state))
        rec = {"band": band, "deal": ds, "turn": turn, "seat": seat, "played": _cards(action)}
        rec["position"] = position(state)
        rec.update(label(state, action, ds * 1000 + turn * 2))
        out.append(rec)
        state.apply(action)
        turn += 1
    return out


def _worker(args: tuple[str, str, int]) -> list[dict[str, Any]]:
    return play_deal(*args)


def read_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stage_label(out: Path, deals: int, threads: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "decisions.jsonl"
    done = {(r["band"], r["deal"]) for r in read_decisions(path)}
    specs = band_specs()
    jobs = [
        (b, specs[b]["spec"], deal_seed(b, i))
        for i in range(deals)
        for b in BANDS
        if (b, deal_seed(b, i)) not in done
    ]
    log(f"label: {len(jobs)} deals to play ({threads} workers)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(threads) as pool, path.open("a", encoding="utf-8") as f:
        for n, recs in enumerate(pool.map(_worker, jobs), 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 10 == 0:
                log(f"  {n}/{len(jobs)} deals, {time.perf_counter() - t0:.0f} s")
    log(f"label done in {time.perf_counter() - t0:.0f} s")


def validate_decision(r: dict[str, Any]) -> list[str]:
    """Problems with one ``decisions.jsonl`` line (empty when valid)."""
    errors = [
        f"missing {k}"
        for k in ("band", "deal", "turn", "seat", "played", "position", "legal", "forced")
        if k not in r
    ]
    if errors:
        return errors
    if len(r["position"].get("hands", [])) != 4:
        errors.append("position needs four hands")
    if r["forced"]:
        return errors
    for k in ("flat", "ref", "best", "played_loss", "knee_best", "knee_loss"):
        if k not in r:
            errors.append(f"missing {k}")
    if errors:
        return errors
    moves = {m["cards"]: m for m in r["ref"]}
    if r["played"] not in moves:
        errors.append("the move played must be among the reference's moves")
    if r["best"] not in moves or moves[r["best"]]["loss"] != min(m["loss"] for m in r["ref"]):
        errors.append("best must be the reference move with the least loss")
    for m in r["ref"]:
        for k in ("cards", "prior", "mean", "se", "loss", "loss_se"):
            if k not in m:
                errors.append(f"move missing {k}")
        if m.get("loss", 0) < 0:
            errors.append("negative loss")
    if r["knee_loss"] is not None and r["knee_loss"] < 0:
        errors.append("negative knee loss")
    return errors


# -- puzzles ------------------------------------------------------------------------------


def position_key(pos: dict[str, Any]) -> str:
    cp = pos["current_play"]
    return json.dumps(
        [pos["hands"], pos["current_player"], None if cp is None else [cp["player"], cp["cards"]]]
    )


def candidate(r: dict[str, Any], elo: float) -> dict[str, Any] | None:
    """A puzzle candidate from a decision, or ``None`` (rules of Task 44, D102 lines): not
    forced or flat, the band's bot lost at least the inaccuracy line, the best move is clear
    (second best loses at least that line, its lead's 95 % range is above 0)."""
    from big2.teach import drills

    if r["forced"] or r["flat"] or r["played_loss"] < LINES[2]:
        return None
    ranked = sorted(r["ref"], key=lambda m: (m["loss"], -m["prior"]))
    if len(ranked) < 2:
        return None
    second = ranked[1]
    if second["loss"] < LINES[2] or second["loss"] - 1.96 * second["loss_se"] <= 0:
        return None

    def act(c: str) -> Any:
        if c == "pass":
            return {"type": "action.pass"}
        from big2._core import classify_cards, parse_cards

        kind = classify_cards(parse_cards(c))[0]
        return {"type": "action.play", "cards": c.split(), "combo": "combo." + kind}

    answers = []
    for i, m in enumerate(ranked):
        grade = "best" if i == 0 else {"great": "good"}.get(word(m["loss"]), word(m["loss"]))
        answers.append(
            {
                "action": act(m["cards"]),
                "grade": grade,
                "reasons": [],
                "loss": m["loss"],
                "loss_se": m["loss_se"],
                "prior": m["prior"],
            }
        )
    pos = r["position"]
    start = {"current_player": pos["current_player"], "hands": pos["hands"]}
    if pos["current_play"] is not None:
        start["current_play"] = pos["current_play"]
    cand = {
        "seat": pos["current_player"],
        "position": {"game": "big2", "history": [], "seed": r["deal"], "start": start},
        "answers": [{**a, "action": a["action"]} for a in answers],
        "following": pos["current_play"] is not None,
    }
    # skill_for reads ``action`` as the miner's "pass" or a card dict
    for a in cand["answers"]:
        if a["action"].get("type") == "action.pass":
            a["action"] = "pass"
        else:
            a["action"] = {"cards": a["action"]["cards"], "combo": a["action"]["combo"]}
    skill = drills.skill_for(cand)
    drill = {
        "id": f"big2-m{r['band']}-{r['deal']}-{r['turn']:02d}",
        "game": "big2",
        "skill": skill,
        "prompt": drills.prompt_for(cand, skill),
        "position": cand["position"],
        "seat": cand["seat"],
        "rating": int(round(elo)),
        "answers": [
            {
                "action": drills._action(a["action"]),
                "grade": a["grade"],
                "reasons": [],
                "loss": a["loss"],
                "loss_se": a["loss_se"],
            }
            for a in cand["answers"]
        ],
        "brilliant_answers": [],
        "source": "solver",
        "band": r["band"],
        "rung_elo": round(elo, 1),
        "bot_move": r["played"],
        "bot_loss": r["played_loss"],
        "bot_loss_se": r["played_loss_se"],
        "theme": skill,
        "turn": r["turn"],
    }
    return drill


def stage_puzzles(out: Path, count: int) -> list[dict[str, Any]]:
    specs = band_specs()
    per_band: dict[str, list[dict[str, Any]]] = {b: [] for b in BANDS}
    seen: set[str] = set()
    for r in read_decisions(out / "decisions.jsonl"):
        c = candidate(r, specs[r["band"]]["elo"])
        if c is None:
            continue
        key = position_key(r["position"])
        if key in seen:
            continue
        seen.add(key)
        per_band[r["band"]].append(c)
    for b in BANDS:
        per_band[b].sort(key=lambda d: -d["bot_loss"])
    chosen: list[dict[str, Any]] = []
    quota = count // len(BANDS)
    for b in BANDS:
        chosen += per_band[b][:quota]
    rest = sorted(
        (d for b in BANDS for d in per_band[b][quota:]), key=lambda d: -d["bot_loss"]
    )
    chosen += rest[: count - len(chosen)]
    chosen.sort(key=lambda d: (BANDS.index(d["band"]), -d["bot_loss"]))
    (out / "candidates.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    log(
        f"puzzles: {len(chosen)} chosen; available per band "
        + json.dumps({b: len(v) for b, v in per_band.items()})
    )
    return chosen


def summarize(out: Path) -> dict[str, Any]:
    """Counts, cost and the knee-vs-reference disagreement, per band (for §0 of the results)."""
    rows: dict[str, dict[str, Any]] = {}
    for r in read_decisions(out / "decisions.jsonl"):
        s = rows.setdefault(
            r["band"],
            dict(decisions=0, forced=0, flat=0, scored=0, agree=0, knee_loss=0.0,
                 ref_s=0.0, knee_s=0.0, losses=[], deals=set()),
        )
        s["decisions"] += 1
        s["deals"].add(r["deal"])
        if r["forced"]:
            s["forced"] += 1
            continue
        s["scored"] += 1
        s["flat"] += bool(r["flat"])
        s["agree"] += r["knee_best"] == r["best"]
        s["knee_loss"] += r["knee_loss"] or 0.0
        s["ref_s"] += r["ref_seconds"]
        s["knee_s"] += r["knee_seconds"]
        if not r["flat"]:
            s["losses"].append(r["played_loss"])
    res = {}
    for b, s in rows.items():
        ls = sorted(s["losses"])

        def q(p: float, ls: list[float] = ls) -> float | None:
            return ls[min(len(ls) - 1, int(p * len(ls)))] if ls else None

        n = max(s["scored"], 1)
        res[b] = {
            "deals": len(s["deals"]),
            "decisions": s["decisions"],
            "forced_share": s["forced"] / s["decisions"],
            "flat_share": s["flat"] / n,
            "knee_disagrees": 1 - s["agree"] / n,
            "knee_mean_loss": s["knee_loss"] / n,
            "ref_s_per_decision": s["ref_s"] / n,
            "knee_s_per_decision": s["knee_s"] / n,
            "played_loss_mean": sum(ls) / len(ls) if ls else None,
            "played_loss_p50_p90_p97": [q(0.5), q(0.9), q(0.97)],
            "share_above_lines": [
                sum(1 for x in ls if x > line) / len(ls) if ls else None for line in LINES
            ],
        }
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# -- Task 42: PUCT-2,000 reference (decision 0010, method decision 0011) ------------------

#: Task 42's bands: every rung of Task 40's ladder except ``random`` (until Task 41 names
#: the final ones), weakest first.
BANDS42 = ["net-t4.34", "easy", "medium", "hard", "net-t1", "expert", "net"]
BELIEF_NET = "runs/p6_mistakes/belief-r1.pknn"
#: The reference: PUCT at 2,000 simulations, c_puct 2.5 (Task 52 not ticked), the phase 3
#: belief network for the hidden cards, the exact information-set endgame (phase 3 §4
#: defaults: at most 12 hidden cards, at most 10 in all hands, 1 M positions). 2,048 belief
#: candidates, not the default 256: at 256 the loss of B3-02's Qh varied by 2.6 points (SD)
#: between seeds, at 2,048 by 1.2 (decision 0011).
PUCT_REF = {
    "iterations": 2000, "c_puct": 2.5, "belief": "net", "infoset_endgame": 12,
    "belief_candidates": 2048,
}
TOP_K = 8
#: One decision in this many is labelled again with a new seed and with ``belief="rules"``.
RECHECK_EVERY = 20


def reference_string(belief: str = "net") -> str:
    r = PUCT_REF
    return (
        f"puct:sims={r['iterations']},c_puct={r['c_puct']},belief={belief},"
        f"candidates={r['belief_candidates']},"
        f"root=per-move,infoset_endgame={r['infoset_endgame']},net={Path(NET).name}"
    )


def priors(state: Any) -> dict[int, float]:
    """The network's prior over every legal move (one forward pass)."""
    from big2._core import net_search

    res = net_search(state, net=NET, k=ALL_LEGAL, n=1, depth=0, seed=0)
    return {int(c["cards"]): float(c["prior"]) for c in res["candidates"]}


def options_of(pri: dict[int, float], extra: list[int]) -> list[int]:
    """Moves to value: prior >= 1 %, the top 8 by prior, and every move in ``extra``."""
    ranked = sorted(pri, key=lambda a: -pri[a])
    keep = [a for a in ranked if pri[a] >= PRIOR_FLOOR] + ranked[:TOP_K] + list(extra)
    return list(dict.fromkeys(keep))


def puct_values(
    state: Any, options: list[int], seed: int, belief: str = "net"
) -> tuple[dict[int, tuple[float, float]], bool]:
    """``{move: (EV, standard error)}`` of each option under the reference, and whether the
    exact endgame answered. Exact: every legal move from one solve, standard error 0. Else
    one PUCT search per option with the root fixed to that move, the same seed for each."""
    from big2._core import analyze

    kw: dict[str, Any] = dict(
        algorithm="puct", net=NET, c_puct=PUCT_REF["c_puct"], belief=belief, seed=seed
    )
    if belief == "net":
        kw["belief_net"] = BELIEF_NET
        kw["belief_candidates"] = PUCT_REF["belief_candidates"]
    probe = analyze(state, iterations=1, infoset_endgame=PUCT_REF["infoset_endgame"], **kw)
    if probe["exact"]:
        return {int(m["cards"]): (float(m["mean_score"]), 0.0) for m in probe["moves"]}, True
    out = {}
    for a in options:
        m = analyze(state, iterations=PUCT_REF["iterations"], root_move=int(a), **kw)["moves"][0]
        out[int(a)] = (float(m["mean_score"]), float(m["score_ci95"]) / 1.96)
    return out, False


def grade_moves(
    values: dict[int, tuple[float, float]], pri: dict[int, float]
) -> list[dict[str, Any]]:
    """Moves best first with EV, SE, prior, loss to the best and the loss's SE (unpaired:
    each option has its own search)."""
    best = max(values, key=lambda a: (values[a][0], pri.get(a, 0.0)))
    ev_b, se_b = values[best]
    rows = []
    for a, (ev, se) in values.items():
        loss = 0.0 if a == best else max(0.0, ev_b - ev)
        loss_se = 0.0 if a == best else math.hypot(se, se_b)
        rows.append(
            {
                "action": _cards(a),
                "ev": round(ev, 4),
                "se": round(se, 4),
                "prior": round(pri.get(a, 0.0), 5),
                "loss": round(loss, 4),
                "loss_se": round(loss_se, 4),
            }
        )
    rows.sort(key=lambda m: (m["loss"], -m["prior"]))
    return rows


def is_flat(moves: list[dict[str, Any]]) -> bool:
    """Every move's loss is below 0.2 points or its 95 % range covers 0."""
    return all(m["loss"] < FLAT_LOSS or m["loss"] - 1.96 * m["loss_se"] <= 0 for m in moves)


def label_puct(
    state: Any, played: int, seed: int, extra: list[int] | None = None, belief: str = "net"
) -> dict[str, Any]:
    """One decision under the PUCT reference (the state before ``played``)."""
    legal = [int(a) for a in state.legal_actions()]
    rec: dict[str, Any] = {"legal_count": len(legal), "forced": len(legal) == 1}
    if rec["forced"]:
        return rec
    t0 = time.perf_counter()
    pri = priors(state)
    opts = options_of(pri, [played, *(extra or [])])
    values, exact = puct_values(state, opts, seed, belief)
    moves = grade_moves(values, pri)
    by = {m["action"]: m for m in moves}
    rec.update(
        exact=exact,
        flat=is_flat(moves),
        moves=moves,
        best=moves[0]["action"],
        loss=by[_cards(played)]["loss"],
        loss_se=by[_cards(played)]["loss_se"],
        reference=reference_string(belief),
        seconds=round(time.perf_counter() - t0, 4),
    )
    return rec


def rung_probs(spec: str, moves: list[dict[str, Any]], pri: dict[int, float]) -> str:
    """Adds ``rung_prob`` to each move: the rung policy's probability of playing it, exact
    for the network-prior rungs (softmax of log prior / τ over the legal moves; τ = 0 the
    top move), else the network prior at τ = 1. Returns which of the two it is."""
    temp = None
    if spec.startswith("netprior:"):
        kv = dict(x.split("=", 1) for x in spec.split(":", 1)[1].split(","))
        temp = float(kv["temperature"])
    if temp is None:
        dist, kind = pri, "net_prior_t1"
    elif temp == 0:
        top = sorted(pri, key=lambda a: -pri[a])[0]  # ties keep the engine order, as the agent
        dist, kind = {a: float(a == top) for a in pri}, "exact"
    else:
        lg = {a: math.log(max(p, 1e-300)) / temp for a, p in pri.items()}
        mx = max(lg.values())
        z = sum(math.exp(v - mx) for v in lg.values())
        dist, kind = {a: math.exp(v - mx) / z for a, v in lg.items()}, "exact"
    for m in moves:
        m["rung_prob"] = round(dist.get(_mask(m["action"]), 0.0), 6)
    return kind


def _mask(cards: str) -> int:
    from big2._core import parse_cards

    return 0 if cards == "pass" else int(parse_cards(cards))


def recheck(ds: int, turn: int) -> bool:
    """The decisions (one in ``RECHECK_EVERY``) labelled again for repeatability."""
    return zlib.crc32(f"{ds}|{turn}".encode()) % RECHECK_EVERY == 0


def play_deal42(band: str, spec: str, elo: float, ds: int) -> list[dict[str, Any]]:
    """One same-rung deal of Task 42: every decision labelled by the PUCT reference; the
    deal's full action history on its first line (``history``)."""
    from big2.agents import make_agent
    from big2.games.big2 import Rules

    state = Rules().new_deal(ds)
    agents = [make_agent(spec, seed=ds * 4 + s) for s in range(4)]
    out = []
    turn = 0
    while not state.is_terminal:
        seat = state.current_player
        action = int(agents[seat].act(state))
        seed = ds * 1000 + turn * 2
        rec: dict[str, Any] = {
            "game": "big2", "deal_id": ds, "turn": turn, "seat": seat, "band": band,
            "rung_elo": round(elo, 1), "played": _cards(action), "seed": seed,
            "position": position(state),
        }
        rec.update(label_puct(state, action, seed))
        if not rec["forced"]:
            rec["rung_prob_kind"] = rung_probs(spec, rec["moves"], priors(state))
            if recheck(ds, turn):
                extra = [_mask(m["action"]) for m in rec["moves"]]
                again = label_puct(state, action, seed + 7_919, extra)
                rules = label_puct(state, action, seed, extra, belief="rules")
                keys = ("best", "loss", "loss_se", "flat")
                rec["repeat"] = {k: again[k] for k in keys}
                rec["rules"] = {k: rules[k] for k in keys}
        out.append(rec)
        state.apply(action)
        turn += 1
    out[0]["history"] = [_cards(a) for _, a in state.history]
    return out


def _worker42(args: tuple[str, str, float, int]) -> list[dict[str, Any]]:
    return play_deal42(*args)


def deal_seed42(band: str, i: int) -> int:
    return (zlib.crc32(f"mistakes42|{band}".encode()) % 100_000) * 100_000 + i


def stage_label42(out: Path, deals: int, threads: int, bands: list[str]) -> None:
    """``deals`` same-rung deals per band into ``<out>/decisions-<band>.jsonl`` (resumable:
    a deal already written is not replayed; bands interleaved so a partial run covers all)."""
    out.mkdir(parents=True, exist_ok=True)
    rungs = {r["name"]: r for r in json.loads((ROOT / LADDER).read_text(encoding="utf-8"))["rungs"]}
    jobs = []
    for b in bands:
        done = {r["deal_id"] for r in read_decisions(out / f"decisions-{b}.jsonl")}
        jobs += [
            (b, rungs[b]["spec"], rungs[b]["elo"], deal_seed42(b, i))
            for i in range(deals)
            if deal_seed42(b, i) not in done
        ]
    jobs.sort(key=lambda j: (j[3] % 100_000, bands.index(j[0])))
    log(f"label42: {len(jobs)} deals to play ({threads} workers)")
    t0 = time.perf_counter()
    files = {b: (out / f"decisions-{b}.jsonl").open("a", encoding="utf-8") for b in bands}
    try:
        with ProcessPoolExecutor(threads) as pool:
            for n, recs in enumerate(pool.map(_worker42, jobs), 1):
                f = files[recs[0]["band"]]
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                if n % 20 == 0:
                    log(f"  {n}/{len(jobs)} deals, {time.perf_counter() - t0:.0f} s")
    finally:
        for f in files.values():
            f.close()
    log(f"label42 done in {time.perf_counter() - t0:.0f} s")


SHAPE_KEYS = ("game", "deal_id", "seat", "band", "rung_elo", "position", "legal_count",
              "played", "forced", "seed")


def validate42(r: dict[str, Any]) -> list[str]:
    """Problems with one Task 42 ``decisions-<band>.jsonl`` line (the plan's shape)."""
    errors = [f"missing {k}" for k in SHAPE_KEYS if k not in r]
    if errors or r["forced"]:
        return errors
    errors += [
        f"missing {k}" for k in ("moves", "loss", "loss_se", "flat", "reference") if k not in r
    ]
    if errors:
        return errors
    acts = {m["action"]: m for m in r["moves"]}
    if r["played"] not in acts:
        errors.append("the move played must be among the moves")
    if r["moves"][0]["loss"] != 0 or any(m["loss"] < 0 for m in r["moves"]):
        errors.append("moves must start with the best (loss 0) and have no negative loss")
    for m in r["moves"]:
        errors += [f"move missing {k}" for k in ("action", "ev", "se", "prior") if k not in m]
    if r["played"] in acts and acts[r["played"]]["loss"] != r["loss"]:
        errors.append("loss must be the played move's loss")
    return errors


# -- the 36 review puzzles (hk-games review 2026-11-27) ------------------------------------

REVIEW = (
    Path(__file__).resolve().parents[3]
    / "hk-games/docs/reviews/2026-11-27-big2-puzzle-review.html"
)
PILOT_CANDIDATES = "runs/p6_mistakes_pilot/candidates.json"


def review_ids(html: str) -> list[tuple[str, str]]:
    """``(review id, candidate id)`` pairs in page order (``B3-02 · ... source big2-m...``)."""
    import html as html_lib
    import re

    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html))
    pairs = re.findall(
        r"(B[1-4]-\d\d)\s+·(?:(?!B[1-4]-\d\d).)*?source\s+(big2-m[\w.\-]+)", text, re.S
    )
    return list(dict.fromkeys(pairs))


def grade_word(i: int, loss: float) -> str:
    """The puzzle grade of the answer ranked ``i`` (0 = best) with ``loss`` to the best."""
    return "best" if i == 0 else {"great": "good"}.get(word(loss), word(loss))


def stage_review36(out: Path, review: Path) -> list[dict[str, Any]]:
    """Re-grade the review's puzzles from their source positions, full history replayed."""
    from big2.agents import make_agent
    from big2.games.big2 import Rules

    pairs = review_ids(review.read_text(encoding="utf-8"))
    cands = {c["id"]: c for c in json.loads((ROOT / PILOT_CANDIDATES).read_text(encoding="utf-8"))}
    rungs = {r["name"]: r for r in json.loads((ROOT / LADDER).read_text(encoding="utf-8"))["rungs"]}
    res = []
    for rid, cid in pairs:
        c = cands[cid]
        ds, turn, spec = c["position"]["seed"], c["turn"], rungs[c["band"]]["spec"]
        state = Rules().new_deal(ds)
        agents = [make_agent(spec, seed=ds * 4 + s) for s in range(4)]
        for _ in range(turn):
            state.apply(int(agents[state.current_player].act(state)))
        assert position(state)["hands"] == c["position"]["start"]["hands"], f"{rid}: replay"
        answers = [
            "pass" if a["action"].get("type") == "action.pass" else " ".join(a["action"]["cards"])
            for a in c["answers"]
        ]
        lab = label_puct(
            state, _mask(c["bot_move"]), ds * 1000 + turn * 2, [_mask(x) for x in answers]
        )
        new = {m["action"]: m for m in lab["moves"]}
        ranked = sorted(answers, key=lambda x: (new[x]["loss"], -new[x]["prior"]))
        base = new[ranked[0]]["loss"]
        options = []
        for a_old, x in zip(c["answers"], answers, strict=True):
            to_best = round(new[x]["loss"] - base, 4)
            options.append({
                "move": x,
                "old_loss": a_old["loss"], "old_word": a_old["grade"],
                "new_loss": to_best, "new_loss_se": new[x]["loss_se"],
                "new_word": grade_word(ranked.index(x), to_best),
                "new_ev": new[x]["ev"],
            })
        res.append({
            "id": rid, "source": cid, "band": c["band"], "seat": c["seat"], "turn": turn,
            "history": [_cards(a) for _, a in state.history],
            "bot_move": c["bot_move"],
            "old_best": answers[0], "new_best": ranked[0],
            "best_changed": ranked[0] != answers[0],
            "best_outside_answers": lab["best"] if lab["best"] not in answers else None,
            "bot_old_loss": c["bot_loss"],
            "bot_new_loss": round(new[c["bot_move"]]["loss"] - base, 4),
            "exact": lab["exact"], "flat": lab["flat"],
            "options": options, "reference": lab["reference"],
        })
        log(f"{rid}: best {answers[0]} -> {ranked[0]}; bot {c['bot_move']} "
            f"{c['bot_loss']:.1f} -> {res[-1]['bot_new_loss']:.1f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "review36-regraded.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res



# -- Task 42 report: per-band table and a histogram per band ---------------------------------

#: Histogram bins (points of loss); the last is open above.
HIST_EDGES = [0.0, 0.2, 0.69, 1.5, 2.75, 4.5, 6.86, 10.0, 15.0, 25.0, 40.0]


def stage_report42(out: Path, bands: list[str]) -> str:
    """``<out>/report.md``, ``<out>/table.json`` and ``<out>/hist-<band>.png``."""
    from big2.teach.grading import losses as L

    paths = [out / f"decisions-{b}.jsonl" for b in bands if (out / f"decisions-{b}.jsonl").exists()]
    decisions = L.read_decisions(paths)
    table = L.band_table(decisions, LINES)
    (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    pct = lambda x: "–" if x is None else f"{100 * x:.1f} %"  # noqa: E731
    num = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
    rows = [
        "| band (rung Elo) | deals | decisions | forced | flat | graded | loss mean / p50 / p75 / "
        "p90 / p97 / p99 | share above 0.2 / 0.69 / 2.75 / 6.86 "
        "| same word: new seed / rules belief |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in bands:
        if b not in table:
            continue
        t = table[b]
        p = t["percentiles"]
        rows.append(
            f"| {b} ({t['rung_elo']:.0f}) | {t['deals']} | {t['decisions']:,} | "
            f"{pct(t['forced_share'])} | {pct(t['flat_share'])} | {t['graded']:,} | "
            f"{num(t['mean'])} / " + " / ".join(num(p[k]) for k in ("50", "75", "90", "97", "99"))
            + " | " + " / ".join(pct(t["share_above"][str(x)]) for x in LINES)
            + f" | {pct(t['repeat_same_word'])} ({t['repeat_checked']}) / "
            f"{pct(t['rules_same_word'])} ({t['rules_checked']}) |"
        )
    rows.append("")
    rows.append(
        f"Pooled: same word with a new seed {pct(L.overall_share(table, 'repeat_same_word'))}, "
        f"with the rules belief {pct(L.overall_share(table, 'rules_same_word'))}."
    )
    scored = [r for r in decisions if not r["forced"]]
    if scored:
        secs = sum(r["seconds"] for r in scored) / len(scored)
        exact = sum(bool(r.get("exact")) for r in scored) / len(scored)
        rows.append(
            f"Cost: {secs:.2f} s per graded decision (one thread); exact endgame answered "
            f"{100 * exact:.1f} % of graded decisions."
        )
    _histograms(out, bands, decisions)
    md = "\n".join(rows) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return md


def _histograms(out: Path, bands: list[str], decisions: list[dict[str, Any]]) -> None:
    """One chart per band, same axes: share of graded decisions by loss bin (bins split at
    the D102 lines), as ``hist-<band>.svg`` and ``.png`` (Playwright, as Task 39's curve)."""
    from expert_search_budget import svg_to_png

    from big2.teach.grading import losses as L

    ink, muted, bar, grid, bg = "#0b0b0b", "#52514e", "#2a78d6", "#c9c8c3", "#fcfcfb"
    pairs = zip(HIST_EDGES, HIST_EDGES[1:], strict=False)
    labels = [f"{a:g}–{b:g}" for a, b in pairs] + [f"≥{HIST_EDGES[-1]:g}"]
    shares = {}
    for b in bands:
        ls = [r["loss"] for r in decisions if r["band"] == b and not r["forced"] and not r["flat"]]
        if ls:
            shares[b] = ([c / len(ls) for c in L.histogram(ls, HIST_EDGES)], len(ls))
    top = max((max(v) for v, _ in shares.values()), default=1.0) * 1.15
    W, H, left, right, ptop, pbot = 900, 560, 24, 24, 60, 80  # the helper's viewport
    pw, ph = W - left - right, H - ptop - pbot
    bw = pw / len(HIST_EDGES)
    for b, (v, n) in shares.items():
        parts = [
            f'<rect width="{W}" height="{H}" fill="{bg}"/>',
            f'<text x="{left}" y="24" font-size="15" fill="{ink}">{b}: loss of the move played, '
            f"{n:,} graded decisions</text>",
            f'<line x1="{left}" x2="{W - right}" y1="{ptop + ph}" y2="{ptop + ph}" '
            f'stroke="{grid}"/>',
        ]
        for i, sh in enumerate(v):
            h = ph * sh / top
            x = left + i * bw + 1
            y = ptop + ph - h
            parts.append(
                f'<path d="M{x:.1f},{ptop + ph} V{y + min(4, h):.1f} '
                f"Q{x:.1f},{y:.1f} {x + min(4, h):.1f},{y:.1f} H{x + bw - 2 - min(4, h):.1f} "
                f'Q{x + bw - 2:.1f},{y:.1f} {x + bw - 2:.1f},{y + min(4, h):.1f} V{ptop + ph} Z" '
                f'fill="{bar}"><title>{labels[i]} points: {100 * sh:.1f} %</title></path>'
            )
            if sh >= 0.01:
                parts.append(
                    f'<text x="{x + bw / 2 - 1:.1f}" y="{y - 5:.1f}" font-size="11" '
                    f'text-anchor="middle" fill="{muted}">{100 * sh:.0f}%</text>'
                )
            parts.append(
                f'<text x="{x + bw / 2 - 1:.1f}" y="{ptop + ph + 16}" font-size="11" '
                f'text-anchor="middle" fill="{muted}">{labels[i]}</text>'
            )
        parts.append(
            f'<text x="{left}" y="{H - 18}" font-size="11" fill="{muted}">points lost to the '
            "reference's best move (bins split at the D102 lines 0.2, 0.69, 2.75, 6.86); "
            "share of non-forced, non-flat decisions</text>"
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'font-family="sans-serif">' + "".join(parts) + "</svg>"
        )
        (out / f"hist-{b}.svg").write_text(svg, encoding="utf-8")
        svg_to_png(out / f"hist-{b}.svg", out / f"hist-{b}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "stage",
        choices=["label", "puzzles", "summary", "all", "label42", "review36", "report42"],
    )
    ap.add_argument("--bands", default=",".join(BANDS42), help="label42: comma-separated rungs")
    ap.add_argument("--review", default=str(REVIEW), help="review36: the hk-games review page")
    ap.add_argument("--out", default="runs/p6_mistakes_pilot")
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    a = ap.parse_args()
    out = ROOT / a.out
    if a.stage == "label42":
        stage_label42(out, a.deals, a.threads, a.bands.split(","))
    if a.stage == "report42":
        stage_report42(out, a.bands.split(","))
    if a.stage == "review36":
        stage_review36(out, Path(a.review))
    if a.stage in ("label", "all"):
        stage_label(out, a.deals, a.threads)
    if a.stage in ("puzzles", "all"):
        stage_puzzles(out, a.count)
    if a.stage in ("summary", "all"):
        print(json.dumps(summarize(out), indent=1))


if __name__ == "__main__":
    math.isfinite(0.0)
    main()
