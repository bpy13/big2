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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["label", "puzzles", "summary", "all"])
    ap.add_argument("--out", default="runs/p6_mistakes_pilot")
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    a = ap.parse_args()
    out = ROOT / a.out
    if a.stage in ("label", "all"):
        stage_label(out, a.deals, a.threads)
    if a.stage in ("puzzles", "all"):
        stage_puzzles(out, a.count)
    if a.stage in ("summary", "all"):
        print(json.dumps(summarize(out), indent=1))


if __name__ == "__main__":
    math.isfinite(0.0)
    main()
