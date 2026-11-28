"""Branch exploration of the qualifying deal-seats (plan Task 50, hk-games D161 choice 2).

    python experiments/branch_explore.py explore --out runs/p6_branches/pilot --pilot 100 --cap 80
    python experiments/branch_explore.py report  --out runs/p6_branches/pilot
    python experiments/branch_explore.py explore --out runs/p6_branches/full --cap 40

Deal-seats: Task 49's sharp rule on Task 42's labels
(``runs/p6_mistakes/decisions-<band>.jsonl``):
at least 3 sharp decisions (not forced, not flat, second-best loses >= 2.75), at most 2 seats
a deal (the two with the most sharp decisions). For each,
:func:`big2.teach.grading.branches.explore`
grows the tree of the seat's own decisions:

- **Stand-in policy** (a player of the deal's bracket): the band rung's own move
  probabilities, exact for the temperature rungs (``netprior``: softmax of log prior / τ over
  every legal move); the network prior at τ = 1 for the search rungs and for the ``net`` band
  (τ = 0 would be deterministic, so the main line alone would cover everything; orchestrator's
  call, 2026-11-28).
- **Bots**: off the main line each bot move comes from a fresh agent of the band's rung,
  seeded from (deal seed, seat, actions so far), so a branch replays from its history alone.
  The main line is the source deal's recorded history.
- **Grades**: the Task 42 reference (``mine_mistakes.label_puct``: PUCT-2,000, belief net,
  exact endgame at <= 12 hidden cards) on every move with prior >= 1 %, the top 8 and the
  bracket's most likely move. Main-line decisions reuse Task 42's grades.
- **Stop** at ``--target`` (0.9) path-probability coverage or ``--cap`` graded decisions,
  main line included; forced decisions are free and not counted.

Output ``<out>/trees.jsonl``, one deal-seat a line (shape: :func:`validate_tree`). A node's key
is hk-games ``deal_puzzles.history_hash``: SHA-256 of the canonical JSON of the app-style
action list before the decision. ``elo`` is null: Task 44 fits it for main line and tree
alike (orchestrator's call, 2026-11-28); options carry the reference's losses, the network
prior and the temperature rungs' probabilities it needs. Resumable (a deal-seat already in the
file is skipped). Runs from this checkout (``PYTHONPATH=python``), ``HKQ_THREADS`` workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

import mine_mistakes as MM  # noqa: E402
from big2.teach.grading import branches as B  # noqa: E402

LABELS = "runs/p6_mistakes"
SHARP_LINE = 2.75
MIN_SHARP = 3
SEATS_PER_DEAL = 2
#: Temperatures whose exact rung probabilities each option carries (for Task 44's fit).
TEMPS = {"net-t4.34": 4.34, "net-t1": 1.0}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# -- deal-seat selection (Task 49's rule) --------------------------------------------------


def select_seats(labels: Path, bands: list[str]) -> list[dict[str, Any]]:
    """Qualifying deal-seats, per band in file order: ``{band, deal_id, seat, sharp,
    bot_loss}``; at most :data:`SEATS_PER_DEAL` a deal, most sharp first (then bot loss)."""
    out = []
    for b in bands:
        seats: dict[tuple[int, int], dict[str, Any]] = {}
        for raw in (labels / f"decisions-{b}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            s = seats.setdefault(
                (r["deal_id"], r["seat"]),
                {
                    "band": b,
                    "deal_id": r["deal_id"],
                    "seat": r["seat"],
                    "sharp": 0,
                    "bot_loss": 0.0,
                },
            )
            if r["forced"]:
                continue
            s["bot_loss"] += r["loss"]
            m = r["moves"]
            if not r["flat"] and len(m) >= 2 and m[1]["loss"] >= SHARP_LINE:
                s["sharp"] += 1
        per_deal: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for s in seats.values():
            if s["sharp"] >= MIN_SHARP:
                per_deal[s["deal_id"]].append(s)
        for d in dict.fromkeys(k[0] for k in seats):
            keep = sorted(
                per_deal.get(d, []), key=lambda s: (-s["sharp"], -s["bot_loss"], s["seat"])
            )
            out += sorted(keep[:SEATS_PER_DEAL], key=lambda s: s["seat"])
    return out


def pilot_pick(seats: list[dict[str, Any]], n: int, bands: list[str]) -> list[dict[str, Any]]:
    """``n`` deal-seats spread over the bands: the first of each band, round robin."""
    per = {b: [s for s in seats if s["band"] == b] for b in bands}
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n and any(i < len(v) for v in per.values()):
        for b in bands:
            if i < len(per[b]) and len(out) < n:
                out.append(per[b][i])
        i += 1
    return out


# -- keys and actions --------------------------------------------------------------------


def app_action(cards: str) -> dict[str, Any]:
    """The app-api action for ``"pass"`` or a card string (hk-games design §7)."""
    if cards == "pass":
        return {"type": "action.pass"}
    from big2._core import classify_cards, parse_cards

    kind = classify_cards(parse_cards(cards))[0]
    return {"type": "action.play", "cards": cards.split(), "combo": "combo." + kind}


def history_hash(actions: list[dict[str, Any]]) -> str:
    """hk-games ``deal_puzzles.history_hash``: SHA-256 of the canonical JSON of the action
    list, card lists sorted."""
    clean = []
    for a in actions:
        a = dict(a)
        if isinstance(a.get("cards"), list):
            a["cards"] = sorted(a["cards"])
        clean.append(a)
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# -- one deal-seat -----------------------------------------------------------------------


def stand_in(band: str, spec: str, pri: dict[int, float]) -> dict[int, float]:
    """The bracket's move probabilities over every legal move (module docstring)."""
    temp = None
    if spec.startswith("netprior:"):
        kv = dict(x.split("=", 1) for x in spec.split(":", 1)[1].split(","))
        temp = float(kv["temperature"])
    if temp is None or temp == 0:
        z = sum(pri.values())
        return {a: p / z for a, p in pri.items()}
    return softmax_t(pri, temp)


def softmax_t(pri: dict[int, float], temp: float) -> dict[int, float]:
    lg = {a: math.log(max(p, 1e-300)) / temp for a, p in pri.items()}
    mx = max(lg.values())
    z = sum(math.exp(v - mx) for v in lg.values())
    return {a: math.exp(v - mx) / z for a, v in lg.items()}


def grade_word(i: int, loss: float) -> str:
    return MM.grade_word(i, loss)


def options_out(
    moves: list[dict[str, Any]], pri: dict[int, float], policy: dict[int, float]
) -> list[dict[str, Any]]:
    """Graded moves (best first) in the deal-puzzle option shape plus what Task 44 needs."""
    rungs = {name: softmax_t(pri, t) for name, t in TEMPS.items()}
    out = []
    for i, m in enumerate(moves):
        a = MM._mask(m["action"])
        out.append(
            {
                "action": app_action(m["action"]),
                "cards": m["action"],
                "loss": m["loss"],
                "loss_se": m["loss_se"],
                "grade": grade_word(i, m["loss"]),
                "ev": m["ev"],
                "se": m["se"],
                "prior": round(pri.get(a, 0.0), 6),
                "rung_prob": round(policy.get(a, 0.0), 6),
                "rung_probs": {k: round(v.get(a, 0.0), 6) for k, v in rungs.items()},
            }
        )
    return out


def theme(state: Any, moves: list[dict[str, Any]], seat: int) -> str:
    """The existing tagger's theme (``drills.skill_for``) for a graded decision."""
    from big2.teach import drills

    pos = MM.position(state)
    answers = []
    for m in moves:
        act = app_action(m["action"])
        answers.append(
            {
                "action": "pass"
                if act["type"] == "action.pass"
                else {"cards": act["cards"], "combo": act["combo"]},
                "reasons": [],
                "loss": m["loss"],
            }
        )
    cand = {
        "seat": seat,
        "position": {"start": {"hands": pos["hands"]}},
        "answers": answers,
        "following": pos["current_play"] is not None,
    }
    return drills.skill_for(cand)


def explore_seat(job: dict[str, Any]) -> dict[str, Any]:
    """Explore one deal-seat; returns its line for ``trees.jsonl``."""
    from big2.agents import make_agent
    from big2.games.big2 import Rules

    t0 = time.perf_counter()
    band, spec, ds, seat = job["band"], job["spec"], job["deal_id"], job["seat"]
    hist = [MM._mask(c) for c in job["history"]]
    main = {r["turn"]: r for r in job["decisions"]}
    stats = {"grade_s": 0.0, "bot_s": 0.0}

    def restore(h: tuple[int, ...]) -> Any:
        s = Rules().new_deal(ds)
        for a in h:
            s.apply(a)
        return s

    def key_of(h: tuple[int, ...]) -> str:
        return history_hash([app_action(MM._cards(a)) for a in h])

    cache: dict[tuple[int, ...], tuple[Any, dict[int, float], dict[int, float]]] = {}

    def info(h: tuple[int, ...]) -> tuple[Any, dict[int, float], dict[int, float]]:
        if h not in cache:
            s = restore(h)
            legal = [int(a) for a in s.legal_actions()]
            pri = {legal[0]: 1.0} if len(legal) == 1 else MM.priors(s)
            cache[h] = (s, pri, stand_in(band, spec, pri))
        return cache[h]

    def describe(h: tuple[int, ...]) -> tuple[str, dict[int, float], bool]:
        s, pri, pol = info(h)
        return key_of(h), pol, len(pri) == 1

    seats_by_turn = job["seats_by_turn"]

    def step(h: tuple[int, ...], m: int) -> tuple[int, ...] | None:
        n = len(h) + 1
        if h + (m,) == tuple(hist[:n]):  # still on the source deal: its recorded moves
            for i in range(n, len(hist)):
                if seats_by_turn[i] == seat:
                    return tuple(hist[:i])
            return None
        s = restore(h)
        s.apply(m)
        h2 = list(h) + [m]
        tb = time.perf_counter()
        while not s.is_terminal and s.current_player != seat:
            p = s.current_player
            seed = zlib.crc32(f"branch|{ds}|{p}|{len(h2)}".encode())
            a = int(make_agent(spec, seed=seed).act(s))
            s.apply(a)
            h2.append(a)
        stats["bot_s"] += time.perf_counter() - tb
        return None if s.is_terminal else tuple(h2)

    def grade(h: tuple[int, ...]) -> dict[str, Any]:
        s, pri, pol = info(h)
        tg = time.perf_counter()
        top = max(pol, key=lambda a: pol[a])
        lab = MM.label_puct(s, top, zlib.crc32(key_of(h).encode()))
        stats["grade_s"] += time.perf_counter() - tg
        return {"moves": lab["moves"], "flat": lab["flat"], "exact": lab["exact"]}

    pregraded = {}
    own_turns = [i for i in range(len(hist)) if seats_by_turn[i] == seat]
    for i in own_turns:
        r = main[i]
        if not r["forced"]:
            pregraded[key_of(tuple(hist[:i]))] = {
                "moves": r["moves"],
                "flat": r["flat"],
                "exact": r.get("exact", False),
            }
    root = tuple(hist[: own_turns[0]])
    res = B.explore(
        B.Game(describe, step, grade),
        root,
        [hist[i] for i in own_turns],
        pregraded,
        cap=job["cap"],
        target=job["target"],
    )
    key_hist = {key_of(h): h for h in cache}
    out_nodes = {}
    for k, n in res.nodes.items():
        h = key_hist[k]
        s, pri, pol = info(h)
        rec: dict[str, Any] = {
            "turn": len(h),
            "history": [MM._cards(a) for a in h],
            "path_p": round(n.path_p, 8),
            "main_line": n.main_line,
            "children": {MM._cards(m): c for m, c in n.children.items()},
        }
        if n.forced:
            rec.update(trivial="forced")
        else:
            g = n.grade
            rec.update(
                trivial="flat" if g["flat"] else None,
                elo=None,
                theme=theme(s, g["moves"], seat),
                exact=g["exact"],
                options=options_out(g["moves"], pri, pol),
            )
        out_nodes[k] = rec
    return {
        "game": "big2",
        "band": band,
        "rung_elo": job["rung_elo"],
        "deal_id": ds,
        "seat": seat,
        "spec": spec,
        "sharp": job["sharp"],
        "cap": job["cap"],
        "target": job["target"],
        "policy": "prior_t1"
        if not spec.startswith("netprior:") or spec.endswith("=0")
        else "exact",
        "root": res.root,
        "coverage": round(res.coverage, 6),
        "coverage_main": round(res.curve[0][1], 6),
        "graded": res.graded,
        "graded_main": res.curve[0][0],
        "new_grades": res.new_grades,
        "curve": [[g, round(c, 6)] for g, c in res.curve],
        "seconds": round(time.perf_counter() - t0, 2),
        "grade_s": round(stats["grade_s"], 2),
        "bot_s": round(stats["bot_s"], 2),
        "reference": MM.reference_string(),
        "nodes": out_nodes,
    }


def validate_tree(t: dict[str, Any]) -> list[str]:
    """Problems with one ``trees.jsonl`` line (empty when valid)."""
    err = [
        f"missing {k}"
        for k in ("band", "deal_id", "seat", "root", "coverage", "nodes", "curve")
        if k not in t
    ]
    if err:
        return err
    if t["root"] not in t["nodes"]:
        err.append("root not among nodes")
    if not 0 <= t["coverage"] <= 1 + 1e-9:
        err.append("coverage out of range")
    for k, n in t["nodes"].items():
        if len(k) != 64:
            err.append("key is not a SHA-256")
        if n["trivial"] == "forced":
            continue
        opts = n.get("options", [])
        if len(opts) < 2:
            err.append(f"{k[:8]}: fewer than two options")
        if opts and opts[0]["loss"] != 0:
            err.append(f"{k[:8]}: options must start with the best")
        for c in n["children"].values():
            if c != B.END and c not in t["nodes"]:
                err.append(f"{k[:8]}: child not among nodes")
    return err


# -- stages ------------------------------------------------------------------------------


def jobs_for(labels: Path, seats: list[dict[str, Any]], cap: int, target: float) -> list[dict]:
    rungs = {
        r["name"]: r for r in json.loads((ROOT / MM.LADDER).read_text(encoding="utf-8"))["rungs"]
    }
    want = defaultdict(set)
    for s in seats:
        want[s["band"]].add((s["deal_id"], s["seat"]))
    deals: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for b in want:
        ids = {d for d, _ in want[b]}
        for raw in (labels / f"decisions-{b}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            if r["deal_id"] in ids:
                deals[(b, r["deal_id"])].append(r)
    out = []
    for s in seats:
        recs = sorted(deals[(s["band"], s["deal_id"])], key=lambda r: r["turn"])
        hist = next(r["history"] for r in recs if "history" in r)
        out.append(
            {
                **s,
                "spec": rungs[s["band"]]["spec"],
                "rung_elo": rungs[s["band"]]["elo"],
                "history": hist,
                "seats_by_turn": [r["seat"] for r in recs],
                "decisions": [r for r in recs if r["seat"] == s["seat"]],
                "cap": cap,
                "target": target,
            }
        )
    return out


def stage_explore(
    out: Path, labels: Path, bands: list[str], pilot: int, cap: int, target: float, threads: int
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trees.jsonl"
    done = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            t = json.loads(raw)
            done.add((t["band"], t["deal_id"], t["seat"]))
    seats = select_seats(labels, bands)
    if pilot:
        seats = pilot_pick(seats, pilot, bands)
    seats = [s for s in seats if (s["band"], s["deal_id"], s["seat"]) not in done]
    log(f"explore: {len(seats)} deal-seats to do ({len(done)} done), cap {cap}, target {target}")
    jobs = jobs_for(labels, seats, cap, target)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(threads) as pool, path.open("a", encoding="utf-8") as f:
        for n, t in enumerate(pool.map(explore_seat, jobs), 1):
            f.write(json.dumps(t) + "\n")
            f.flush()
            if n % 10 == 0:
                log(f"  {n}/{len(jobs)} deal-seats, {time.perf_counter() - t0:.0f} s")
    log(f"explore done in {time.perf_counter() - t0:.0f} s")


def stage_report(out: Path, caps: list[int], bands: list[str]) -> str:
    """``<out>/report.md`` and ``report.json``: coverage at each cap per band, cost."""
    trees = [json.loads(x) for x in (out / "trees.jsonl").read_text(encoding="utf-8").splitlines()]
    bad = sum(1 for t in trees if validate_tree(t))
    rows: dict[str, dict[str, Any]] = {}
    groups = {b: [t for t in trees if t["band"] == b] for b in bands}
    groups["all"] = trees
    for b, ts in groups.items():
        if not ts:
            continue
        r: dict[str, Any] = {"seats": len(ts)}
        r["main"] = sum(t["coverage_main"] for t in ts) / len(ts)
        r["graded_main"] = sum(t["graded_main"] for t in ts) / len(ts)
        for c in caps:
            covs = sorted(B.coverage_at([tuple(x) for x in t["curve"]], c) for t in ts)
            r[f"cov{c}"] = sum(covs) / len(covs)
            r[f"p10_{c}"] = covs[int(0.1 * len(covs))]
            r[f"reach90_{c}"] = sum(1 for x in covs if x >= 0.9) / len(covs)
            r[f"graded{c}"] = sum(
                max([g for g, _ in t["curve"] if g <= c] or [t["curve"][0][0]]) for t in ts
            ) / len(ts)
        r["new_grades"] = sum(t["new_grades"] for t in ts) / len(ts)
        r["seconds"] = sum(t["seconds"] for t in ts) / len(ts)
        r["grade_s_per"] = sum(t["grade_s"] for t in ts) / max(1, sum(t["new_grades"] for t in ts))
        r["bot_s"] = sum(t["bot_s"] for t in ts) / len(ts)
        rows[b] = r
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    head = " | ".join(f"cap {c}: mean / p10 / >= 90 % / graded" for c in caps)
    md = [
        f"| band | deal-seats | main line: coverage / graded | {head} | s a deal-seat (bots) |",
        "|---|---|---|" + "---|" * len(caps) + "---|",
    ]
    for b, r in rows.items():
        cells = " | ".join(
            f"{pct(r[f'cov{c}'])} / {pct(r[f'p10_{c}'])} / {pct(r[f'reach90_{c}'])} / "
            f"{r[f'graded{c}']:.1f}"
            for c in caps
        )
        md.append(
            f"| {b} | {r['seats']} | {pct(r['main'])} / {r['graded_main']:.1f} | {cells} | "
            f"{r['seconds']:.0f} ({r['bot_s']:.0f}) |"
        )
    md.append("")
    md.append(f"Invalid trees: {bad}. Seconds per new grade: {rows['all']['grade_s_per']:.2f}.")
    text = "\n".join(md) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["explore", "report", "count"])
    ap.add_argument("--out", default="runs/p6_branches/pilot")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--bands", default=",".join(MM.BANDS42))
    ap.add_argument("--pilot", type=int, default=0, help="only this many deal-seats (0 = all)")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--caps", default="20,40,80")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    a = ap.parse_args()
    bands = a.bands.split(",")
    if a.stage == "explore":
        stage_explore(ROOT / a.out, ROOT / a.labels, bands, a.pilot, a.cap, a.target, a.threads)
    if a.stage == "report":
        stage_report(ROOT / a.out, [int(c) for c in a.caps.split(",")], bands)
    if a.stage == "count":
        seats = select_seats(ROOT / a.labels, bands)
        print(len(seats), {b: sum(1 for s in seats if s["band"] == b) for b in bands})


if __name__ == "__main__":
    main()
