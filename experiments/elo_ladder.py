"""Elo ladder of the bots below the expert (plan Task 40).

    python experiments/elo_ladder.py span     # step 1: random, easy, medium, net t=1, net, expert
    python experiments/elo_ladder.py rungs    # step 2: temperatures at equal Elo steps (bisection)
    python experiments/elo_ladder.py fit      # step 3: each rung vs the one and two above, sized
    python experiments/elo_ladder.py report   # ladder.json and ladder.md from everything played
    python experiments/elo_ladder.py all      # the four in order (each stage resumes)
    python experiments/elo_ladder.py all --quick   # a 1/25-size run with cheap searches

Every pairing is an ABAB duplicate match (both rungs hold two seats; four rotations per
deal; last-card duty on, tiered scoring, no four-card plays: the engine's defaults). Each
game is one record ``{"deal", "players", "scores", "seconds"}`` in
``runs/p6_ladder/pairs/<a>__<b>/block_*.json``; the fit (:mod:`big2.teach.grading.elo`)
reads only those records, so the other games can reuse it by writing the same shape.

Ratings: maximum-likelihood Bradley-Terry on the app's pairwise rule (higher final score
wins, equal draws, every seat of one rung against every seat of the other), 400 log10 scale,
anchored so today's ``medium`` is 1200 (hk-games D98, until Task 41 picks the app's medium),
95 % ranges by bootstrap over deals.

Runs from this checkout with ``PYTHONPATH=python`` (the job script sets it); worker
processes come from ``HKQ_THREADS`` (default 8). Deterministic: pairing seeds come from the
rung names, agents from the deal seed and rotation.
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

from big2.teach.grading import elo  # noqa: E402

NET = "runs/p6_net/selfplay-h24.pknn"
"""The self-play network (Task 37's export of hour 24), relative to the checkout."""
ANCHOR = ("medium", 1200.0)
SEED = 11
#: Today's app levels exactly as crates/app-api runs them (api.rs, solver.rs): the
#: structured greedy with 10 % random moves, the Q-network at temperature 1, ISMCTS-500 and
#: ISMCTS-10,000 (exploration 0.25, greedy rollout epsilon 0.1, one thread).
OLD_LEVELS = {
    "random": "random",
    "easy": "greedy:structured=1,epsilon=0.1",
    "medium": "qnet:temperature=1.0",
    "hard": "ismcts:iterations=500",
    "expert": "ismcts:iterations=10000",
}
SLOW = {"hard", "expert"}
SPAN_RUNGS = ["random", "easy", "medium", "net-t1", "net"]
"""Step 1 plays these against each other and all but ``medium`` against ``expert``
(``medium`` is in for the anchor; it is cheap)."""
SPAN_DEALS = 500
PROBE_DEALS = 200
FIT_DEALS = 200
TARGET_HALF_WIDTH = 25.0
STEP_RANGE = (150.0, 250.0)
CAP_DEALS = 3000
CAP_DEALS_SLOW = 500
TAU_RANGE = (0.02, 50.0)
MAX_BISECTIONS = 8
BLOCK = 50
RESAMPLES = 1000


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def net_spec(temperature: float) -> str:
    return f"netprior:net={NET},temperature={temperature:g}"


def tau_name(temperature: float) -> str:
    return "net" if temperature == 0 else f"net-t{temperature:g}"


class Ladder:
    """The study's state on disk: rung specs, played games, fitted ratings."""

    def __init__(self, out: Path, quick: bool, threads: int) -> None:
        self.out = out
        self.quick = quick
        self.threads = threads
        self.scale = 1 / 25 if quick else 1.0
        self.rungs: dict[str, str] = {}  # name -> spec (rungs, old levels and probes)
        self.kinds: dict[str, str] = {}  # name -> "ladder" | "old-level" | "probe"
        self.rules_kwargs: dict[str, Any] = {}
        self.load()
        for name, spec in OLD_LEVELS.items():
            if quick and name == "hard":
                spec = "ismcts:iterations=100"
            if quick and name == "expert":
                spec = "ismcts:iterations=300"
            self.add(name, spec, "ladder" if name == "random" else "old-level")
        self.add("net", net_spec(0.0), "ladder")
        self.add("net-t1", net_spec(1.0), "ladder")

    # -- registry -------------------------------------------------------------------------
    def load(self) -> None:
        p = self.out / "rungs.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            self.rungs, self.kinds = d["specs"], d["kinds"]

    def save(self) -> None:
        write_json(self.out / "rungs.json", {"specs": self.rungs, "kinds": self.kinds})

    def add(self, name: str, spec: str, kind: str) -> None:
        if name not in self.rungs:
            self.rungs[name] = spec
            self.kinds[name] = kind
            self.save()
        elif kind == "ladder" and self.kinds[name] != "ladder":
            self.kinds[name] = kind
            self.save()

    def deals(self, n: int) -> int:
        return max(BLOCK if not self.quick else 10, int(round(n * self.scale)))

    # -- games ----------------------------------------------------------------------------
    def pair_dir(self, a: str, b: str) -> Path:
        a, b = sorted((a, b))
        return self.out / "pairs" / f"{a}__{b}"

    def pair_seed(self, a: str, b: str) -> int:
        a, b = sorted((a, b))
        return (zlib.crc32(f"{a}|{b}".encode()) % 100_000) * 100_000

    def played(self, a: str, b: str) -> int:
        """Deals played so far between ``a`` and ``b`` (whole blocks)."""
        return sum(self._block_range(p)[1] - self._block_range(p)[0] for p in self._blocks(a, b))

    def _blocks(self, a: str, b: str) -> list[Path]:
        return sorted(
            self.pair_dir(a, b).glob("block_d*.json"), key=lambda p: self._block_range(p)[0]
        )

    @staticmethod
    def _block_range(p: Path) -> tuple[int, int]:
        lo, hi = p.stem[len("block_d") :].split("-")
        return int(lo), int(hi)

    def play(self, a: str, b: str, deals: int) -> None:
        """Brings the pairing up to ``deals`` deals (ABAB, four rotations each)."""
        have = self.played(a, b)
        if have >= deals:
            return
        block = BLOCK if not self.quick else 10
        base = self.pair_seed(a, b)
        names = [a, b, a, b]
        specs = {a: self.rungs[a], b: self.rungs[b]}
        log(f"play {a} vs {b}: {have} -> {deals} deals ({self.threads} workers)")
        t0 = time.perf_counter()
        for lo in range(have, deals, block):
            hi = min(lo + block, deals)
            seeds = [base + d for d in range(lo, hi)]
            games = play_deals(names, specs, seeds, self.threads)
            write_json(self.pair_dir(a, b) / f"block_d{lo}-{hi}.json", {"games": games})
        log(f"  done in {time.perf_counter() - t0:.0f} s")

    def games(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in sorted((self.out / "pairs").glob("*__*")) if (self.out / "pairs").exists() else []:
            for p in sorted(d.glob("block_d*.json")):
                out.extend(json.loads(p.read_text(encoding="utf-8"))["games"])
        return out

    def pairings(self) -> list[dict[str, Any]]:
        """Every pairing with its deals and the thread-seconds its moves took."""
        rows = []
        if not (self.out / "pairs").exists():
            return rows
        for d in sorted((self.out / "pairs").glob("*__*")):
            a, b = d.name.split("__")
            secs = 0.0
            n = 0
            for p in d.glob("block_d*.json"):
                for g in json.loads(p.read_text(encoding="utf-8"))["games"]:
                    secs += sum(g["seconds"])
                n += self._block_range(p)[1] - self._block_range(p)[0]
            rows.append({"a": a, "b": b, "deals": n, "thread_seconds": round(secs, 1)})
        return rows

    # -- fits -----------------------------------------------------------------------------
    def fit(self) -> dict[str, float]:
        return elo.fit_elo(self.games(), anchor=ANCHOR)

    def report(
        self, order: list[str] | None = None, resamples: int = RESAMPLES
    ) -> elo.LadderReport:
        return elo.ladder_report(
            self.games(), anchor=ANCHOR, resamples=resamples, seed=SEED, order=order
        )

    def ladder_names(self) -> list[str]:
        return [n for n, k in self.kinds.items() if k == "ladder"]

    def level_names(self) -> list[str]:
        return [n for n, k in self.kinds.items() if k in ("ladder", "old-level")]


def _play_block(
    names: list[str], specs: dict[str, str], seeds: list[int], rules_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Worker: plays the deals in duplicate, one record per game."""
    from big2.agents import make_agent
    from big2.eval.tournament import _seatings, play_game
    from big2.games.big2 import Rules

    rules = Rules(**rules_kwargs)
    out = []
    for ds in seeds:
        deal = rules.new_deal(ds)
        for rot, seat_names in enumerate(_seatings(names, ds, False)):
            agents = [make_agent(specs[nm], seed=ds * 4 + rot) for nm in seat_names]
            move_time = [0.0] * 4
            final = play_game(deal.copy(), agents, move_time)
            assert final.scores is not None
            out.append(
                {
                    "deal": ds,
                    "players": seat_names,
                    "scores": [float(x) for x in final.scores],
                    "seconds": [round(t, 4) for t in move_time],
                }
            )
    return out


def play_deals(
    names: list[str], specs: dict[str, str], seeds: list[int], workers: int
) -> list[dict[str, Any]]:
    if workers <= 1 or len(seeds) <= 1:
        return _play_block(names, specs, seeds, {})
    workers = min(workers, len(seeds))
    chunks = [seeds[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(
            pool.map(_play_block, [names] * workers, [specs] * workers, chunks, [{}] * workers)
        )
    by_seed = {g["deal"]: [] for g in (x for part in parts for x in part)}
    for part in parts:
        for g in part:
            by_seed[g["deal"]].append(g)
    return [g for s in seeds for g in by_seed[s]]


# -- stages -------------------------------------------------------------------------------


def stage_span(lad: Ladder) -> dict[str, Any]:
    pairs = [(a, b) for i, a in enumerate(SPAN_RUNGS) for b in SPAN_RUNGS[i + 1 :]]
    pairs += [(r, "expert") for r in SPAN_RUNGS if r != "medium"]
    for a, b in pairs:
        lad.play(a, b, lad.deals(SPAN_DEALS))
    rep = lad.report(order=None, resamples=RESAMPLES // 4)
    span = rep.elo["net"] - rep.elo["random"]
    res = {
        "deals_per_pairing": lad.deals(SPAN_DEALS),
        "elo": rep.elo,
        "range95": rep.range95,
        "order": rep.order,
        "span_random_to_net": span,
        "pairings": lad.pairings(),
    }
    write_json(lad.out / "span.json", res)
    log(
        f"span: {json.dumps({k: round(v) for k, v in rep.elo.items()})}; random->net {span:.0f} Elo"
    )
    return res


def targets_for(span_lo: float, span_hi: float) -> list[float]:
    """Equal Elo steps over the span, 150-250 apart (fewer, wider steps when it is narrow)."""
    span = span_hi - span_lo
    if span < STEP_RANGE[0]:
        return []
    n = max(1, round(span / 200.0))
    while n > 1 and span / n < STEP_RANGE[0]:
        n -= 1
    while span / n > STEP_RANGE[1]:
        n += 1
    step = span / n
    return [span_lo + step * k for k in range(1, n)]


def nearest_known(elo_by: dict[str, float], known: list[str], target: float) -> list[str]:
    """The two known rungs nearest to ``target`` in Elo (one below and one above if both exist)."""
    below = [n for n in known if elo_by[n] <= target]
    above = [n for n in known if elo_by[n] > target]
    picks = []
    if below:
        picks.append(max(below, key=lambda n: elo_by[n]))
    if above:
        picks.append(min(above, key=lambda n: elo_by[n]))
    if len(picks) < 2:
        rest = sorted((n for n in known if n not in picks), key=lambda n: abs(elo_by[n] - target))
        picks += rest[: 2 - len(picks)]
    return picks


def stage_rungs(lad: Ladder) -> dict[str, Any]:
    """Step 2: a temperature per target Elo, by bisection on log(temperature)."""
    fit = lad.fit()
    lo, hi = fit["random"], fit["net"]
    targets = targets_for(lo, hi)
    log(f"rungs: span {lo:.0f} -> {hi:.0f}, targets {[round(t) for t in targets]}")
    cheap = [n for n in lad.level_names() if n not in SLOW and n in fit]
    chosen: dict[str, Any] = {}
    probes: dict[float, str] = {0.0: "net", 1.0: "net-t1"}
    for name, spec in lad.rungs.items():
        if spec.startswith("netprior:"):
            probes[float(spec.rsplit("temperature=", 1)[1])] = name
    for target in targets:
        fit = lad.fit()
        known = [n for n in cheap + list(probes.values()) if n in fit]
        close = min(known, key=lambda n: abs(fit[n] - target))
        if abs(fit[close] - target) <= TARGET_HALF_WIDTH and lad.rungs[close].startswith(
            "netprior:"
        ):
            log(f"  target {target:.0f}: {close} already at {fit[close]:.0f}")
            lad.add(close, lad.rungs[close], "ladder")
            chosen[close] = {
                "target": target,
                "elo": fit[close],
                "temperature": float(lad.rungs[close].rsplit("=", 1)[1]),
            }
            continue
        # Bracket: Elo falls as the temperature rises.
        lo_t, hi_t = math.log(TAU_RANGE[0]), math.log(TAU_RANGE[1])
        for t, nm in probes.items():
            if t <= 0 or nm not in fit:
                continue
            if fit[nm] > target:
                lo_t = max(lo_t, math.log(t))
            else:
                hi_t = min(hi_t, math.log(t))
        best: tuple[float, str, float] | None = None
        for _ in range(MAX_BISECTIONS):
            tau = float(f"{math.exp((lo_t + hi_t) / 2):.3g}")
            name = tau_name(tau)
            lad.add(name, net_spec(tau), "probe")
            probes[tau] = name
            opponents = nearest_known(fit, [n for n in known if n != name], target)
            for opp in opponents:
                lad.play(name, opp, lad.deals(PROBE_DEALS))
            fit = lad.fit()
            known = [n for n in cheap + list(probes.values()) if n in fit]
            e = fit[name]
            log(f"  target {target:.0f}: t={tau:g} -> {e:.0f} Elo (vs {', '.join(opponents)})")
            if best is None or abs(e - target) < abs(best[2] - target):
                best = (tau, name, e)
            if abs(e - target) <= TARGET_HALF_WIDTH:
                break
            if e > target:
                lo_t = math.log(tau)
            else:
                hi_t = math.log(tau)
        assert best is not None
        tau, name, e = best
        lad.add(name, net_spec(tau), "ladder")
        chosen[name] = {"target": target, "elo": e, "temperature": tau}
    res = {"span": [lo, hi], "targets": targets, "chosen": chosen, "elo": lad.fit()}
    write_json(lad.out / "rungs_stage.json", res)
    return res


def required_pairs(order: list[str]) -> list[tuple[str, str]]:
    """Each rung with the one and two above it."""
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1 : i + 3]:
            pairs.append((a, b))
    return pairs


def cap_for(lad: Ladder, a: str, b: str) -> int:
    return lad.deals(CAP_DEALS_SLOW if a in SLOW or b in SLOW else CAP_DEALS)


def stage_fit(lad: Ladder) -> dict[str, Any]:
    """Step 3: every rung (and old level) plays the one and two above; then each adjacent
    gap's pairing grows until its 95 % half-width is at most 25 Elo or its cap is reached."""
    names = lad.level_names()
    fit = lad.fit()
    for n in names:
        if n not in fit:  # hard has not played yet: start it between medium and expert
            fit[n] = (fit.get("medium", 1200.0) + fit.get("expert", 1500.0)) / 2
    for round_ in range(6):
        order = sorted(names, key=lambda n: fit[n])
        todo = [(a, b) for a, b in required_pairs(order) if lad.played(a, b) < lad.deals(FIT_DEALS)]
        log(
            f"fit round {round_}: order {order}; {len(todo)} pairings to bring to "
            f"{lad.deals(FIT_DEALS)} deals"
        )
        for a, b in todo:
            lad.play(a, b, lad.deals(FIT_DEALS))
        fit = lad.fit()
        if sorted(names, key=lambda n: fit[n]) == order and not todo:
            break
    notes: dict[str, str] = {}
    for _ in range(5):
        order = sorted(names, key=lambda n: fit[n])
        rep = lad.report(order=order, resamples=RESAMPLES // 4)
        grew = False
        for g in rep.gaps:
            a, b = g["lower"], g["upper"]
            have = lad.played(a, b)
            if have == 0:
                lad.play(a, b, lad.deals(FIT_DEALS))
                grew = True
                continue
            if g["half_width"] <= TARGET_HALF_WIDTH:
                continue
            need = elo.deals_for_half_width(have, g["half_width"], TARGET_HALF_WIDTH)
            need = int(math.ceil(need / BLOCK) * BLOCK)
            cap = cap_for(lad, a, b)
            if need > cap:
                need = cap
            if need > have:
                log(
                    f"  gap {a} -> {b}: half-width {g['half_width']:.0f} on {have} deals; to {need}"
                )
                lad.play(a, b, need)
                grew = True
            elif g["half_width"] > TARGET_HALF_WIDTH:
                notes[f"{a}->{b}"] = (
                    f"half-width {g['half_width']:.0f} Elo at the cap of {cap} deals"
                    + (" (a search bot: 0.26 s a move)" if a in SLOW or b in SLOW else "")
                )
        fit = lad.fit()
        if not grew:
            break
    # The ladder's own adjacent gaps (old levels sit between some rungs): direct pairings,
    # grown the same way. These rungs are all network or random players, so this is cheap.
    for _ in range(5):
        ladder = sorted(lad.ladder_names(), key=lambda n: fit[n])
        rep = lad.report(order=ladder, resamples=RESAMPLES // 4)
        grew = False
        for g in rep.gaps:
            a, b = g["lower"], g["upper"]
            have = lad.played(a, b)
            need = lad.deals(FIT_DEALS) if have == 0 else have
            if g["half_width"] > TARGET_HALF_WIDTH and have > 0:
                need = elo.deals_for_half_width(have, g["half_width"], TARGET_HALF_WIDTH)
                need = min(int(math.ceil(need / BLOCK) * BLOCK), cap_for(lad, a, b))
            if need > have:
                log(f"  ladder gap {a} -> {b}: half-width {g['half_width']:.0f}; to {need} deals")
                lad.play(a, b, need)
                grew = True
            elif g["half_width"] > TARGET_HALF_WIDTH:
                notes[f"{a}->{b}"] = (
                    f"half-width {g['half_width']:.0f} Elo at the cap of {cap_for(lad, a, b)} deals"
                )
        fit = lad.fit()
        if not grew:
            break
    res = {"order": sorted(names, key=lambda n: fit[n]), "elo": fit, "notes": notes}
    write_json(lad.out / "fit_stage.json", res)
    return res


def stage_report(lad: Ladder) -> dict[str, Any]:
    names = lad.level_names()
    fit = lad.fit()
    names = [n for n in names if n in fit]
    order = sorted(names, key=lambda n: fit[n])
    rep = lad.report(order=order, resamples=RESAMPLES if not lad.quick else 200)
    pairings = lad.pairings()
    deals_of = {n: 0 for n in rep.elo}
    for p in pairings:
        deals_of[p["a"]] += p["deals"]
        deals_of[p["b"]] += p["deals"]
    ladder = [n for n in order if lad.kinds[n] == "ladder"]
    rungs = []
    for n in order:
        spec = lad.rungs[n]
        rungs.append(
            {
                "name": n,
                "kind": lad.kinds[n],
                "spec": spec,
                "temperature": float(spec.rsplit("temperature=", 1)[1])
                if spec.startswith("netprior:")
                else None,
                "elo": round(rep.elo[n], 1),
                "range95": [round(x, 1) for x in rep.range95[n]],
                "deals": deals_of[n],
            }
        )
    gaps = [
        {
            **g,
            "gap": round(g["gap"], 1),
            "range95": [round(x, 1) for x in g["range95"]],
            "half_width": round(g["half_width"], 1),
            "deals": lad.played(g["lower"], g["upper"]),
            "within_25": g["half_width"] <= TARGET_HALF_WIDTH,
        }
        for g in rep.gaps
    ]
    ladder_gaps = [
        {
            **g,
            "gap": round(g["gap"], 1),
            "range95": [round(x, 1) for x in g["range95"]],
            "half_width": round(g["half_width"], 1),
            "deals": lad.played(g["lower"], g["upper"]),
            "within_25": g["half_width"] <= TARGET_HALF_WIDTH,
        }
        for g in lad.report(order=ladder, resamples=rep.resamples).gaps
    ]
    fit_notes = {}
    p = lad.out / "fit_stage.json"
    if p.exists():
        fit_notes = json.loads(p.read_text(encoding="utf-8")).get("notes", {})
    thread_hours = sum(p["thread_seconds"] for p in pairings) / 3600
    res = {
        "task": "Task 40: ladder rungs below the expert and their Elo",
        "network": NET,
        "anchor": {
            "rung": ANCHOR[0],
            "elo": ANCHOR[1],
            "why": "hk-games D98: today's medium = 1200 until Task 41 picks the app's medium",
        },
        "quick": lad.quick,
        "settings": {
            "lineup": "ABAB, four rotations per deal (duplicate), duty on, tiered scoring",
            "pairwise_rule": "higher final score = 1, equal = 0.5, every seat of one rung "
            "vs every seat of the other",
            "fit": "maximum-likelihood Bradley-Terry, draws as half, 400 log10 scale",
            "ranges": f"95 % percentile bootstrap over deals, {rep.resamples} resamples",
            "span_deals": lad.deals(SPAN_DEALS),
            "probe_deals": lad.deals(PROBE_DEALS),
            "fit_deals_first": lad.deals(FIT_DEALS),
            "target_gap_half_width": TARGET_HALF_WIDTH,
            "caps": {
                "network rungs": lad.deals(CAP_DEALS),
                "hard/expert": lad.deals(CAP_DEALS_SLOW),
            },
            "seed": SEED,
        },
        "rungs": rungs,
        "ladder": ladder,
        "ladder_gaps": ladder_gaps,
        "adjacent_gaps_all": gaps,
        "strict_order": all(g["gap"] > 0 for g in gaps),
        "ladder_gaps_within_25": all(g["within_25"] for g in ladder_gaps),
        "old_levels": {
            n: {"elo": round(rep.elo[n], 1), "range95": [round(x, 1) for x in rep.range95[n]]}
            for n in OLD_LEVELS
            if n in rep.elo
        },
        "gap_notes": fit_notes,
        "pairings": pairings,
        "thread_hours": round(thread_hours, 2),
        "games": rep.games,
        "deals": rep.deals,
    }
    write_json(lad.out / "ladder.json", res)
    (lad.out / "ladder.md").write_text(markdown(res), encoding="utf-8")
    log(
        f"report: {len(rungs)} rungs, strict order {res['strict_order']}, ladder gaps within "
        f"25: {res['ladder_gaps_within_25']}, {thread_hours:.1f} thread-hours"
    )
    return res


def markdown(res: dict[str, Any]) -> str:
    lines = [
        "| rung | kind | setting | Elo | 95 % range | deals |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["rungs"]:
        lines.append(
            f"| `{r['name']}` | {r['kind']} | `{r['spec'].replace(NET, '<net>')}` | "
            f"{r['elo']:.0f} | [{r['range95'][0]:.0f}, {r['range95'][1]:.0f}] | {r['deals']} |"
        )
    lines += [
        "",
        "| lower | upper | gap | 95 % range | deals | within ±25 |",
        "|---|---|---|---|---|---|",
    ]
    for g in res["adjacent_gaps_all"]:
        lines.append(
            f"| `{g['lower']}` | `{g['upper']}` | {g['gap']:+.0f} | "
            f"[{g['range95'][0]:.0f}, {g['range95'][1]:.0f}] | "
            f"{g['deals']} | {'yes' if g['within_25'] else 'no'} |"
        )
    lines += ["", "| pairing | deals | thread-h |", "|---|---|---|"]
    for p in res["pairings"]:
        lines.append(
            f"| `{p['a']}` vs `{p['b']}` | {p['deals']} | {p['thread_seconds'] / 3600:.2f} |"
        )
    lines.append(
        f"\nTotal: {res['thread_hours']:.1f} thread-hours, {res['games']} games "
        f"on {res['deals']} deals."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["span", "rungs", "fit", "report", "all"])
    ap.add_argument(
        "--quick",
        action="store_true",
        help="1/25 of the deals, cheap searches, runs/p6_ladder_quick",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    out = (
        Path(args.out)
        if args.out
        else ROOT / "runs" / ("p6_ladder_quick" if args.quick else "p6_ladder")
    )
    lad = Ladder(out, args.quick, args.threads)
    stages = ["span", "rungs", "fit", "report"] if args.stage == "all" else [args.stage]
    for st in stages:
        log(f"== stage {st} ({'quick' if args.quick else 'full'}) -> {out}")
        {"span": stage_span, "rungs": stage_rungs, "fit": stage_fit, "report": stage_report}[st](
            lad
        )


if __name__ == "__main__":
    main()
