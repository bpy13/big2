"""The expert solver's depth and budget study: where the curve bends (plan Task 39).

Results: ``docs/study-results/expert-search-budget.md``. Search: ``crates/search/src/netsearch.rs``
(network play-outs) and ``puct.rs`` (PUCT with the network), both with the self-play network
``runs/p6_net/selfplay-h24.pknn`` (Task 37).

Usage (each stage is one queue job, see ``experiments/expert_search_budget_job.ps1``)::

    python experiments/expert_search_budget.py screen  [--deals 300] [--settings a,b]   # cpu
    python experiments/expert_search_budget.py timing  --where native|node [--settings …] # quiet
    python experiments/expert_search_budget.py finalists                      # cheap
    python experiments/expert_search_budget.py confirm [--deals 2000] [--expert-deals 1000]
    python experiments/expert_search_budget.py report                         # table, curve, knee

* **Settings**: play-outs ``k`` (candidates) in {2, 3, 5, 8}, ``n`` (deals) in {16, 32, 64,
  128, 256}, ``depth`` in {0, 4, 8, 16, full} plies, one knob at a time from ``k 3, n 64, depth
  full``; ``value_mix`` 0.5 at depths 8 and 16 (orchestrator, 2026-11-27); PUCT with the
  network at 200, 500, 1,000 and 2,000 simulations (``c_puct`` 2.5, no rollout, rules mask: the
  phase 3 settings). Both searches draw the hidden cards with the rules mask.
* **Raw network**: the prior's top move, played in Rust from the same ``.pknn``
  (``netsearch`` with ``k = 1, n = 1, depth = 0``: one candidate, so the search is skipped in all
  but name; about 0.15 ms a move). Every setting plays it on the same fixed deals (seed 7, the
  deals of Task 36b), lineups ``1+3`` and ``ABAB``, duplicate seats. **Gain** = the setting's
  score minus the raw network's, paired by deal, averaged over the two lineups; 95 % bootstrap
  range over deals.
* **screen**: every setting at 300 deals per lineup. **finalists**: the five best per unit of
  time: the settings on the efficient frontier of (native median ms, gain) — no cheaper setting
  gains more — taking the five with the highest gain, filled from the next-best gains if the
  frontier is short. **confirm**: the finalists at 2,000 deals against the raw network (the
  screen's deals are the first 300) and 1,000 deals against the app's `expert`
  (``ismcts:iterations=10000``, as Task 36b; its deals too).
* **timing**: ``crates/search-wasm`` steps ``netsearch`` and ``netpuct`` on 300 positions from
  games the network plays against itself (seed 7, after 30 warm-up positions), one thread:
  natively (``puct_timing`` example) and in Node WebAssembly (``runs/p6_budget/pkg``). Phone
  time = 3 x Node (assumed: hk-games has no phone timing yet, ``docs/bot-speed.md``).
* **Knee rule** (orchestrator's default): among settings whose phone 95th percentile is at most
  500 ms for a bot move, the cheapest whose gain is within 0.2 points a deal of the best
  measured gain or whose 95 % range overlaps the best's. Also the cheapest within 600 ms for
  `grade`, and the gain per doubling of time along the frontier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p6_budget"
NET = "runs/p6_net/selfplay-h24.pknn"
#: The raw network: one candidate, so no play-out can change the move.
RAW = f"netsearch:net={NET},k=1,n=1,depth=0"
#: `expert` as the app plays it (Task 36b).
EXPERT = "ismcts:iterations=10000"
SEED = 7
LINEUPS = ("1+3", "ABAB")
SCREEN_DEALS = 300
CONFIRM_DEALS = 2000
EXPERT_DEALS = 1000
BLOCK = 100
FINALISTS = 5
TIMING_COUNT = 300
TIMING_WARMUP = 30
#: Phone time per Node time (assumed; no phone timing in hk-games yet).
PHONE_PER_NODE = 3.0
#: Phone budgets (hk-games docs/architecture.md 2a): a bot move, a background grade.
MOVE_BUDGET_MS = 500.0
GRADE_BUDGET_MS = 600.0
KNEE_MARGIN = 0.2
C_PUCT = 2.5


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def threads() -> int:
    return int(os.environ.get("HKQ_THREADS", "8"))


# settings ---------------------------------------------------------------------------


def playout(k: int, n: int, depth: int | str, value_mix: float = 1.0) -> dict[str, Any]:
    name = f"k{k}-n{n}-d{depth}" + (f"-mix{value_mix:g}" if value_mix != 1.0 else "")
    spec = f"netsearch:net={NET},k={k},n={n},depth={depth}"
    if value_mix != 1.0:
        spec += f",value_mix={value_mix}"
    step = f"netsearch {k} {n} {depth} {value_mix:g}"
    return {
        "name": name,
        "family": "play-outs",
        "spec": spec,
        "step": step,
        "k": k,
        "n": n,
        "depth": depth,
        "value_mix": value_mix,
    }


def puct(simulations: int) -> dict[str, Any]:
    return {
        "name": f"puct{simulations}",
        "family": "PUCT",
        "spec": f"puct:net={NET},simulations={simulations},c_puct={C_PUCT},belief=rules",
        "step": f"netpuct {simulations} 1",
        "simulations": simulations,
    }


SETTINGS: list[dict[str, Any]] = [
    playout(3, 64, "full"),
    *(playout(k, 64, "full") for k in (2, 5, 8)),
    *(playout(3, n, "full") for n in (16, 32, 128, 256)),
    *(playout(3, 64, d) for d in (0, 4, 8, 16)),
    playout(3, 64, 8, 0.5),
    playout(3, 64, 16, 0.5),
    *(puct(s) for s in (200, 500, 1000, 2000)),
]
BY_NAME = {s["name"]: s for s in SETTINGS}


def chosen(text: str | None) -> list[dict[str, Any]]:
    """``--settings``: names, ``all`` or ``finalists``."""
    if not text or text == "all":
        return SETTINGS
    if text == "finalists":
        return [BY_NAME[n] for n in read_json(OUT / "finalists.json")["finalists"]]
    return [BY_NAME[n.strip()] for n in text.split(",") if n.strip()]


# matches ----------------------------------------------------------------------------


def match(cand: str, opp: str, lineup: str, deals: int, cache: Path) -> dict[str, Any]:
    """One lineup of a duplicate match, cached in blocks; adds the per-deal paired scores."""
    from big2.eval.harness import play_match

    r = play_match(
        cand, opp, lineup, deals, seed=SEED, workers=threads(), cache_dir=cache, block=BLOCK
    ).to_dict()
    r["deal_diffs"] = deal_diffs(cache, cand, opp, deals).tolist()
    return r


def deal_diffs(cache: Path, cand: str, opp: str, deals: int) -> np.ndarray:
    """Per deal, the candidate's mean score minus the opponent's, in deal order."""
    rows: dict[int, float] = {}
    for f in cache.glob(f"block_s{SEED}_d*.json"):
        saved = read_json(f)
        a, _ = saved["deals"]
        for k, rec in enumerate(saved["records"]):
            c = rec[f"score|{cand}"]
            o = rec[f"score|{opp}"]
            rows[a + k] = sum(c) / len(c) - sum(o) / len(o)
    return np.array([rows[d] for d in range(deals)])


def bootstrap(x: np.ndarray, seed: int = 3) -> list[float]:
    from big2.eval.harness import bootstrap_ci

    return [float(v) for v in bootstrap_ci(x, seed=seed)]


def versus(setting: dict[str, Any], opp: str, deals: int, folder: Path) -> dict[str, Any]:
    """The setting against ``opp`` in both lineups (cached); gain = both lineups' paired mean."""
    path = folder / f"results_{deals}.json"
    if path.exists():
        return read_json(path)
    res: dict[str, Any] = {
        "setting": setting["name"],
        "spec": setting["spec"],
        "opponent": opp,
        "deals": deals,
        "lineups": {},
    }
    for lu in LINEUPS:
        log(f"{setting['name']} vs {opp}, {lu}, {deals} deals")
        res["lineups"][lu] = match(setting["spec"], opp, lu, deals, folder / lu.replace("+", "p"))
    both = np.mean([res["lineups"][lu]["deal_diffs"] for lu in LINEUPS], axis=0)
    res["gain"] = {"mean": float(both.mean()), "ci95": bootstrap(both)}
    res["seconds_per_move"] = float(
        np.mean([res["lineups"][lu]["seconds_per_move"] for lu in LINEUPS])
    )
    write_json(path, res)
    return res


def screen_stage(args: argparse.Namespace) -> None:
    for s in chosen(args.settings):
        versus(s, RAW, args.deals, OUT / "screen" / s["name"])
    report_stage(args)


def confirm_stage(args: argparse.Namespace) -> None:
    fin = read_json(OUT / "finalists.json")["finalists"]
    for name in fin:
        s = BY_NAME[name]
        versus(s, RAW, args.deals, OUT / "screen" / name)  # the screen's blocks are reused
    raw_vs_expert(args.expert_deals)
    for name in fin:
        versus(BY_NAME[name], EXPERT, args.expert_deals, OUT / "expert" / name)
    report_stage(args)


def raw_vs_expert(deals: int) -> dict[str, Any] | None:
    """The raw network against `expert`: Task 36b's hour-24 result if it exists (the same
    network in PyTorch, the same deals), else played here."""
    folder = OUT / "expert" / "raw"
    path = folder / f"results_{deals}.json"
    if path.exists():
        return read_json(path)
    t36 = ROOT / "runs" / "selfplay_luck" / "expert" / "h24" / f"results_{deals}.json"
    if t36.exists():
        src = read_json(t36)
        res = {
            "setting": "raw",
            "spec": src["checkpoint"],
            "opponent": EXPERT,
            "deals": deals,
            "source": str(t36.relative_to(ROOT)),
            "lineups": {},
        }
        for lu in LINEUPS:
            r = src["vs_expert"][lu]
            res["lineups"][lu] = {k: r[k] for k in ("paired_diff", "paired_ci95", "mean", "deals")}
        write_json(path, res)
        return res
    raw = {"name": "raw", "spec": RAW}
    return versus(raw, EXPERT, deals, folder)


# timing -----------------------------------------------------------------------------


def native_exe() -> Path:
    return ROOT / "target" / "release" / "examples" / "puct_timing.exe"


def quantiles(ms: list[float]) -> dict[str, float]:
    v = sorted(ms)
    q = lambda p: v[min(len(v) - 1, int(math.floor(p * len(v))))]  # noqa: E731
    return {"n": len(v), "median": q(0.5), "p95": q(0.95), "mean": float(np.mean(v)), "max": v[-1]}


def timing_stage(args: argparse.Namespace) -> None:
    settings = chosen(args.settings)
    count = args.count + TIMING_WARMUP
    steps = [f"{s['step']} {count} {SEED}" for s in settings]
    out = OUT / "timing" / f"{args.where}.json"
    done: dict[str, Any] = read_json(out) if out.exists() else {}
    for s, step in zip(settings, steps, strict=True):
        if s["name"] in done and done[s["name"]]["n"] == args.count:
            continue
        log(f"timing {args.where}: {step}")
        steps_file = OUT / "timing" / f"step_{args.where}.txt"
        steps_file.parent.mkdir(parents=True, exist_ok=True)
        steps_file.write_text(step + "\n", encoding="utf-8")
        if args.where == "native":
            cmd = [str(native_exe()), NET, str(steps_file)]
        else:
            cmd = [
                "node",
                str(ROOT / "experiments" / "search_budget_timing.mjs"),
                str(OUT / "pkg"),
                NET,
                str(steps_file),
            ]
        text = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
        rows = json.loads(text.strip().splitlines()[-1])
        ms = [r["ms"] for r in rows[TIMING_WARMUP:]]
        done[s["name"]] = {
            **quantiles(ms),
            "step": step,
            "legal_mean": float(np.mean([r["legal"] for r in rows])),
            "net_calls_mean": float(
                np.mean([r.get("net_calls", r.get("simulations", 0)) for r in rows])
            ),
        }
        write_json(out, done)
    report_stage(args)


# finalists, knee, report -------------------------------------------------------------


def screen_results(deals: int = SCREEN_DEALS) -> dict[str, Any]:
    """Every setting's latest result against the raw network (most deals first)."""
    out = {}
    for s in SETTINGS:
        folder = OUT / "screen" / s["name"]
        files = sorted(folder.glob("results_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if files:
            out[s["name"]] = read_json(files[-1])
    return out


def timing(where: str) -> dict[str, Any]:
    p = OUT / "timing" / f"{where}.json"
    return read_json(p) if p.exists() else {}


def frontier(rows: list[tuple[str, float, float]]) -> list[str]:
    """Names on the efficient frontier of (ms, gain): no cheaper row gains at least as much."""
    best = -math.inf
    out = []
    for name, _ms, gain in sorted(rows, key=lambda r: (r[1], -r[2])):
        if gain > best:
            out.append(name)
            best = gain
    return out


def finalists_stage(args: argparse.Namespace) -> list[str]:
    res = screen_results()
    nat = timing("native")
    rows = [(n, nat[n]["median"], res[n]["gain"]["mean"]) for n in res if n in nat]
    missing = [s["name"] for s in SETTINGS if s["name"] not in res or s["name"] not in nat]
    if missing:
        log(f"finalists: no screen result or native timing yet for {missing}")
    front = frontier(rows)
    by_gain = sorted(front, key=lambda n: -res[n]["gain"]["mean"])[:FINALISTS]
    if len(by_gain) < FINALISTS:
        rest = sorted(
            (n for n, _, _ in rows if n not in by_gain), key=lambda n: -res[n]["gain"]["mean"]
        )
        by_gain += rest[: FINALISTS - len(by_gain)]
    fin = {
        "finalists": by_gain,
        "frontier": front,
        "rule": "efficient frontier of (native "
        "median ms, gain over the raw network), the five with the highest gain",
        "screen_deals": {n: res[n]["deals"] for n in res},
        "missing": missing,
        "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(OUT / "finalists.json", fin)
    log(f"finalists: {by_gain} (frontier {front})")
    return by_gain


def fmt(m: float, ci: list[float]) -> str:
    return f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def phone_estimates() -> dict[str, dict[str, float]]:
    """Phone ms (median, p95) for every timed setting: finalists from Node x 3, the others
    from native x the median Node/native factor of the finalists (4 if none yet)."""
    nat, node = timing("native"), timing("node")
    factors = [node[n]["median"] / nat[n]["median"] for n in node if n in nat]
    f = float(np.median(factors)) if factors else 4.0
    out = {}
    for n, t in nat.items():
        if n in node:
            out[n] = {
                "median": PHONE_PER_NODE * node[n]["median"],
                "p95": PHONE_PER_NODE * node[n]["p95"],
                "from": "node",
            }
        else:
            out[n] = {
                "median": PHONE_PER_NODE * f * t["median"],
                "p95": PHONE_PER_NODE * f * t["p95"],
                "from": f"native x {f:.2f}",
            }
    out["_node_per_native"] = {"factor": f, "finalists_timed": len(factors)}
    return out


def knee(
    res: dict[str, Any], nat: dict[str, Any], phone: dict[str, Any], budget_ms: float
) -> dict[str, Any]:
    names = [n for n in res if n in nat]
    if not names:
        return {"setting": None}
    best = max(names, key=lambda n: res[n]["gain"]["mean"])
    bg, bci = res[best]["gain"]["mean"], res[best]["gain"]["ci95"]
    ok = []
    for n in names:
        g, ci = res[n]["gain"]["mean"], res[n]["gain"]["ci95"]
        within = g >= bg - KNEE_MARGIN or ci[1] >= bci[0]
        in_budget = n in phone and phone[n]["p95"] <= budget_ms
        if within and in_budget:
            ok.append(n)
    pick = min(ok, key=lambda n: nat[n]["median"]) if ok else None
    return {
        "setting": pick,
        "best": best,
        "best_gain": bg,
        "best_ci95": bci,
        "budget_ms": budget_ms,
        "candidates": ok,
        "gain": res[pick]["gain"] if pick else None,
        "native_median_ms": nat[pick]["median"] if pick else None,
        "phone_p95_ms": phone[pick]["p95"] if pick else None,
    }


def gain_per_doubling(res: dict[str, Any], nat: dict[str, Any]) -> dict[str, Any]:
    """Least-squares slope of gain against log2(native median ms) along the frontier, and
    the step-by-step slopes."""
    rows = [(n, nat[n]["median"], res[n]["gain"]["mean"]) for n in res if n in nat]
    front = frontier(rows)
    pts = [(math.log2(nat[n]["median"]), res[n]["gain"]["mean"], n) for n in front]
    steps = []
    for (x0, g0, a), (x1, g1, b) in zip(pts, pts[1:], strict=False):
        if x1 > x0:
            steps.append({"from": a, "to": b, "gain_per_doubling": (g1 - g0) / (x1 - x0)})
    slope = None
    if len(pts) >= 2:
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        slope = float(np.polyfit(x, y, 1)[0])
    return {"frontier": front, "slope": slope, "steps": steps}


def report_stage(args: argparse.Namespace | None = None) -> str:
    res = screen_results()
    nat, node = timing("native"), timing("node")
    phone = phone_estimates()
    fin_path = OUT / "finalists.json"
    fin = read_json(fin_path)["finalists"] if fin_path.exists() else []
    expert = {}
    for n in fin + ["raw"]:
        folder = OUT / "expert" / n
        files = sorted(folder.glob("results_*.json")) if folder.exists() else []
        if files:
            expert[n] = read_json(files[-1])
    lines = [
        "| Setting | Deals | Gain over raw network (both lineups) | 1+3 | ABAB | native ms median / p95 "  # noqa: E501
        "| Node ms median / p95 | phone ms median / p95 (est.) | vs `expert` 1+3 | vs `expert` ABAB |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = sorted(res, key=lambda n: nat[n]["median"] if n in nat else math.inf)
    for n in order:
        r = res[n]
        lu = r["lineups"]
        cells = [
            n + (" (finalist)" if n in fin else ""),
            f"{r['deals']:,}",
            fmt(r["gain"]["mean"], r["gain"]["ci95"]),
            fmt(lu["1+3"]["paired_diff"], lu["1+3"]["paired_ci95"]),
            fmt(lu["ABAB"]["paired_diff"], lu["ABAB"]["paired_ci95"]),
            f"{nat[n]['median']:.0f} / {nat[n]['p95']:.0f}" if n in nat else "",
            f"{node[n]['median']:.0f} / {node[n]['p95']:.0f}" if n in node else "",
            f"{phone[n]['median']:.0f} / {phone[n]['p95']:.0f}" if n in phone else "",
        ]
        if n in expert:
            e = expert[n]["lineups"]
            cells += [
                fmt(e["1+3"]["paired_diff"], e["1+3"]["paired_ci95"]),
                fmt(e["ABAB"]["paired_diff"], e["ABAB"]["paired_ci95"]),
            ]
        else:
            cells += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")
    if "raw" in expert:
        e = expert["raw"]["lineups"]
        lines.append(
            "| raw network | "
            + f"{expert['raw']['deals']:,}"
            + " | (0) | | | 0.15 / 0.3 | | | "
            + fmt(e["1+3"]["paired_diff"], e["1+3"]["paired_ci95"])
            + " | "
            + fmt(e["ABAB"]["paired_diff"], e["ABAB"]["paired_ci95"])
            + " |"
        )
    k_move = knee(res, nat, phone, MOVE_BUDGET_MS)
    k_grade = knee(res, nat, phone, GRADE_BUDGET_MS)
    slope = gain_per_doubling(res, nat)
    text = "\n".join(lines) + "\n\n"
    text += f"Knee (bot move, phone p95 <= {MOVE_BUDGET_MS:.0f} ms): {json.dumps(k_move)}\n\n"
    text += f"Knee (grade, phone p95 <= {GRADE_BUDGET_MS:.0f} ms): {json.dumps(k_grade)}\n\n"
    text += f"Gain per doubling of time: {json.dumps(slope)}\n\n"
    text += f"Phone factor: {json.dumps(phone.get('_node_per_native'))}, phone = {PHONE_PER_NODE:g} x Node (assumed)\n"  # noqa: E501
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    write_json(
        OUT / "report.json",
        {
            "knee_move": k_move,
            "knee_grade": k_grade,
            "gain_per_doubling": slope,
            "phone": phone,
            "finalists": fin,
        },
    )
    curve(res, nat, phone, fin, k_move)
    print(text, flush=True)
    return text


# the curve --------------------------------------------------------------------------


def curve(
    res: dict[str, Any],
    nat: dict[str, Any],
    phone: dict[str, Any],
    fin: list[str],
    k_move: dict[str, Any],
) -> None:
    """Gain against native ms a move (log scale), one mark per setting, ranges as bars;
    SVG always, PNG through Playwright's Chromium when it is installed."""
    pts = [
        (n, nat[n]["median"], res[n]["gain"]["mean"], res[n]["gain"]["ci95"])
        for n in res
        if n in nat
    ]
    if not pts:
        return
    W, H, L, R, T, B = 900, 560, 70, 30, 30, 70
    xs = [p[1] for p in pts]
    x0, x1 = math.log10(min(xs) / 1.3), math.log10(max(xs) * 1.3)
    gs = [p[3][0] for p in pts] + [p[3][1] for p in pts] + [0.0]
    y0, y1 = min(gs) - 0.3, max(gs) + 0.3
    sx = lambda ms: L + (math.log10(ms) - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    sy = lambda g: T + (y1 - g) / (y1 - y0) * (H - T - B)  # noqa: E731
    colours = {"play-outs": "#1f77b4", "PUCT": "#d62728"}
    el = [f'<rect width="{W}" height="{H}" fill="white"/>']
    # axes and grid
    for e in range(int(math.floor(x0)), int(math.ceil(x1)) + 1):
        for m in (1, 2, 5):
            v = m * 10**e
            if x0 <= math.log10(v) <= x1:
                el.append(
                    f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{H - B}" stroke="#eee"/>'
                )
                el.append(
                    f'<text x="{sx(v):.1f}" y="{H - B + 18}" font-size="12" text-anchor="middle">{v:g}</text>'  # noqa: E501
                )
    step = 0.5 if y1 - y0 < 5 else 1.0
    g = math.ceil(y0 / step) * step
    while g <= y1:
        el.append(f'<line x1="{L}" y1="{sy(g):.1f}" x2="{W - R}" y2="{sy(g):.1f}" stroke="#eee"/>')
        el.append(
            f'<text x="{L - 6}" y="{sy(g) + 4:.1f}" font-size="12" text-anchor="end">{g:+.1f}</text>'  # noqa: E501
        )
        g += step
    el.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{W - R}" y2="{sy(0):.1f}" stroke="#999"/>')
    el.append(
        f'<text x="{(L + W - R) / 2}" y="{H - B + 42}" font-size="13" text-anchor="middle">'
        "native ms per move, one thread (median, log scale)</text>"
    )
    el.append(
        f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" font-size="13" text-anchor="middle">'  # noqa: E501
        "points a deal over the raw network</text>"
    )
    # points
    for n, ms, gain, ci in pts:
        fam = BY_NAME[n]["family"]
        c = colours[fam]
        x, y = sx(ms), sy(gain)
        el.append(
            f'<line x1="{x:.1f}" y1="{sy(ci[0]):.1f}" x2="{x:.1f}" y2="{sy(ci[1]):.1f}" stroke="{c}" stroke-width="1.5"/>'  # noqa: E501
        )
        r = 6 if n in fin else 4
        el.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c if n in fin else "white"}" stroke="{c}" stroke-width="2"/>'  # noqa: E501
        )
        label = n + (" ◆" if n == k_move.get("setting") else "")
        el.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" font-size="11" fill="{c}">{label}</text>')
    # frontier line (play-outs and PUCT together)
    front = frontier([(n, ms, gain) for n, ms, gain, _ in pts])
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(nat[n]['median']):.1f},{sy(res[n]['gain']['mean']):.1f}"
        for i, n in enumerate(front)
    )
    el.append(f'<path d="{path}" fill="none" stroke="#555" stroke-dasharray="4 3"/>')
    el.append(
        f'<text x="{W - R}" y="{T + 12}" font-size="12" text-anchor="end" fill="#1f77b4">● play-outs (k, n, depth)</text>'  # noqa: E501
    )
    el.append(
        f'<text x="{W - R}" y="{T + 28}" font-size="12" text-anchor="end" fill="#d62728">● PUCT with the network</text>'  # noqa: E501
    )
    el.append(
        f'<text x="{W - R}" y="{T + 44}" font-size="12" text-anchor="end" fill="#555">filled = finalist, ◆ = knee, dashed = frontier</text>'  # noqa: E501
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">'  # noqa: E501
        + "".join(el)
        + "</svg>"
    )
    (OUT / "curve.svg").write_text(svg, encoding="utf-8")
    svg_to_png(OUT / "curve.svg", OUT / "curve.png")


def svg_to_png(svg: Path, png: Path) -> bool:
    """Screenshots the SVG with Playwright's Chromium (hk-games' app has the package); False
    when it is not available."""
    pw = ROOT.parent.parent / "hk-games" / "app" / "node_modules" / "playwright"
    if not pw.exists() or shutil.which("node") is None:
        log("curve: no Playwright, PNG not written (SVG is)")
        return False
    script = f"""
const {{ chromium }} = require({json.dumps(str(pw))});
(async () => {{
  const b = await chromium.launch();
  const p = await b.newPage({{ viewport: {{ width: 900, height: 560 }}, deviceScaleFactor: 2 }});
  await p.goto('file:///' + {json.dumps(str(svg).replace(os.sep, "/"))});
  await p.screenshot({{ path: {json.dumps(str(png))} }});
  await b.close();
}})().catch(e => {{ console.error(e); process.exit(1); }});
"""
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    if r.returncode:
        log(f"curve: PNG failed: {r.stderr.strip()[:300]}")
        return False
    return True


# beyond 2,000 simulations (Task 51) --------------------------------------------------

BEYOND = OUT / "beyond"
BEYOND_SIMS = (200, 500, 1000, 2000, 4000, 8000, 16000, 32000)


def beyond_stage(args: argparse.Namespace) -> None:
    """PUCT at ``--simulations`` against the raw network, 2,000 deals per lineup (same seed and
    deals as the confirm); cached in blocks, so adding it again resumes."""
    s = puct(args.simulations)
    versus(s, RAW, args.deals, BEYOND / s["name"])


def beyond_result(sims: int, deals: int) -> dict[str, Any] | None:
    for folder in (OUT / "screen" / f"puct{sims}", BEYOND / f"puct{sims}"):
        path = folder / f"results_{deals}.json"
        if path.exists():
            return read_json(path)
    return None


def beyond_report_stage(args: argparse.Namespace) -> None:
    """Table (markdown) and chart (SVG, PNG) of gain against simulations, 200 to the last run."""
    rows = []
    for sims in BEYOND_SIMS:
        r = beyond_result(sims, args.deals)
        if r is None:
            continue
        lu = {k: v["paired_diff"] for k, v in r["lineups"].items()}
        lci = {k: v["paired_ci95"] for k, v in r["lineups"].items()}
        rows.append(
            {
                "sims": sims,
                "gain": r["gain"]["mean"],
                "ci": r["gain"]["ci95"],
                "lineups": lu,
                "lineup_ci": lci,
                "ms": r["seconds_per_move"] * 1000.0,
            }
        )
    for prev, row in zip([None, *rows], rows, strict=False):
        row["per_doubling"] = (
            None
            if prev is None
            else (row["gain"] - prev["gain"]) / math.log2(row["sims"] / prev["sims"])
        )
    lines = [
        "| Simulations | Gain over raw network [95 %] | 1+3 | ABAB | ms a move | Gain per doubling |",  # noqa: E501
        "|---:|---|---|---|---:|---:|",
    ]
    for r in rows:
        pd = "" if r["per_doubling"] is None else f"{r['per_doubling']:+.3f}"
        lines.append(
            f"| {r['sims']:,} | {fmt(r['gain'], r['ci'])} | "
            f"{fmt(r['lineups']['1+3'], r['lineup_ci']['1+3'])} | "
            f"{fmt(r['lineups']['ABAB'], r['lineup_ci']['ABAB'])} | {r['ms']:.0f} | {pd} |"
        )
    BEYOND.mkdir(parents=True, exist_ok=True)
    text = chr(10).join(lines) + chr(10)
    (BEYOND / "table.md").write_text(text, encoding="utf-8")
    write_json(BEYOND / "table.json", rows)
    log(text)
    if len(rows) < 2:
        return
    W, H, L, R, T, B = 900, 560, 90, 40, 60, 90
    xs = [math.log2(r["sims"]) for r in rows]
    x0, x1 = min(xs), max(xs)
    lo = min(r["ci"][0] for r in rows)
    hi = max(r["ci"][1] for r in rows)
    lo, hi = min(0.0, lo) - 0.05, hi + 0.05
    px = lambda v: L + (v - x0) / (x1 - x0 or 1) * (W - L - R)  # noqa: E731
    py = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)  # noqa: E731
    el = [
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W / 2}" y="34" font-size="24" text-anchor="middle" font-weight="bold">'
        "Gain over the raw network, PUCT by simulations</text>",
        f'<line x1="{L}" y1="{py(0)}" x2="{W - R}" y2="{py(0)}" stroke="#999" stroke-dasharray="6"/>',  # noqa: E501
        f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#333"/>',
        f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#333"/>',
        f'<text x="{W / 2}" y="{H - 20}" font-size="20" text-anchor="middle">'
        "simulations a move (log scale)</text>",
        f'<text transform="translate(26 {H / 2}) rotate(-90)" font-size="20" '
        'text-anchor="middle">points a deal</text>',
    ]
    t = math.ceil(lo * 2) / 2
    while t <= hi:
        el.append(
            f'<text x="{L - 10}" y="{py(t) + 7}" font-size="18" text-anchor="end">{t:.1f}</text>'
        )
        t += 0.5
    pts = []
    for r, x in zip(rows, xs, strict=True):
        cx = px(x)
        el.append(
            f'<line x1="{cx}" y1="{py(r["ci"][0])}" x2="{cx}" y2="{py(r["ci"][1])}" stroke="#1f6feb" stroke-width="4"/>'  # noqa: E501
        )
        el.append(f'<circle cx="{cx}" cy="{py(r["gain"])}" r="8" fill="#1f6feb"/>')
        el.append(
            f'<text x="{cx}" y="{H - B + 30}" font-size="18" text-anchor="middle">{r["sims"]:,}</text>'  # noqa: E501
        )
        pts.append(f"{cx},{py(r['gain'])}")
    el.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1f6feb" stroke-width="2"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">'  # noqa: E501
        + "".join(el)
        + "</svg>"
    )
    (BEYOND / "beyond.svg").write_text(svg, encoding="utf-8")
    svg_to_png(BEYOND / "beyond.svg", BEYOND / "beyond.png")


# main -------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--deals", type=int, default=SCREEN_DEALS)
    p.add_argument("--settings", default="all")
    p.set_defaults(fn=screen_stage)
    p = sub.add_parser("timing")
    p.add_argument("--where", choices=("native", "node"), required=True)
    p.add_argument("--settings", default="all")
    p.add_argument("--count", type=int, default=TIMING_COUNT)
    p.set_defaults(fn=timing_stage)
    p = sub.add_parser("finalists")
    p.set_defaults(fn=finalists_stage)
    p = sub.add_parser("confirm")
    p.add_argument("--deals", type=int, default=CONFIRM_DEALS)
    p.add_argument("--expert-deals", type=int, default=EXPERT_DEALS)
    p.set_defaults(fn=confirm_stage)
    p = sub.add_parser("report")
    p.set_defaults(fn=report_stage)
    p = sub.add_parser("beyond")
    p.add_argument("--simulations", type=int, required=True)
    p.add_argument("--deals", type=int, default=CONFIRM_DEALS)
    p.set_defaults(fn=beyond_stage)
    p = sub.add_parser("beyond-report")
    p.add_argument("--deals", type=int, default=CONFIRM_DEALS)
    p.set_defaults(fn=beyond_report_stage)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
