"""Self-play pilot with the luck taken out (plan Task 36).

Results: docs/study-results/selfplay-luck.md. Trainer: :mod:`big2.rl.selfplay`.

Usage (each stage is one queue job, see ``experiments/selfplay_luck_job.ps1``)::

    python experiments/selfplay_luck.py smoke                 # gpu: picks the pull, ~35 min
    python experiments/selfplay_luck.py train --until-hours 4 # gpu: trains up to hour 4
    python experiments/selfplay_luck.py eval --hour 4         # cpu: hour 4 (and 0) vs expert
    python experiments/selfplay_luck.py h2h --hour 12         # cpu: hour 12 vs hour 8 (Task 36c)
    python experiments/selfplay_luck.py table                 # table_expert.md from every eval

* **Start network**: round 3 (``runs/net_r3/best.pt``, copied read-only from the standalone
  checkout). `expert` is ISMCTS-10,000 with no network inside; r3 is the best network there
  is (it ties its teacher ISMCTS-5000 inside PUCT-2000) and has the policy head self-play
  needs.
* **smoke**: three 10-minute runs with the pull 0.02, 0.1 and 0.5 (Dou Dizhu's grid), each
  end network against the start, 2,000 duplicate deals per lineup. The pull taken is the
  smallest whose paired difference against the start has a 95 % range reaching 0 or above
  in both lineups (Dou Dizhu's rule); none: 0.5. The baseline (duplicate, critic or the
  mean of both) is the one that left the least return variance over the second half of the
  three runs' logs. Both go to ``runs/selfplay_luck/choice.json``.
* **train**: resumes ``runs/selfplay_luck`` until ``--until-hours`` of training in all; one
  ``hour_XX.pt`` per hour. Does nothing once ``stop.json`` says stop or at 24 hours.
* **eval** (Task 36b): the hour's network (playing its prior's top move, no search) against
  the app's `expert` (ISMCTS-10,000), the only benchmark since 2026-11-27; the same 1,000
  deals (seed 7) each time, both lineups, 95 % bootstrap ranges over deals. The start r3 plays
  the same deals; the gain over r3 is paired deal by deal. At hour 8 the stop rule: unless
  that gain (both lineups' mean per deal) has its whole 95 % range above 0, ``stop.json``
  stops the run. ``--final``: the last trained checkpoint and r3 again at 2,000 deals.
  (Hours 0 and 4 were first measured against r3 and ISMCTS-1000, 4,000 deals: ``table.md``.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "selfplay_luck"
SMOKE = ROOT / "runs" / "selfplay_luck_smoke"
START = "runs/net_r3/best.pt"
#: `expert` as the app plays it (crates/app-api solver.rs: ISMCTS-10,000, exploration 0.25,
#: greedy rollout epsilon 0.1, score objective, one thread, no exact endgame, no network).
#: The only benchmark opponent since 2026-11-27 (plan Task 36b, Brian).
EXPERT = "ismcts:iterations=10000"
EXPERT_DEALS = 1000
FINAL_DEALS = 2000
PULLS = (0.02, 0.1, 0.5)
SMOKE_MINUTES = 10.0
SMOKE_DEALS = 2000
EVAL_SEED = 7
LINEUPS = ("1+3", "ABAB")
HOURS = (0, 4, 8, 12, 16, 20, 24)
STOP_HOUR = 8
MAX_HOURS = 24
#: Baseline name -> ``oracle_weight``.
BASELINES = {"duplicate": 0.0, "oracle": 1.0, "mix": 0.5}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def threads() -> int:
    return int(os.environ.get("HKQ_THREADS", "8"))


def spec(checkpoint: str) -> str:
    return f"pvnet:checkpoint={checkpoint},mode=prior"


def hour_ckpt(hour: int) -> str:
    return START if hour == 0 else f"runs/selfplay_luck/hour_{hour:02d}.pt"


def match(cand: str, opp: str, lineup: str, deals: int, cache: Path) -> dict[str, Any]:
    """One lineup of a duplicate match, cached block by block; adds the per-deal scores."""
    from big2.eval.harness import play_match

    r = play_match(
        cand,
        opp,
        lineup,
        deals,
        seed=EVAL_SEED,
        workers=threads(),
        cache_dir=cache,
        block=500,
    ).to_dict()
    r["deal_scores"] = deal_scores(cache, cand, deals).tolist()
    return r


def deal_scores(cache: Path, cand: str, deals: int) -> np.ndarray:
    """The candidate's mean score per deal, in deal order, from the match's cached blocks."""
    rows: dict[int, float] = {}
    for f in cache.glob(f"block_s{EVAL_SEED}_d*.json"):
        saved = json.loads(f.read_text())
        a, _ = saved["deals"]
        for k, rec in enumerate(saved["records"]):
            v = rec[f"score|{cand}"]
            rows[a + k] = sum(v) / len(v)
    return np.array([rows[d] for d in range(deals)])


def bootstrap(x: np.ndarray, seed: int = 3) -> list[float]:
    from big2.eval.harness import bootstrap_ci

    return [float(v) for v in bootstrap_ci(x, seed=seed)]


def train_cfg(run: Path, pull: float, oracle_weight: float, **kw: Any) -> Any:
    from big2.rl.selfplay import SelfPlayConfig

    return SelfPlayConfig(
        run_dir=str(run),
        init=str(ROOT / START),
        pull=pull,
        oracle_weight=oracle_weight,
        device=kw.pop("device", "cuda"),
        **kw,
    )


# smoke ----------------------------------------------------------------------------


def smoke(args: argparse.Namespace) -> None:
    from big2.rl.selfplay import train

    rows: dict[str, Any] = {}
    resid: dict[str, list[float]] = {k: [] for k in BASELINES}
    for pull in PULLS:
        run = SMOKE / f"pull_{pull}"
        if not (run / "done.json").exists():
            log(f"smoke: pull {pull}, {args.minutes} min")
            cfg = train_cfg(run, pull, 0.0, max_minutes=args.minutes, device=args.device)
            train(cfg, log=lambda r: None)
            write_json(run / "done.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        recs = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
        late = recs[len(recs) // 2 :]
        for k in BASELINES:
            resid[k] += [r[f"resid_{k}"] for r in late]
        end = f"runs/selfplay_luck_smoke/pull_{pull}/latest.pt"
        row: dict[str, Any] = {
            "iters": recs[-1]["iter"],
            "games": recs[-1]["games"],
            "minutes": recs[-1]["minutes_total"],
            "kl_end": recs[-1]["kl"],
        }
        for lu in LINEUPS:
            log(f"smoke: pull {pull} vs start, {lu}")
            cache = SMOKE / f"pull_{pull}" / f"vs_start_{lu.replace('+', 'p')}"
            r = match(spec(end), spec(START), lu, args.deals, cache)
            row[lu] = {k: r[k] for k in ("deals", "paired_diff", "paired_ci95", "mean")}
        rows[str(pull)] = row
    ok = [p for p in PULLS if all(rows[str(p)][lu]["paired_ci95"][1] >= 0 for lu in LINEUPS)]
    pull = min(ok) if ok else max(PULLS)
    means = {k: float(np.mean(v)) for k, v in resid.items()}
    baseline = min(means, key=lambda k: means[k])
    choice = {
        "pull": pull,
        "baseline": baseline,
        "oracle_weight": BASELINES[baseline],
        "residual_variance": means,
        "runs": rows,
        "rule": "smallest pull whose paired diff vs start has a 95 % range reaching 0 in both "
        "lineups (none: 0.5); baseline with the least residual variance (second half of logs)",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(OUT / "choice.json", choice)
    log(json.dumps(choice, indent=2))


# train ----------------------------------------------------------------------------


def stopped() -> bool:
    path = OUT / "stop.json"
    return path.exists() and bool(json.loads(path.read_text())["stop"])


def train_stage(args: argparse.Namespace) -> None:
    from big2.rl.selfplay import train

    if stopped():
        log("stop.json says stop: nothing to train")
        return
    hours = min(args.until_hours, MAX_HOURS)
    choice = json.loads((OUT / "choice.json").read_text())
    import torch

    torch.set_num_threads(threads())
    cfg = train_cfg(
        OUT,
        choice["pull"],
        choice["oracle_weight"],
        max_minutes=args.max_minutes,
        until_minutes=hours * 60.0,
        device=args.device,
    )
    log(f"train until hour {hours}: pull {cfg.pull}, oracle_weight {cfg.oracle_weight}")
    train(cfg, log=lambda r: None)
    log(f"train: stopped at {json.loads((OUT / 'progress.json').read_text())}")


# eval -----------------------------------------------------------------------------


def expert_hour(hour: int, deals: int) -> dict[str, Any] | None:
    """The hour's network against `expert` (cached); None without a network.

    Results go to ``expert/hXX/results_<deals>.json``. A larger deal count reuses the
    smaller one's 500-deal blocks (same seed, same cache folder).
    """
    folder = OUT / "expert" / f"h{hour:02d}"
    path = folder / f"results_{deals}.json"
    if path.exists():
        res: dict[str, Any] = json.loads(path.read_text())
        return res
    ckpt = hour_ckpt(hour)
    if not (ROOT / ckpt).exists():
        log(f"eval: no network for hour {hour} ({ckpt})")
        return None
    res = {"hour": hour, "checkpoint": ckpt, "deals": deals, "opponent": EXPERT, "vs_expert": {}}
    for lu in LINEUPS:
        log(f"eval hour {hour}: vs expert ({EXPERT}), {lu}, {deals} deals")
        res["vs_expert"][lu] = match(
            spec(ckpt), EXPERT, lu, deals, folder / f"vs_expert_{lu.replace('+', 'p')}"
        )
    write_json(path, res)
    return res


def gain_over_start(res: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
    """The hour's score against `expert` minus the start's, paired by deal."""
    out: dict[str, Any] = {}
    both = []
    for lu in LINEUPS:
        d = np.array(res["vs_expert"][lu]["deal_scores"]) - np.array(
            start["vs_expert"][lu]["deal_scores"]
        )
        both.append(d)
        out[lu] = {"mean": float(d.mean()), "ci95": bootstrap(d)}
    m = (both[0] + both[1]) / 2
    out["both"] = {"mean": float(m.mean()), "ci95": bootstrap(m)}
    return out


def last_trained(until: int) -> int:
    """The latest 4-hour checkpoint at or before ``until`` that exists (0: the start)."""
    return max(h for h in HOURS if h <= until and (ROOT / hour_ckpt(h)).exists())


def eval_stage(args: argparse.Namespace) -> None:
    hour = last_trained(args.hour) if args.final else args.hour
    deals = FINAL_DEALS if args.final else args.deals
    start = expert_hour(0, deals)
    assert start is not None
    res = expert_hour(hour, deals) if hour else start
    if res is None:
        log(f"eval: hour {hour} was not trained (stopped run?); nothing to measure")
        table()
        return
    if hour == STOP_HOUR and not args.final and not (OUT / "stop.json").exists():
        g = gain_over_start(res, start)["both"]
        stop = not g["ci95"][0] > 0
        write_json(
            OUT / "stop.json",
            {
                "stop": stop,
                "hour": STOP_HOUR,
                "deals": deals,
                "gain_vs_expert": g,
                "rule": "stop unless hour 8's gain over the start (r3), both measured against "
                f"`expert` ({EXPERT}) and paired by deal, has its whole 95 % range above 0",
                "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        log(f"stop rule at hour {STOP_HOUR}: gain {g} -> {'STOP' if stop else 'go on'}")
    table()


# head to head (Task 36c) ----------------------------------------------------------

H2H_DEALS = 2000
FIRST_ABOVE_EXPERT = 8


def h2h_pairs(hour: int) -> list[tuple[str, int]]:
    """The older selves hour ``hour`` plays: the previous 4-hour checkpoint and hour 8."""
    pairs = [("prev", hour - 4)]
    if hour > FIRST_ABOVE_EXPERT + 4:
        pairs.append(("h8", FIRST_ABOVE_EXPERT))
    return pairs


def h2h_stage(args: argparse.Namespace) -> None:
    """Hour N against its older selves, networks only (prior top move, no search)."""
    hour = args.hour
    for name, older in h2h_pairs(hour):
        folder = OUT / "h2h" / f"h{hour:02d}_vs_h{older:02d}"
        path = folder / f"results_{args.deals}.json"
        if path.exists():
            continue
        if not (ROOT / hour_ckpt(hour)).exists() or not (ROOT / hour_ckpt(older)).exists():
            log(f"h2h: missing checkpoint for hour {hour} or {older}; skipped")
            continue
        res: dict[str, Any] = {"hour": hour, "older": older, "kind": name, "deals": args.deals}
        for lu in LINEUPS:
            log(f"h2h: hour {hour} vs hour {older}, {lu}, {args.deals} deals")
            r = match(
                spec(hour_ckpt(hour)),
                spec(hour_ckpt(older)),
                lu,
                args.deals,
                folder / lu.replace("+", "p"),
            )
            res[lu] = {k: r[k] for k in ("deals", "paired_diff", "paired_ci95", "mean")}
        write_json(path, res)
    h2h_table()


def h2h_table() -> str:
    """``runs/selfplay_luck/table_h2h.md``: every head-to-head result so far."""
    lines = ["| Hour | Against | Deals | 1+3 | ABAB |", "|---:|---|---:|---:|---:|"]
    for f in sorted((OUT / "h2h").glob("h*/results_*.json")):
        r = json.loads(f.read_text())
        cells = [fmt(r[lu]["paired_diff"], r[lu]["paired_ci95"]) for lu in LINEUPS]
        what = f"hour {r['older']} ({'previous' if r['kind'] == 'prev' else 'first above expert'})"
        lines.append(f"| {r['hour']} | {what} | {r['deals']:,} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (OUT / "table_h2h.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


# table ----------------------------------------------------------------------------


def fmt(m: float, ci: list[float]) -> str:
    return f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def table() -> str:
    """``runs/selfplay_luck/table_expert.md`` and ``results_expert.json`` from every eval.

    (``table.md`` keeps the first, superseded measure: hours 0 and 4 against r3 and
    ISMCTS-1000, 4,000 deals.)
    """
    lines = [
        "| Hour | Deals | vs `expert` 1+3 | vs `expert` ABAB "
        "| Gain over r3 vs `expert` (both lineups, paired) |",
        "|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for deals in (EXPERT_DEALS, FINAL_DEALS):
        start = expert_hour_cached(0, deals)
        for hour in HOURS:
            res = expert_hour_cached(hour, deals)
            if res is None or start is None:
                continue
            vs = res["vs_expert"]
            cells = [fmt(vs[lu]["paired_diff"], vs[lu]["paired_ci95"]) for lu in LINEUPS]
            g = gain_over_start(res, start) if hour else None
            cells.append(fmt(g["both"]["mean"], g["both"]["ci95"]) if g else "(start r3)")
            lines.append(f"| {hour} | {deals:,} | " + " | ".join(cells) + " |")
            rows.append(
                {
                    "hour": hour,
                    "deals": deals,
                    "vs_expert": {
                        lu: {k: vs[lu][k] for k in ("paired_diff", "paired_ci95", "mean")}
                        for lu in vs
                    },
                    "gain_over_start": g,
                }
            )
    text = "\n".join(lines) + "\n"
    if rows:
        (OUT / "table_expert.md").write_text(text, encoding="utf-8")
        write_json(
            OUT / "results_expert.json", {"rows": rows, "seed": EVAL_SEED, "opponent": EXPERT}
        )
    print(text, flush=True)
    return text


def expert_hour_cached(hour: int, deals: int) -> dict[str, Any] | None:
    path = OUT / "expert" / f"h{hour:02d}" / f"results_{deals}.json"
    if not path.exists():
        return None
    res: dict[str, Any] = json.loads(path.read_text())
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--minutes", type=float, default=SMOKE_MINUTES)
    s.add_argument("--deals", type=int, default=SMOKE_DEALS)
    s.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--until-hours", type=int, required=True)
    t.add_argument("--max-minutes", type=float, default=250.0)
    t.add_argument("--device", default="cuda")
    e = sub.add_parser("eval")
    e.add_argument("--hour", type=int, required=True)
    e.add_argument("--deals", type=int, default=EXPERT_DEALS)
    e.add_argument(
        "--final",
        action="store_true",
        help=f"the last trained checkpoint up to --hour (and r3), {FINAL_DEALS:,} deals",
    )
    h = sub.add_parser("h2h")
    h.add_argument("--hour", type=int, required=True)
    h.add_argument("--deals", type=int, default=H2H_DEALS)
    sub.add_parser("table")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"smoke": smoke, "train": train_stage, "eval": eval_stage, "h2h": h2h_stage, "table": lambda a: table()}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
