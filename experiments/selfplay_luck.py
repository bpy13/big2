"""Self-play pilot with the luck taken out (plan Task 36).

Results: docs/study-results/selfplay-luck.md. Trainer: :mod:`big2.rl.selfplay`.

Usage (each stage is one queue job, see ``experiments/selfplay_luck_job.ps1``)::

    python experiments/selfplay_luck.py smoke                 # gpu: picks the pull, ~35 min
    python experiments/selfplay_luck.py train --until-hours 4 # gpu: trains up to hour 4
    python experiments/selfplay_luck.py eval --hour 4         # cpu: measures hour 4 (and 0)
    python experiments/selfplay_luck.py table                 # table.md from every eval

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
* **eval**: the hour's network (playing its prior's top move, no search) against the start
  network and against ISMCTS-1000, the outside opponent self-play never trains against; the
  same 4,000 deals (seed 7) each time, both lineups, 95 % bootstrap ranges over deals. The
  gain against ISMCTS-1000 is paired deal by deal with the start's own match. At hour 8 the
  stop rule: unless that gain (both lineups' mean per deal) has its whole 95 % range above 0,
  ``stop.json`` stops the run.
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
OUTSIDE = "ismcts:iterations=1000"
PULLS = (0.02, 0.1, 0.5)
SMOKE_MINUTES = 10.0
SMOKE_DEALS = 2000
EVAL_DEALS = 4000
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


def eval_hour(hour: int) -> dict[str, Any] | None:
    """Plays the hour's matches (cached); returns its results, or None without a network."""
    path = OUT / f"eval_h{hour:02d}" / "results.json"
    if path.exists():
        res: dict[str, Any] = json.loads(path.read_text())
        return res
    ckpt = hour_ckpt(hour)
    if not (ROOT / ckpt).exists():
        log(f"eval: no network for hour {hour} ({ckpt})")
        return None
    res = {"hour": hour, "checkpoint": ckpt, "vs_start": {}, "vs_outside": {}}
    folder = OUT / f"eval_h{hour:02d}"
    for lu in LINEUPS:
        tag = lu.replace("+", "p")
        if hour:
            log(f"eval hour {hour}: vs start, {lu}")
            r = match(spec(ckpt), spec(START), lu, EVAL_DEALS, folder / f"vs_start_{tag}")
            r.pop("deal_scores")
            res["vs_start"][lu] = r
        log(f"eval hour {hour}: vs {OUTSIDE}, {lu}")
        res["vs_outside"][lu] = match(
            spec(ckpt), OUTSIDE, lu, EVAL_DEALS, folder / f"vs_outside_{tag}"
        )
    write_json(path, res)
    return res


def gain_over_start(res: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
    """The hour's score against the outside opponent minus the start's, paired by deal."""
    out: dict[str, Any] = {}
    both = []
    for lu in LINEUPS:
        d = np.array(res["vs_outside"][lu]["deal_scores"]) - np.array(
            start["vs_outside"][lu]["deal_scores"]
        )
        both.append(d)
        out[lu] = {"mean": float(d.mean()), "ci95": bootstrap(d)}
    m = (both[0] + both[1]) / 2
    out["both"] = {"mean": float(m.mean()), "ci95": bootstrap(m)}
    return out


def eval_stage(args: argparse.Namespace) -> None:
    start = eval_hour(0)
    assert start is not None
    res = eval_hour(args.hour) if args.hour else start
    if res is None:
        log(f"eval: hour {args.hour} was not trained (stopped run?); nothing to measure")
        table()
        return
    if args.hour == STOP_HOUR and not (OUT / "stop.json").exists():
        g = gain_over_start(res, start)["both"]
        stop = not g["ci95"][0] > 0
        write_json(
            OUT / "stop.json",
            {
                "stop": stop,
                "hour": STOP_HOUR,
                "gain_vs_outside": g,
                "rule": "stop unless hour 8's gain over the start against "
                f"{OUTSIDE} has its whole 95 % range above 0",
                "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        log(f"stop rule at hour {STOP_HOUR}: gain {g} -> {'STOP' if stop else 'go on'}")
    table()


# table ----------------------------------------------------------------------------


def fmt(m: float, ci: list[float]) -> str:
    return f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def table() -> str:
    """``runs/selfplay_luck/table.md`` and ``results.json`` from every finished eval."""
    start = eval_hour_cached(0)
    lines = [
        "| Hour | vs start 1+3 | vs start ABAB | vs ISMCTS-1000 1+3 | vs ISMCTS-1000 ABAB "
        "| Gain vs ISMCTS-1000 over start (both lineups) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for hour in HOURS:
        res = eval_hour_cached(hour)
        if res is None or start is None:
            continue
        vs = res["vs_start"]
        cells = [
            fmt(vs[lu]["paired_diff"], vs[lu]["paired_ci95"]) if hour else "(start)"
            for lu in LINEUPS
        ]
        outside = res["vs_outside"]
        cells += [fmt(outside[lu]["paired_diff"], outside[lu]["paired_ci95"]) for lu in LINEUPS]
        g = gain_over_start(res, start) if hour else None
        cells.append(fmt(g["both"]["mean"], g["both"]["ci95"]) if g else "(start)")
        lines.append(f"| {hour} | " + " | ".join(cells) + " |")
        rows.append(
            {
                "hour": hour,
                "vs_start": {
                    lu: {k: vs[lu][k] for k in ("paired_diff", "paired_ci95")} for lu in vs
                },
                "vs_outside": {
                    lu: {k: outside[lu][k] for k in ("paired_diff", "paired_ci95", "mean")}
                    for lu in outside
                },
                "gain_over_start": g,
            }
        )
    text = "\n".join(lines) + "\n"
    if rows:
        (OUT / "table.md").write_text(text, encoding="utf-8")
        write_json(OUT / "results.json", {"rows": rows, "deals": EVAL_DEALS, "seed": EVAL_SEED})
    print(text, flush=True)
    return text


def eval_hour_cached(hour: int) -> dict[str, Any] | None:
    path = OUT / f"eval_h{hour:02d}" / "results.json"
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
    sub.add_parser("table")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"smoke": smoke, "train": train_stage, "eval": eval_stage, "table": lambda a: table()}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
