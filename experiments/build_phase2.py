"""Build-run phase 2 matches (Task 15). Results: docs/study-results/build-phase2.md.

Usage::

    python experiments/build_phase2.py all --workers 8
    python experiments/build_phase2.py puct-vs-ismcts --deals 1000 --workers 8
    python experiments/build_phase2.py c-puct --deals 300 --workers 8
    python experiments/build_phase2.py heads --deals 500 --workers 8
    python experiments/build_phase2.py equal-time --deals 1000 --seconds 0.5 --workers 8

One subcommand per experiment (plan ``docs/plans/2026-10-29-big2-bot-build.md``):

* ``puct-vs-ismcts`` (a, bar A9): PUCT-2000 with the round-0 network against
  ISMCTS-1000; A9 needs ≥ +1.0 with the interval above 0.
* ``c-puct`` (d): PUCT-2000 with ``c_puct`` 1.0, 1.5 and 2.5 against ISMCTS-1000 on
  the first deals; 1.5 is read back from (a)'s saved blocks (same deal seeds).
* ``heads`` (c, bar A10): the v2 network's distribution head (``pvnet`` agent,
  argmax of the expected score under the game's scoring) against Task 7's
  mean-score network (``qnet``, argmax), under ``tiered`` and under ``win_loss``.
  A10: the distribution head loses at most 0.5 under tiered and wins under win_loss.
* ``equal-time`` (b): PUCT against ISMCTS with the same wall time per move on one
  thread (``time_limit``, no simulation cap). Depends on CPU load: the results
  record the load at the start.
* ``all``: the four above in the order a, d, c, b (cheap and fixed-count first).

Every match runs both lineups (``1+3`` and ``ABAB``) on duplicate deals through
:func:`big2.eval.harness.play_match`, saved in blocks of ``--block`` deals under
``runs/<out>/<cell>/<lineup>/``. A subcommand whose ``results.json`` exists is
skipped; an interrupted one resumes from its saved blocks. ``progress.json`` in
``runs/<out>/`` moves after every block.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ISMCTS_1000 = "ismcts:iterations=1000"
DEFAULT_NET = "runs/net_r0/pvnet-r0.pknn"
DEFAULT_CHECKPOINT = "runs/net_r0/best.pt"
#: Task 7's mean-score Q-network (the ladder's L2), played greedily.
QNET_TASK7 = "qnet:path=models/big2-qnet-duty.pknn,temperature=0"
LINEUPS = ["1+3", "ABAB"]
#: A9: PUCT-2000 must beat ISMCTS-1000 by at least this many points per game.
A9_BAR = 1.0
#: A10: the distribution head may lose at most this much under tiered scoring.
A10_TIERED_LOSS = 0.5
C_PUCT_VALUES = [1.0, 1.5, 2.5]
#: Huge counts so that only ``time_limit`` stops the equal-time searches.
UNCAPPED = 10_000_000


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def puct_spec(net: str, simulations: int = 2000, c_puct: float | None = None) -> str:
    spec = f"puct:net={net},simulations={simulations}"
    return spec if c_puct is None else f"{spec},c_puct={c_puct}"


class Progress:
    """``progress.json``: finished cells, the one running, the ones still to come."""

    def __init__(self, out: Path, todo: list[str]) -> None:
        self.path = out / "progress.json"
        self.todo = todo
        self.done: list[str] = []
        self.t0 = time.time()

    def update(self, running: str | None, done: str | None = None) -> None:
        if done is not None and done not in self.done:
            self.done.append(done)
        _write_json(
            self.path,
            {
                "done": self.done,
                "running": running,
                "next": [t for t in self.todo if t not in self.done and t != running],
                "minutes_this_run": round((time.time() - self.t0) / 60, 1),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def play_cell(
    args: argparse.Namespace,
    progress: Progress,
    cell: str,
    candidate: str,
    opponent: str,
    deals: int,
    rules: Any = None,
    folder: str | None = None,
) -> dict[str, Any]:
    """Both lineups of one match, block by block (so progress moves every block)."""
    from big2.eval.harness import play_match

    reports = {}
    for lu in LINEUPS:
        cache = ROOT / "runs" / args.out / (folder or cell) / lu.replace("+", "p")
        steps = list(range(args.block, deals, args.block)) + [deals]
        for n in steps:
            progress.update(f"{cell} {lu}, deals up to {n} of {deals}")
            r = play_match(
                candidate,
                opponent,
                lu,
                n,
                seed=args.seed,
                workers=args.workers,
                rules=rules,
                cache_dir=cache,
                block=args.block,
            )
        reports[lu] = r.to_dict()
        progress.update(None, done=f"{cell} {lu}")
        print(
            f"done {cell} {lu}: {r.mean:+.2f} [{r.ci95[0]:+.2f}, {r.ci95[1]:+.2f}], "
            f"win {r.win_rate:.1%}, {r.seconds / 60:.0f} min",
            flush=True,
        )
    return reports


def _row(label: str, lu: str, r: dict[str, Any]) -> str:
    return (
        f"| {label} | {lu} | {r['deals']:,} | {r['mean']:+.2f} [{r['ci95'][0]:+.2f}, "
        f"{r['ci95'][1]:+.2f}] | {r['win_rate']:.1%} [{r['win_ci95'][0]:.1%}, "
        f"{r['win_ci95'][1]:.1%}] | {r['opponent_mean']:+.2f} | {r['opponent_win_rate']:.1%} "
        f"| {r['paired_diff']:+.2f} [{r['paired_ci95'][0]:+.2f}, {r['paired_ci95'][1]:+.2f}] "
        f"| {r['seconds_per_move']:.3f} / {r['opponent_seconds_per_move']:.3f} |"
    )


TABLE_HEAD = [
    "| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) "
    "| Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]


def _finish(args: argparse.Namespace, name: str, results: dict[str, Any], lines: list[str]) -> None:
    folder = ROOT / "runs" / args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    table = "\n".join(lines) + "\n"
    _write_json(folder / "results.json", results)
    (folder / "table.md").write_text(table, encoding="utf-8")
    print(table, flush=True)


def _done(args: argparse.Namespace, name: str) -> bool:
    if (ROOT / "runs" / args.out / name / "results.json").exists():
        print(f"skip {name}: results.json exists", flush=True)
        return True
    return False


# (a) puct-vs-ismcts -------------------------------------------------------------


def puct_vs_ismcts(args: argparse.Namespace, progress: Progress) -> None:
    name = "a_puct_vs_ismcts"
    if _done(args, name):
        return
    cand = puct_spec(args.net, args.simulations)
    reports = play_cell(args, progress, name, cand, ISMCTS_1000, args.deals)
    passed = {lu: r["mean"] >= A9_BAR and r["ci95"][0] > 0 for lu, r in reports.items()}
    lines = [
        f"(a) `{cand}` vs `{ISMCTS_1000}`, {args.deals:,} duplicate deals (seeds 0…"
        f"{args.deals - 1}, each 4×, seats rotated), duty on, tiered, no four-card plays. "
        "95 % bootstrap intervals over deals. Fixed counts: results do not depend on CPU load.",
        "",
        *TABLE_HEAD,
        *(_row("PUCT-2000 vs ISMCTS-1000", lu, r) for lu, r in reports.items()),
        "",
        f"A9 bar (≥ +{A9_BAR:.1f}, interval above 0): "
        + ", ".join(f"{lu} {'passes' if ok else 'fails'}" for lu, ok in passed.items())
        + ".",
    ]
    _finish(
        args,
        name,
        {"candidate": cand, "opponent": ISMCTS_1000, "reports": reports, "a9_passed": passed},
        lines,
    )


# (d) c-puct ---------------------------------------------------------------------


def c_puct(args: argparse.Namespace, progress: Progress) -> None:
    name = "d_c_puct"
    if _done(args, name):
        return
    rows: dict[str, Any] = {}
    for c in C_PUCT_VALUES:
        if c == 1.5:
            # The default: the same spec and deal seeds as (a), so its blocks are reused.
            cand = puct_spec(args.net, args.simulations)
            folder = "a_puct_vs_ismcts"
        else:
            cand = puct_spec(args.net, args.simulations, c)
            folder = f"{name}/c{c}"
        rows[str(c)] = {
            "candidate": cand,
            "reports": play_cell(
                args, progress, f"{name} c={c}", cand, ISMCTS_1000, args.c_deals, folder=folder
            ),
        }
    best = max(rows, key=lambda c: sum(r["mean"] for r in rows[c]["reports"].values()))
    lines = [
        f"(d) PUCT-{args.simulations} with `c_puct` in {C_PUCT_VALUES} vs `{ISMCTS_1000}`, "
        f"{args.c_deals:,} duplicate deals (seeds 0…{args.c_deals - 1}; c = 1.5 is the first "
        f"{args.c_deals} deals of (a)).",
        "",
        *TABLE_HEAD,
        *(
            _row(f"c_puct = {c}", lu, r)
            for c, row in rows.items()
            for lu, r in row["reports"].items()
        ),
        "",
        f"Best mean over both lineups: c_puct = {best}.",
    ]
    _finish(args, name, {"rows": rows, "best": float(best)}, lines)


# (c) heads ----------------------------------------------------------------------


def heads(args: argparse.Namespace, progress: Progress) -> None:
    from big2.games.big2 import Rules

    name = "c_heads"
    if _done(args, name):
        return
    cand = f"pvnet:checkpoint={args.checkpoint}"
    rows: dict[str, Any] = {}
    for scoring in ("tiered", "win_loss"):
        rows[scoring] = play_cell(
            args,
            progress,
            f"{name} {scoring}",
            cand,
            QNET_TASK7,
            args.heads_deals,
            rules=Rules(scoring=scoring),
            folder=f"{name}/{scoring}",
        )
    t = rows["tiered"]
    w = rows["win_loss"]
    a10 = {
        "tiered_ok": all(r["mean"] >= -A10_TIERED_LOSS for r in t.values()),
        "win_loss_wins": all(r["mean"] > 0 for r in w.values()),
    }
    lines = [
        f"(c) The v2 distribution head (`{cand}`: argmax of the actor's expected score from "
        f"the four cards-left distributions under the game's scoring) vs Task 7's mean-score "
        f"network (`{QNET_TASK7}`), {args.heads_deals:,} duplicate deals per scoring, duty on, "
        "no search on either side. Under win_loss the winner scores +3 and each loser −1.",
        "",
        *TABLE_HEAD,
        *(
            _row(f"v2 head vs Task 7 qnet, {s}", lu, r)
            for s, rr in rows.items()
            for lu, r in rr.items()
        ),
        "",
        f"A10 (lose ≤ {A10_TIERED_LOSS} under tiered: "
        f"{'yes' if a10['tiered_ok'] else 'no'}; win under win_loss: "
        f"{'yes' if a10['win_loss_wins'] else 'no'}).",
    ]
    _finish(
        args, name, {"candidate": cand, "opponent": QNET_TASK7, "rows": rows, "a10": a10}, lines
    )


# (b) equal-time -----------------------------------------------------------------


def cpu_load() -> dict[str, Any]:
    """Machine load when the match starts (Windows: typeperf; elsewhere os.getloadavg)."""
    import subprocess

    info: dict[str, Any] = {"logical_cpus": os.cpu_count()}
    try:
        out = subprocess.run(
            ["typeperf", r"\Processor(_Total)\% Processor Time", "-sc", "5"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        vals = [float(line.split(",")[1].strip('"')) for line in out.splitlines()[2:7]]
        info["cpu_percent_before_start"] = round(sum(vals) / len(vals), 1)
    except Exception as e:  # noqa: BLE001 - load is informative only
        info["cpu_percent_before_start"] = f"not measured ({e})"
    return info


def equal_time(args: argparse.Namespace, progress: Progress) -> None:
    name = "b_equal_time"
    if _done(args, name):
        return
    folder = ROOT / "runs" / args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    load_path = folder / "load.json"
    if not load_path.exists():
        _write_json(load_path, {**cpu_load(), "workers": args.workers, "at": time.ctime()})
    load = json.loads(load_path.read_text())
    s = args.seconds
    cand = f"puct:net={args.net},simulations={UNCAPPED},time_limit={s},threads=1"
    opp = f"ismcts:iterations={UNCAPPED},time_limit={s},threads=1"
    reports = play_cell(args, progress, name, cand, opp, args.deals)
    lines = [
        f"(b) PUCT vs ISMCTS at equal wall time: {s} s per move each, one thread, no simulation "
        f"cap (`{cand}` vs `{opp}`); forced moves play at once. {args.deals:,} duplicate deals. "
        f"Ran under outside CPU load: {load.get('cpu_percent_before_start')} % of "
        f"{load.get('logical_cpus')} logical CPUs busy before the start (13 cards and mahjong "
        f"jobs and other projects), {load.get('workers')} workers. Both sides share each "
        "worker in turn, so load slows both alike.",
        "",
        *TABLE_HEAD,
        *(_row(f"PUCT vs ISMCTS, {s} s per move", lu, r) for lu, r in reports.items()),
    ]
    _finish(
        args, name, {"candidate": cand, "opponent": opp, "load": load, "reports": reports}, lines
    )


# main ---------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[argparse.Namespace, Progress], None]] = {
    "puct-vs-ismcts": puct_vs_ismcts,
    "c-puct": c_puct,
    "heads": heads,
    "equal-time": equal_time,
}
CELLS = {
    "puct-vs-ismcts": ["a_puct_vs_ismcts"],
    "c-puct": [f"d_c_puct c={c}" for c in C_PUCT_VALUES],
    "heads": ["c_heads tiered", "c_heads win_loss"],
    "equal-time": ["b_equal_time"],
}
RESULT_DIRS = {
    "puct-vs-ismcts": "a_puct_vs_ismcts",
    "c-puct": "d_c_puct",
    "heads": "c_heads",
    "equal-time": "b_equal_time",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--deals", type=int, default=1000, help="deals for (a) and (b)")
    ap.add_argument("--c-deals", type=int, default=300, help="deals per c_puct value (d)")
    ap.add_argument("--heads-deals", type=int, default=500, help="deals per scoring (c)")
    ap.add_argument("--seconds", type=float, default=0.5, help="seconds per move in (b)")
    ap.add_argument("--simulations", type=int, default=2000)
    ap.add_argument("--net", default=DEFAULT_NET, help=".pknn network v2 for PUCT")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="v2 .pt for (c)")
    ap.add_argument("--block", type=int, default=100, help="deals per saved block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="p2_matches", help="folder under runs/")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.chdir(ROOT)  # network paths in agent specs are relative to the repo root
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    todo = [f"{c} {lu}" for n in names for c in CELLS[n] for lu in LINEUPS]
    progress = Progress(out, todo)
    for n in names:  # finished subcommands count as done from the start
        if (out / RESULT_DIRS[n] / "results.json").exists():
            progress.done += [f"{c} {lu}" for c in CELLS[n] for lu in LINEUPS]
    for n in names:
        COMMANDS[n](args, progress)
    progress.update(None)
    if args.command == "all":
        _write_json(out / "results.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
