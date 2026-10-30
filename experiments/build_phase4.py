"""Build-run phase 4 experiments (teaching depth). Results: docs/study-results/build-phase4.md.

Usage::

    python experiments/build_phase4.py audit                 # Task 26: runs/p4_audit/hints-50.json
    python experiments/build_phase4.py regrade --threads 8   # Task 27: runs/p4_regrade/results.json
    python experiments/build_phase4.py regrade --reviewed 4 --out p4_regrade_tiny  # smoke run
    python experiments/build_phase4.py ladder --threads 8    # Task 29: runs/p4_ladder/results.json
    python experiments/build_phase4.py ladder --deals 8 --hard-iterations 100 \
        --expert-iterations 300 --out p4_ladder_tiny                 # smoke run
    python experiments/build_phase4.py mine --threads 8      # Task 28: drills/big2.json
    python experiments/build_phase4.py mine --games 8 --out p4_mine_tiny \
        --drills runs/p4_mine_tiny/big2.json                         # smoke run

* ``audit`` (Task 26, A20): 50 positions from noisy greedy self-play; on each, the paired
  move table (every legal move on the same 300 sampled deals), its best move as the hint and
  the hint's reason keys. Brian rates each entry right / irrelevant / wrong (B11).
* ``regrade`` (Task 27, A21): ``--reviewed`` self-play deals by ISMCTS-300 in every seat;
  every decision with a choice graded twice by the paired grader (300 deals, doubled while the
  grade is ``unclear`` up to ``--max-deals``, two seeds), every
  5th decision of a deal also by a 10x grader (3,000 deals); the luck ledger of every seat.
  Prints the agreement shares (A21 bar: >= 95 %), the share of ``unclear`` grades, the
  phone-vs-10x agreement (bar >= 90 %) and the ledger's largest sum error (bar 1e-9), and
  writes a Markdown table for build-phase4.md §1 beside the JSON.

* ``ladder`` (Task 29, A23): ladder v2 (``crates/teach/src/ladder.rs``), each pair of
  adjacent levels on ``--deals`` duplicate deals, lineup A B A B, each deal 4x with the hands
  rotated; A23: strict order and every adjacent gap >= 1.5 points. Resume: none needed per pair
  (about 45 min in all at 8 threads with the default budgets, estimate); rerun to redo.
* ``mine`` (Task 28, A22): ``big2-teach mine`` over ``--games`` self-play deals (resumes from
  ``runs/<out>/progress.json``), then ``drills/big2.json`` (``big2.teach.drills``), the
  50-drill audit sample ``runs/p4_audit/drills-50.json`` and the mistake clusters beside the
  app's skill list (``big2.teach.cluster``).

All run the native ``big2-teach`` program (``crates/teach/src/main.rs``), built here with
``cargo build --release -p big2-teach``, so the experiments need no rebuild of the Python
extension.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target" / "release" / ("big2-teach.exe" if sys.platform == "win32" else "big2-teach")
A21_BAR = 0.95
REFERENCE_BAR = 0.90


def build() -> None:
    subprocess.run(["cargo", "build", "--release", "-p", "big2-teach"], cwd=ROOT, check=True)


def run(args: list[str]) -> None:
    build()
    subprocess.run([str(EXE), *args], cwd=ROOT, check=True)


def audit(ns: argparse.Namespace) -> None:
    out = ROOT / "runs" / ns.out / "hints-50.json"
    run(["audit", "--out", str(out), "--positions", str(ns.positions), "--deals", str(ns.deals)])
    data = json.loads(out.read_text(encoding="utf-8"))
    print(f"{len(data['entries'])} entries; key counts: {data['key_counts']}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def regrade_table(r: dict) -> str:
    """The build-phase4.md §1 table from a regrade results.json."""
    a, ref, led = r["agreement"], r["phone_vs_reference"], r["ledger"]
    m = r["method"]
    verdict = "met" if a["strict"] >= A21_BAR else "not met"
    ref_verdict = "met" if ref["strict"] >= REFERENCE_BAR else "not met"
    led_verdict = "met" if led["max_abs_sum_error"] <= 1e-9 else "not met"
    mean = led["mean_abs"]
    rows = [
        (
            f"Same grade, two seeds ({r['decisions_graded']} decisions with a choice)",
            _pct(a["strict"]),
            "≥ 95 % (A21)",
            verdict,
        ),
        ("Compatible (an `unclear` range covers the other grade)", _pct(a["compatible"]), "–", "–"),
        ("Same grade with `best` and `good` merged", _pct(a["best_good_merged"]), "–", "–"),
        ("Share of `unclear` grades", _pct(r["unclear_share"]), "–", "–"),
        (
            f"Phone grader = 10x grader ({ref['decisions']} decisions)",
            _pct(ref["strict"]),
            "≥ 90 %",
            ref_verdict,
        ),
        ("Phone vs 10x, compatible", _pct(ref["compatible"]), "–", "–"),
        (
            f"Ledger: largest sum error over {led['seats']} seat-deals",
            f"{led['max_abs_sum_error']:.1e}",
            "≤ 1e-9 (A21)",
            led_verdict,
        ),
    ]
    lines_pts = ", ".join(f"{x:.2f}" for x in m["thresholds_points"])
    lines = [
        f"Reviewed deals: {m['reviewed_deals']} ({m['players']}); grader: {m['grader']}; "
        f"reference: {m['reference']}; lines {lines_pts} points.",
        "",
        "| Measure | Value | Bar | Verdict |",
        "|---|---|---|---|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
        "Ledger, mean absolute size per seat-deal (points): "
        f"score {mean['score']:.2f}, deal strength {mean['deal_strength']:.2f}, "
        f"opponents' errors {mean['opponents_errors']:.2f}, "
        f"own errors {mean['own_errors']:.2f}, residual {mean['residual']:.2f}.",
        "",
        f"Grades (seed A): {r['grade_counts_seed_a']}. "
        f"Forced decisions not graded: {r['forced_skipped']}. "
        f"Time: {r['seconds']:.0f} s on {m['threads']} threads.",
    ]
    return "\n".join(lines) + "\n"


def regrade(ns: argparse.Namespace) -> None:
    out_dir = ROOT / "runs" / ns.out
    out = out_dir / "results.json"
    run(
        [
            "regrade",
            "--out",
            str(out),
            "--reviewed",
            str(ns.reviewed),
            "--deals",
            str(ns.deals),
            "--threads",
            str(ns.threads),
            "--iterations",
            str(ns.iterations),
            "--max-deals",
            str(ns.max_deals),
        ]
    )
    r = json.loads(out.read_text(encoding="utf-8"))
    table = regrade_table(r)
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    print(table)


def ladder(ns: argparse.Namespace) -> None:
    out_dir = ROOT / "runs" / ns.out
    out = out_dir / "results.json"
    args = ["ladder", "--out", str(out), "--deals", str(ns.deals), "--threads", str(ns.threads)]
    if ns.hard_iterations:
        args += ["--hard-iterations", str(ns.hard_iterations)]
    if ns.expert_iterations:
        args += ["--expert-iterations", str(ns.expert_iterations)]
    run(args)
    r = json.loads(out.read_text(encoding="utf-8"))
    lines = [
        "| Upper vs lower | Mean (points per game) | 95 % | Upper wins | Gap ≥ 1.5 |",
        "|---|---|---|---|---|",
    ]
    for p in r["pairs"]:
        ok = "yes" if p["mean"] >= 1.5 and p["mean"] - p["ci95"] > 0 else "no"
        lines.append(
            f"| {p['upper']} vs {p['lower']} | {p['mean']:+.2f} | ± {p['ci95']:.2f} "
            f"| {100 * p['upper_win_rate']:.1f} % | {ok} |"
        )
    shares = ", ".join(f"{m['level']} {100 * m['share']:.1f} %" for m in r["mistake_share"])
    lines += ["", f"Mistakes fired per decision with a choice: {shares}."]
    lines += [f"Strict order: {r['strict_order']}; every gap ≥ 1.5: {r['gaps_at_least_1_5']}."]
    table = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    print(table)


def mine(ns: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from big2.teach.cluster import clusters, format_clusters
    from big2.teach.drills import read_raw, run_miner, write_drills

    run_dir = ROOT / "runs" / ns.out
    raw = run_miner(run_dir, ns.games, ns.threads, ["--deals", str(ns.deals)])
    summary = write_drills(raw, ROOT / ns.drills, ROOT / ns.audit)
    app = ROOT.parent / "hk-games" / "app" / "src" / "drills" / "big2.json"
    skills = None
    if app.exists():
        skills = sorted({d["skill"] for d in json.loads(app.read_text(encoding="utf-8"))})
    text = format_clusters(clusters(read_raw(raw), ns.min_cluster), skills)
    (run_dir / "clusters.txt").write_text(text + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="Task 26: 50 hints with reason keys for Brian")
    a.add_argument("--positions", type=int, default=50)
    a.add_argument("--deals", type=int, default=300)
    a.add_argument("--out", default="p4_audit")
    a.set_defaults(func=audit)
    g = sub.add_parser("regrade", help="Task 27: grade repeatability and the luck ledger")
    g.add_argument("--reviewed", type=int, default=50)
    g.add_argument("--deals", type=int, default=300)
    g.add_argument("--iterations", type=int, default=300)
    g.add_argument(
        "--max-deals", type=int, default=1200, help="double the deals while unclear, up to this"
    )
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--out", default="p4_regrade")
    g.set_defaults(func=regrade)
    lad = sub.add_parser("ladder", help="Task 29: ladder v2 round of adjacent levels")
    lad.add_argument("--deals", type=int, default=500)
    lad.add_argument("--threads", type=int, default=8)
    lad.add_argument("--hard-iterations", type=int, default=0, help="0 = the level's default")
    lad.add_argument("--expert-iterations", type=int, default=0, help="0 = the level's default")
    lad.add_argument("--out", default="p4_ladder")
    lad.set_defaults(func=ladder)
    m = sub.add_parser("mine", help="Task 28: drill miner and mistake clusters")
    m.add_argument("--games", type=int, default=700)
    m.add_argument("--deals", type=int, default=600)
    m.add_argument("--threads", type=int, default=8)
    m.add_argument("--min-cluster", type=int, default=10)
    m.add_argument("--out", default="p4_mine")
    m.add_argument("--drills", default="drills/big2.json")
    m.add_argument("--audit", default="runs/p4_audit/drills-50.json")
    m.set_defaults(func=mine)
    ns = p.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
