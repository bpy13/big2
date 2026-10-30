"""Build-run phase 4 experiments (teaching depth). Results: docs/study-results/build-phase4.md.

Usage::

    python experiments/build_phase4.py audit                 # Task 26: runs/p4_audit/hints-50.json
    python experiments/build_phase4.py regrade --threads 8   # Task 27: runs/p4_regrade/results.json
    python experiments/build_phase4.py regrade --reviewed 4 --out p4_regrade_tiny  # smoke run

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

Both run the native ``big2-teach`` program (``crates/teach/src/main.rs``), built here with
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
    ns = p.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
