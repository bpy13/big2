"""Expert iteration, round N (plan Task 17). Results: docs/study-results/build-phase3.md §1.

Usage::

    python experiments/expert_iteration.py round 1 --stages label --workers 8
    python experiments/expert_iteration.py round 1 --stages train
    python experiments/expert_iteration.py round 1 --stages match --workers 8
    python experiments/expert_iteration.py round 1            # all three in order

Round 1 follows the recommendation in docs/open-items.md (2026-10-30) after Task 15:

* **label** — a stronger teacher than round 0's ISMCTS-1000: ISMCTS-5000 (greedy rollout),
  which beat ISMCTS-1000 by +1.09 [+0.35, +1.87] (paired, 500 deals; build-phase1.md, ladder
  v0). PUCT λ = 0.5 only tied ISMCTS-1000 (build-phase2.md §6.2) at about 4× ISMCTS-5000's
  time per move, and it would feed the round-0 network's own errors back. Same generator and
  shard format as round 0 (``big2 rl distill-gen``), 25 % random top-3 moves.
* **train** — network v2 starting from round 0's weights, with the value trained on every
  candidate's search score as the main target (``score_weight`` 100 against 0.5 in round 0,
  where the score loss was ~0.4 % of the total), the cards-left distributions kept as
  auxiliary heads (weight 0.25 each). Then the calibration check of build-phase2.md §6.1 on
  round 1's held-out shards: the slope of the search's score on the prediction should be
  near 1 (round 0: 0.86).
* **match** — PUCT-2000 with the new network, ``c_puct`` 2.5, leaf weight λ = 1 and λ = 0.5,
  against ISMCTS-1000 on 300 duplicate deals each (both lineups); the better λ goes on to
  1,000 deals (bar A9: ≥ +1.0, interval above 0). Only if it passes A9: round 1 against
  round 0 (both PUCT-2000 at that λ), 1,000 deals (bar A12). Otherwise the loop stops
  (proposal §10 row 2: ISMCTS ships as ``expert``).

Every stage resumes: labelling counts the shards already written, training continues from
``latest.pt``, matches reuse their saved blocks; a stage whose results exist is skipped.
``runs/round<N>/stages.json`` says which stages are done; ``progress.json`` there moves
with the matches, ``label/progress.json`` with the labelling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from build_phase2 import (  # also opts this process out of power throttling
    A9_BAR,
    ISMCTS_1000,
    TABLE_HEAD,
    Progress,
    _row,
    _write_json,
    play_cell,
)

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
#: Round 1's teacher (see the module docstring).
TEACHER_ITERATIONS = 5000
LEAF_WEIGHTS = [1.0, 0.5]
C_PUCT = 2.5
#: Loss weights of round 1's training (round 0: actor 1, seats 1, prior 1, score 0.5).
WEIGHTS = {"actor_weight": 0.25, "seats_weight": 0.25, "prior_weight": 1.0, "score_weight": 100.0}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def paths(n: int, tag: str = "") -> dict[str, Path]:
    """Folders of round ``n`` (``tag`` makes separate ones, e.g. for a test run)."""
    base = ROOT / "runs" / f"round{n}{tag}"
    net = ROOT / "runs" / f"net_r{n}{tag}"
    prev = ROOT / "runs" / f"net_r{n - 1}"
    return {
        "base": base,
        "data": base / "data",
        "label": base / "label",
        "net": net,
        "prev_net": prev,
        "pknn": net / f"pvnet-r{n}.pknn",
        "prev_pknn": prev / f"pvnet-r{n - 1}.pknn",
    }


def stage_progress(base: Path, stage: str, note: str) -> None:
    path = base / "stages.json"
    state = json.loads(path.read_text()) if path.exists() else {"done": []}
    state.update({"running": stage, "note": note, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_json(path, state)


def stage_done(base: Path, stage: str) -> None:
    path = base / "stages.json"
    state = json.loads(path.read_text()) if path.exists() else {"done": []}
    if stage not in state["done"]:
        state["done"].append(stage)
    state.update({"running": None, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_json(path, state)


# label --------------------------------------------------------------------------


def label(args: argparse.Namespace, p: dict[str, Path]) -> None:
    from big2.rl.distill import generate

    if (p["label"] / "results.json").exists():
        log("label: results.json exists, skipped")
        return
    stage_progress(p["base"], "label", f"ISMCTS-{args.teacher_iterations} labels, see label/")
    made = generate(
        str(p["data"]),
        workers=args.workers,
        minutes=args.label_minutes,
        iterations=args.teacher_iterations,
        rollout="greedy",
        target_decisions=args.target_decisions,
        progress_dir=str(p["label"]),
    )
    log(f"label: {made:,} decisions this run")
    stage_done(p["base"], "label")


# train --------------------------------------------------------------------------


def gpu_busy_percent() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no GPU tool: just train
        return None


def train(args: argparse.Namespace, p: dict[str, Path]) -> None:
    from big2.rl.distill import DistillV2Config, train_v2
    from big2.rl.export import export_pknn

    net_dir = p["net"]
    if not (net_dir / "results.json").exists():
        # Plan rule: wait while another job keeps the GPU over 20 % busy (at most 2 h).
        waited = 0
        while (busy := gpu_busy_percent()) is not None and busy > 20 and waited < 120:
            stage_progress(p["base"], "train", f"waiting for the GPU ({busy:.0f} % busy)")
            time.sleep(300)
            waited += 5
        latest = net_dir / "latest.pt"
        cfg = DistillV2Config(
            data_dir=str(p["data"]),
            run_dir=str(net_dir),
            device=args.device,
            max_minutes=args.train_minutes,
            resume=str(latest) if latest.exists() else None,
            init=str(p["prev_net"] / "best.pt"),
            **WEIGHTS,
        )
        stage_progress(p["base"], "train", f"network r{args.n}, see {net_dir.name}/progress.json")
        train_v2(cfg, log=lambda r: log(f"train: {r}"))
    if not p["pknn"].exists():
        export_pknn(net_dir / "best.pt", p["pknn"])
        log(f"export: {p['pknn']} ({p['pknn'].stat().st_size:,} bytes)")
    calib = p["base"] / "f_calibration" / "results.json"
    if not calib.exists():
        stage_progress(p["base"], "train", "calibration on the held-out shards")
        subprocess.run(
            [
                PY,
                str(ROOT / "experiments" / "build_phase2.py"),
                "calibration",
                "--data-dir",
                str(p["data"]),
                "--checkpoint",
                str(net_dir / "best.pt"),
                "--out",
                p["base"].name,
                "--device",
                args.device,
                "--workers",
                str(args.workers),
            ],
            check=True,
        )
    s = json.loads(calib.read_text())["score_vs_search"]
    log(
        f"calibration: slope {s['slope_search_on_pred']:.2f}, bias {s['bias']:+.2f}, "
        f"RMS {s['rms']:.2f}, correlation {s['corr']:.3f}"
    )
    stage_done(p["base"], "train")


# match --------------------------------------------------------------------------


def puct(pknn: Path, leaf: float) -> str:
    rel = pknn.relative_to(ROOT).as_posix()
    return f"puct:net={rel},simulations=2000,c_puct={C_PUCT},leaf_network_weight={leaf}"


def match(args: argparse.Namespace, p: dict[str, Path]) -> None:
    out = p["base"]
    if (out / "results.json").exists():
        log("match: results.json exists, skipped")
        return
    # play_cell reads out/block/seed/workers from an args-like object.
    margs = argparse.Namespace(
        out=out.name, block=100, seed=0, workers=args.workers, deals=args.deals
    )
    cells = [f"screen lambda={w}" for w in LEAF_WEIGHTS] + ["a9", "a12"]
    progress = Progress(out, [f"{c} {lu}" for c in cells for lu in ("1+3", "ABAB")])
    screens: dict[str, Any] = {}
    for w in LEAF_WEIGHTS:
        cand = puct(p["pknn"], w)
        screens[str(w)] = {
            "candidate": cand,
            "reports": play_cell(
                margs,
                progress,
                f"screen lambda={w}",
                cand,
                ISMCTS_1000,
                args.screen_deals,
                folder=f"vs_ismcts/lambda{w}",
            ),
        }
    best = max(screens, key=lambda w: sum(r["paired_diff"] for r in screens[w]["reports"].values()))
    cand = screens[best]["candidate"]
    # The same folder: the screen's first blocks are reused (same deal seeds).
    a9_reports = play_cell(
        margs, progress, "a9", cand, ISMCTS_1000, args.deals, folder=f"vs_ismcts/lambda{best}"
    )
    a9 = {
        lu: r["paired_diff"] >= A9_BAR and r["paired_ci95"][0] > 0 for lu, r in a9_reports.items()
    }
    a12_reports: dict[str, Any] | None = None
    a12: dict[str, bool] | None = None
    prev = puct(p["prev_pknn"], float(best))
    if all(a9.values()):
        a12_reports = play_cell(
            margs, progress, "a12", cand, prev, args.deals, folder=f"vs_r{args.n - 1}"
        )
        a12 = {lu: r["paired_ci95"][0] > 0 for lu, r in a12_reports.items()}
    verdict = (
        "beats round 0 and ISMCTS-1000: ships"
        if a12 and all(a12.values())
        else "stopped: does not beat ISMCTS-1000 by the A9 bar (ISMCTS ships as expert)"
        if not all(a9.values())
        else f"stopped: does not beat round {args.n - 1} (A12)"
    )
    calib = json.loads((out / "f_calibration" / "results.json").read_text())["score_vs_search"]
    lines = [
        f"Round {args.n}: labels from ISMCTS-{args.teacher_iterations} (greedy rollout); "
        f"network from round {args.n - 1}'s weights, loss weights {WEIGHTS}. Calibration on "
        f"held-out round-{args.n} shards: slope of the search's score on the prediction "
        f"{calib['slope_search_on_pred']:.2f} (round 0: 0.86), bias {calib['bias']:+.2f}, "
        f"RMS {calib['rms']:.2f} points.",
        "",
        *TABLE_HEAD,
        *(
            _row(f"r{args.n} PUCT-2000 λ = {w} vs ISMCTS-1000", lu, r)
            for w, s in screens.items()
            for lu, r in s["reports"].items()
        ),
        *(_row(f"r{args.n} λ = {best} vs ISMCTS-1000 (A9)", lu, r) for lu, r in a9_reports.items()),
        *(
            _row(f"r{args.n} vs r{args.n - 1}, λ = {best} (A12)", lu, r)
            for lu, r in (a12_reports or {}).items()
        ),
        "",
        f"A9 (≥ +{A9_BAR:.1f}, interval above 0): "
        + ", ".join(f"{lu} {'passes' if ok else 'fails'}" for lu, ok in a9.items())
        + (
            "; A12 (beats the previous round): "
            + ", ".join(f"{lu} {'passes' if ok else 'fails'}" for lu, ok in a12.items())
            if a12
            else "; A12 not run (A9 failed)"
        )
        + f". Verdict: {verdict}.",
    ]
    table = "\n".join(lines) + "\n"
    (out / "table.md").write_text(table, encoding="utf-8")
    _write_json(
        out / "results.json",
        {
            "round": args.n,
            "teacher": f"ismcts:iterations={args.teacher_iterations}",
            "weights": WEIGHTS,
            "calibration": calib,
            "screens": screens,
            "best_lambda": float(best),
            "a9_reports": a9_reports,
            "a9": a9,
            "a12_reports": a12_reports,
            "a12": a12,
            "verdict": verdict,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(table, flush=True)
    progress.update(None)
    stage_done(out, "match")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("command", choices=["round"])
    ap.add_argument("n", type=int, help="round number (1 = first round after round 0)")
    ap.add_argument("--stages", nargs="+", default=["label", "train", "match"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--teacher-iterations", type=int, default=TEACHER_ITERATIONS)
    ap.add_argument("--target-decisions", type=int, default=800_000)
    ap.add_argument("--label-minutes", type=float, default=720.0)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--screen-deals", type=int, default=300)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--tag", default="", help="suffix for the run folders (e.g. a test run)")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("round 0 is Tasks 11-12")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    p = paths(args.n, args.tag)
    p["base"].mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        {"label": label, "train": train, "match": match}[stage](args, p)


if __name__ == "__main__":
    main()
