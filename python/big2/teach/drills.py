"""Drill miner (plan Task 28): solver positions where the bot levels disagree, as drills in
the app's shape (hk-games ``docs/engine-interface.md`` §6).

The heavy part runs in Rust: ``big2-teach mine`` (``crates/teach/src/main.rs``) plays
self-play deals with the ladder levels and writes ``raw.jsonl``. Each ``candidate`` line is
a position whose best move is clear on a paired move table (gap to the runner-up >= 2 points,
interval above 0) that ``easy`` and ``medium`` get wrong at least half the time and
``expert`` gets right; each ``mistake`` line is a self-play decision graded worse than
``best``. This module turns candidates into drills:

* ``rating = 1000 + 400 * (share of levels wrong) - 100 * log2(gap)`` (the plan's seed
  rating; the share averages random, easy, medium, hard and expert);
* ``skill`` from the best move's reason keys (``SKILL_RULES``; app skills first);
* ``prompt`` a ``drill.*`` key; ``source: "solver"``; ``solver: {samples, gap}``.

Positions come from ISMCTS self-play (the ladder's ``hard`` level), not PUCT self-play as
the plan said: PUCT has not beaten ISMCTS yet (``docs/open-items.md``).
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXE = ROOT / "target" / "release" / ("big2-teach.exe" if sys.platform == "win32" else "big2-teach")

#: Fields every drill must carry (engine-interface.md §6).
REQUIRED = ("id", "game", "skill", "prompt", "position", "seat", "rating", "answers", "source")
GRADES = {"best", "good", "inaccuracy", "mistake", "blunder", "unclear"}

#: The app's current skills (hk-games app/src/drills/big2.json) and their prompts.
APP_PROMPTS = {
    "skill.when_to_pass": "drill.play_or_pass",
    "skill.finish_in_two": "drill.finish_in_two",
    "skill.saving_twos": "drill.keep_the_two",
    "skill.endgame_counting": "drill.count_the_rest",
}
#: New skills this miner proposes (keys for the app's dictionaries; bot-notes lists them).
NEW_SKILLS = (
    "skill.keep_the_plan",
    "skill.control",
    "skill.score_vs_win",
    "skill.kingmaking",
    "skill.general",
)
#: First matching rule wins: (reason key of the best move, skill).
SKILL_RULES: tuple[tuple[str, str], ...] = (
    ("reason.counterfactual.forced", "skill.endgame_counting"),
    ("reason.control.pass_to_keep", "skill.when_to_pass"),
    ("reason.plan.keeps", "skill.keep_the_plan"),
    ("reason.plan.breaks", "skill.keep_the_plan"),
    ("reason.control.safe_single", "skill.control"),
    ("reason.control.highest_out", "skill.control"),
    ("reason.value.score_vs_win", "skill.score_vs_win"),
    ("reason.helps_seat", "skill.kingmaking"),
)


def _keys(answer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["key"]: r.get("args", {}) for r in answer.get("reasons", [])}


def _has_two(action: Any) -> bool:
    return isinstance(action, dict) and any(c.startswith("2") for c in action.get("cards", []))


def skill_for(candidate: dict[str, Any]) -> str:
    """The drill's skill from its best answer's reason keys (and two special cases)."""
    answers = candidate["answers"]
    best = answers[0]
    keys = _keys(best)
    plays = keys.get("reason.tempo.plays_to_out")
    if (
        plays is not None
        and plays.get("mine", 99) <= 2
        and "reason.counterfactual.forced" not in keys
    ):
        return "skill.finish_in_two"
    hand = candidate["position"]["start"]["hands"][candidate["seat"]]
    if (
        not _has_two(best["action"])
        and any(c.startswith("2") for c in hand)
        and len(answers) > 1
        and _has_two(answers[1]["action"])
    ):
        return "skill.saving_twos"
    for key, skill in SKILL_RULES:
        if key in keys:
            return skill
    return "skill.general"


def prompt_for(candidate: dict[str, Any], skill: str) -> str:
    if skill in APP_PROMPTS and skill != "skill.when_to_pass":
        return APP_PROMPTS[skill]
    if candidate.get("following") and any(a["action"] == "pass" for a in candidate["answers"]):
        return "drill.play_or_pass"
    return "drill.best_move"


def rating_for(candidate: dict[str, Any]) -> int:
    wrong = candidate["wrong"]
    share = sum(wrong.values()) / len(wrong)
    return int(round(1000 + 400 * share - 100 * math.log2(float(candidate["gap"]))))


def _action(a: Any) -> dict[str, Any]:
    """The miner writes a pass as ``"pass"``; the app's Action is an object."""
    if a == "pass":
        return {"type": "action.pass"}
    return {"type": "action.play", **a}


def to_drill(candidate: dict[str, Any]) -> dict[str, Any]:
    skill = skill_for(candidate)
    return {
        "id": f"big2-s{candidate['deal']:05d}-{candidate['turn']:02d}",
        "game": "big2",
        "skill": skill,
        "prompt": prompt_for(candidate, skill),
        "position": candidate["position"],
        "seat": candidate["seat"],
        "rating": rating_for(candidate),
        "answers": [
            {"action": _action(a["action"]), "grade": a["grade"], "reasons": a["reasons"]}
            for a in candidate["answers"]
        ],
        "source": "solver",
        "solver": {"samples": candidate["samples"], "gap": candidate["gap"]},
    }


def read_raw(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_drills(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drills from the miner's candidate lines, by rating."""
    drills = [to_drill(c) for c in raw if c.get("type") == "candidate"]
    drills.sort(key=lambda d: (d["rating"], d["id"]))
    return drills


def validate_drill(d: dict[str, Any]) -> list[str]:
    """Problems with one drill against the engine-interface §6 shape (empty when valid)."""
    errors = [f"missing {k}" for k in REQUIRED if k not in d]
    if errors:
        return errors
    if d["game"] != "big2" or d["source"] not in ("hand", "solver"):
        errors.append("game must be big2 and source hand or solver")
    if not (d["skill"].startswith("skill.") and d["prompt"].startswith("drill.")):
        errors.append("skill and prompt must be keys")
    start = d["position"].get("start", {})
    if len(start.get("hands", [])) != 4 or start.get("current_player") != d["seat"]:
        errors.append("position.start needs four hands and current_player == seat")
    if not isinstance(d["rating"], int):
        errors.append("rating must be an integer")
    if not d["answers"] or d["answers"][0]["grade"] != "best":
        errors.append("the first answer must be the best move")
    for a in d["answers"]:
        if a["grade"] not in GRADES or a["action"].get("type") not in (
            "action.play",
            "action.pass",
        ):
            errors.append(f"bad answer {a}")
        if any(not r["key"].startswith("reason.") for r in a["reasons"]):
            errors.append("reasons must be reason keys")
    return errors


def audit_sample(drills: list[dict[str, Any]], n: int = 50, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` drills spread over the rating range, for Brian's tag audit (B11): each with
    ``tag_rating`` left empty (right / irrelevant / wrong)."""
    if len(drills) <= n:
        picked = list(drills)
    else:
        # Drills are sorted by rating, so a uniform sample of indices spreads over ratings.
        picked = [drills[i] for i in sorted(random.Random(seed).sample(range(len(drills)), n))]
    return [dict(d, tag_rating=None) for d in picked]


def run_miner(
    run_dir: str | Path, games: int, threads: int, extra: list[str] | None = None
) -> Path:
    """Builds and runs ``big2-teach mine`` (resumes from ``run_dir/progress.json``)."""
    subprocess.run(["cargo", "build", "--release", "-p", "big2-teach"], cwd=ROOT, check=True)
    cmd = [
        str(EXE),
        "mine",
        "--out",
        str(run_dir),
        "--games",
        str(games),
        "--threads",
        str(threads),
    ]
    subprocess.run([*cmd, *(extra or [])], cwd=ROOT, check=True)
    return Path(run_dir) / "raw.jsonl"


def write_drills(raw_path: str | Path, out: str | Path, audit: str | Path | None) -> dict[str, Any]:
    """Writes the drill file (and the audit sample); returns a summary."""
    drills = build_drills(read_raw(raw_path))
    bad = {d["id"]: e for d in drills if (e := validate_drill(d))}
    if bad:
        raise ValueError(f"invalid drills: {bad}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(drills, indent=1) + "\n", encoding="utf-8")
    if audit:
        Path(audit).parent.mkdir(parents=True, exist_ok=True)
        Path(audit).write_text(json.dumps(audit_sample(drills), indent=1) + "\n", encoding="utf-8")
    ratings = [d["rating"] for d in drills]
    skills: dict[str, int] = {}
    for d in drills:
        skills[d["skill"]] = skills.get(d["skill"], 0) + 1
    return {
        "drills": len(drills),
        "rating_min": min(ratings, default=0),
        "rating_max": max(ratings, default=0),
        "skills": dict(sorted(skills.items(), key=lambda kv: -kv[1])),
    }
