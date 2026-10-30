"""Drill miner and mistake clustering (plan Task 28)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
from big2.teach.cluster import clusters, format_clusters
from big2.teach.drills import (
    EXE,
    REQUIRED,
    ROOT,
    build_drills,
    rating_for,
    read_raw,
    skill_for,
    validate_drill,
    write_drills,
)


def position(hands: list[list[str]], seat: int, play: dict | None) -> dict:
    return {
        "game": "big2",
        "seed": 1,
        "start": {"hands": hands, "current_player": seat, "current_play": play},
        "history": [],
    }


KD = {"cards": ["Kd"], "combo": "combo.single"}
TWO = {"cards": ["2c"], "combo": "combo.single"}
HANDS = [["4d", "9c", "Kd", "2c"], ["7s", "8h"], ["3c", "4h", "6d"], ["4c", "8d", "9d"]]


def candidate(**over: object) -> dict:
    c = {
        "type": "candidate",
        "deal": 3,
        "turn": 7,
        "seat": 0,
        "position": position(HANDS, 0, {"player": 2, "cards": ["Ah"], "combo": "combo.single"}),
        "following": True,
        "best_keys": ["reason.control.safe_single", "reason.value.gap"],
        "answers": [
            {
                "action": {"cards": ["2c"], "combo": "combo.single"},
                "grade": "best",
                "loss": 0.0,
                "reasons": [{"key": "reason.control.safe_single", "args": {"card": "2c"}}],
            },
            {"action": "pass", "grade": "mistake", "loss": 4.0, "reasons": []},
        ],
        "gap": 4.0,
        "gap_ci": 1.0,
        "samples": 600,
        "wrong": {"random": 0.5, "easy": 1.0, "medium": 0.5, "hard": 0.0, "expert": 0.0},
    }
    c.update(over)
    return c


def mistake(keys: list[str], loss: float = 2.0, level: str = "easy") -> dict:
    return {"type": "mistake", "best_keys": keys, "loss": loss, "grade": "mistake", "level": level}


def test_drills_have_the_engine_interface_shape() -> None:
    drills = build_drills([candidate(), mistake(["reason.plan.keeps"])])
    assert len(drills) == 1
    d = drills[0]
    assert all(k in d for k in REQUIRED)
    assert validate_drill(d) == []
    assert d["id"] == "big2-s00003-07"
    assert d["source"] == "solver" and d["solver"] == {"samples": 600, "gap": 4.0}
    assert d["answers"][1]["action"] == {"type": "action.pass"}
    assert d["answers"][0]["action"]["type"] == "action.play"
    assert d["prompt"] == "drill.play_or_pass"


def test_rating_follows_the_seed_formula() -> None:
    c = candidate()
    share = (0.5 + 1.0 + 0.5) / 5
    assert rating_for(c) == round(1000 + 400 * share - 100 * math.log2(4.0))
    # A smaller gap is harder: higher rating.
    assert rating_for(candidate(gap=2.0)) > rating_for(c)


def test_skill_mapping() -> None:
    assert skill_for(candidate()) == "skill.control"
    pass_best = candidate(
        answers=[
            {
                "action": "pass",
                "grade": "best",
                "reasons": [{"key": "reason.control.pass_to_keep", "args": {"card": "Kd"}}],
            },
            {
                "action": {"cards": ["Kd"], "combo": "combo.single"},
                "grade": "mistake",
                "reasons": [],
            },
        ]
    )
    assert skill_for(pass_best) == "skill.when_to_pass"
    finish = candidate(
        answers=[
            {
                "action": {"cards": ["9c"], "combo": "combo.single"},
                "grade": "best",
                "reasons": [{"key": "reason.tempo.plays_to_out", "args": {"mine": 2}}],
            }
        ]
    )
    assert skill_for(finish) == "skill.finish_in_two"
    twos = candidate(
        answers=[
            {"action": KD, "grade": "best", "reasons": []},
            {
                "action": {"cards": ["2c"], "combo": "combo.single"},
                "grade": "mistake",
                "reasons": [],
            },
        ]
    )
    assert skill_for(twos) == "skill.saving_twos"


def test_validate_catches_bad_drills() -> None:
    d = build_drills([candidate()])[0]
    assert validate_drill({k: v for k, v in d.items() if k != "skill"}) == ["missing skill"]
    bad = dict(d, seat=1)
    assert validate_drill(bad)
    bad = dict(d, answers=[dict(d["answers"][1])])
    assert "the first answer must be the best move" in validate_drill(bad)


def test_clusters_group_by_best_keys() -> None:
    raw = [mistake(["reason.plan.keeps", "reason.value.gap"]) for _ in range(6)]
    raw += [mistake(["reason.helps_seat"], loss=5.0, level="medium") for _ in range(5)]
    raw += [mistake(["reason.control.safe_single"]) for _ in range(2)]  # under min_size
    cs = clusters(raw, min_size=5)
    assert [c.keys for c in cs] == [("reason.plan.keeps",), ("reason.helps_seat",)]
    assert cs[0].skill == "skill.keep_the_plan" and cs[1].mean_loss == 5.0
    text = format_clusters(cs, ["skill.when_to_pass", "skill.keep_the_plan"])
    assert "new-skill candidates: skill.kingmaking" in text
    assert "app skills with no cluster: skill.when_to_pass" in text


def test_write_drills_and_audit(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    lines = [candidate(deal=i, gap=2.0 + i) for i in range(60)] + [mistake(["reason.plan.keeps"])]
    raw.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    summary = write_drills(raw, tmp_path / "big2.json", tmp_path / "audit.json")
    assert summary["drills"] == 60
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert len(audit) == 50 and all(a["tag_rating"] is None for a in audit)
    assert len({a["id"] for a in audit}) == 50


@pytest.mark.skipif(
    not EXE.exists(), reason="big2-teach is not built (cargo build --release -p big2-teach)"
)
def test_native_miner_output_validates(tmp_path: Path) -> None:
    """A tiny run of the Rust miner: every candidate becomes a valid drill."""
    run = tmp_path / "mine"
    cmd = [str(EXE), "mine", "--out", str(run), "--games", "4", "--threads", "2", "--deals", "60"]
    cmd += ["--hard-iterations", "60", "--expert-iterations", "200", "--samples", "4"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    raw = read_raw(run / "raw.jsonl")
    assert any(r["type"] == "mistake" for r in raw)
    for d in build_drills(raw):
        assert validate_drill(d) == [], d["id"]
