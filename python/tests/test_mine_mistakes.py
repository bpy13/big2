"""Tests for experiments/mine_mistakes.py (plan Task 42a)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mine_mistakes", ROOT / "experiments/mine_mistakes.py")
mm = importlib.util.module_from_spec(spec)
sys.modules["mine_mistakes"] = mm
spec.loader.exec_module(mm)  # type: ignore[union-attr]

needs_net = pytest.mark.skipif(not (ROOT / mm.NET).exists(), reason="network file missing")


def test_word_lines() -> None:
    assert [mm.word(x) for x in (0.0, 0.2, 0.5, 2.0, 5.0, 7.0)] == [
        "great", "great", "good", "inaccuracy", "mistake", "blunder"
    ]


def decision(**over: object) -> dict:
    hands = [["3d"], ["4d"], ["5d"], ["6d"]]
    d = {
        "band": "b", "deal": 1, "turn": 0, "seat": 0, "played": "pass", "legal": 2,
        "forced": False, "flat": False, "best": "3d", "played_loss": 3.5,
        "played_loss_se": 0.4, "knee_best": "3d", "knee_loss": 0.0,
        "position": {"hands": hands, "current_player": 0, "current_play": None},
        "ref": [
            {"cards": "3d", "prior": 0.7, "mean": 1.0, "se": 0.1, "loss": 0.0, "loss_se": 0.0},
            {"cards": "pass", "prior": 0.3, "mean": -2.5, "se": 0.1, "loss": 3.5, "loss_se": 0.4},
        ],
    }
    d.update(over)
    return d


def test_validate_decision() -> None:
    assert mm.validate_decision(decision()) == []
    assert mm.validate_decision({**decision(), "forced": True}) == []
    assert mm.validate_decision(decision(played="9s"))  # played move missing from the reference
    assert mm.validate_decision(decision(best="pass"))  # best is not the least loss
    bad = decision()
    del bad["knee_loss"]
    assert mm.validate_decision(bad)


@needs_net
def test_label_and_play_deal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mm, "REF", {"k": 2, "n": 4, "depth": 2})
    monkeypatch.setattr(mm, "KNEE", {"k": 2, "n": 2, "depth": 1})
    recs = mm.play_deal("net", f"netprior:net={mm.NET},temperature=1", 5)
    assert recs and all(mm.validate_decision(r) == [] for r in recs)
    assert any(r["forced"] for r in recs) or any(not r["forced"] for r in recs)
    # the move played is always labelled, and a played pass is the 0-card action
    for r in recs:
        if not r["forced"]:
            assert r["played"] in {m["cards"] for m in r["ref"]}


def test_candidate_needs_clear_best() -> None:
    from big2.teach import drills

    elo = 1100.0
    clear = decision(position={
        "hands": [["3d", "4d"], ["5d"], ["6d"], ["7d"]], "current_player": 0,
        "current_play": None})
    clear["ref"][1]["loss"] = 3.5
    d = mm.candidate(clear, elo)
    assert d is not None and drills.validate_drill(d) == []
    assert d["bot_move"] == "pass" and d["band"] == "b" and d["rung_elo"] == 1100.0
    assert d["answers"][0]["grade"] == "best" and d["answers"][1]["grade"] == "mistake"
    assert mm.candidate(decision(played_loss=1.0), elo) is None  # below the line
    assert mm.candidate(decision(flat=True), elo) is None
    noisy = decision()
    noisy["ref"][1]["loss_se"] = 3.0  # lead's range covers 0
    assert mm.candidate(noisy, elo) is None
