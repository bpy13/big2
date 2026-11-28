"""Loss distributions per band (plan Task 42): ``big2.teach.grading.losses``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from big2.teach.grading import losses as L


def rec(band: str, loss: float, *, forced: bool = False, flat: bool = False, **kw: object) -> dict:
    return {"band": band, "rung_elo": 1000.0, "deal_id": 1, "forced": forced, "flat": flat,
            "loss": loss, **kw}


def test_words_and_percentiles() -> None:
    assert [L.word(x) for x in (0.0, 0.5, 2.0, 5.0, 9.0)] == [
        "great", "good", "inaccuracy", "mistake", "blunder"
    ]
    xs = [float(i) for i in range(101)]
    assert L.percentile(xs, 50) == 50 and L.percentile(xs, 99) == 99
    assert L.percentile([], 50) is None
    assert L.histogram([0.0, 0.5, 3.0, 100.0], [0, 1, 5]) == [2, 1, 1]


def test_band_table(tmp_path: Path) -> None:
    rows = [rec("a", 0.0, forced=True), rec("a", 0.1, flat=True)]
    rows += [rec("a", float(x)) for x in (0, 1, 2, 3, 8)]
    rows += [rec("a", 3.0, repeat={"loss": 4.0}, rules={"loss": 7.0})]  # same word, then not
    rows += [rec("b", 0.5)]
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    t = L.band_table(L.read_decisions([p]))
    a = t["a"]
    assert a["decisions"] == 8 and a["forced_share"] == pytest.approx(1 / 8)
    assert a["flat_share"] == pytest.approx(1 / 7) and a["graded"] == 6
    assert a["mean"] == pytest.approx(17 / 6)
    assert a["share_above"]["2.75"] == pytest.approx(3 / 6)
    assert a["share_above"]["6.86"] == pytest.approx(1 / 6)
    assert a["repeat_same_word"] == 1.0 and a["rules_same_word"] == 0.0
    assert t["b"]["repeat_same_word"] is None
    assert L.overall_share(t, "repeat_same_word") == 1.0
