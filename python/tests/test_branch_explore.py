"""Tests for experiments/branch_explore.py (plan Task 50): selection, policy, keys, one seat."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "branch_explore", ROOT / "experiments/branch_explore.py"
)
be = importlib.util.module_from_spec(spec)
sys.modules["branch_explore"] = be
spec.loader.exec_module(be)  # type: ignore[union-attr]

LABELS = ROOT / "runs/p6_mistakes"
needs_data = pytest.mark.skipif(
    not (LABELS / "decisions-expert.jsonl").exists()
    or not (ROOT / be.MM.NET).exists()
    or not (ROOT / be.MM.BELIEF_NET).exists(),
    reason="Task 42 labels or network files missing",
)


def _dec(deal: int, seat: int, second: float, forced: bool = False, flat: bool = False) -> dict:
    return {
        "deal_id": deal,
        "seat": seat,
        "forced": forced,
        "flat": flat,
        "loss": 1.0,
        "moves": [{"loss": 0.0}, {"loss": second}],
    }


def test_select_seats_sharp_rule(tmp_path: Path) -> None:
    rows = []
    # deal 1: seats 0, 1, 2 have 3, 4, 5 sharp decisions; only the two sharpest are kept
    for seat, n in ((0, 3), (1, 4), (2, 5)):
        rows += [_dec(1, seat, 3.0) for _ in range(n)]
    # deal 2: seat 0 has 2 sharp plus a flat, a forced and a not-sharp one: does not pass
    rows += [
        _dec(2, 0, 3.0),
        _dec(2, 0, 3.0),
        _dec(2, 0, 9.0, flat=True),
        _dec(2, 0, 9.0, forced=True),
        _dec(2, 0, 2.7),
    ]
    (tmp_path / "decisions-b.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    seats = be.select_seats(tmp_path, ["b"])
    assert [(s["deal_id"], s["seat"], s["sharp"]) for s in seats] == [(1, 1, 4), (1, 2, 5)]


def test_pilot_pick_round_robin() -> None:
    seats = [{"band": b, "i": i} for b in ("x", "y") for i in range(5)] + [{"band": "z", "i": 0}]
    got = be.pilot_pick(seats, 5, ["x", "y", "z"])
    assert [(s["band"], s["i"]) for s in got] == [("x", 0), ("y", 0), ("z", 0), ("x", 1), ("y", 1)]


def test_stand_in_policy() -> None:
    pri = {1: 0.6, 2: 0.3, 3: 0.1}
    net = "netprior:net=n.pknn,temperature="
    # search rungs and the temperature-0 rung: the prior at tau = 1
    assert be.stand_in("expert", "ismcts:iterations=10000", pri) == pytest.approx(pri)
    assert be.stand_in("net", net + "0", pri) == pytest.approx(pri)
    assert be.stand_in("net-t1", net + "1", pri) == pytest.approx(pri)
    hot = be.stand_in("net-t4.34", net + "4.34", pri)
    assert sum(hot.values()) == pytest.approx(1.0)
    assert hot[1] > hot[2] > hot[3] and hot[1] < 0.6 and hot[3] > 0.1


def test_history_hash_matches_hk_games_convention() -> None:
    # hk-games server/hkgames/deal_puzzles.history_hash of the same list (checked 2026-11-28)
    acts = [be.app_action("3d 4d 8d 9d Kd"), be.app_action("pass")]
    assert acts[0] == {
        "type": "action.play",
        "cards": ["3d", "4d", "8d", "9d", "Kd"],
        "combo": "combo.flush",
    }
    assert be.history_hash(acts) == (
        "8a543b5ed13e38af3c17bc1e12ff73b7fe3ad8397eadb9d964abe001205a1ede"
    )
    # the key sorts card lists, so the order cards are written in does not matter
    shuffled = [{**acts[0], "cards": ["Kd", "3d", "9d", "4d", "8d"]}, acts[1]]
    assert be.history_hash(shuffled) == be.history_hash(acts)


@needs_data
def test_explore_one_seat() -> None:
    seats = be.pilot_pick(be.select_seats(LABELS, ["net-t1"]), 1, ["net-t1"])
    (job,) = be.jobs_for(LABELS, seats, cap=0, target=0.9)
    t = be.explore_seat(job)
    assert be.validate_tree(t) == []
    assert t["new_grades"] == 0  # cap 0: the main line only, graded by Task 42
    assert all(n["main_line"] for n in t["nodes"].values())
    assert t["coverage"] == pytest.approx(t["coverage_main"])
    main = [n for n in t["nodes"].values() if n["trivial"] != "forced"]
    assert len(main) == t["graded_main"]
