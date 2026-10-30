"""Convention-gap check (plan Task 30) on a synthetic log."""

from __future__ import annotations

import json
import random
from pathlib import Path

from big2.cli.main import app
from big2.teach.convention_gap import GapConfig, analyse, disagrees, key_set, load_log
from typer.testing import CliRunner

PLAY = {"type": "action.play", "cards": ["3d"]}
OTHER = {"type": "action.play", "cards": ["4d"]}


def synthetic_log(seed: int = 0) -> list[dict]:
    """Two key sets over ratings 1000-2000.

    * pass_to_keep: players disagree 50 % of the time at every rating (a table habit);
    * plan.breaks: 70 % at 1000, falling to 10 % at 2000 (a skill players learn).
    """
    rng = random.Random(seed)
    items = []
    for i in range(2000):
        rating = 1000 + (i % 50) * 20
        if i % 2 == 0:
            keys, p = ["reason.control.pass_to_keep"], 0.5
        else:
            keys, p = ["reason.plan.breaks", "reason.value.gap"], 0.7 - 0.6 * (rating - 1000) / 1000
        wrong = rng.random() < p
        items.append(
            {
                "position": {"id": i},
                "played": OTHER if wrong else PLAY,
                "best": PLAY,
                "loss": 1.5 if wrong else 0.0,
                "rating": rating,
                "reasons": [{"key": k, "args": {}} for k in keys],
            }
        )
    return items


def test_rating_independent_and_dependent_sets_are_split() -> None:
    reports = analyse(synthetic_log())
    by_keys = {r.keys: r for r in reports}
    conv = by_keys[("reason.control.pass_to_keep",)]
    skill = by_keys[("reason.plan.breaks", "reason.value.gap")]
    assert conv.tag == "convention", conv
    assert skill.tag == "skill", skill
    assert abs(conv.slope_per_100) < 0.01
    assert skill.slope_per_100 < -0.04
    assert reports[0] is conv  # conventions come first


def test_small_and_rare_sets() -> None:
    log = synthetic_log()[:20]
    assert {r.tag for r in analyse(log)} == {"too_few"}
    # A flat set that players almost never get wrong is "rare", not a convention.
    flat = [dict(it, played=PLAY, loss=0.0) for it in synthetic_log() if it["rating"] < 1600]
    for it in flat[:15]:
        it["played"] = OTHER
    tags = {r.tag for r in analyse(flat, GapConfig(min_items=10))}
    assert "convention" not in tags


def test_helpers() -> None:
    assert key_set({"reasons": []}) == ("(none)",)
    assert key_set({"reasons": [{"key": "b"}, {"key": "a"}, {"key": "a"}]}) == ("a", "b")
    assert disagrees({"played": OTHER, "best": PLAY})
    assert not disagrees({"played": PLAY, "best": PLAY})
    assert disagrees({"played": PLAY, "loss": 0.4})


def test_cli_prints_conventions(tmp_path: Path) -> None:
    path = tmp_path / "log.json"
    path.write_text(json.dumps({"items": synthetic_log()}), encoding="utf-8")
    assert len(load_log(path)) == 2000
    result = CliRunner().invoke(app, ["teach", "convention-gap", str(path)])
    assert result.exit_code == 0, result.output
    first = result.output.splitlines()[1]
    assert first.startswith("convention") and "pass_to_keep" in first
