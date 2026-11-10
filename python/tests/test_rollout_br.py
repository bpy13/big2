"""Rollout best response (plan Task 32): runs on a tiny deal and gives legal, repeatable play."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import rollout_br  # noqa: E402
from big2.games.big2 import Rules  # noqa: E402


def test_br_picks_a_legal_move_and_is_repeatable() -> None:
    state = Rules().new_deal(5)
    a = rollout_br.RolloutBR(2, 5, seed=1).act(state)
    b = rollout_br.RolloutBR(2, 5, seed=1).act(state)
    assert a == b
    assert a in [int(x) for x in state.legal_actions()]


def test_play_deal_tiny(monkeypatch) -> None:
    monkeypatch.setattr(rollout_br, "EXPERT", rollout_br.expert_spec(5))
    row = rollout_br.play_deal(7, "ABAB", rollouts=1, rollout_iterations=3)
    assert row["lineup"] == "ABAB" and len(row["per_rotation"]) == 4
    assert all(len(r) == 2 for r in row["per_rotation"])
