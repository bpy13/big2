import numpy as np
import pytest

from big2_rl.rules import SCORE_ORIGINAL, SCORE_TIERED
from big2_rl.training import (
    checkpoint_bytes,
    evaluate_self_play,
    load_checkpoint,
    load_checkpoint_data,
    new_training_run,
    save_checkpoint,
    train_batch,
)


def _assert_same_solver_state(first, second):
    first_sets = first.solver.export_information_sets()
    second_sets = second.solver.export_information_sets()
    assert first_sets.keys() == second_sets.keys()
    for key in first_sets:
        np.testing.assert_allclose(
            first_sets[key]["regrets"], second_sets[key]["regrets"]
        )
        np.testing.assert_allclose(
            first_sets[key]["average_strategy"],
            second_sets[key]["average_strategy"],
        )


def test_checkpoint_round_trip_and_resume_are_deterministic(tmp_path):
    run = new_training_run(SCORE_TIERED, seed=83)
    train_batch(run, 1)
    checkpoint = save_checkpoint(run, tmp_path / "training.json")
    resumed = load_checkpoint(checkpoint, SCORE_TIERED)
    downloaded = load_checkpoint_data(checkpoint_bytes(run), SCORE_TIERED)

    assert resumed.iterations == run.iterations
    assert resumed.scoring_mode == SCORE_TIERED
    _assert_same_solver_state(run, resumed)
    _assert_same_solver_state(run, downloaded)

    train_batch(run, 1)
    train_batch(resumed, 1)
    _assert_same_solver_state(run, resumed)


def test_checkpoint_rejects_a_different_scoring_rule(tmp_path):
    run = new_training_run(SCORE_ORIGINAL, seed=5)
    checkpoint = save_checkpoint(run, tmp_path / "original.json")
    with pytest.raises(ValueError, match="scoring mode"):
        load_checkpoint(checkpoint, SCORE_TIERED)


def test_self_play_reports_normalized_win_rates_and_zero_sum_returns():
    run = new_training_run(seed=19)
    train_batch(run, 1)
    metrics = evaluate_self_play(run, episodes=2, seed=29)

    assert metrics["episodes"] == 2
    assert sum(metrics["win_rates"]) == pytest.approx(1.0)
    assert sum(metrics["mean_returns"]) == pytest.approx(0.0)
