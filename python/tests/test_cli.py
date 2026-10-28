"""Smoke tests for the CLI."""

from big2.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_analyze_opening_position() -> None:
    r = runner.invoke(
        app,
        [
            "analyze",
            "--seat",
            "0",
            "--hand",
            "3d 4d 4h 5h 6d 7d 7s 8d 8s Jd Jh Kc 2s",
            "--iterations",
            "500",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Win rate" in r.output


def test_analyze_rejects_bad_history() -> None:
    r = runner.invoke(app, ["analyze", "--seat", "0", "--hand", "3d", "--history", "garbage"])
    assert r.exit_code != 0


def test_tournament_runs() -> None:
    r = runner.invoke(app, ["tournament", "greedy", "random", "random", "random", "--deals", "3"])
    assert r.exit_code == 0, r.output
    assert "greedy" in r.output


def test_play_can_quit() -> None:
    r = runner.invoke(app, ["play", "--bots", "greedy", "--seed", "1"], input="quit\n")
    assert r.exit_code == 0, r.output
