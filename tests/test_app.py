from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_training_first_app_runs_and_resumes_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("BIG2_MODEL_DIR", str(tmp_path))
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Train", "Play", "Rules"]

    app.button(key="train_big2").click().run()
    assert not app.exception
    trained_iterations = next(
        int(metric.value) for metric in app.metric if metric.label == "MCCFR iterations"
    )
    assert trained_iterations > 0

    app.button(key="save_big2").click().run()
    checkpoint_path = tmp_path / "custom_tiers.json"
    assert checkpoint_path.exists()

    app.button(key="train_big2").click().run()
    assert not app.exception
    assert any(
        metric.label == "MCCFR iterations" and int(metric.value) > trained_iterations
        for metric in app.metric
    )

    app.button(key="load_big2").click().run()
    assert not app.exception
    assert any(
        metric.label == "MCCFR iterations" and int(metric.value) == trained_iterations
        for metric in app.metric
    )
