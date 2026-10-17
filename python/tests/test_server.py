"""HTTP API tests (no browser)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import poker.server.app as server
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "GAME_LOG_DIR", tmp_path)
    return TestClient(server.app)


def new_game(client: TestClient, **kw: Any) -> dict[str, Any]:
    body = {"bots": "greedy", "seed": 7, **kw}
    r = client.post("/api/games", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def run_bots(client: TestClient, view: dict[str, Any]) -> dict[str, Any]:
    while not view["over"] and not view["your_turn"]:
        r = client.post(f"/api/games/{view['id']}/step")
        assert r.status_code == 200
        view = r.json()["view"]
    return view


def test_config_lists_difficulties_and_rules(client: TestClient) -> None:
    c = client.get("/api/config").json()
    assert "hard" in c["difficulties"]
    assert "no_twos" in c["rules"]["straights"]


def test_view_hides_other_hands_until_the_end(client: TestClient) -> None:
    v = new_game(client)
    assert len(v["hand"]) == 13
    assert v["final_hands"] is None and v["initial_hands"] is None
    assert "hands" not in v


def test_full_game_with_greedy_human(client: TestClient, tmp_path: Path) -> None:
    v = new_game(client, seat=2)
    gid = v["id"]
    while not v["over"]:
        v = run_bots(client, v)
        if v["over"]:
            break
        # Play the first legal move the server offers.
        r = client.post(f"/api/games/{gid}/play", json={"cards": v["legal"][0]})
        assert r.status_code == 200, r.text
        v = r.json()
    assert v["winner"] is not None
    assert sum(v["scores"]) == 0
    assert len(v["final_hands"]) == 4
    assert list(tmp_path.glob("*.json")), "game log saved"
    log = client.get(f"/api/games/{gid}/log").json()
    assert len(log["actions"]) == len(v["history"])
    review = client.get(f"/api/games/{gid}/review").json()
    assert review, "at least one non-forced decision"
    for item in review:
        assert item["score_loss"] >= 0  # the best move has the highest mean
        assert item["agrees"] == (
            item["score_loss"] == 0 and item["best"]["cards"] == item["played"]
        )


def test_illegal_and_out_of_turn_moves_are_rejected(client: TestClient) -> None:
    v = run_bots(client, new_game(client))
    gid = v["id"]
    # A card the human does not hold
    missing = next(c for c in ["3d", "3c", "3h", "3s", "4d", "4c"] if c not in v["hand"])
    r = client.post(f"/api/games/{gid}/play", json={"cards": [missing]})
    assert r.status_code == 400 and "not in hand" in r.json()["detail"]
    r = client.post(f"/api/games/{gid}/play", json={"cards": ["xx"]})
    assert r.status_code == 400


def test_analyze_only_on_your_turn(client: TestClient) -> None:
    v = run_bots(client, new_game(client))
    r = client.post(f"/api/games/{v['id']}/analyze", json={"iterations": 500})
    assert r.status_code == 200
    a = r.json()
    assert 1 <= len(a["moves"]) <= 3
    legal = {tuple(m) for m in v["legal"]}
    assert all(tuple(m["cards"]) in legal for m in a["moves"])
    # Review before the end is refused
    assert client.get(f"/api/games/{v['id']}/review").status_code == 409


def test_bad_requests(client: TestClient) -> None:
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games", json={"difficulty": "godlike"}).status_code == 422
    assert client.post("/api/games", json={"bots": "alphago"}).status_code == 422
    assert client.post("/api/games", json={"rules": {"scoring": "bogus"}}).status_code == 422


MODELS = Path(__file__).resolve().parents[2] / "models"


def test_rl_opponents_and_network_hints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from poker.server import models

    if not (MODELS / "big2-distilled.pt").exists():
        pytest.skip("no trained model in models/")
    monkeypatch.setattr(models, "MODELS_DIR", MODELS)
    monkeypatch.delenv("POKER_RL_CHECKPOINT", raising=False)
    c = client.get("/api/config").json()
    assert {"rl", "rl_search"} <= set(c["difficulties"])
    assert "distilled-v1" in {m["id"] for m in c["models"]}
    v = run_bots(client, new_game(client, difficulty="rl", bots=None, model="distilled-v1"))
    assert v["model"] == "distilled-v1"
    a = client.post(f"/api/games/{v['id']}/analyze", json={"iterations": 300}).json()
    net = a["network"]
    assert net["model"]["id"] == "distilled-v1" and net["model"]["kind"] == "q"
    assert 1 <= len(net["moves"]) <= 3
    legal = {tuple(m) for m in v["legal"]}
    assert all(tuple(m["cards"]) in legal for m in net["moves"])
    # The solver rates every legal move, so the network's pick always has a rank.
    pick = net["pick"]
    assert pick["cards"] == net["moves"][0]["cards"]
    assert pick["solver_rank"] >= 1 and pick["solver"]["cards"] == pick["cards"]
    assert pick["loss"] >= 0 and (pick["solver_rank"] > 1 or pick["loss"] == 0)
    # The network was trained on the default rules only.
    r = client.post("/api/games", json={"difficulty": "rl", "rules": {"scoring": "original"}})
    assert r.status_code == 422


def test_switching_the_hint_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    import shutil

    from poker.server import models

    if not (MODELS / "big2-distilled.pt").exists():
        pytest.skip("no trained model in models/")
    # Two copies of one network under two catalog names, neither loaded yet.
    for name in ["big2-distilled-10h.pt", "big2-ppo-10h.pt"]:
        shutil.copy(MODELS / "big2-distilled.pt", tmp_path / name)
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("POKER_RL_CHECKPOINT", raising=False)
    monkeypatch.setattr("poker.rl.agent._CACHE", {})
    listed = client.get("/api/config").json()["models"]
    assert [m["id"] for m in listed] == ["distilled-10h", "ppo-10h"]
    assert not any(m["loaded"] for m in listed)

    v = run_bots(client, new_game(client))  # greedy bots: nothing loaded
    assert v["model"] == "distilled-10h"  # the default
    assert not any(m["loaded"] for m in client.get("/api/config").json()["models"])
    r = client.post(f"/api/games/{v['id']}/model", json={"model": "ppo-10h"})
    assert r.status_code == 200 and r.json()["loaded"]
    assert client.get(f"/api/games/{v['id']}").json()["model"] == "ppo-10h"
    loaded = {m["id"]: m["loaded"] for m in client.get("/api/config").json()["models"]}
    assert loaded == {"distilled-10h": False, "ppo-10h": True}
    a = client.post(f"/api/games/{v['id']}/analyze", json={"iterations": 200}).json()
    assert a["network"]["model"]["id"] == "ppo-10h"
    bad = client.post(f"/api/games/{v['id']}/model", json={"model": "nope"})
    assert bad.status_code == 422
    assert client.post("/api/games", json={"model": "nope"}).status_code == 422


def test_without_a_model_rl_is_not_offered(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from poker.server import models

    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("POKER_RL_CHECKPOINT", raising=False)
    c = client.get("/api/config").json()
    assert "rl" not in c["difficulties"] and c["models"] == []
    v = run_bots(client, new_game(client))
    assert v["model"] is None
    assert (
        client.post(f"/api/games/{v['id']}/analyze", json={"iterations": 200}).json()["network"]
        is None
    )
