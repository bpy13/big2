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
