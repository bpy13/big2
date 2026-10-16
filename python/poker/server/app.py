"""FastAPI app: the HTTP API behind the web UI.

Run with ``poker serve`` (serves the built UI from ``web/dist`` too) or, during
UI development, ``uvicorn poker.server.app:app --reload`` next to ``npm run dev``.

Endpoints (all JSON):

| Method | Path | Purpose |
|---|---|---|
| GET  | /api/config | Difficulty presets and rule variants |
| POST | /api/games | New game → view |
| GET  | /api/games/{id} | Current view |
| POST | /api/games/{id}/play | Human move ``{"cards": ["3d", "3c"]}`` (``[]`` = pass) → view |
| POST | /api/games/{id}/step | One bot move → ``{"move": …, "view": …}`` |
| POST | /api/games/{id}/analyze | Solver's top moves for the human's turn |
| GET  | /api/games/{id}/review | Post-game review of the human's decisions |
| GET  | /api/games/{id}/log | Replayable game log |

Card codes are rank + suit, e.g. ``"Td"``; see docs/rules/big2.md.
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from poker.agents import make_agent
from poker.games.big2 import Rules, cards
from poker.server.session import DIFFICULTIES, GameSession

MAX_SESSIONS = 200
GAME_LOG_DIR = Path(os.environ.get("POKER_GAME_LOG_DIR", "runs/games"))
WEB_DIST = Path(
    os.environ.get("POKER_WEB_DIST", Path(__file__).resolve().parents[3] / "web" / "dist")
)

RULE_OPTIONS: dict[str, list[Any]] = {
    "straights": ["with_low_twos", "no_twos"],
    "flush_order": ["rank_first", "suit_first"],
    "pass_rule": ["may_reenter", "locked_out"],
    "scoring": ["tiered", "original", "win_loss"],
    "last_card_duty": [False, True],
}

app = FastAPI(
    title="poker", version="0.1.0", description="Big Two vs solver bots, with study mode."
)
_sessions: OrderedDict[str, GameSession] = OrderedDict()


class RulesIn(BaseModel):
    straights: str = "with_low_twos"
    flush_order: str = "rank_first"
    pass_rule: str = "may_reenter"
    scoring: str = "tiered"
    last_card_duty: bool = False


class NewGame(BaseModel):
    difficulty: str | None = Field(None, description="easy | medium | hard | expert")
    bots: str | None = Field(None, description="Agent spec; overrides difficulty")
    rules: RulesIn = Field(default_factory=RulesIn)
    seed: int | None = None
    seat: int = Field(0, ge=0, le=3)


class PlayIn(BaseModel):
    cards: list[str] = Field(default_factory=list, description="Card codes; empty = pass")


class AnalyzeIn(BaseModel):
    iterations: int = Field(20_000, ge=100, le=200_000)
    objective: str = Field("score", pattern="^(score|win)$")
    top: int = Field(3, ge=1, le=500)


def _get(game_id: str) -> GameSession:
    s = _sessions.get(game_id)
    if s is None:
        raise HTTPException(
            404, f"no game {game_id!r} (games are kept in memory; the server may have restarted)"
        )
    return s


def _after_move(s: GameSession) -> None:
    if s.state.is_terminal:
        s.save(GAME_LOG_DIR)


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {"difficulties": DIFFICULTIES, "rules": RULE_OPTIONS, "defaults": Rules().kwargs()}


@app.post("/api/games")
def new_game(body: NewGame) -> dict[str, Any]:
    spec = body.bots or DIFFICULTIES.get(body.difficulty or "hard")
    if spec is None:
        raise HTTPException(422, f"unknown difficulty {body.difficulty!r}")
    try:
        make_agent(spec)
        rules = Rules(**body.rules.model_dump())
        rules.new_deal(0)  # validates rule names
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    seed = body.seed if body.seed is not None else random.getrandbits(32)
    s = GameSession(rules=rules, seat=body.seat, bot_spec=spec, seed=seed)
    _sessions[s.id] = s
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return s.view()


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict[str, Any]:
    return _get(game_id).view()


@app.post("/api/games/{game_id}/play")
def play(game_id: str, body: PlayIn) -> dict[str, Any]:
    s = _get(game_id)
    with s.lock:
        try:
            s.human_play(cards(" ".join(body.cards)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        _after_move(s)
        return s.view()


@app.post("/api/games/{game_id}/step")
def step(game_id: str) -> dict[str, Any]:
    s = _get(game_id)
    with s.lock:
        move = s.bot_step()
        _after_move(s)
        return {"move": None if move is None else s._action_view(*move), "view": s.view()}


@app.post("/api/games/{game_id}/analyze")
def analyze(game_id: str, body: AnalyzeIn) -> dict[str, Any]:
    s = _get(game_id)
    with s.lock:
        try:
            return s.analyze(body.iterations, body.top, body.objective)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e


@app.get("/api/games/{game_id}/review")
def review(game_id: str) -> list[dict[str, Any]]:
    s = _get(game_id)
    with s.lock:
        try:
            return s.review()
        except ValueError as e:
            raise HTTPException(409, str(e)) from e


@app.get("/api/games/{game_id}/log")
def log(game_id: str) -> dict[str, Any]:
    return _get(game_id).log()


# The built UI (web/dist), if present. Unknown non-API paths fall back to index.html.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404)
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
