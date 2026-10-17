"""One human-vs-bots game, independent of HTTP.

A :class:`GameSession` owns the true game state and exposes only what the human
player may see (:meth:`GameSession.view`). Bots move one at a time
(:meth:`GameSession.bot_step`) so a client can animate each move.

Study features:

* :meth:`GameSession.analyze` — the solver's view of the human's current turn.
* :meth:`GameSession.review` — after the game, the solver's verdict on each of
  the human's decisions, with the expected-score cost of any deviation.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker._core import analyze as core_analyze
from poker.agents import Agent, make_agent
from poker.games.big2 import Big2State, Rules, card_list, kind_of
from poker.server import models

#: Bot strength presets offered by the UI. Values are agent specs.
DIFFICULTIES: dict[str, str] = {
    "easy": "greedy:epsilon=0.1",
    "medium": "ismcts:iterations=300",
    "hard": "ismcts:iterations=2000",
    "expert": "ismcts:iterations=10000,threads=4",
}


def rl_available(rules: Rules | None = None) -> bool:
    """True if a trained network exists (and, given ``rules``, they are the default
    rules, the only ones the networks were trained on)."""
    return models.default_model() is not None and (rules is None or rules == Rules())


def difficulties(model: models.ModelInfo | None = None) -> dict[str, str]:
    """Opponent presets: the solver levels, plus the RL agents when a network exists
    (``model``, or the default one)."""
    d = dict(DIFFICULTIES)
    m = model or models.default_model()
    if m is not None:
        d["rl"] = f"dmc:checkpoint={m.path}"
        d["rl_search"] = f"dmcsearch:checkpoint={m.path},determinizations=32,top=6"
    return d


#: Solver settings for study-mode analysis (ISMCTS simulations) and the
#: post-game review (PIMC sampled deals per decision).
ANALYSIS_ITERATIONS = 20_000
REVIEW_DEALS = 300


@dataclass
class Decision:
    """A human decision point, kept for the post-game review."""

    state: Big2State
    chosen: int


@dataclass
class GameSession:
    rules: Rules
    seat: int
    bot_spec: str
    seed: int
    #: Network for the study-mode hint (a :mod:`poker.server.models` id; ``None`` = default).
    model: str | None = None
    id: str = field(default_factory=lambda: secrets.token_urlsafe(8))
    created: float = field(default_factory=time.time)
    state: Big2State = field(init=False)
    bots: dict[int, Agent] = field(init=False)
    # (player, cards) per action, plus the index in `actions` where each trick starts.
    actions: list[tuple[int, int]] = field(default_factory=list)
    trick_starts: list[int] = field(default_factory=lambda: [0])
    decisions: list[Decision] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _review: list[dict[str, Any]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.seat < 4:
            raise ValueError("seat must be 0-3")
        self.state = self.rules.new_deal(self.seed)
        self.initial_hands = self.state.hands
        self.bots = {
            p: make_agent(self.bot_spec, seed=self.seed * 4 + p) for p in range(4) if p != self.seat
        }

    # -- moves -------------------------------------------------------------

    def _apply(self, cards: int) -> None:
        p = self.state.current_player
        assert p is not None
        self.state.apply(cards)
        self.actions.append((p, cards))
        if self.state.current_play is None and not self.state.is_terminal:
            self.trick_starts.append(len(self.actions))

    def human_play(self, cards: int) -> None:
        """Applies the human's move. Raises ``ValueError`` if it is not their turn
        or the move is illegal."""
        if self.state.current_player != self.seat:
            raise ValueError("it is not your turn")
        before = self.state.copy()
        self._apply(cards)  # raises ValueError if illegal
        self.decisions.append(Decision(before, cards))

    def bot_step(self) -> tuple[int, int] | None:
        """Plays one bot move if it is a bot's turn; returns ``(player, cards)``."""
        p = self.state.current_player
        if p is None or p == self.seat:
            return None
        cards = self.bots[p].act(self.state)
        self._apply(cards)
        return p, cards

    # -- what the human sees -------------------------------------------------

    def view(self) -> dict[str, Any]:
        """The game from the human's seat. Hidden hands appear only once the game is over."""
        s = self.state
        over = s.is_terminal
        trick_start = self.trick_starts[-1]
        cp = s.current_play
        return {
            "id": self.id,
            "seat": self.seat,
            "rules": self.rules.kwargs(),
            "bots": self.bot_spec,
            "seed": self.seed,
            "model": None if (m := self.hint_model()) is None else m.id,
            "hand": card_list(s.hand(self.seat)),
            "hand_sizes": s.hand_sizes,
            "current_player": s.current_player,
            "your_turn": s.current_player == self.seat,
            "is_opening": s.is_opening,
            "current_play": None
            if cp is None
            else {"player": cp[0], "cards": card_list(cp[1]), "kind": cp[2]},
            "trick": [self._action_view(p, c) for p, c in self.actions[trick_start:]],
            "history": [self._action_view(p, c) for p, c in self.actions],
            "trick_starts": self.trick_starts,
            "legal": [card_list(a) for a in s.legal_actions()]
            if s.current_player == self.seat
            else [],
            "locked_out": s.locked_out,
            "over": over,
            "winner": s.winner,
            "scores": s.scores,
            "penalties": s.penalties if over else None,
            "final_hands": [card_list(h) for h in s.hands] if over else None,
            "initial_hands": [card_list(h) for h in self.initial_hands] if over else None,
        }

    def _action_view(self, player: int, cards: int) -> dict[str, Any]:
        return {
            "player": player,
            "cards": card_list(cards),
            "kind": kind_of(cards, self.rules) if cards else "pass",
        }

    # -- study ---------------------------------------------------------------

    def analyze(
        self, iterations: int = ANALYSIS_ITERATIONS, top: int = 3, objective: str = "score"
    ) -> dict[str, Any]:
        """The solver's top moves for the human's current turn and, when a network
        is available, the network's top moves beside them (``"network"``, see
        :meth:`network_view`)."""
        if self.state.current_player != self.seat:
            raise ValueError("analysis is only available on your turn")
        view = _analysis_view(
            self.state, self.rules, "ismcts", iterations, None, objective, seed=len(self.actions)
        )
        solver = view["moves"]
        view["moves"] = solver[:top]
        view["network"] = self.network_view(solver, objective)
        return view

    def hint_model(self) -> models.ModelInfo | None:
        """The network used for the hint, or ``None`` (none available, or non-default rules)."""
        if not rl_available(self.rules):
            return None
        return models.get(self.model) if self.model else models.default_model()

    def set_model(self, model_id: str) -> dict[str, Any]:
        """Switches the hint to another network, loading it now. Raises ``ValueError``
        for an unknown model or non-default rules."""
        if not rl_available(self.rules):
            raise ValueError("the networks were trained on the default rules only")
        from poker.rl.agent import load_model

        m = models.get(model_id)
        load_model(m.path)
        self.model = m.id
        return models.describe(m)

    def network_view(self, solver: list[dict[str, Any]], objective: str) -> dict[str, Any] | None:
        """The hint network's top 3 moves, and how the solver rates its first choice.

        ``solver`` is the solver's ranking of every legal move (best first). The
        result has ``model`` (:func:`poker.server.models.describe`), ``moves`` and
        ``pick``: the network's first choice with its ``solver_rank`` (1 = the
        solver's best), the solver's ``solver`` stats for it, and ``loss`` — how
        much worse the solver rates it than its own best, in points (objective
        "score") or win probability ("win"). Q-networks give each move a predicted
        ``mean_score`` and ``win_rate``; a policy network (PPO) gives ``prob``.
        """
        m = self.hint_model()
        if m is None:
            return None
        from poker.rl.agent import evaluate_moves, is_policy, load_model
        from poker.rl.model import SCORE_SCALE

        net = load_model(m.path)
        moves = evaluate_moves(net, self.state)
        policy = is_policy(net)
        if policy:
            logits = [mv["mean_score"] / SCORE_SCALE for mv in moves]
            hi = max(logits)
            w = [math.exp(x - hi) for x in logits]
            for mv, x in zip(moves, w, strict=True):
                mv["prob"] = x / sum(w)
        elif objective == "win":
            moves.sort(key=lambda mv: -mv["win_rate"])
        shown = [
            {
                "cards": card_list(mv["cards"]),
                "kind": "pass" if not mv["cards"] else kind_of(mv["cards"], self.rules),
                "mean_score": None if policy else mv["mean_score"],
                "win_rate": None if policy else mv["win_rate"],
                "prob": mv.get("prob"),
            }
            for mv in moves[:3]
        ]
        pick = shown[0]["cards"]
        rank = next((i for i, s in enumerate(solver) if s["cards"] == pick), None)
        key = "mean_score" if objective == "score" else "win_rate"
        return {
            "model": {**models.describe(m), "kind": "policy" if policy else "q"},
            "moves": shown,
            "pick": {
                "cards": pick,
                "solver_rank": None if rank is None else rank + 1,
                "solver": None if rank is None else solver[rank],
                "loss": None if rank is None else solver[0][key] - solver[rank][key],
            },
        }

    def review(self, iterations: int = REVIEW_DEALS) -> list[dict[str, Any]]:
        """After the game: each human decision with the solver's top moves and the
        expected-score difference between the best move and the one played.

        Uses PIMC (``iterations`` sampled deals, every legal move simulated in each),
        so the move you played gets as many simulations as the best one and the
        comparison is fair. Forced moves are skipped.
        """
        if not self.state.is_terminal:
            raise ValueError("the review is available when the game is over")
        if self._review is None:
            out = []
            for i, d in enumerate(self.decisions):
                legal = d.state.legal_actions()
                if len(legal) == 1:
                    continue  # forced move, nothing to review
                a = _analysis_view(
                    d.state, self.rules, "pimc", iterations, len(legal), "score", seed=i
                )
                best = a["moves"][0]
                chosen = next(m for m in a["moves"] if m["cards"] == card_list(d.chosen))
                out.append(
                    {
                        "turn": i,
                        "hand": card_list(d.state.hand(self.seat)),
                        "to_beat": None
                        if d.state.current_play is None
                        else card_list(d.state.current_play[1]),
                        "played": card_list(d.chosen),
                        "played_stats": chosen,
                        "best": best,
                        "top": a["moves"][:3],
                        "score_loss": best["mean_score"] - chosen["mean_score"],
                        "agrees": best["cards"] == chosen["cards"],
                    }
                )
            self._review = out
        return self._review

    # -- persistence ---------------------------------------------------------

    def log(self) -> dict[str, Any]:
        """Everything needed to replay the game exactly."""
        return {
            "id": self.id,
            "created": self.created,
            "rules": self.rules.kwargs(),
            "seed": self.seed,
            "seat": self.seat,
            "bots": self.bot_spec,
            "initial_hands": [card_list(h) for h in self.initial_hands],
            "actions": [{"player": p, "cards": card_list(c)} for p, c in self.actions],
            "winner": self.state.winner,
            "scores": self.state.scores,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = (
            directory
            / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.created))}-{self.id}.json"
        )
        path.write_text(json.dumps(self.log(), indent=2))
        return path


def _analysis_view(
    state: Big2State,
    rules: Rules,
    algorithm: str,
    iterations: int,
    top: int | None,
    objective: str,
    seed: int,
) -> dict[str, Any]:
    """The solver's ranking of the legal moves (all of them if ``top`` is ``None``)."""
    r = core_analyze(
        state, algorithm=algorithm, iterations=iterations, threads=4, seed=seed, objective=objective
    )
    return {
        "simulations": r["simulations"],
        "seconds": r["seconds"],
        "objective": objective,
        "moves": [
            {
                "cards": card_list(m["cards"]),
                "kind": "pass" if not m["cards"] else kind_of(m["cards"], rules),
                "win_rate": m["win_rate"],
                "win_ci95": m["win_ci95"],
                "mean_score": m["mean_score"],
                "score_ci95": m["score_ci95"],
                "visits": m["visits"],
                "share": m["visits"] / max(r["simulations"], 1),
            }
            for m in r["moves"][:top]
        ],
    }
