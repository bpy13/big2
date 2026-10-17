"""The trained networks the UI can use, for the RL opponents and the study-mode hint.

Each network was trained under one set of rules (:attr:`ModelInfo.rules`) and is
offered only for games under those rules. A network is listed when its file
exists in ``models/`` (override the directory with ``POKER_MODELS_DIR``); for each
rule set, the first one listed is the default. ``POKER_RL_CHECKPOINT`` adds any
other checkpoint, for the default rules, under the id ``"custom"``.

The hint can also compare the solver with the greedy bot (:data:`GREEDY`), which
needs no file and works under any rules, or show the solver alone (:data:`NONE`).

Nothing is loaded at startup. The default model loads on first use; any other
model loads only when the player confirms it (``POST /api/games/{id}/model``) or
starts a game with it. Loaded models stay in memory (:func:`poker.rl.agent.load_model`
caches them).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker.games.big2 import Rules

MODELS_DIR = Path(
    os.environ.get("POKER_MODELS_DIR", Path(__file__).resolve().parents[3] / "models")
)


@dataclass(frozen=True)
class ModelInfo:
    """One selectable network."""

    id: str
    label: str
    filename: str
    #: Strength in one line, from docs/study-results/phase3-rl.md.
    note: str
    #: ``"q"``: predicts each move's score and win chance (DMC, distillation).
    #: ``"policy"``: gives each move a probability (PPO); no score predictions.
    #: ``"bot"``: a rule-based bot; only its move.
    #: ``"none"``: no comparison, the solver alone.
    kind: str = "q"
    #: The rules it was trained under (``None``: any rules).
    rules: Rules | None = field(default_factory=Rules)

    def fits(self, rules: Rules) -> bool:
        """True if this model can be used for a game under ``rules``."""
        return self.rules is None or self.rules == rules

    @property
    def path(self) -> Path:
        p = Path(self.filename)
        return p if p.is_absolute() else MODELS_DIR / p


#: The rules of the Phase 3 networks: last-card duty was off by default then.
NO_DUTY = Rules(last_card_duty=False)

#: Best first: the first available model for a game's rules is its default.
CATALOG: list[ModelInfo] = [
    ModelInfo(
        "distilled-duty",
        "Distilled from the solver (last-card duty)",
        "big2-distilled-duty.pt",
        "Trained for the default rules (last-card duty on).",
    ),
    ModelInfo(
        "distilled-10h",
        "Distilled from the solver (+10 h)",
        "big2-distilled-10h.pt",
        "Ties the solver: −0.5 ± 0.8 points per game vs ISMCTS-1000. Last-card duty off.",
        rules=NO_DUTY,
    ),
    ModelInfo(
        "dmc-10h",
        "Self-play DMC (+10 h)",
        "big2-dmc-selfplay-10h.pt",
        "Learned without the solver; ties it: −0.2 ± 0.8 points per game. Last-card duty off.",
        rules=NO_DUTY,
    ),
    ModelInfo(
        "ppo-10h",
        "Self-play PPO (+10 h)",
        "big2-ppo-10h.pt",
        "Weaker: −1.4 ± 0.7 points per game vs the solver. Shows move probabilities."
        " Last-card duty off.",
        kind="policy",
        rules=NO_DUTY,
    ),
    ModelInfo(
        "distilled-v1",
        "Distilled, original",
        "big2-distilled.pt",
        "The Phase 3 model: −0.5 ± 1.0 vs the solver; the +10 h one beats it by 1.3."
        " Last-card duty off.",
        rules=NO_DUTY,
    ),
]

#: The greedy bot as a comparison for the hint (no file, any rules).
GREEDY = ModelInfo(
    "greedy",
    "Greedy bot",
    "",
    "The rule-based bot the solver imagines its opponents to be: sheds its weakest cards first.",
    kind="bot",
    rules=None,
)

#: No comparison: the hint shows the solver alone.
NONE = ModelInfo("none", "No comparison", "", "Only the solver's moves.", kind="none", rules=None)


def available(rules: Rules | None = None) -> list[ModelInfo]:
    """The networks whose files exist (for ``rules``, if given), defaults first."""
    out = [m for m in CATALOG if m.path.is_file()]
    custom = os.environ.get("POKER_RL_CHECKPOINT")
    if custom and Path(custom).is_file():
        p = Path(custom).resolve()
        out = [m for m in out if m.path.resolve() != p]
        out.insert(0, ModelInfo("custom", p.name, str(p), "Set by POKER_RL_CHECKPOINT."))
    return [m for m in out if rules is None or m.fits(rules)]


def default_model(rules: Rules) -> ModelInfo | None:
    """The network used for games under ``rules`` unless the player picks another,
    or ``None`` if no network was trained for them."""
    models = available(rules)
    return models[0] if models else None


def get(model_id: str, rules: Rules | None = None) -> ModelInfo:
    """The model with this id (a network, :data:`GREEDY` or :data:`NONE`). ``ValueError``
    if it does not exist or, given ``rules``, was trained under other rules."""
    for m in [*available(), GREEDY, NONE]:
        if m.id == model_id:
            if rules is not None and not m.fits(rules):
                raise ValueError(f"model {model_id!r} was trained under other rules")
            return m
    raise ValueError(f"no model {model_id!r} (available: {[m.id for m in available()]})")


def is_loaded(m: ModelInfo) -> bool:
    """True if the network is already in memory (always true for the greedy bot)."""
    if m.kind in ("bot", "none"):
        return True
    try:
        from poker.rl.agent import is_cached
    except ImportError:  # no PyTorch
        return False
    return is_cached(m.path)


def describe(m: ModelInfo) -> dict[str, Any]:
    """The model as the API shows it."""
    return {
        "id": m.id,
        "label": m.label,
        "note": m.note,
        "kind": m.kind,
        "loaded": is_loaded(m),
        "rules": None if m.rules is None else m.rules.kwargs(),
    }
