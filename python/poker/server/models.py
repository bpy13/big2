"""The trained networks the UI can use, for the RL opponents and the study-mode hint.

A model is listed when its file exists in ``models/`` (override the directory with
``POKER_MODELS_DIR``). The first available one is the default. ``POKER_RL_CHECKPOINT``
adds any other checkpoint as the default, under the id ``"custom"``.

Nothing is loaded at startup. The default model loads on first use; any other
model loads only when the player confirms it (``POST /api/games/{id}/model``) or
starts a game with it. Loaded models stay in memory (:func:`poker.rl.agent.load_model`
caches them).

All networks were trained under the default rules only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    kind: str = "q"

    @property
    def path(self) -> Path:
        p = Path(self.filename)
        return p if p.is_absolute() else MODELS_DIR / p


#: Best first: the first available model is the default.
CATALOG: list[ModelInfo] = [
    ModelInfo(
        "distilled-10h",
        "Distilled from the solver (+10 h)",
        "big2-distilled-10h.pt",
        "Ties the solver: −0.5 ± 0.8 points per game vs ISMCTS-1000.",
    ),
    ModelInfo(
        "dmc-10h",
        "Self-play DMC (+10 h)",
        "big2-dmc-selfplay-10h.pt",
        "Learned without the solver; ties it: −0.2 ± 0.8 points per game.",
    ),
    ModelInfo(
        "ppo-10h",
        "Self-play PPO (+10 h)",
        "big2-ppo-10h.pt",
        "Weaker: −1.4 ± 0.7 points per game vs the solver. Shows move probabilities.",
        kind="policy",
    ),
    ModelInfo(
        "distilled-v1",
        "Distilled, original",
        "big2-distilled.pt",
        "The Phase 3 model: −0.5 ± 1.0 vs the solver; the +10 h one beats it by 1.3.",
    ),
]


def available() -> list[ModelInfo]:
    """The models whose files exist, default first."""
    out = [m for m in CATALOG if m.path.is_file()]
    custom = os.environ.get("POKER_RL_CHECKPOINT")
    if custom and Path(custom).is_file():
        p = Path(custom).resolve()
        out = [m for m in out if m.path.resolve() != p]
        out.insert(0, ModelInfo("custom", p.name, str(p), "Set by POKER_RL_CHECKPOINT."))
    return out


def default_model() -> ModelInfo | None:
    """The model used unless the player picks another, or ``None`` if there is none."""
    models = available()
    return models[0] if models else None


def get(model_id: str) -> ModelInfo:
    """The available model with this id; ``ValueError`` if there is none."""
    for m in available():
        if m.id == model_id:
            return m
    raise ValueError(f"no model {model_id!r} (available: {[m.id for m in available()]})")


def is_loaded(m: ModelInfo) -> bool:
    """True if the network is already in memory."""
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
    }
