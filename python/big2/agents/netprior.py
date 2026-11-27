"""The network alone: its prior over the legal moves, played at a temperature.

``NetPriorAgent`` reads a ``.pknn`` policy-value network (``big2 rl export``) through the
Rust engine (:func:`big2._core.net_search` with every legal move as a candidate and no
play-out, so one forward pass per decision) and samples a move from the prior raised to
``1 / temperature``: ``temperature = 0`` plays the top move (the raw network);
``temperature = 1`` samples the prior as it is; larger temperatures flatten it towards the
uniform random player. These are the Elo ladder's network rungs (plan Task 40).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from big2._core import Big2State, net_search

ALL_LEGAL = 1_000_000
"""A ``k`` larger than any legal-move count: every legal move is a candidate."""


@dataclass
class NetPriorAgent:
    """Samples the network's prior at ``temperature`` (0 = the top move)."""

    net: str
    temperature: float = 1.0
    seed: int | None = None
    name: str = ""
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be at least 0, not {self.temperature}")
        if not self.name:
            self.name = f"netprior:temperature={self.temperature}"
        self._rng = random.Random(self.seed)

    def prior(self, state: Big2State) -> tuple[list[int], np.ndarray]:
        """The legal moves (best first) and the network's prior over them."""
        res = net_search(state, net=self.net, k=ALL_LEGAL, n=1, depth=0, seed=0)
        cands = sorted(res["candidates"], key=lambda c: -c["prior"])
        moves = [int(c["cards"]) for c in cands]
        p = np.array([c["prior"] for c in cands], dtype=np.float64)
        return moves, p

    def act(self, state: Big2State) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return int(legal[0])
        moves, p = self.prior(state)
        if self.temperature == 0:
            return moves[0]  # sorted best first; ties keep the engine's order
        logits = np.log(np.maximum(p, 1e-300)) / self.temperature
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        u = self._rng.random()
        return moves[min(int(np.searchsorted(np.cumsum(w), u, side="right")), len(moves) - 1)]


__all__ = ["ALL_LEGAL", "NetPriorAgent"]
