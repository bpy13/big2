"""Network-guided search: policy-rollout Monte Carlo with a trained Q-network.

The Phase 1 solver estimates a move by playing sampled deals to the end with the
*greedy bot* playing every seat. Phase 1 found that this model, not the search
budget, limits its strength. This search keeps the same Monte Carlo structure but
lets the learned network do the playing:

1. The network ranks the legal moves; keep the ``top`` best (pruning).
2. Sample ``determinizations`` deals of the hidden cards (uniform, as in Phase 1).
3. For every (deal, candidate move) pair, play the game to the end with the
   network choosing every seat's moves (greedy, no exploration).
4. Each candidate's value is its average final score (or win rate) over the deals.

All rollouts of one decision run in lockstep in a :class:`Big2VecEnv`, so the
network evaluates every pending position in one batch per step.

This is Tesauro & Galperin's rollout idea (backgammon, 1996) with determinized
hidden information.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from poker._core import Big2State, Big2VecEnv
from poker.rl.agent import evaluate_moves, load_model
from poker.rl.dmc import choose
from poker.rl.model import ACTION_LEN, STATE_LEN


@torch.no_grad()
def rollout_search(
    net: nn.Module,
    state: Big2State,
    determinizations: int = 32,
    top: int = 6,
    objective: str = "score",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Analyzes the player to act's options.

    Returns the candidate moves, best first, each with ``cards``, ``mean_score``,
    ``score_ci95``, ``win_rate``, ``win_ci95``, ``visits`` (rollouts) and
    ``net_score`` (the network's own prediction before search).
    """
    player = state.current_player
    if player is None:
        raise ValueError("the deal is over")
    ranked = evaluate_moves(net, state)
    candidates = ranked[: max(1, top)]
    if len(ranked) == 1:
        m = ranked[0]
        return [
            {**m, "score_ci95": 0.0, "win_ci95": 0.0, "visits": 0, "net_score": m["mean_score"]}
        ]

    rng = random.Random(seed)
    starts: list[Big2State] = []
    for _ in range(determinizations):
        d = state.determinize(player, seed=rng.getrandbits(63))
        starts.extend(d.child(m["cards"]) for m in candidates)
    env = Big2VecEnv.from_states(starts)
    np_rng = np.random.default_rng(seed)
    while True:
        st, ac, counts_l, ids, _ = env.observe()
        if not ids:
            break
        counts = np.asarray(counts_l, dtype=np.int64)
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        env.step(choose(net, states, actions, counts, 0.0, np_rng).tolist())

    k = len(candidates)
    scores = np.zeros((determinizations, k))
    wins = np.zeros((determinizations, k))
    for i in range(len(starts)):
        final = env.scores(i)
        assert final is not None
        scores[i // k, i % k] = final[player]
        wins[i // k, i % k] = env.winner(i) == player
    out = []
    n = determinizations
    for j, m in enumerate(candidates):
        p = float(wins[:, j].mean())
        out.append(
            {
                "cards": m["cards"],
                "mean_score": float(scores[:, j].mean()),
                "score_ci95": float(1.96 * scores[:, j].std(ddof=1) / math.sqrt(n))
                if n > 1
                else math.inf,
                "win_rate": p,
                "win_ci95": float(1.96 * math.sqrt(p * (1 - p) / n)),
                "visits": n,
                "net_score": m["mean_score"],
            }
        )
    key = "mean_score" if objective == "score" else "win_rate"
    return sorted(out, key=lambda m: -m[key])


@dataclass
class RolloutSearchAgent:
    """Plays the best move found by :func:`rollout_search`."""

    checkpoint: str
    determinizations: int = 32
    top: int = 6
    objective: str = "score"
    seed: int = 0
    name: str = ""
    _calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.net = load_model(self.checkpoint)
        self.name = self.name or f"dmcsearch:checkpoint={self.checkpoint}"

    def act(self, state: Big2State) -> int:
        self._calls += 1
        moves = rollout_search(
            self.net,
            state,
            self.determinizations,
            self.top,
            self.objective,
            seed=self.seed * 7919 + self._calls,
        )
        return int(moves[0]["cards"])
