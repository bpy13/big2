"""The Q-network: scores every legal move of a decision in one pass.

Inputs are the 0/1 features from ``crates/big2/src/features.rs``:

* the state (``STATE_LEN`` values), encoded once per decision;
* each legal action (``ACTION_LEN`` values), plus the 52-card hand that would be
  left after playing it (computed here from the state's hand bits).

Outputs per action:

* ``score`` — predicted final score of the acting player, divided by
  :data:`SCORE_SCALE` (pairwise settlement, rules §7);
* ``win_logit`` — logit of the probability that the acting player goes out first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from big2._core import Big2VecEnv

STATE_LEN, ACTION_LEN = Big2VecEnv.feature_sizes()
SCORE_SCALE = 50.0
"""Scores are divided by this for training; typical final scores are within ±50."""


@dataclass
class NetConfig:
    state_hidden: int = 512
    state_out: int = 256
    action_out: int = 128
    head_hidden: int = 256


class QNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg
        self.state_tower = nn.Sequential(
            nn.Linear(STATE_LEN, c.state_hidden),
            nn.ReLU(),
            nn.Linear(c.state_hidden, c.state_out),
            nn.ReLU(),
        )
        self.action_tower = nn.Sequential(nn.Linear(ACTION_LEN + 52, c.action_out), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(c.state_out + c.action_out, c.head_hidden),
            nn.ReLU(),
            nn.Linear(c.head_hidden, c.head_hidden // 2),
            nn.ReLU(),
            nn.Linear(c.head_hidden // 2, 2),
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores actions.

        Args:
            states: ``[B, STATE_LEN]`` float, one row per decision.
            actions: ``[N, ACTION_LEN]`` float, one row per candidate action.
            index: ``[N]`` long, the decision (row of ``states``) of each action.

        Returns:
            ``(score, win_logit)``, each ``[N]``.
        """
        hand_left = states[index, :52] - actions[:, :52]
        h = self.state_tower(states)[index]
        a = self.action_tower(torch.cat([actions, hand_left], dim=1))
        out = self.head(torch.cat([h, a], dim=1))
        return out[:, 0], out[:, 1]

    def config_dict(self) -> dict[str, int]:
        return asdict(self.cfg)


# ---------------------------------------------------------------------------
# Network v2: cards-left distributions and a prior (plan Task 12)
# ---------------------------------------------------------------------------

MAX_CARDS = 13
"""Cards in a full hand; the distributions have ``MAX_CARDS + 1`` classes (0–13)."""
SEATS = 4
DIST_LEN = SEATS * (MAX_CARDS + 1)
"""Distribution logits per candidate: 4 seats × 14 card counts."""
V2_OUTPUTS = DIST_LEN + 1
"""The head's outputs: the 56 distribution logits, then the policy logit."""

SCORINGS = ("tiered", "original", "win_loss")


def payoff(scoring: str) -> torch.Tensor:
    """Penalty per cards-left count 0–13 under ``scoring`` (rules §7.1), as ``[14]`` floats.

    ``tiered``: ``[0, 1, …, 7, 16, 18, 30, 33, 36, 52]``. ``original``: 1–9 ×1, 10–12 ×2,
    13 ×3. ``win_loss``: 0 for the winner, 1 for everyone else; with this vector the
    pairwise settlement of :func:`seat_scores` gives the winner +3 and the others −1,
    so one formula serves every scoring.
    """
    n = torch.arange(MAX_CARDS + 1, dtype=torch.float32)
    if scoring == "tiered":
        mult = torch.where(n <= 7, 1.0, torch.where(n <= 9, 2.0, torch.where(n <= 12, 3.0, 4.0)))
        return n * mult
    if scoring == "original":
        mult = torch.where(n <= 9, 1.0, torch.where(n <= 12, 2.0, 3.0))
        return n * mult
    if scoring == "win_loss":
        return (n > 0).float()
    raise ValueError(f"unknown scoring {scoring!r}; expected one of {SCORINGS}")


def seat_scores(probs: torch.Tensor, scoring: str = "tiered") -> torch.Tensor:
    """Expected final score of each seat from per-seat cards-left distributions.

    ``probs`` is ``[..., 4, 14]`` (rows sum to 1, seats relative to the actor). With
    ``E[p_i]`` the expected penalty of seat ``i``, the pairwise settlement (rules §7)
    ``score_i = Σ_j (p_j − p_i) = Σ_j p_j − 4·p_i`` is linear, so the expectation of
    the score is the same formula over the expected penalties. Returns ``[..., 4]``.
    """
    pen = probs @ payoff(scoring).to(probs.device, probs.dtype)
    return pen.sum(dim=-1, keepdim=True) - SEATS * pen


def segment_log_softmax(logits: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    """``log_softmax`` of ``logits`` [N] within each group ``index`` [N] (``groups`` of them)."""
    m = torch.full((groups,), -torch.inf, device=logits.device, dtype=logits.dtype)
    m = m.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
    z = logits - m[index].detach()
    s = torch.zeros(groups, device=logits.device, dtype=logits.dtype).index_add(0, index, z.exp())
    return z - s.log()[index]


@dataclass
class NetConfigV2:
    state_hidden: int = 512
    state_out: int = 256
    action_out: int = 128
    head_hidden: int = 256
    #: Dropout after each hidden ReLU of the state tower and the head, in training
    #: only (the exported file and the Rust reader have no dropout).
    dropout: float = 0.0


class QNetV2(nn.Module):
    """Network v2: per candidate move, four cards-left distributions and a prior logit.

    The towers are those of :class:`QNet` (state 512–256, move 128, so Task 7's export
    path still applies); the head ends in :data:`V2_OUTPUTS` numbers per candidate:
    ``4 × 14`` distribution logits (seat 0 = the actor, then next, across, previous;
    class ``k`` = finishing with ``k`` cards) and one policy logit. The mean score of
    any scoring is :func:`seat_scores` of the distributions, so one network serves
    ``tiered``, ``original`` and ``win_loss``.
    """

    def __init__(self, cfg: NetConfigV2 | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfigV2()
        c = self.cfg

        def act() -> list[nn.Module]:
            return [nn.ReLU(), nn.Dropout(c.dropout)] if c.dropout > 0 else [nn.ReLU()]

        self.state_tower = nn.Sequential(
            nn.Linear(STATE_LEN, c.state_hidden),
            *act(),
            nn.Linear(c.state_hidden, c.state_out),
            *act(),
        )
        self.action_tower = nn.Sequential(nn.Linear(ACTION_LEN + 52, c.action_out), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(c.state_out + c.action_out, c.head_hidden),
            *act(),
            nn.Linear(c.head_hidden, c.head_hidden // 2),
            *act(),
            nn.Linear(c.head_hidden // 2, V2_OUTPUTS),
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the network; inputs as :meth:`QNet.forward`.

        Returns ``(dist_logits [N, 4, 14], prior_logit [N])``.
        """
        hand_left = states[index, :52] - actions[:, :52]
        h = self.state_tower(states)[index]
        a = self.action_tower(torch.cat([actions, hand_left], dim=1))
        out = self.head(torch.cat([h, a], dim=1))
        return out[:, :DIST_LEN].reshape(-1, SEATS, MAX_CARDS + 1), out[:, DIST_LEN]

    def config_dict(self) -> dict[str, int]:
        return asdict(self.cfg)


class V2Scorer(nn.Module):
    """A :class:`QNetV2` seen as a Q-network (``forward`` returns ``(score, 0)``), so
    the DMC tools (``choose``, ``evaluate_vs_greedy``) can play it.

    ``mode="prior"`` scores moves by their policy logit; ``mode="score"`` by the
    actor's expected score under ``scoring``, divided by :data:`SCORE_SCALE`.
    """

    def __init__(self, net: QNetV2, mode: str = "score", scoring: str = "tiered") -> None:
        super().__init__()
        if mode not in ("prior", "score"):
            raise ValueError(f"mode must be 'prior' or 'score', not {mode!r}")
        self.net, self.mode, self.scoring = net, mode, scoring

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist, prior = self.net(states, actions, index)
        if self.mode == "prior":
            return prior, torch.zeros_like(prior)
        score = seat_scores(dist.softmax(-1), self.scoring)[:, 0] / SCORE_SCALE
        return score, torch.zeros_like(score)
