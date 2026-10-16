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

from poker._core import Big2VecEnv

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
