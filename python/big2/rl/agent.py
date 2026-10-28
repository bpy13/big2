"""Play Big Two with a trained Q-network checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2._core import Big2State, encode
from big2.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig, QNet

torch.set_num_threads(1)  # many agents run in parallel processes during tournaments

_CACHE: dict[str, nn.Module] = {}


def load_model(path: str | Path) -> nn.Module:
    """Loads (and caches) a checkpoint from :mod:`big2.rl.dmc`, :mod:`big2.rl.distill`
    or :mod:`big2.rl.ppo`. For a PPO policy the first output is a logit, not a score:
    the argmax is still the move to play, but the numbers are not points."""
    key = str(Path(path).resolve())
    if key not in _CACHE:
        ckpt = torch.load(key, map_location="cpu", weights_only=False)
        net: nn.Module
        if ckpt.get("arch") == "ppo":
            from big2.rl.ppo import PolicyValueNet

            net = PolicyValueNet(NetConfig(**ckpt["net_config"]))
        else:
            net = QNet(NetConfig(**ckpt["net_config"]))
        net.load_state_dict(ckpt["model"])
        net.eval()
        _CACHE[key] = net
    return _CACHE[key]


def is_cached(path: str | Path) -> bool:
    """True if :func:`load_model` has already loaded this checkpoint."""
    return str(Path(path).resolve()) in _CACHE


def is_policy(net: nn.Module) -> bool:
    """True for a PPO policy, whose first output is a logit rather than a score."""
    from big2.rl.ppo import PolicyValueNet

    return isinstance(net, PolicyValueNet)


@torch.no_grad()
def evaluate_moves(net: nn.Module, state: Big2State) -> list[dict[str, Any]]:
    """Every legal move with the network's predicted score and win probability, best first."""
    s, a, legal = encode(state)
    st = torch.from_numpy(np.frombuffer(s, dtype=np.uint8).reshape(1, STATE_LEN).astype(np.float32))
    ac = torch.from_numpy(
        np.frombuffer(a, dtype=np.uint8).reshape(-1, ACTION_LEN).astype(np.float32)
    )
    score, win = net(st, ac, torch.zeros(len(legal), dtype=torch.long))
    moves = [
        {"cards": c, "mean_score": float(sc) * SCORE_SCALE, "win_rate": float(w)}
        for c, sc, w in zip(legal, score, torch.sigmoid(win), strict=True)
    ]
    return sorted(moves, key=lambda m: -m["mean_score"])


@dataclass
class DMCAgent:
    """Greedy with respect to the network: plays the move with the best predicted
    score (``objective="score"``) or win probability (``objective="win"``)."""

    checkpoint: str
    objective: str = "score"
    name: str = ""
    net: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.net = load_model(self.checkpoint)
        self.name = self.name or f"dmc:checkpoint={self.checkpoint}"

    def act(self, state: Big2State) -> int:
        moves = evaluate_moves(self.net, state)
        key = "mean_score" if self.objective == "score" else "win_rate"
        return int(max(moves, key=lambda m: m[key])["cards"])
