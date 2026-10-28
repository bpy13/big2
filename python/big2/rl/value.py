"""A value network for search leaves: "how will this fully dealt position end?"

The Phase 1 solver judges each simulated position by playing it to the end with
the greedy bot. This module trains a small network to predict the result of
playing it out with a much stronger policy instead — a trained Q-network (by
default the distilled one) playing every seat — so the search can ask the
network in one call instead of simulating:

1. **Generate:** the policy network plays self-play games (5% random moves for
   variety). Every position is recorded with *full information* (all four hands,
   :func:`big2._core.encode_full`, relative to the player to act).
2. **Label:** when a game ends, each position gets every seat's final score and
   who won, again relative to the player to act.
3. **Fit:** a 273 → 256 → 256 → 8 MLP regresses the four scores (MSE) and
   classifies the winner (cross-entropy over the four seats).

The network is exported to the dependency-free ``.pknn`` format that the Rust
search loads (``crates/nn``). Use it with
``analyze(..., value="models/big2-value.pknn")`` or the agent spec
``ismcts:iterations=1000,value=models/big2-value.pknn``.

Because the policy is close to deterministic, the label of a position is close to
a deterministic function of it: the network learns *how the policy would play
this out*, which is cheaper to query than playing it out.
"""

from __future__ import annotations

import copy
import json
import math
import struct
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2._core import Big2VecEnv
from big2.rl.device import actor_device, resolve_device
from big2.rl.dmc import _to_float, choose
from big2.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN

FULL_LEN = 273
FEATURES_NAME = "big2-full-v1"


class ValueMLP(nn.Module):
    """Full-information features → 4 scores / ``SCORE_SCALE`` and 4 win logits,
    for the relative seats (0 = player to act, 1 = next, 2 = across, 3 = previous)."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FULL_LEN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


def export_pknn(net: ValueMLP, path: str | Path, meta: dict[str, Any]) -> None:
    """Writes ``net`` in the ``.pknn`` format read by ``crates/nn`` (see its docs)."""
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    text = "\n".join(f"{k}={v}" for k, v in meta.items()).encode()
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(b"PKNN" + struct.pack("<II", 1, len(text)) + text)
        f.write(struct.pack("<I", len(linears)))
        for lin in linears:
            w = lin.weight.detach().cpu().numpy().astype("<f4")
            b = lin.bias.detach().cpu().numpy().astype("<f4")
            f.write(struct.pack("<II", lin.in_features, lin.out_features))
            f.write(w.tobytes(order="C") + b.tobytes())
    tmp.replace(path)


@dataclass
class ValueConfig:
    policy: str = "models/big2-distilled.pt"  # plays the self-play games
    out: str = "models/big2-value.pknn"
    run_dir: str = "runs/value/v1"
    envs: int = 512
    games_per_round: int = 2048
    epsilon: float = 0.05
    replay_rounds: int = 8
    lr: float = 1e-3
    batch_size: int = 4096
    hidden: int = 256
    win_weight: float = 0.5
    max_minutes: float = 30.0
    seed: int = 0
    device: str = "auto"


@dataclass
class Samples:
    x: np.ndarray  # [N, FULL_LEN] uint8
    scores: np.ndarray  # [N, 4] float32, relative seats, / SCORE_SCALE
    winner: np.ndarray  # [N] int64, relative seat

    def __len__(self) -> int:
        return len(self.winner)


def generate(
    policy: nn.Module,
    env: Big2VecEnv,
    games: int,
    epsilon: float,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Samples:
    """Self-play with ``policy`` until ``games`` games finish; every position labelled."""
    pending: dict[int, list[tuple[np.ndarray, int]]] = {}
    xs: list[np.ndarray] = []
    sc: list[np.ndarray] = []
    wn: list[int] = []
    finished = 0
    while finished < games:
        st, ac, counts_l, ids, players = env.observe()
        full = np.frombuffer(env.full_features(), dtype=np.uint8).reshape(len(ids), FULL_LEN)
        counts = np.asarray(counts_l, dtype=np.int64)
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        choices = choose(policy, states, actions, counts, epsilon, rng, device)
        for k, i in enumerate(ids):
            pending.setdefault(i, []).append((full[k], players[k]))
        for i, scores, winner, _ in env.step(choices.tolist()):
            s = np.asarray(scores, dtype=np.float32) / SCORE_SCALE
            for x, p in pending.pop(i, []):
                xs.append(x)
                sc.append(np.roll(s, -p))  # relative seat r = absolute seat (p + r) % 4
                wn.append((winner - p) % 4)
            finished += 1
    return Samples(np.stack(xs), np.stack(sc), np.asarray(wn, dtype=np.int64))


def _losses(
    net: ValueMLP, data: Samples, idx: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    out = net(_to_float(data.x[idx], device))
    mse = ((out[:, :4] - _to_float(data.scores[idx], device)) ** 2).mean()
    ce = nn.functional.cross_entropy(out[:, 4:], torch.from_numpy(data.winner[idx]).to(device))
    return mse, ce


def train(cfg: ValueConfig, log: Any = print) -> Path:
    """Trains a value network for ``max_minutes``; writes ``cfg.out`` (best on held-out
    data) and ``run_dir/log.jsonl``. Returns the run directory."""
    from big2.rl.agent import load_model

    cfg = replace(cfg, device=resolve_device(cfg.device))
    dev, adev = cfg.device, actor_device(cfg.device)
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # load_model caches one CPU copy shared by the server and the agents; Module.to moves
    # in place, so move a private copy to the actor device, never the cached one.
    policy = copy.deepcopy(load_model(cfg.policy)).to(adev)
    net = ValueMLP(cfg.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    env = Big2VecEnv(cfg.envs, seed=cfg.seed)
    # Held-out games from separate deals, to pick the best network.
    held_out = generate(
        policy, Big2VecEnv(cfg.envs, seed=cfg.seed + 777_777), 1024, cfg.epsilon, rng, adev
    )
    zero_mse = float((held_out.scores**2).mean())  # loss of always predicting 0
    replay: deque[Samples] = deque(maxlen=cfg.replay_rounds)
    meta = {
        "kind": "big2-value",
        "features": FEATURES_NAME,
        "score_scale": SCORE_SCALE,
        "policy": Path(cfg.policy).name,
    }
    t0, rnd, games, best = time.time(), 0, 0, math.inf
    with (run / "log.jsonl").open("a") as logf:
        while (time.time() - t0) / 60 < cfg.max_minutes:
            rnd += 1
            tc = time.time()
            replay.append(generate(policy, env, cfg.games_per_round, cfg.epsilon, rng, adev))
            collect_s = time.time() - tc
            games += cfg.games_per_round
            data = Samples(
                np.concatenate([d.x for d in replay]),
                np.concatenate([d.scores for d in replay]),
                np.concatenate([d.winner for d in replay]),
            )
            net.train()
            order = rng.permutation(len(data))
            for lo in range(0, len(order), cfg.batch_size):
                mse, ce = _losses(net, data, order[lo : lo + cfg.batch_size], dev)
                loss = mse + cfg.win_weight * ce
                opt.zero_grad()
                torch.autograd.backward(loss)
                opt.step()
            net.eval()
            with torch.no_grad():
                vm, vc = _losses(net, held_out, np.arange(len(held_out)), dev)
            rec = {
                "round": rnd,
                "minutes": (time.time() - t0) / 60,
                "games": games,
                "samples": len(data),
                "collect_s": collect_s,
                "val_mse": vm.item(),
                "val_r2": 1 - vm.item() / zero_mse,
                "val_ce": vc.item(),
            }
            if vm.item() < best:
                best = vm.item()
                export_pknn(net, cfg.out, {**meta, "round": rnd, "val_r2": f"{rec['val_r2']:.4f}"})
                torch.save({k: v.cpu() for k, v in net.state_dict().items()}, run / "best.pt")
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(rec)
    return run
