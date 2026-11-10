"""A learned belief about the hidden cards (plan Task 19).

From the actor's view (the Q-network's ``STATE_LEN`` state features) the network predicts,
for each of the 52 cards, which of the three opponents holds it: a softmax over relative
seats 1–3 per card (``52 × 3`` logits added to the hand-size baseline; cards the actor can see are
masked in the loss). Under 0.5 M parameters.

Data: :func:`generate` plays self-play games with a network-v2 checkpoint (prior, greedy
with a little exploration; the plan's "PUCT self-play" is replaced by this, see
``docs/study-results/build-phase3.md`` §3: PUCT costs ~0.2 s per move, 1 M positions would
take ~50 CPU-hours) and records, per decision, the actor's state features and the true
holder of every unseen card. Shards are ``.npz`` with bit-packed states.

:func:`train` fits the network, reports the held-out log-likelihood per unseen card next to
the uniform baseline (each card held by seat ``s`` with probability ``cards_s / unseen``),
and :func:`export_belief_pknn` writes the ``.pknn`` file ``big2_big2::belief_net::BeliefNet``
reads.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2.rl.device import resolve_device
from big2.rl.export import F16, F32, write_pknn_v2
from big2.rl.model import STATE_LEN

OFF_UNSEEN = 52
OFF_COUNTS = 312
"""Offsets in the state features (``crates/big2/src/features.rs``)."""

BELIEF_KIND = "big2-belief"
"""Must match ``BELIEF_KIND`` in ``crates/big2/src/belief_net.rs``."""
BELIEF_FEATURES = "big2-state-v1"
SHARD_FORMAT = 1


@dataclass
class BeliefConfig:
    """Network and training settings."""

    hidden1: int = 320
    hidden2: int = 256
    lr: float = 2e-3
    batch_size: int = 2048
    epochs: int = 6
    weight_decay: float = 1e-5
    seed: int = 0


class BeliefNet(nn.Module):
    """``STATE_LEN`` → hidden1 → hidden2 → ``52 × 3`` logits (relative seats 1, 2, 3)."""

    def __init__(self, cfg: BeliefConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BeliefConfig()
        c = self.cfg
        self.net = nn.Sequential(
            nn.Linear(STATE_LEN, c.hidden1),
            nn.ReLU(),
            nn.Linear(c.hidden1, c.hidden2),
            nn.ReLU(),
            nn.Linear(c.hidden2, 52 * 3),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """``[B, STATE_LEN]`` float → ``[B, 52, 3]`` logits: the hand-size baseline
        (:func:`uniform_logprobs`) plus the network's correction, so an untrained
        network is the uniform belief. The Rust reader adds the same baseline."""
        return self.net(states).reshape(-1, 52, 3) + uniform_logprobs(states).unsqueeze(1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def holder_labels(states: np.ndarray, hands: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Relative holder (0 = seat 1, 1 = seat 2, 2 = seat 3) of every unseen card, else -1.

    ``states`` is ``[D, STATE_LEN]`` u8, ``hands`` ``[D, 4]`` u64 absolute hand masks,
    ``players`` ``[D]`` the seat to act.
    """
    d = len(states)
    out = np.full((d, 52), -1, dtype=np.int8)
    unseen = states[:, OFF_UNSEEN : OFF_UNSEEN + 52].astype(bool)
    bits = np.uint64(1) << np.arange(52, dtype=np.uint64)
    for rel in (1, 2, 3):
        seat = (players.astype(np.int64) + rel) % 4
        mask = hands[np.arange(d), seat]
        has = (mask[:, None] & bits[None, :]) != 0
        out[unseen & has] = rel - 1
    return out


def generate(
    net_checkpoint: str,
    out_dir: str | Path,
    positions: int = 1_000_000,
    shard_positions: int = 50_000,
    seed: int = 0,
    epsilon: float = 0.05,
    envs: int = 256,
    log: Any = print,
) -> int:
    """Self-play with the network v2 prior; writes shards of ``(states, labels)`` to
    ``out_dir``. Returns the positions written. Existing shards count: a rerun adds only
    the missing ones (shard ``k`` is seeded by ``seed + k``)."""
    from big2._core import Big2VecEnv
    from big2.rl.dmc import choose
    from big2.rl.export import load_pvnet
    from big2.rl.model import ACTION_LEN, V2Scorer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, _ = load_pvnet(net_checkpoint)
    player = V2Scorer(net, "prior")
    player.eval()
    total = 0
    n_shards = math.ceil(positions / shard_positions)
    for k in range(n_shards):
        path = out / f"belief_{k:05d}.npz"
        if path.exists():
            total += _shard_len(path)
            continue
        env = Big2VecEnv(envs, seed=seed + k)
        rng = np.random.default_rng(seed + k)
        packed: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        got = 0
        t0 = time.time()
        with torch.no_grad():
            while got < shard_positions:
                st, ac, counts_l, ids, players_l = env.observe()
                states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
                actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
                counts = np.asarray(counts_l, dtype=np.int64)
                hands = np.asarray([env.state(i).hands for i in ids], dtype=np.uint64)
                players = np.frombuffer(players_l, dtype=np.uint8).astype(np.int64)
                packed.append(np.packbits(states, axis=1))
                labels.append(holder_labels(states, hands, players))
                got += len(ids)
                env.step(choose(player, states, actions, counts, epsilon, rng).tolist())
        np.savez_compressed(
            path.with_suffix(".tmp.npz"),
            format=np.int32(SHARD_FORMAT),
            states=np.concatenate(packed)[:shard_positions],
            labels=np.concatenate(labels)[:shard_positions],
        )
        path.with_suffix(".tmp.npz").replace(path)
        total += shard_positions
        log(f"shard {k + 1}/{n_shards}: {shard_positions:,} positions in {time.time() - t0:.0f} s")
    return total


def _shard_len(path: Path) -> int:
    with np.load(path) as d:
        return int(d["labels"].shape[0])


def load_shards(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(packed_states, labels, is_heldout)``; every 20th shard is held out."""
    files = sorted(Path(data_dir).glob("belief_*.npz"))
    if not files:
        raise FileNotFoundError(f"no belief_*.npz shards in {data_dir}")
    st, lb, held = [], [], []
    for i, f in enumerate(files):
        with np.load(f) as d:
            st.append(d["states"])
            lb.append(d["labels"])
            is_held = i % 20 == 19 if len(files) >= 20 else i == len(files) - 1
            held.append(np.full(len(d["labels"]), is_held))
    return np.concatenate(st), np.concatenate(lb), np.concatenate(held)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _unpack(packed: np.ndarray, device: str) -> torch.Tensor:
    bits = np.unpackbits(packed, axis=1)[:, :STATE_LEN]
    return torch.from_numpy(bits).to(device=device, dtype=torch.float32)


def uniform_logprobs(states: torch.Tensor) -> torch.Tensor:
    """The baseline ``[B, 3]`` log-probabilities: seat ``s`` holds an unseen card with
    probability ``cards_s / unseen`` (hand sizes are in the state features)."""
    counts = states[:, OFF_COUNTS : OFF_COUNTS + 42].reshape(-1, 3, 14)
    n = (counts * torch.arange(14, device=states.device)).sum(dim=2)
    return torch.log(n.clamp(min=1e-9) / n.sum(dim=1, keepdim=True).clamp(min=1e-9))


@torch.no_grad()
def evaluate(
    net: BeliefNet, packed: np.ndarray, labels: np.ndarray, device: str, batch: int = 8192
) -> dict[str, float]:
    """Mean log-likelihood (nats) per unseen card, for the network and the uniform
    baseline, and the share of unseen cards whose true holder is the network's top pick."""
    net.eval()
    ll = base = top = cards = 0.0
    for i in range(0, len(packed), batch):
        x = _unpack(packed[i : i + batch], device)
        y = torch.from_numpy(labels[i : i + batch].astype(np.int64)).to(device)
        valid = y >= 0
        lp = torch.log_softmax(net(x), dim=2)
        yy = y.clamp(min=0)
        ll += float(lp.gather(2, yy.unsqueeze(2)).squeeze(2)[valid].sum())
        ub = uniform_logprobs(x).unsqueeze(1).expand(-1, 52, -1)
        base += float(ub.gather(2, yy.unsqueeze(2)).squeeze(2)[valid].sum())
        top += float((lp.argmax(dim=2) == y)[valid].sum())
        cards += float(valid.sum())
    return {
        "loglik_per_card": ll / cards,
        "uniform_loglik_per_card": base / cards,
        "top1": top / cards,
        "cards": cards,
    }


def train(
    data_dir: str | Path,
    run_dir: str | Path,
    cfg: BeliefConfig | None = None,
    device: str = "auto",
    log: Any = print,
) -> Path:
    """Trains on the shards in ``data_dir``; writes ``run_dir/best.pt``,
    ``results.json`` (held-out numbers per epoch) and returns the checkpoint path."""
    cfg = cfg or BeliefConfig()
    dev = resolve_device(device)
    torch.manual_seed(cfg.seed)
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    packed, labels, held = load_shards(data_dir)
    tr, va = np.flatnonzero(~held), np.flatnonzero(held)
    log(f"{len(tr):,} training positions, {len(va):,} held out")
    net = BeliefNet(cfg).to(dev)
    log(f"{net.n_parameters():,} parameters on {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(tr) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []
    best = -1e9
    t0 = time.time()
    for epoch in range(cfg.epochs):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), cfg.batch_size):
            idx = np.sort(order[i : i + cfg.batch_size])
            x = _unpack(packed[idx], dev)
            y = torch.from_numpy(labels[idx].astype(np.int64)).to(dev)
            valid = y >= 0
            lp = torch.log_softmax(net(x), dim=2)
            nll = -lp.gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
            loss = (nll * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        res = evaluate(net, packed[va], labels[va], dev)
        res["epoch"] = epoch + 1
        res["minutes"] = (time.time() - t0) / 60
        history.append(res)
        log(
            f"epoch {epoch + 1}: held-out log-lik {res['loglik_per_card']:.4f} "
            f"(uniform {res['uniform_loglik_per_card']:.4f}), top-1 {res['top1']:.3f}"
        )
        if res["loglik_per_card"] > best:
            best = res["loglik_per_card"]
            torch.save(
                {"arch": "belief", "net_config": asdict(cfg), "model": net.state_dict()},
                run / "best.pt",
            )
    (run / "results.json").write_text(
        json.dumps(
            {"config": asdict(cfg), "parameters": net.n_parameters(), "epochs": history}, indent=1
        )
    )
    return run / "best.pt"


def load_belief(checkpoint: str | Path) -> tuple[BeliefNet, dict[str, Any]]:
    """Loads a checkpoint written by :func:`train` on the CPU."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if ckpt.get("arch") != "belief":
        raise ValueError(f"{checkpoint} is not a belief checkpoint")
    net = BeliefNet(BeliefConfig(**ckpt["net_config"]))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def export_belief_pknn(checkpoint: str | Path, out: str | Path, precision: str = "f16") -> Path:
    """Writes ``out`` (``.pknn`` version 2, one tower ``belief``) for the Rust reader."""
    if precision not in ("f16", "f32"):
        raise ValueError("precision must be 'f16' or 'f32'")
    net, _ = load_belief(checkpoint)
    code = F16 if precision == "f16" else F32
    layers = [(m, code, None) for m in net.net if isinstance(m, nn.Linear)]
    meta = {
        "kind": BELIEF_KIND,
        "features": BELIEF_FEATURES,
        "outputs": "card52x3",
        "precision": precision,
        "source": Path(checkpoint).name,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_pknn_v2(out, [("belief", layers)], meta)
    return Path(out)
