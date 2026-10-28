"""Search distillation (expert iteration): train the Q-network on the solver's estimates.

Self-play DMC learns from one noisy number per decision — the final score of one
game. The Phase 1 solver, at every decision, estimates the value of many moves
from hundreds of simulated games each. Distillation turns those estimates into
training targets:

1. **Generate** (``generate``): worker processes play games in which every
   decision is analyzed by ISMCTS. For each decision we store the state features
   and, for every move the search simulated at least ``min_visits`` times, its
   mean final score, win rate and visit count. The search's best move is then
   played (with probability ``epsilon`` a random one among its top 3, for
   variety).
2. **Train** (``train``): regress the network's score and win heads toward those
   targets, weighting each move by its visit count (more simulations = less noise).

The result plays like the solver at a fraction of the cost, and can then be
improved further with self-play (``big2 rl train --init ...``).

This is the "expert iteration" idea (Anthony et al., 2017) that AlphaZero also
uses, with ISMCTS as the expert.
"""

from __future__ import annotations

import json
import math
import random
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2._core import analyze, encode
from big2.games.big2 import Rules
from big2.rl.device import actor_device, resolve_device
from big2.rl.dmc import (
    TrainConfig,
    _to_float,
    best_eval_so_far,
    evaluate_vs_greedy,
    save_checkpoint,
)
from big2.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig, QNet

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_worker(
    out_dir: str,
    worker: int,
    minutes: float,
    iterations: int,
    min_visits: int,
    epsilon: float,
    shard_games: int,
    rules: Rules,
) -> int:
    rng = random.Random(1_000_003 * worker + int(time.time()))
    deadline = time.time() + 60 * minutes
    # A per-run prefix, so a new run adds shards instead of overwriting earlier ones.
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    shard = 0
    decisions = 0
    while time.time() < deadline:
        states: list[bytes] = []
        actions: list[bytes] = []
        owner: list[int] = []
        score: list[float] = []
        win: list[float] = []
        visits: list[int] = []
        for _ in range(shard_games):
            s = rules.new_deal(rng.getrandbits(32))
            while not s.is_terminal:
                legal = s.legal_actions()
                if len(legal) == 1:
                    s.apply(legal[0])
                    continue
                res = analyze(s, iterations=iterations, threads=1, seed=rng.getrandbits(32))
                st, ac, order = encode(s)
                pos = {c: k for k, c in enumerate(order)}
                kept = [m for m in res["moves"] if m["visits"] >= min_visits]
                if kept:
                    idx = len(states)
                    states.append(st)
                    for m in kept:
                        k = pos[m["cards"]]
                        actions.append(ac[k * ACTION_LEN : (k + 1) * ACTION_LEN])
                        owner.append(idx)
                        score.append(m["mean_score"])
                        win.append(m["win_rate"])
                        visits.append(m["visits"])
                    decisions += 1
                top = res["moves"][: min(3, len(res["moves"]))]
                pick = rng.choice(top) if rng.random() < epsilon else top[0]
                s.apply(pick["cards"])
        # Write under a temporary name, then rename: readers never see a partial shard.
        final = Path(out_dir) / f"{run_tag}_w{worker:02d}_{shard:05d}.npz"
        tmp = final.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            states=np.frombuffer(b"".join(states), dtype=np.uint8).reshape(-1, STATE_LEN),
            actions=np.frombuffer(b"".join(actions), dtype=np.uint8).reshape(-1, ACTION_LEN),
            owner=np.asarray(owner, dtype=np.int32),
            score=np.asarray(score, dtype=np.float32),
            win=np.asarray(win, dtype=np.float32),
            visits=np.asarray(visits, dtype=np.int32),
        )
        tmp.replace(final)
        shard += 1
    return decisions


def generate(
    out_dir: str,
    workers: int = 4,
    minutes: float = 60.0,
    iterations: int = 1000,
    min_visits: int = 20,
    epsilon: float = 0.25,
    shard_games: int = 20,
    rules: Rules | None = None,
) -> int:
    """Generates solver-labelled decisions into ``out_dir`` (one .npz per shard),
    from games under ``rules`` (default: the default rules)."""
    rules = rules or Rules()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        futures = [
            pool.submit(
                _generate_worker,
                out_dir,
                w,
                minutes,
                iterations,
                min_visits,
                epsilon,
                shard_games,
                rules,
            )
            for w in range(workers)
        ]
        return sum(f.result() for f in futures)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    data_dir: str = "runs/distill/data"
    run_dir: str = "runs/distill/v1"
    lr: float = 3e-4
    batch_size: int = 1024  # decisions per minibatch
    max_decisions: int = 600_000  # most recent decisions kept in memory
    visit_cap: int = 200  # weight = min(visits, cap) / cap
    win_weight: float = 0.5
    eval_every_minutes: float = 3.0
    eval_deals: int = 256
    init: str | None = None  # start from these weights
    resume: str | None = None  # continue a run: weights, optimizer, step count
    max_minutes: float = 60.0
    seed: int = 0
    device: str = "auto"  # see big2.rl.device
    #: Rule variants of the data (``Rules`` keyword arguments), for the evaluation games.
    rules: dict[str, Any] = field(default_factory=dict)
    net: NetConfig = field(default_factory=NetConfig)


class _Data:
    """Loaded shards: per-decision states, per-move targets."""

    def __init__(self) -> None:
        self.files: set[str] = set()
        self.parts: list[dict[str, np.ndarray]] = []

    def refresh(self, data_dir: str, max_decisions: int) -> int:
        # Oldest first (by write time), so trimming below drops the oldest data. A
        # generator may rename or replace a file between listing and stat: skip it.
        stamped = []
        for p in Path(data_dir).glob("*.npz"):
            if str(p) in self.files or p.name.endswith(".tmp.npz"):
                continue
            try:
                stamped.append((p.stat().st_mtime, str(p)))
            except FileNotFoundError:
                continue
        new = [f for _, f in sorted(stamped)]
        for f in new:
            try:
                with np.load(f) as z:
                    part = {k: z[k] for k in z.files}
                # Moves are stored grouped by decision (owner ascending): index ranges.
                d = np.arange(len(part["states"]))
                part["lo"] = np.searchsorted(part["owner"], d, side="left")
                part["hi"] = np.searchsorted(part["owner"], d, side="right")
                self.parts.append(part)
                self.files.add(f)
            except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                continue  # unreadable (e.g. still being written by an older generator); retry later
        while sum(len(p["states"]) for p in self.parts) > max_decisions and len(self.parts) > 1:
            self.parts.pop(0)
        return len(new)

    def decisions(self) -> int:
        return sum(len(p["states"]) for p in self.parts)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """Random decisions with all their labelled moves."""
        sizes = np.array([len(p["states"]) for p in self.parts])
        which = rng.choice(len(self.parts), size=n, p=sizes / sizes.sum())
        st, ac, idx, sc, wn, vs = [], [], [], [], [], []
        row = 0
        for part_i in np.unique(which):
            p = self.parts[part_i]
            picks = rng.choice(len(p["states"]), size=int((which == part_i).sum()))
            for d in picks:
                m = np.arange(p["lo"][d], p["hi"][d])
                st.append(p["states"][d])
                ac.append(p["actions"][m])
                idx.append(np.full(len(m), row))
                sc.append(p["score"][m])
                wn.append(p["win"][m])
                vs.append(p["visits"][m])
                row += 1
        return (
            np.stack(st),
            np.concatenate(ac),
            np.concatenate(idx),
            np.concatenate(sc),
            np.concatenate(wn),
            np.concatenate(vs),
        )


def train(cfg: DistillConfig, log: Any = print) -> Path:
    """Trains on the data in ``cfg.data_dir`` (picking up new shards as they appear)."""
    cfg = replace(cfg, device=resolve_device(cfg.device))
    dev = cfg.device
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = QNet(cfg.net).to(dev)
    if cfg.init:
        net.load_state_dict(torch.load(cfg.init, map_location=dev, weights_only=False)["model"])
    adev = actor_device(dev)
    actor = QNet(cfg.net).to(adev)  # plays the evaluation games
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    steps, best = 0, -math.inf
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        steps = int(ckpt["info"]["iter"])
        best = best_eval_so_far(ckpt, Path(cfg.resume).parent / "best.pt")
    data = _Data()
    t0 = time.time()
    last_eval = t0
    # Checkpoints use the DMC format so DMCAgent can load them.
    ckpt_cfg = TrainConfig(run_dir=cfg.run_dir, device=dev, net=cfg.net)
    while data.refresh(cfg.data_dir, cfg.max_decisions) == 0 and data.decisions() == 0:
        if time.time() - t0 > 60 * cfg.max_minutes:
            raise RuntimeError(f"no data appeared in {cfg.data_dir}")
        time.sleep(10)
    with (run / "log.jsonl").open("a") as logf:
        losses: list[tuple[float, float]] = []
        while (time.time() - t0) / 60 < cfg.max_minutes:
            if steps % 200 == 0:
                data.refresh(cfg.data_dir, cfg.max_decisions)
            st, ac, idx, sc, wn, vs = data.sample(cfg.batch_size, rng)
            w = _to_float(np.minimum(vs, cfg.visit_cap) / cfg.visit_cap, dev)
            score, win_logit = net(
                _to_float(st, dev), _to_float(ac, dev), torch.from_numpy(idx).long().to(dev)
            )
            l_s = (w * (score - _to_float(sc / SCORE_SCALE, dev)) ** 2).sum() / w.sum()
            bce = nn.functional.binary_cross_entropy_with_logits(
                win_logit, _to_float(wn, dev), reduction="none"
            )
            l_w = (w * bce).sum() / w.sum()
            loss = l_s + cfg.win_weight * l_w
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append((l_s.item(), l_w.item()))
            steps += 1
            if time.time() - last_eval > 60 * cfg.eval_every_minutes:
                last_eval = time.time()
                actor.load_state_dict(net.state_dict())
                actor.eval()
                rec: dict[str, Any] = {
                    "step": steps,
                    "minutes": (time.time() - t0) / 60,
                    "decisions": data.decisions(),
                    "loss_score": float(np.mean([x[0] for x in losses])),
                    "loss_win": float(np.mean([x[1] for x in losses])),
                    **evaluate_vs_greedy(
                        actor, cfg.eval_deals, seed=10_000, rules=cfg.rules, device=adev
                    ),
                }
                losses.clear()
                info = {
                    "iter": steps,
                    "games": 0,
                    "samples": data.decisions(),
                    "best_eval": best,
                    **rec,
                }
                if rec["eval_score"] > best:
                    best = rec["eval_score"]
                    save_checkpoint(run / "best.pt", net, ckpt_cfg, info)
                save_checkpoint(run / "latest.pt", net, ckpt_cfg, info, opt)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log(rec)
    return run
