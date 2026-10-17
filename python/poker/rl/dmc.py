"""Deep Monte Carlo (DMC) self-play training, after DouZero (Zha et al., ICML 2021).

The idea is simple:

1. **Play.** Run many games of the current network against itself. At each
   decision, score every legal move with the network and play the best one
   (with probability ``epsilon``, a random legal move instead, to explore).
2. **Label.** When a game ends, label every (state, move) pair with the final
   score of the player who made the move — a Monte Carlo return, no bootstrapping.
3. **Fit.** Regress the network's predicted score toward those labels (MSE),
   and its win head toward "did this player go out first" (cross-entropy).
4. Repeat.

Each iteration plays ``games_per_iter`` games, then trains on the samples of the
last ``replay_iters`` iterations. Every ``eval_every`` iterations the greedy
network plays a duplicate-deal match against three greedy bots.

Run with ``poker rl train`` (see ``--help``) or :func:`train`. Output in
``run_dir``: ``log.jsonl`` (one line per iteration), ``latest.pt``, ``best.pt``
(best evaluation score) and ``iter_XXXXX.pt`` snapshots.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from poker._core import Big2VecEnv
from poker.rl.device import actor_device, resolve_device
from poker.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig, QNet


@dataclass
class TrainConfig:
    run_dir: str = "runs/dmc/default"
    envs: int = 256
    games_per_iter: int = 1024
    epsilon: float = 0.05
    lr: float = 3e-4
    batch_size: int = 2048
    replay_iters: int = 4
    win_weight: float = 0.5
    eval_every: int = 10
    eval_deals: int = 256
    snapshot_every: int = 50
    max_iters: int = 1_000_000
    max_minutes: float = 60.0
    seed: int = 0
    device: str = "auto"  # see poker.rl.device
    rules: dict[str, Any] = field(default_factory=dict)
    net: NetConfig = field(default_factory=NetConfig)


@dataclass
class Batch:
    """Training samples as uint8 features plus labels."""

    states: np.ndarray  # [N, STATE_LEN] uint8
    actions: np.ndarray  # [N, ACTION_LEN] uint8
    scores: np.ndarray  # [N] float32 (raw final score)
    wins: np.ndarray  # [N] float32 (0/1)

    def __len__(self) -> int:
        return len(self.scores)


def _to_float(x: np.ndarray, device: str) -> torch.Tensor:
    """Array → float32 tensor on ``device``. Integer arrays (the 0/1 features) are
    converted after the copy, so they cross to a GPU as uint8 (4× less data)."""
    x = np.asarray(x)
    if x.dtype.kind == "f":
        x = x.astype(np.float32, copy=False)
    t = torch.from_numpy(x if x.flags.writeable else x.copy())
    return t.to(device).float()


@torch.no_grad()
def choose(
    net: nn.Module,
    states: np.ndarray,
    actions: np.ndarray,
    counts: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: str = "cpu",
) -> np.ndarray:
    """Picks one action per decision: argmax predicted score, or random with prob. epsilon."""
    b = len(counts)
    index = np.repeat(np.arange(b), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    score, _ = net(
        _to_float(states, device), _to_float(actions, device), torch.from_numpy(index).to(device)
    )
    q = np.full((b, int(counts.max())), -np.inf, dtype=np.float32)
    q[index, np.arange(len(index)) - starts[index]] = score.float().cpu().numpy()
    choices = q.argmax(axis=1)
    explore = rng.random(b) < epsilon
    if explore.any():
        choices[explore] = (rng.random(int(explore.sum())) * counts[explore]).astype(np.int64)
    return choices


def collect(
    net: nn.Module,
    env: Big2VecEnv,
    games: int,
    epsilon: float,
    rng: np.random.Generator,
    device: str = "cpu",
) -> tuple[Batch, dict[str, float]]:
    """Self-play until ``games`` games finish; returns labelled samples."""
    net.eval()
    trajectories: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    out_s, out_a, out_r, out_w = [], [], [], []
    finished_games = 0
    lengths = []
    while finished_games < games:
        st, ac, counts_l, ids, players = env.observe()
        counts = np.asarray(counts_l, dtype=np.int64)
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        choices = choose(net, states, actions, counts, epsilon, rng, device)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        chosen = actions[starts + choices]
        for k, i in enumerate(ids):
            trajectories.setdefault(i, []).append((states[k], chosen[k], players[k]))
        for i, scores, winner, _ in env.step(choices.tolist()):
            traj = trajectories.pop(i, [])
            lengths.append(len(traj))
            for s, a, p in traj:
                out_s.append(s)
                out_a.append(a)
                out_r.append(scores[p])
                out_w.append(1.0 if winner == p else 0.0)
            finished_games += 1
    batch = Batch(
        np.stack(out_s),
        np.stack(out_a),
        np.asarray(out_r, np.float32),
        np.asarray(out_w, np.float32),
    )
    return batch, {"iter_games": finished_games, "mean_length": float(np.mean(lengths))}


def train_epoch(
    net: QNet, opt: torch.optim.Optimizer, data: Batch, cfg: TrainConfig, rng: np.random.Generator
) -> dict[str, float]:
    """One pass over ``data`` in shuffled minibatches."""
    net.train()
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    order = rng.permutation(len(data))
    totals = {"loss_score": 0.0, "loss_win": 0.0}
    n = 0
    for lo in range(0, len(order), cfg.batch_size):
        idx = order[lo : lo + cfg.batch_size]
        s = _to_float(data.states[idx], cfg.device)
        a = _to_float(data.actions[idx], cfg.device)
        r = torch.from_numpy(data.scores[idx] / SCORE_SCALE).to(cfg.device)
        w = torch.from_numpy(data.wins[idx]).to(cfg.device)
        score, win_logit = net(s, a, torch.arange(len(idx), device=cfg.device))
        loss_s, loss_w = mse(score, r), bce(win_logit, w)
        loss = loss_s + cfg.win_weight * loss_w
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        totals["loss_score"] += loss_s.item() * len(idx)
        totals["loss_win"] += loss_w.item() * len(idx)
        n += len(idx)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_vs_greedy(
    net: nn.Module, deals: int, seed: int, rules: dict[str, Any] | None = None, device: str = "cpu"
) -> dict[str, float]:
    """Duplicate match: the greedy network in one seat vs three greedy bots, every
    deal played with the network in each of the four seats."""
    net.eval()
    env = Big2VecEnv(4 * deals, seed=seed, mode="vs_greedy", auto_reset=False, **(rules or {}))
    rng = np.random.default_rng(0)
    per_deal_score = np.zeros(deals)
    per_deal_win = np.zeros(deals)
    while env.num_done < env.n:
        st, ac, counts_l, ids, _ = env.observe()
        if not ids:
            break
        counts = np.asarray(counts_l, dtype=np.int64)
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        choices = choose(net, states, actions, counts, 0.0, rng, device)
        for i, scores, winner, seat in env.step(choices.tolist()):
            assert seat is not None  # vs_greedy mode always has a learner seat
            per_deal_score[i // 4] += scores[seat] / 4
            per_deal_win[i // 4] += (winner == seat) / 4
    n = len(per_deal_score)
    return {
        "eval_score": float(per_deal_score.mean()),
        "eval_score_ci95": float(1.96 * per_deal_score.std(ddof=1) / math.sqrt(n)),
        "eval_win_rate": float(per_deal_win.mean()),
    }


def save_checkpoint(
    path: Path,
    net: QNet,
    cfg: TrainConfig,
    info: dict[str, Any],
    opt: torch.optim.Optimizer | None = None,
) -> None:
    """Writes weights, configs and progress; with ``opt``, also the optimizer (for resuming)."""
    payload: dict[str, Any] = {
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "net_config": net.config_dict(),
        "train_config": asdict(cfg),
        "info": info,
    }
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, path)


def train(
    cfg: TrainConfig, resume: str | None = None, log: Any = print, init: str | None = None
) -> Path:
    """Runs DMC training until ``max_iters`` or ``max_minutes``; returns the run directory.

    ``resume`` continues a run (weights, optimizer, counters); ``init`` only copies
    weights from a checkpoint (e.g. a distilled network) and starts a fresh run.
    """
    cfg = replace(cfg, device=resolve_device(cfg.device))
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    net = QNet(cfg.net).to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    start_iter, total_games, total_samples, best = 0, 0, 0, -math.inf
    if init:
        net.load_state_dict(torch.load(init, map_location=cfg.device, weights_only=False)["model"])
    if resume:
        ckpt = torch.load(resume, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        info = ckpt["info"]
        start_iter, total_games, total_samples = info["iter"], info["games"], info["samples"]
        best = info.get("best_eval", -math.inf)

    adev = actor_device(cfg.device)
    actor = QNet(cfg.net).to(adev)
    env = Big2VecEnv(cfg.envs, seed=cfg.seed, **cfg.rules)
    replay: deque[Batch] = deque(maxlen=cfg.replay_iters)
    t0 = time.time()
    it = start_iter
    with (run / "log.jsonl").open("a") as logf:
        while it < cfg.max_iters and (time.time() - t0) / 60 < cfg.max_minutes:
            it += 1
            actor.load_state_dict(net.state_dict())
            tc = time.time()
            batch, cstats = collect(actor, env, cfg.games_per_iter, cfg.epsilon, rng, adev)
            collect_s = time.time() - tc
            replay.append(batch)
            data = Batch(
                np.concatenate([b.states for b in replay]),
                np.concatenate([b.actions for b in replay]),
                np.concatenate([b.scores for b in replay]),
                np.concatenate([b.wins for b in replay]),
            )
            tt = time.time()
            losses = train_epoch(net, opt, data, cfg, rng)
            train_s = time.time() - tt
            total_games += int(cstats["iter_games"])
            total_samples += len(batch)
            rec: dict[str, Any] = {
                "iter": it,
                "minutes": (time.time() - t0) / 60,
                "games": total_games,
                "samples": total_samples,
                "collect_s": collect_s,
                "train_s": train_s,
                **cstats,
                **losses,
            }
            if it % cfg.eval_every == 0:
                actor.load_state_dict(net.state_dict())
                rec.update(
                    evaluate_vs_greedy(
                        actor, cfg.eval_deals, seed=10_000 + cfg.seed, rules=cfg.rules, device=adev
                    )
                )
            info = {
                "iter": it,
                "games": total_games,
                "samples": total_samples,
                "best_eval": best,
                **rec,
            }
            if "eval_score" in rec and rec["eval_score"] > best:
                best = rec["eval_score"]
                info["best_eval"] = best
                save_checkpoint(run / "best.pt", net, cfg, info)
            if it % cfg.snapshot_every == 0:
                save_checkpoint(run / f"iter_{it:05d}.pt", net, cfg, info)
            save_checkpoint(run / "latest.pt", net, cfg, info, opt)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(rec)
    return run
