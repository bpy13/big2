"""PPO self-play (Schulman et al., 2017), as used for Big Two by Charlesworth (2018).

Where DMC learns the *value* of every move and plays the best, PPO learns a
*policy* — a probability for each legal move — plus a value estimate of the
position:

1. **Play** self-play games, sampling each move from the policy.
2. **Advantage**: for each decision, ``A = final score − V(position)``: how much
   better the game went than the network expected. (Rewards only arrive at the
   end, so with discount 1 this is the Monte Carlo advantage.)
3. **Update** for a few epochs: raise the probability of moves with positive
   advantage and lower the others, but clip the change (the "proximal" part) so
   one batch cannot move the policy too far; also regress V toward the final
   score and add a small entropy bonus to keep exploring.

The policy scores each legal move with the same state/move towers as the DMC
network; a softmax over the legal moves of a decision gives the probabilities.
Checkpoints load with :func:`poker.rl.agent.load_model` (the agent then plays the
most likely move).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from poker._core import Big2VecEnv
from poker.rl.device import actor_device, resolve_device
from poker.rl.dmc import _to_float, best_eval_so_far, evaluate_vs_greedy
from poker.rl.model import ACTION_LEN, SCORE_SCALE, STATE_LEN, NetConfig


class PolicyValueNet(nn.Module):
    """``forward`` returns ``(logits, value)`` per candidate action, so code that
    takes the argmax of the first output (evaluation, agents) works unchanged."""

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
        self.value_head = nn.Sequential(
            nn.Linear(c.state_out, c.head_hidden // 2), nn.ReLU(), nn.Linear(c.head_hidden // 2, 1)
        )
        self.action_tower = nn.Sequential(nn.Linear(ACTION_LEN + 52, c.action_out), nn.ReLU())
        self.policy_head = nn.Sequential(
            nn.Linear(c.state_out + c.action_out, c.head_hidden),
            nn.ReLU(),
            nn.Linear(c.head_hidden, 1),
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.state_tower(states)
        value = self.value_head(h).squeeze(1)
        hand_left = states[index, :52] - actions[:, :52]
        a = self.action_tower(torch.cat([actions, hand_left], dim=1))
        logits = self.policy_head(torch.cat([h[index], a], dim=1)).squeeze(1)
        return logits, value[index]

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        v: torch.Tensor = self.value_head(self.state_tower(states)).squeeze(1)
        return v

    def config_dict(self) -> dict[str, int]:
        return asdict(self.cfg)


def segment_log_softmax(logits: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Log-softmax of ``logits`` within each segment (decision) given by ``index``."""
    seg_max = torch.full((n_segments,), -math.inf, device=logits.device).scatter_reduce(
        0, index, logits.detach(), reduce="amax"
    )
    z = logits - seg_max[index]
    seg_sum = torch.zeros(n_segments, device=logits.device).index_add(0, index, z.exp())
    return z - seg_sum.log()[index]


@dataclass
class PPOConfig:
    run_dir: str = "runs/ppo/v1"
    envs: int = 256
    games_per_iter: int = 512
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 2048  # decisions
    clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    eval_every: int = 10
    eval_deals: int = 256
    max_minutes: float = 60.0
    seed: int = 0
    device: str = "auto"  # see poker.rl.device
    rules: dict[str, Any] = field(default_factory=dict)  # Rules keyword arguments
    net: NetConfig = field(default_factory=NetConfig)


@dataclass
class Rollout:
    states: np.ndarray  # [D, S] uint8
    actions: np.ndarray  # [M, A] uint8 (all legal actions of all decisions)
    owner: np.ndarray  # [M] decision of each action row
    chosen: np.ndarray  # [D] index into actions (global row)
    logp: np.ndarray  # [D] log-prob of the chosen action when played
    ret: np.ndarray  # [D] final score / SCORE_SCALE


@torch.no_grad()
def collect(
    net: PolicyValueNet,
    env: Big2VecEnv,
    games: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Rollout:
    net.eval()
    traj: dict[int, list[tuple[np.ndarray, np.ndarray, int, float, int]]] = {}
    st_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    chosen: list[int] = []
    logps: list[float] = []
    rets: list[float] = []
    finished = 0
    n_act = 0
    while finished < games:
        st, ac, counts_l, ids, players = env.observe()
        counts = np.asarray(counts_l, dtype=np.int64)
        states = np.frombuffer(st, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        actions = np.frombuffer(ac, dtype=np.uint8).reshape(-1, ACTION_LEN)
        index = torch.from_numpy(np.repeat(np.arange(len(ids)), counts)).to(device)
        logits, _ = net(_to_float(states, device), _to_float(actions, device), index)
        logp = segment_log_softmax(logits, index, len(ids)).cpu().numpy()
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        choices = np.empty(len(ids), dtype=np.int64)
        for k in range(len(ids)):
            probs = np.exp(logp[starts[k] : starts[k] + counts[k]])
            choices[k] = rng.choice(counts[k], p=probs / probs.sum())
        for k, i in enumerate(ids):
            lo = int(starts[k])
            traj.setdefault(i, []).append(
                (
                    states[k],
                    actions[lo : lo + counts[k]],
                    int(choices[k]),
                    float(logp[lo + choices[k]]),
                    players[k],
                )
            )
        for i, scores, _, _ in env.step(choices.tolist()):
            for s, acts, c, lp, p in traj.pop(i, []):
                st_rows.append(s)
                owners.append(np.full(len(acts), len(st_rows) - 1, dtype=np.int64))
                act_rows.append(acts)
                chosen.append(n_act + c)
                n_act += len(acts)
                logps.append(lp)
                rets.append(scores[p] / SCORE_SCALE)
            finished += 1
    return Rollout(
        np.stack(st_rows),
        np.concatenate(act_rows),
        np.concatenate(owners),
        np.asarray(chosen),
        np.asarray(logps, np.float32),
        np.asarray(rets, np.float32),
    )


def update(
    net: PolicyValueNet,
    opt: torch.optim.Optimizer,
    ro: Rollout,
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    net.train()
    dev = cfg.device
    d = len(ro.states)
    lo = np.searchsorted(ro.owner, np.arange(d), side="left")
    hi = np.searchsorted(ro.owner, np.arange(d), side="right")
    with torch.no_grad():
        values = (
            torch.cat(
                [
                    net.state_value(_to_float(ro.states[i : i + 8192], dev))
                    for i in range(0, d, 8192)
                ]
            )
            .cpu()
            .numpy()
        )
    adv = ro.ret - values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"loss_policy": 0.0, "loss_value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n = 0
    for _ in range(cfg.epochs):
        for b0 in range(0, d, cfg.minibatch):
            dec = rng.permutation(d)[: cfg.minibatch] if b0 == 0 else rng.choice(d, cfg.minibatch)
            rows = np.concatenate([np.arange(lo[i], hi[i]) for i in dec])
            seg = np.repeat(np.arange(len(dec)), hi[dec] - lo[dec])
            chosen_local = np.concatenate([[0], np.cumsum(hi[dec] - lo[dec])[:-1]]) + (
                ro.chosen[dec] - lo[dec]
            )
            seg_t = torch.from_numpy(seg).to(dev)
            logits, value = net(
                _to_float(ro.states[dec], dev), _to_float(ro.actions[rows], dev), seg_t
            )
            logp_all = segment_log_softmax(logits, seg_t, len(dec))
            logp = logp_all[torch.from_numpy(chosen_local).to(dev)]
            ratio = (logp - _to_float(ro.logp[dec], dev)).exp()
            a = _to_float(adv[dec], dev)
            pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            v = net.state_value(_to_float(ro.states[dec], dev))
            vl = ((v - _to_float(ro.ret[dec], dev)) ** 2).mean()
            ent = -torch.zeros(len(dec), device=dev).index_add(0, seg_t, logp_all.exp() * logp_all)
            ent = ent.mean()
            loss: torch.Tensor = pg + cfg.value_weight * vl - cfg.entropy_weight * ent
            opt.zero_grad()
            torch.autograd.backward(loss)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["loss_policy"] += pg.item()
            stats["loss_value"] += vl.item()
            stats["entropy"] += ent.item()
            stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            n += 1
            del value
    return {k: v / max(n, 1) for k, v in stats.items()}


def train(cfg: PPOConfig, log: Any = print, resume: str | None = None) -> Path:
    """Runs PPO self-play for ``max_minutes``; returns the run directory.

    ``resume`` continues from a checkpoint: weights, optimizer (if saved) and counters.
    """
    cfg = replace(cfg, device=resolve_device(cfg.device))
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = PolicyValueNet(cfg.net).to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    adev = actor_device(cfg.device)
    actor = PolicyValueNet(cfg.net).to(adev)
    t0, it, games, best = time.time(), 0, 0, -math.inf
    if resume:
        ckpt = torch.load(resume, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        it, games = ckpt["info"]["iter"], ckpt["info"]["games"]
        best = best_eval_so_far(ckpt, Path(resume).parent / "best.pt")
    # Offset by the iteration so a resumed run does not replay the same deals.
    env = Big2VecEnv(cfg.envs, seed=cfg.seed + it, **cfg.rules)
    with (run / "log.jsonl").open("a") as logf:
        while (time.time() - t0) / 60 < cfg.max_minutes:
            it += 1
            tc = time.time()
            actor.load_state_dict(net.state_dict())
            ro = collect(actor, env, cfg.games_per_iter, rng, adev)
            collect_s = time.time() - tc
            games += cfg.games_per_iter
            stats = update(net, opt, ro, cfg, rng)
            rec: dict[str, Any] = {
                "iter": it,
                "minutes": (time.time() - t0) / 60,
                "games": games,
                "decisions": len(ro.states),
                "collect_s": collect_s,
                **stats,
            }
            if it % cfg.eval_every == 0:
                actor.load_state_dict(net.state_dict())
                rec.update(
                    evaluate_vs_greedy(
                        actor, cfg.eval_deals, seed=10_000, rules=cfg.rules, device=adev
                    )
                )
                payload = {
                    "arch": "ppo",
                    "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "net_config": net.config_dict(),
                    "train_config": asdict(cfg),
                    "info": {**rec, "best_eval": max(best, rec["eval_score"])},
                }
                if rec["eval_score"] > best:
                    best = rec["eval_score"]
                    torch.save(payload, run / "best.pt")
                torch.save({**payload, "optimizer": opt.state_dict()}, run / "latest.pt")
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(rec)
    return run
