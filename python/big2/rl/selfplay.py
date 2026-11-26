"""Duplicate self-play PPO from a network v2, with the luck taken out (plan Task 36).

The method that worked for mahjong (hk-mahjong-lab Task 29) and the lessons of Dou Dizhu
(Tasks 26/27), applied to Big Two:

1. **Policy.** The prior head of a :class:`~big2.rl.model.QNetV2` (round 3 by default): a
   softmax over each decision's legal moves. Moves are sampled from it.
2. **Luck removed (duplicate).** Every deal is played ``4 x copies`` times: its four hands
   rotated over the seats, each rotation ``copies`` times. A seat's return is its final score
   (``/ SCORE_SCALE``); its baseline is the mean return of the *same hand* in the other games
   of the deal in which the hand faced the same lineup (leave-one-out). So the strength of the
   cards cancels and only how the hand was played is rewarded.
3. **Oracle critic.** A small value network that sees every hand (``Big2VecEnv.full_features``)
   learns each decision's return. It never plays. ``oracle_weight`` mixes it into the
   baseline (``b = (1 - w) * duplicate + w * critic``); the log reports how much of the return
   variance each baseline leaves, so the smoke run shows whether the critic helps.
4. **KL pull.** ``pull * KL(start || policy)`` on every learner decision, constant.
5. **Frozen opponents.** In a ``frozen_share`` of the deals the lineup is learner, frozen,
   learner, frozen: the frozen seats play the start network or an earlier hourly snapshot,
   so the learner does not only learn its own copy's habits (Dou Dizhu's lesson). Rotations
   by two keep that lineup, so the duplicate baseline still pairs like with like.
6. **Distributions kept.** The cards-left heads keep learning the real outcome of the move
   played (``dist_weight``), so the network still works as PUCT's leaf value.

Checkpoints are network v2 checkpoints (``arch = "qnet_v2"``): ``pvnet:checkpoint=…`` plays
them and :func:`big2.rl.export.export_pvnet_pknn` exports them. Output in ``run_dir``:
``log.jsonl``, ``progress.json``, ``latest.pt`` (resume), ``hour_XX.pt`` (one per hour of
training, counted over resumes).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2._core import Big2State, Big2VecEnv
from big2.rl.device import resolve_device
from big2.rl.dmc import _to_float
from big2.rl.model import (
    ACTION_LEN,
    MAX_CARDS,
    SCORE_SCALE,
    SEATS,
    STATE_LEN,
    NetConfigV2,
    QNetV2,
    segment_log_softmax,
)

FULL_LEN = 273
"""Bytes of ``Big2VecEnv.full_features`` per decision (``big2_big2::features::FULL_LEN``)."""
LEARNER = 0
"""Agent id of the learner in a lineup; frozen pool members are 1, 2, …"""


@dataclass
class SelfPlayConfig:
    run_dir: str = "runs/selfplay/default"
    init: str = "runs/net_r3/best.pt"
    deals_per_iter: int = 256
    copies: int = 2
    frozen_share: float = 0.5
    #: Most recent hourly snapshots kept as frozen opponents (the start is always one).
    pool_size: int = 4
    lr: float = 3e-5
    critic_lr: float = 3e-4
    epochs: int = 2
    minibatch: int = 2048
    clip: float = 0.2
    entropy_weight: float = 0.0
    pull: float = 0.02
    oracle_weight: float = 0.0
    dist_weight: float = 0.1
    #: Stop this run after this many minutes (0: no limit).
    max_minutes: float = 240.0
    #: Stop when the training time over all resumes reaches this (0: no limit).
    until_minutes: float = 0.0
    snapshot_minutes: float = 60.0
    save_minutes: float = 10.0
    seed: int = 0
    device: str = "auto"
    rules: dict[str, Any] = field(default_factory=dict)
    #: False (Task 36d ablation): no rotations or copies; every game is its own fresh deal
    #: (same games per iteration) and the baseline is the batch mean return, mixed with the
    #: oracle critic by ``oracle_weight``.
    duplicate: bool = True


class OracleCritic(nn.Module):
    """Full-information value: every hand visible -> the actor's return. Training only."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FULL_LEN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, full: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(full).squeeze(1)
        return out


@dataclass
class Rollout:
    """Learner decisions of one iteration."""

    states: np.ndarray  # [D, STATE_LEN] uint8
    full: np.ndarray  # [D, FULL_LEN] uint8
    actions: np.ndarray  # [M, ACTION_LEN] uint8, every candidate of every decision
    lo: np.ndarray  # [D] first candidate row of each decision
    hi: np.ndarray  # [D] one past its last
    chosen: np.ndarray  # [D] global candidate row played
    logp: np.ndarray  # [D] log-prob of the played move when sampled
    start_logp: np.ndarray  # [M] the start network's log-probs of every candidate
    ret: np.ndarray  # [D] the actor's final score / SCORE_SCALE
    dup: np.ndarray  # [D] the duplicate baseline of the actor's hand
    left: np.ndarray  # [D, 4] cards left at the end, seats relative to the actor
    games: int
    frozen_games: int


def _logp_dense(
    net: nn.Module, states: np.ndarray, actions: np.ndarray, counts: np.ndarray, device: str
) -> torch.Tensor:
    """Log-probs of every candidate (flat, ``[sum(counts)]``) under ``net``'s prior."""
    index = torch.from_numpy(np.repeat(np.arange(len(counts)), counts)).to(device)
    _, prior = net(_to_float(states, device), _to_float(actions, device), index)
    return segment_log_softmax(prior.float(), index, len(counts))


def _sample(logp: torch.Tensor, counts: np.ndarray, gen: torch.Generator) -> np.ndarray:
    """One candidate per decision, drawn from ``exp(logp)`` (Gumbel trick, on the device)."""
    b, width = len(counts), int(counts.max())
    dev = logp.device
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    index = np.repeat(np.arange(b), counts)
    col = np.arange(len(index)) - starts[index]
    dense = torch.full((b, width), -torch.inf, device=dev)
    dense[torch.from_numpy(index).to(dev), torch.from_numpy(col).to(dev)] = logp
    u = torch.rand((b, width), device=dev, generator=gen).clamp_(1e-20, 1.0)
    out: np.ndarray = (dense - torch.log(-torch.log(u))).argmax(1).cpu().numpy()
    return out


def deal_hands(rng: np.random.Generator, rules: dict[str, Any]) -> list[int]:
    """Four 13-card hands (masks) of a fresh deal."""
    return list(Big2State.new_deal(int(rng.integers(2**62)), **rules).hands)


def duplicate_baselines(
    games: list[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    """Leave-one-out duplicate baseline per ``(game, seat)``.

    Each game is ``{"group", "rotation", "lineup", "scores"}``: seat ``s`` holds hand
    ``(s + rotation) % 4`` of its group's deal; ``lineup[s]`` is the agent id there. A seat's
    baseline is the mean score of the same hand in the other games of the group where the
    agents around it (from its seat on, in turn order) were the same; with no such game, the
    hand's mean over every other game of the group; with none at all, 0.
    """
    keyed: dict[tuple[Any, ...], list[tuple[int, int, float]]] = {}
    by_hand: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for gi, g in enumerate(games):
        lu = g["lineup"]
        for s in range(SEATS):
            hand = (s + g["rotation"]) % SEATS
            around = tuple(lu[(s + k) % SEATS] for k in range(SEATS))
            keyed.setdefault((g["group"], hand, around), []).append((gi, s, g["scores"][s]))
            by_hand.setdefault((g["group"], hand), []).append((gi, g["scores"][s]))
    out: dict[tuple[int, int], float] = {}
    for (group, hand, _), rows in keyed.items():
        total = sum(r[2] for r in rows)
        for gi, s, v in rows:
            if len(rows) > 1:
                out[(gi, s)] = (total - v) / (len(rows) - 1)
            else:
                others = [x for gj, x in by_hand[(group, hand)] if gj != gi]
                out[(gi, s)] = float(np.mean(others)) if others else 0.0
    return out


@torch.no_grad()
def collect(
    learner: QNetV2,
    start: QNetV2,
    pool: list[QNetV2],
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> Rollout:
    """Plays one iteration of duplicate games; returns the learner's decisions.

    ``pool`` holds the frozen opponents (index 0 is the start network); agent id ``k >= 1``
    plays ``pool[k - 1]``.
    """
    dev = cfg.device
    games: list[dict[str, Any]] = []
    states: list[Big2State] = []
    frozen_games = 0
    rotations, copies = (SEATS, cfg.copies) if cfg.duplicate else (1, 1)
    for group in range(cfg.deals_per_iter * (1 if cfg.duplicate else SEATS * cfg.copies)):
        hands = deal_hands(rng, cfg.rules)
        if pool and rng.random() < cfg.frozen_share:
            f = 1 + int(rng.integers(len(pool)))
            lineup = [LEARNER, f, LEARNER, f]
        else:
            lineup = [LEARNER] * SEATS
        for rotation in range(rotations):
            for _ in range(copies):
                seat_hands = [hands[(s + rotation) % SEATS] for s in range(SEATS)]
                states.append(Big2State.from_hands(seat_hands, **cfg.rules))
                games.append({"group": group, "rotation": rotation, "lineup": lineup})
                frozen_games += lineup[1] != LEARNER
    env = Big2VecEnv.from_states(states)
    # Per game: the learner's decisions so far as (row in the lists below, seat).
    pending: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(games))}
    st_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    start_rows: list[np.ndarray] = []
    chosen_local: list[int] = []
    logps: list[float] = []
    seats_of: list[int] = []
    game_of: list[int] = []
    finished = 0
    while finished < len(games):
        st_b, ac_b, counts_l, ids_l, players_l = env.observe()
        counts = np.asarray(counts_l, dtype=np.int64)
        ids = np.asarray(ids_l, dtype=np.int64)
        players = np.frombuffer(bytes(players_l), dtype=np.uint8).astype(np.int64)
        st = np.frombuffer(st_b, dtype=np.uint8).reshape(len(ids), STATE_LEN)
        ac = np.frombuffer(ac_b, dtype=np.uint8).reshape(-1, ACTION_LEN)
        full = np.frombuffer(env.full_features(), dtype=np.uint8).reshape(len(ids), FULL_LEN)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        agent = np.array([games[i]["lineup"][p] for i, p in zip(ids, players, strict=True)])
        choices = np.empty(len(ids), dtype=np.int64)
        for a in np.unique(agent):
            sel = np.flatnonzero(agent == a)
            rows = np.concatenate([np.arange(starts[k], starts[k] + counts[k]) for k in sel])
            net = learner if a == LEARNER else pool[a - 1]
            lp = _logp_dense(net, st[sel], ac[rows], counts[sel], dev)
            pick = _sample(lp, counts[sel], gen)
            choices[sel] = pick
            if a != LEARNER:
                continue
            slp = _logp_dense(start, st[sel], ac[rows], counts[sel], dev).cpu().numpy()
            lp_np = lp.cpu().numpy()
            off = 0
            for j, k in enumerate(sel):
                n = int(counts[k])
                pending[int(ids[k])].append((len(st_rows), int(players[k])))
                st_rows.append(st[k])
                full_rows.append(full[k])
                act_rows.append(ac[starts[k] : starts[k] + n])
                start_rows.append(slp[off : off + n])
                chosen_local.append(int(pick[j]))
                logps.append(float(lp_np[off + pick[j]]))
                seats_of.append(int(players[k]))
                game_of.append(int(ids[k]))
                off += n
        for i, scores, _, _ in env.step(choices.tolist()):
            games[i]["scores"] = [float(x) for x in scores]
            games[i]["left"] = list(env.state(i).hand_sizes)
            finished += 1
    base = duplicate_baselines(games)
    d = len(st_rows)
    counts_all = np.array([len(a) for a in act_rows], dtype=np.int64)
    lo = np.concatenate([[0], np.cumsum(counts_all)[:-1]])
    ret = np.empty(d, np.float32)
    dup = np.empty(d, np.float32)
    left = np.empty((d, SEATS), np.int64)
    for r in range(d):
        g, s = game_of[r], seats_of[r]
        ret[r] = games[g]["scores"][s] / SCORE_SCALE
        dup[r] = base[(g, s)] / SCORE_SCALE
        left[r] = [games[g]["left"][(s + k) % SEATS] for k in range(SEATS)]
    return Rollout(
        states=np.stack(st_rows),
        full=np.stack(full_rows),
        actions=np.concatenate(act_rows),
        lo=lo,
        hi=lo + counts_all,
        chosen=lo + np.asarray(chosen_local),
        logp=np.asarray(logps, np.float32),
        start_logp=np.concatenate(start_rows).astype(np.float32),
        ret=ret,
        dup=dup,
        left=left,
        games=len(games),
        frozen_games=frozen_games,
    )


def _residual(ret: np.ndarray, base: np.ndarray) -> float:
    """Share of the return's variance left after subtracting ``base``."""
    return float(np.mean((ret - base) ** 2) / max(np.var(ret), 1e-12))


def update(
    net: QNetV2,
    critic: OracleCritic,
    opt: torch.optim.Optimizer,
    copt: torch.optim.Optimizer,
    ro: Rollout,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One PPO update (policy, pull, distributions) and one critic fit on ``ro``."""
    dev = cfg.device
    d = len(ro.states)
    with torch.no_grad():
        oracle = (
            torch.cat([critic(_to_float(ro.full[i : i + 8192], dev)) for i in range(0, d, 8192)])
            .cpu()
            .numpy()
        )
    w = cfg.oracle_weight
    plain = ro.dup if cfg.duplicate else np.full_like(ro.ret, ro.ret.mean())
    base = (1 - w) * plain + w * oracle
    adv = ro.ret - base
    adv = adv / (adv.std() + 1e-8)
    stats: dict[str, float] = {
        "resid_duplicate": _residual(ro.ret, ro.dup),
        "resid_oracle": _residual(ro.ret, oracle),
        "resid_mix": _residual(ro.ret, 0.5 * (ro.dup + oracle)),
        "ret_sd": float(ro.ret.std() * SCORE_SCALE),
    }
    sums = {k: 0.0 for k in ("pg", "kl", "entropy", "dist", "critic", "clip_frac")}
    n = 0
    net.eval()  # no dropout: the ratio must compare like with like
    for _ in range(cfg.epochs):
        perm = rng.permutation(d)
        for b0 in range(0, d, cfg.minibatch):
            dec = perm[b0 : b0 + cfg.minibatch]
            cnt = ro.hi[dec] - ro.lo[dec]
            rows = np.concatenate([np.arange(ro.lo[i], ro.hi[i]) for i in dec])
            seg_np = np.repeat(np.arange(len(dec)), cnt)
            seg = torch.from_numpy(seg_np).to(dev)
            pick = torch.from_numpy(
                np.concatenate([[0], np.cumsum(cnt)[:-1]]) + ro.chosen[dec] - ro.lo[dec]
            ).to(dev)
            dist, prior = net(_to_float(ro.states[dec], dev), _to_float(ro.actions[rows], dev), seg)
            logp_all = segment_log_softmax(prior.float(), seg, len(dec))
            ratio = (logp_all[pick] - _to_float(ro.logp[dec], dev)).exp()
            a = _to_float(adv[dec], dev)
            pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            p_all = logp_all.exp()
            slp = _to_float(ro.start_logp[rows], dev)
            zero = torch.zeros(len(dec), device=dev)
            kl = zero.index_add(0, seg, slp.exp() * (slp - logp_all)).mean()
            ent = -zero.index_add(0, seg, p_all * logp_all).mean()
            target = torch.from_numpy(ro.left[dec]).to(dev).clamp_(0, MAX_CARDS)
            dl = nn.functional.cross_entropy(
                dist[pick].float().reshape(-1, MAX_CARDS + 1), target.reshape(-1)
            )
            loss = pg + cfg.pull * kl - cfg.entropy_weight * ent + cfg.dist_weight * dl
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            v = critic(_to_float(ro.full[dec], dev))
            cl = ((v - _to_float(ro.ret[dec], dev)) ** 2).mean()
            copt.zero_grad()
            cl.backward()
            copt.step()
            sums["pg"] += pg.item()
            sums["kl"] += kl.item()
            sums["entropy"] += ent.item()
            sums["dist"] += dl.item()
            sums["critic"] += cl.item()
            sums["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            n += 1
    return stats | {k: v / max(n, 1) for k, v in sums.items()}


def load_v2(path: str | Path, device: str) -> tuple[QNetV2, dict[str, Any]]:
    """A network v2 checkpoint on ``device``, in eval mode."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if ckpt.get("arch") != "qnet_v2":
        raise ValueError(f"{path} is not a network v2 checkpoint")
    net = QNetV2(NetConfigV2(**ckpt["net_config"])).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def _payload(net: QNetV2, cfg: SelfPlayConfig, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "arch": "qnet_v2",
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "net_config": net.config_dict(),
        "train_config": asdict(cfg),
        "info": info,
    }


def _save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(cfg: SelfPlayConfig, log: Any = print) -> Path:
    """Runs duplicate self-play until ``max_minutes`` (this run) or ``until_minutes`` (all
    runs); resumes from ``run_dir/latest.pt`` when it exists. Returns the run folder."""
    cfg = replace(cfg, device=resolve_device(cfg.device))
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dev = cfg.device
    start, _ = load_v2(cfg.init, dev)
    for p in start.parameters():
        p.requires_grad_(False)
    net, _ = load_v2(cfg.init, dev)
    critic = OracleCritic().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.0)
    copt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    info: dict[str, Any] = {"iter": 0, "games": 0, "decisions": 0, "minutes_total": 0.0}
    snapshots: list[str] = []
    latest = run / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["optimizer"])
        copt.load_state_dict(ck["critic_optimizer"])
        info = ck["info"]
        snapshots = list(ck.get("snapshots", []))
    pool = [start] + [load_v2(s, dev)[0] for s in snapshots[-cfg.pool_size :]]
    rng = np.random.default_rng([cfg.seed, info["iter"]])
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1_000_003 + info["iter"])
    t0 = time.time()
    base_total = info["minutes_total"]
    last_save = 0.0

    def save(path: Path) -> None:
        ck = _payload(net, cfg, info)
        ck |= {
            "critic": critic.state_dict(),
            "optimizer": opt.state_dict(),
            "critic_optimizer": copt.state_dict(),
            "snapshots": snapshots,
        }
        _save(ck, path)

    with (run / "log.jsonl").open("a") as logf:
        while True:
            minutes = (time.time() - t0) / 60
            if cfg.max_minutes and minutes >= cfg.max_minutes:
                break
            if cfg.until_minutes and info["minutes_total"] >= cfg.until_minutes:
                break
            tc = time.time()
            ro = collect(net, start, pool, cfg, rng, gen)
            collect_s = time.time() - tc
            stats = update(net, critic, opt, copt, ro, cfg, rng)
            info["iter"] += 1
            info["games"] += ro.games
            info["decisions"] += len(ro.states)
            info["minutes_total"] = base_total + (time.time() - t0) / 60
            rec = {
                "iter": info["iter"],
                "minutes_total": round(info["minutes_total"], 3),
                "games": info["games"],
                "decisions": len(ro.states),
                "frozen_games": ro.frozen_games,
                "collect_s": round(collect_s, 2),
                "update_s": round(time.time() - tc - collect_s, 2),
                **{k: round(v, 5) for k, v in stats.items()},
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(rec)
            hour = int(info["minutes_total"] // cfg.snapshot_minutes)
            if hour > len(snapshots):
                path = run / f"hour_{hour:02d}.pt"
                _save(_payload(net, cfg, dict(info)), path)
                snapshots.append(str(path))
                frozen, _ = load_v2(path, dev)
                pool = [start] + pool[1:] + [frozen]
                pool = [start] + pool[1:][-cfg.pool_size :]
                save(latest)
                last_save = time.time()
            elif time.time() - last_save > cfg.save_minutes * 60:
                save(latest)
                last_save = time.time()
            progress = {
                **{k: rec[k] for k in ("iter", "minutes_total", "games")},
                "kl": rec["kl"],
                "until_minutes": cfg.until_minutes,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (run / "progress.json").write_text(json.dumps(progress, indent=2))
    save(latest)
    return run
