"""Search distillation (expert iteration): train the Q-network on the solver's estimates.

Self-play DMC learns from one noisy number per decision — the final score of one
game. The Phase 1 solver, at every decision, estimates the value of many moves
from hundreds of simulated games each. Distillation turns those estimates into
training targets:

1. **Generate** (``generate``): worker processes play games in which every
   decision is analyzed by ISMCTS. For each decision we store the state features
   and every legal move (labelled by ``plan.candidates``) with the search's mean
   final score, win rate and visit count, plus the deal's outcome: every seat's
   cards left and score relative to the actor, and who led the next trick. The
   search's best move is then played (with probability ``epsilon`` a random one
   among its top 3, for variety).
2. **Train** (``train``): regress the network's score and win heads toward those
   targets, weighting each move by its visit count (more simulations = less noise;
   moves under ``min_visits`` simulations get no weight).

Shard format 2 (``generate``), one ``.npz`` per shard. Per decision (``D`` rows):
``states`` [D, STATE_LEN] u8; ``player`` [D] i8 (absolute seat);
``final_cards`` [D, 4] i8 and ``final_scores`` [D, 4] f32 at the deal's end, seats
relative to the actor (0 = actor, 1 = next, 2 = across, 3 = previous);
``next_leader`` [D] i8, the relative seat that led the trick after this decision's
trick (-1: the deal ended first); ``chosen`` [D] i16, the played move's index among
the decision's candidates. Per candidate (``N`` rows, grouped by decision):
``owner`` [N] i32 (decision row, ascending); ``actions`` [N, ACTION_LEN] u8;
``cards`` [N] u64; ``keeps`` [N] i8 (decomposition kept whole, -1 none);
``breaks`` [N] u64 (group broken, 0 none); ``visits`` [N] i32, ``score`` [N] f32,
``win`` [N] f32 (all 0 for a move the search did not try). Plus ``format`` = 2.

The result plays like the solver at a fraction of the cost, and can then be
improved further with self-play (``big2 rl train --init ...``).

This is the "expert iteration" idea (Anthony et al., 2017) that AlphaZero also
uses, with ISMCTS as the expert.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from big2._core import analyze, candidates, encode
from big2.games.big2 import Rules
from big2.rl.device import actor_device, resolve_device
from big2.rl.dmc import (
    TrainConfig,
    _to_float,
    best_eval_so_far,
    evaluate_vs_greedy,
    save_checkpoint,
)
from big2.rl.model import (
    ACTION_LEN,
    MAX_CARDS,
    SCORE_SCALE,
    STATE_LEN,
    NetConfig,
    NetConfigV2,
    QNet,
    QNetV2,
    V2Scorer,
    seat_scores,
    segment_log_softmax,
)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


#: Shard layout version written by :func:`generate` (stored as ``format``).
#: Version 1 (no ``format`` array) kept only moves with ``visits >= 20``.
SHARD_FORMAT = 2


def _rel(seat: int, actor: int) -> int:
    """Seat relative to ``actor``: 0 = actor, 1 = next, 2 = across, 3 = previous
    (the order of ``crates/big2/src/features.rs``)."""
    return (seat - actor) % 4


def _generate_worker(
    out_dir: str,
    worker: int,
    minutes: float,
    iterations: int,
    epsilon: float,
    shard_games: int,
    rules: Rules,
    rollout: str,
    stop: Any = None,
) -> int:
    rng = random.Random(1_000_003 * worker + int(time.time()))
    deadline = time.time() + 60 * minutes
    # A per-run prefix, so a new run adds shards instead of overwriting earlier ones.
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    shard = 0
    decisions = 0

    def stopping() -> bool:
        return time.time() >= deadline or (stop is not None and stop.is_set())

    while shard == 0 or not stopping():  # always at least one shard
        # Per decision.
        states: list[bytes] = []
        player: list[int] = []
        final_cards: list[list[int]] = []
        final_scores: list[list[float]] = []
        next_leader: list[int] = []
        chosen: list[int] = []
        # Per candidate (grouped by decision, ``owner`` ascending).
        actions: list[bytes] = []
        owner: list[int] = []
        cards: list[int] = []
        keeps: list[int] = []
        breaks: list[int] = []
        score: list[float] = []
        win: list[float] = []
        visits: list[int] = []
        for _ in range(shard_games):
            s = rules.new_deal(rng.getrandbits(32))
            first = len(states)  # this game's first decision row
            pending: list[int] = []  # decisions whose trick has not ended yet
            while not s.is_terminal:
                legal = s.legal_actions()
                if len(legal) == 1:
                    s.apply(legal[0])
                else:
                    actor = s.current_player
                    assert actor is not None
                    res = analyze(
                        s,
                        iterations=iterations,
                        threads=1,
                        seed=rng.getrandbits(32),
                        rollout=rollout,
                    )
                    st, ac, order = encode(s)
                    pos = {c: k for k, c in enumerate(order)}
                    stats = {m["cards"]: m for m in res["moves"]}
                    idx = len(states)
                    states.append(st)
                    player.append(actor)
                    top = res["moves"][: min(3, len(res["moves"]))]
                    pick = rng.choice(top) if rng.random() < epsilon else top[0]
                    # Every legal action, in the planner's order, with the search's
                    # statistics (zero for a move the search never tried).
                    for j, (c, keep, brk) in enumerate(candidates(s)):
                        k = pos[c]
                        m = stats.get(c)
                        actions.append(ac[k * ACTION_LEN : (k + 1) * ACTION_LEN])
                        owner.append(idx)
                        cards.append(c)
                        keeps.append(keep)
                        breaks.append(brk)
                        score.append(m["mean_score"] if m else 0.0)
                        win.append(m["win_rate"] if m else 0.0)
                        visits.append(m["visits"] if m else 0)
                        if c == pick["cards"]:
                            chosen.append(j)
                    pending.append(idx)
                    decisions += 1
                    s.apply(pick["cards"])
                # A new trick starts: its leader is the "next leader" of every
                # decision taken in the trick that just ended.
                cp = s.current_play
                if not s.is_terminal and cp is None and pending:
                    leader = s.current_player
                    assert leader is not None
                    for d in pending:
                        next_leader.append(_rel(leader, player[d]))
                    pending.clear()
            next_leader.extend(-1 for _ in pending)  # the deal ended within that trick
            assert len(next_leader) == len(states) == len(chosen)
            left, scores = s.hand_sizes, s.scores
            assert scores is not None
            for d in range(first, len(states)):
                a = player[d]
                final_cards.append([left[(a + r) % 4] for r in range(4)])
                final_scores.append([scores[(a + r) % 4] for r in range(4)])
        if not states:
            break
        # Write under a temporary name, then rename: readers never see a partial shard.
        final = Path(out_dir) / f"{run_tag}_w{worker:02d}_{shard:05d}.npz"
        tmp = final.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            format=np.int32(SHARD_FORMAT),
            states=np.frombuffer(b"".join(states), dtype=np.uint8).reshape(-1, STATE_LEN),
            player=np.asarray(player, dtype=np.int8),
            final_cards=np.asarray(final_cards, dtype=np.int8).reshape(-1, 4),
            final_scores=np.asarray(final_scores, dtype=np.float32).reshape(-1, 4),
            next_leader=np.asarray(next_leader, dtype=np.int8),
            chosen=np.asarray(chosen, dtype=np.int16),
            actions=np.frombuffer(b"".join(actions), dtype=np.uint8).reshape(-1, ACTION_LEN),
            owner=np.asarray(owner, dtype=np.int32),
            cards=np.asarray(cards, dtype=np.uint64),
            keeps=np.asarray(keeps, dtype=np.int8),
            breaks=np.asarray(breaks, dtype=np.uint64),
            score=np.asarray(score, dtype=np.float32),
            win=np.asarray(win, dtype=np.float32),
            visits=np.asarray(visits, dtype=np.int32),
        )
        tmp.replace(final)
        shard += 1
    return decisions


def _shard_counts(path: Path) -> tuple[int, int]:
    """``(decisions, candidate rows)`` of one shard, reading only small arrays."""
    with np.load(path) as z:
        rows = len(z["owner"])
        n = len(z["player"]) if "player" in z.files else len(z["states"])
    return n, rows


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def generate(
    out_dir: str,
    workers: int = 4,
    minutes: float = 60.0,
    iterations: int = 1000,
    epsilon: float = 0.25,
    shard_games: int = 20,
    rules: Rules | None = None,
    rollout: str = "greedy",
    target_decisions: int | None = None,
    progress_dir: str | None = None,
    progress_every: float = 60.0,
) -> int:
    """Generates solver-labelled decisions into ``out_dir`` (one .npz per shard),
    from games under ``rules`` (default: the default rules), searched by ISMCTS with
    ``iterations`` simulations and the given ``rollout`` policy.

    Every searched decision (two or more legal moves) is stored with all its legal
    moves as ``plan.candidates`` orders and labels them (shard format 2, see
    the module docstring), each with the search's visits, mean score and win rate.

    Stops after ``minutes`` or once ``out_dir`` holds ``target_decisions`` decisions
    (counting shards of earlier runs, so rerunning the same command resumes).
    Writes ``progress.json`` every ``progress_every`` seconds and ``results.json`` at
    the end into ``progress_dir`` (default: the parent of ``out_dir``).
    Returns the decisions this call generated.
    """
    rules = rules or Rules()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(progress_dir) if progress_dir else out.parent
    pdir.mkdir(parents=True, exist_ok=True)
    counted: dict[str, tuple[int, int]] = {}
    t0 = time.time()
    start_decisions = -1
    config = {
        "out_dir": str(out),
        "workers": workers,
        "minutes": minutes,
        "iterations": iterations,
        "rollout": rollout,
        "epsilon": epsilon,
        "rules": rules.kwargs(),
        "target_decisions": target_decisions,
        "shard_format": SHARD_FORMAT,
    }

    def report(final: bool) -> int:
        nonlocal start_decisions
        for p in out.glob("*.npz"):
            if p.name.endswith(".tmp.npz") or p.name in counted:
                continue
            try:
                counted[p.name] = _shard_counts(p)
            except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                continue  # retried at the next report
        total = sum(n for n, _ in counted.values())
        if start_decisions < 0:
            start_decisions = total
        elapsed = (time.time() - t0) / 60
        rate = (total - start_decisions) / elapsed if elapsed > 0 else 0.0
        state = {
            "status": "done" if final else "running",
            "decisions": total,
            "candidate_rows": sum(r for _, r in counted.values()),
            "shards": len(counted),
            "decisions_this_run": total - start_decisions,
            "minutes_this_run": round(elapsed, 1),
            "decisions_per_minute": round(rate),
            "minutes_left_to_target": (
                round((target_decisions - total) / rate)
                if target_decisions and rate > 0 and total < target_decisions
                else None
            ),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "next": "nothing" if final else "more shards",
            "config": config,
        }
        _write_json(pdir / ("results.json" if final else "progress.json"), state)
        if final:
            _write_json(pdir / "progress.json", state)
        return total

    # The first report sets the baseline (shards of earlier runs) for the rate.
    if report(False) >= (target_decisions if target_decisions is not None else math.inf):
        report(True)
        return 0
    with mp.Manager() as manager, ProcessPoolExecutor(workers) as pool:
        stop = manager.Event()
        futures = [
            pool.submit(
                _generate_worker,
                out_dir,
                w,
                minutes,
                iterations,
                epsilon,
                shard_games,
                rules,
                rollout,
                stop,
            )
            for w in range(workers)
        ]
        pending = set(futures)
        while pending:
            _, pending = wait(pending, timeout=progress_every)
            total = report(False)
            if target_decisions is not None and total >= target_decisions:
                stop.set()
        made = sum(f.result() for f in futures)
    report(True)
    return made


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
    #: Moves simulated fewer times get no weight (format-2 shards hold every legal move).
    min_visits: int = 20
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
            vs = np.where(vs >= cfg.min_visits, vs, 0)
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


# ---------------------------------------------------------------------------
# Network v2 training (plan Task 12): distributions, all seats, prior
# ---------------------------------------------------------------------------


@dataclass
class DistillV2Config:
    """Settings of :func:`train_v2`. Loss weights default to 1 : 1 : 1 for the three
    heads (open question 8 of the build plan), plus ``score_weight`` for the
    auxiliary mean-score loss (see :func:`train_v2`)."""

    data_dir: str = "runs/round0/data"
    run_dir: str = "runs/net_r0"
    # Regularized after the first round-0 run (lr 1e-3, weight decay 1e-4, no dropout),
    # whose distribution heads overfitted within 10 minutes (~7 passes over the data;
    # the ~45 decisions of a deal share one outcome).
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    batch_size: int = 2048  # decisions per minibatch
    #: Every ``val_every``-th shard (by sorted name) is held out for validation.
    val_every: int = 50
    actor_weight: float = 1.0  # the actor's cards-left distribution
    seats_weight: float = 1.0  # the three opponents' distributions
    prior_weight: float = 1.0  # the policy logit vs the search's visit shares
    #: Mean score from the distributions vs the search's mean score for every
    #: candidate it simulated (weight min(visits, cap) / cap; 0 under ``min_visits``).
    score_weight: float = 0.5
    visit_cap: int = 200
    min_visits: int = 20
    eval_every_minutes: float = 5.0
    eval_deals: int = 128  # per evaluation mode (prior, score) vs 3 greedy
    #: Stop after this many evaluations without a better validation loss (0: never).
    patience: int = 6
    max_minutes: float = 180.0  # total over resumes
    resume: str | None = None
    seed: int = 0
    device: str = "auto"
    rules: dict[str, Any] = field(default_factory=dict)
    net: NetConfigV2 = field(default_factory=lambda: NetConfigV2(dropout=0.2))


class _DataV2:
    """Every shard of a folder in contiguous arrays (moved to the training device).

    Per decision: ``states`` [D, STATE_LEN] u8, ``lo``/``count`` (its candidate rows),
    ``chosen`` (the played candidate's global row), ``final_cards`` [D, 4]. Per
    candidate: ``actions`` [N, ACTION_LEN] u8, ``visits``, ``score``. ``val`` marks
    the decisions of held-out shards.
    """

    def __init__(
        self, data_dir: str, val_every: int, log: Any = print, files: list[Path] | None = None
    ) -> None:
        """``files`` (default: every shard of ``data_dir``) picks the shards to load;
        ``val_every`` then counts within that list."""
        if files is None:
            files = sorted(
                p for p in Path(data_dir).glob("*.npz") if not p.name.endswith(".tmp.npz")
            )
        if not files:
            raise RuntimeError(f"no shards in {data_dir}")
        counts = [_shard_counts(p) for p in files]
        d_total, n_total = sum(c[0] for c in counts), sum(c[1] for c in counts)
        log(f"loading {len(files)} shards: {d_total:,} decisions, {n_total:,} candidates")
        self.states = np.empty((d_total, STATE_LEN), dtype=np.uint8)
        self.final_cards = np.empty((d_total, 4), dtype=np.int64)
        self.lo = np.empty(d_total, dtype=np.int64)
        self.count = np.empty(d_total, dtype=np.int64)
        self.chosen = np.empty(d_total, dtype=np.int64)
        self.val = np.zeros(d_total, dtype=bool)
        self.actions = np.empty((n_total, ACTION_LEN), dtype=np.uint8)
        self.visits = np.empty(n_total, dtype=np.float32)
        self.score = np.empty(n_total, dtype=np.float32)
        d0 = n0 = 0
        for k, p in enumerate(files):
            with np.load(p) as z:
                shard_format = int(z["format"]) if "format" in z.files else 1
                if shard_format != SHARD_FORMAT:
                    raise ValueError(f"{p.name}: v2 training needs shard format {SHARD_FORMAT}")
                d, n = len(z["states"]), len(z["owner"])
                lo = np.searchsorted(z["owner"], np.arange(d), side="left")
                hi = np.searchsorted(z["owner"], np.arange(d), side="right")
                self.states[d0 : d0 + d] = z["states"]
                self.final_cards[d0 : d0 + d] = z["final_cards"]
                self.lo[d0 : d0 + d] = n0 + lo
                self.count[d0 : d0 + d] = hi - lo
                self.chosen[d0 : d0 + d] = n0 + lo + z["chosen"].astype(np.int64)
                self.val[d0 : d0 + d] = k % val_every == 0
                self.actions[n0 : n0 + n] = z["actions"]
                self.visits[n0 : n0 + n] = z["visits"]
                self.score[n0 : n0 + n] = z["score"]
            d0, n0 = d0 + d, n0 + n
        assert (d0, n0) == (d_total, n_total)
        assert (self.count >= 1).all() and (self.chosen < self.lo + self.count).all()

    def to(self, device: str) -> dict[str, torch.Tensor]:
        """The arrays as tensors on ``device`` (features stay uint8 until a batch)."""
        names = ("states", "final_cards", "lo", "count", "chosen", "actions", "visits", "score")
        t = {k: torch.from_numpy(getattr(self, k)).to(device) for k in names}
        val = torch.from_numpy(self.val)
        t["train_ids"] = torch.nonzero(~val).squeeze(1).to(device)
        t["val_ids"] = torch.nonzero(val).squeeze(1).to(device)
        return t


def _v2_batch(t: dict[str, torch.Tensor], d: torch.Tensor) -> dict[str, torch.Tensor]:
    """The decisions ``d`` (global ids) with all their candidates, as float inputs."""
    cnt = t["count"][d]
    starts = torch.cumsum(cnt, 0) - cnt
    index = torch.repeat_interleave(torch.arange(len(d), device=d.device), cnt)
    rows = t["lo"][d][index] + torch.arange(len(index), device=d.device) - starts[index]
    return {
        "states": t["states"][d].float(),
        "actions": t["actions"][rows].float(),
        "index": index,
        "chosen": starts + (t["chosen"][d] - t["lo"][d]),
        "final_cards": t["final_cards"][d],
        "visits": t["visits"][rows],
        "score": t["score"][rows],
    }


def v2_losses(
    net: QNetV2, b: dict[str, torch.Tensor], cfg: DistillV2Config
) -> dict[str, torch.Tensor]:
    """Per-head losses of one batch (see :func:`train_v2`), plus ``total`` and the
    prior's top-1 agreement with the search's most-visited move (``prior_top1``)."""
    groups = len(b["states"])
    dist, prior = net(b["states"], b["actions"], b["index"])
    played = dist[b["chosen"]]  # [B, 4, 14]
    cards = b["final_cards"].clamp(0, MAX_CARDS)
    actor = nn.functional.cross_entropy(played[:, 0], cards[:, 0])
    seats = nn.functional.cross_entropy(
        played[:, 1:].reshape(-1, MAX_CARDS + 1), cards[:, 1:].reshape(-1)
    )
    # Prior: cross-entropy against the visit shares within each decision.
    logp = segment_log_softmax(prior, b["index"], groups)
    vis = b["visits"]
    total_vis = torch.zeros(groups, device=vis.device).index_add(0, b["index"], vis)
    share = vis / total_vis.clamp(min=1.0)[b["index"]]
    prior_ce = -(share * logp).sum() / groups
    # Auxiliary: the tiered mean score from the distributions vs the search's mean score.
    w = torch.where(vis >= cfg.min_visits, vis.clamp(max=cfg.visit_cap) / cfg.visit_cap, 0.0)
    pred = seat_scores(dist.softmax(-1), "tiered")[:, 0]
    score_mse = (w * ((pred - b["score"]) / SCORE_SCALE) ** 2).sum() / w.sum().clamp(min=1e-6)
    total = (
        cfg.actor_weight * actor
        + cfg.seats_weight * seats
        + cfg.prior_weight * prior_ce
        + cfg.score_weight * score_mse
    )
    with torch.no_grad():
        m = torch.full((groups,), -torch.inf, device=vis.device)
        best_vis = m.scatter_reduce(0, b["index"], vis, reduce="amax")
        best_pri = m.scatter_reduce(0, b["index"], prior, reduce="amax")
        hit = (vis == best_vis[b["index"]]) & (prior == best_pri[b["index"]])
        top1 = torch.zeros(groups, device=vis.device).index_add(0, b["index"], hit.float())
    return {
        "total": total,
        "actor": actor,
        "seats": seats,
        "prior": prior_ce,
        "score": score_mse,
        "prior_top1": (top1 > 0).float().mean(),
    }


@torch.no_grad()
def _validate(
    net: QNetV2, t: dict[str, torch.Tensor], cfg: DistillV2Config, batch: int = 4096
) -> dict[str, float]:
    """Mean losses over the held-out decisions, weighted by batch size."""
    net.eval()
    ids = t["val_ids"]
    sums: dict[str, float] = {}
    for i in range(0, len(ids), batch):
        d = ids[i : i + batch]
        out = v2_losses(net, _v2_batch(t, d), cfg)
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(d)
    net.train()
    res = {f"val_{k}": v / len(ids) for k, v in sums.items()}
    res["val_score_rms_points"] = math.sqrt(res["val_score"]) * SCORE_SCALE
    return res


def _save_v2(
    path: Path,
    net: QNetV2,
    cfg: DistillV2Config,
    info: dict[str, Any],
    opt: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "arch": "qnet_v2",
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "net_config": net.config_dict(),
        "train_config": asdict(cfg),
        "info": info,
    }
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_qnet_v2(checkpoint: str | Path, device: str = "cpu") -> tuple[QNetV2, dict[str, Any]]:
    """Loads a :func:`train_v2` checkpoint; returns ``(network in eval mode, checkpoint)``."""
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if ckpt.get("arch") != "qnet_v2":
        raise ValueError(f"{checkpoint} is not a v2 network checkpoint")
    net = QNetV2(NetConfigV2(**ckpt["net_config"])).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def train_v2(cfg: DistillV2Config, log: Any = print) -> Path:
    """Trains :class:`QNetV2` on format-2 shards (Task 11's round-0 data).

    Losses per decision: cross-entropy of the actor's cards-left distribution
    (``actor``) and of the three opponents' (``seats``) for the move actually played
    (the shards hold the real deal's outcome for that move only), cross-entropy of
    the prior against the search's visit shares (``prior``), and the auxiliary
    ``score`` loss (squared error, in units of :data:`SCORE_SCALE`, between the tiered
    mean score of each candidate's distributions and the search's mean score,
    weighted by visits): it gives the distributions of the moves not played a
    low-noise target.

    Every ``eval_every_minutes``: validation losses on held-out shards, two short
    matches vs 3 greedy (playing by the prior, and by the expected score),
    ``latest.pt``, ``progress.json``; ``best.pt`` when the validation total improves.
    Ends after ``max_minutes`` (counting resumed time) or ``patience`` evaluations
    without improvement, writing ``results.json``.
    """
    cfg = replace(cfg, device=resolve_device(cfg.device))
    dev = cfg.device
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    net = QNetV2(cfg.net).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps, prev_minutes, best, stale = 0, 0.0, math.inf, 0
    history: list[dict[str, Any]] = []
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        info = ckpt["info"]
        steps, prev_minutes = int(info["steps"]), float(info["minutes"])
        best, stale = float(info["best_val_total"]), int(info["stale"])
        history = list(info.get("history", []))
        torch.manual_seed(cfg.seed + steps)
    params = sum(p.numel() for p in net.parameters())
    t_load = time.time()
    data = _DataV2(cfg.data_dir, cfg.val_every, log=log)
    t = data.to(dev)
    sizes = {
        "decisions": len(data.states),
        "candidates": len(data.actions),
        "val_decisions": int(data.val.sum()),
    }
    del data
    log(f"data on {dev} in {time.time() - t_load:.0f} s: {sizes}; {params:,} parameters")
    adev = actor_device(dev)
    actor = QNetV2(cfg.net).to(adev)
    t0 = time.time()

    def minutes() -> float:
        return prev_minutes + (time.time() - t0) / 60

    def lr_at(m: float) -> float:
        warm = min(1.0, (steps + 1) / cfg.warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, m / cfg.max_minutes)))
        return cfg.lr * warm * (0.05 + 0.95 * cos)

    def write_progress(status: str, extra: dict[str, Any]) -> None:
        state = {
            "status": status,
            "steps": steps,
            "minutes": round(minutes(), 1),
            "max_minutes": cfg.max_minutes,
            "best_val_total": best,
            "evals_without_improvement": stale,
            "parameters": params,
            **sizes,
            **extra,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "next": "nothing" if status == "done" else "more steps",
        }
        _write_json(run / ("results.json" if status == "done" else "progress.json"), state)
        if status == "done":
            _write_json(run / "progress.json", state)

    write_progress("running", {"note": "data loaded, training"})
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + steps)
    train_ids = t["train_ids"]
    running: dict[str, list[float]] = {}
    logf = (run / "log.jsonl").open("a")

    def evaluate_and_save() -> None:
        """Validation, matches vs greedy, checkpoints, progress (once per interval)."""
        nonlocal best, stale
        rec: dict[str, Any] = {
            "step": steps,
            "minutes": round(minutes(), 2),
            "lr": lr_at(minutes()),
            **{f"train_{k}": float(np.mean(v)) for k, v in running.items()},
            **_validate(net, t, cfg),
        }
        running.clear()
        actor.load_state_dict(net.state_dict())
        actor.eval()
        for mode in ("prior", "score"):
            ev = evaluate_vs_greedy(
                V2Scorer(actor, mode), cfg.eval_deals, seed=10_000, rules=cfg.rules, device=adev
            )
            rec[f"greedy_{mode}"] = round(ev["eval_score"], 3)
            rec[f"greedy_{mode}_ci95"] = round(ev["eval_score_ci95"], 3)
        improved = rec["val_total"] < best
        if improved:
            best, stale = rec["val_total"], 0
        else:
            stale += 1
        rec["best"] = improved
        history.append(rec)
        info = {
            "steps": steps,
            "minutes": minutes(),
            "best_val_total": best,
            "stale": stale,
            "history": history,
            **rec,
        }
        if improved:
            _save_v2(run / "best.pt", net, cfg, info)
        _save_v2(run / "latest.pt", net, cfg, info, opt)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        log(rec)
        write_progress("running", {"last": rec})

    last_eval = time.time()
    done_reason = "max_minutes"
    trained_since_eval = False
    try:
        while minutes() < cfg.max_minutes:
            for g in opt.param_groups:
                g["lr"] = lr_at(minutes())
            pick = torch.randint(len(train_ids), (cfg.batch_size,), device=dev, generator=gen)
            out = v2_losses(net, _v2_batch(t, train_ids[pick]), cfg)
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(out["total"])
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            steps += 1
            trained_since_eval = True
            if steps % 50 == 1:
                for k, v in out.items():
                    running.setdefault(k, []).append(float(v.detach()))
            if time.time() - last_eval < 60 * cfg.eval_every_minutes:
                continue
            last_eval = time.time()
            evaluate_and_save()
            trained_since_eval = False
            if cfg.patience and stale >= cfg.patience:
                done_reason = f"no better validation loss in {stale} evaluations"
                break
        if trained_since_eval:  # the time ran out between two evaluations
            evaluate_and_save()
    finally:
        logf.close()
    best_rec = min(history, key=lambda r: r["val_total"]) if history else {}
    write_progress(
        "done",
        {
            "stopped": done_reason,
            "best": best_rec,
            "val_loss_per_head": {
                k: best_rec.get(f"val_{k}") for k in ("actor", "seats", "prior", "score")
            },
            "last": history[-1] if history else {},
        },
    )
    return run
