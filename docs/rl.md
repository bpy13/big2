# Reinforcement learning

Phase 3 trains a neural network to play Big Two by self-play and compares it
with the Phase 1 search solver. Results are in
[study-results/phase3-rl.md](study-results/phase3-rl.md).

## 1. Search vs learning

| | Search solver (Phase 1) | RL network (Phase 3) |
|---|---|---|
| When it thinks | At the table, every move | Beforehand, during training |
| Time per move | 15 ms – 0.2 s | ~1 ms |
| Hidden cards | Uniform random guesses | Learns what the history implies |
| Model of opponents | Greedy rollout bot | Itself (trained against copies of itself) |
| Uncertainty | 95% intervals from simulations | None built in |

The network does not replace search; the most successful game AIs (AlphaZero,
ReBeL, Student of Games) combine both. Section 5 does a simple version.

## 2. The method: Deep Monte Carlo (DouZero)

DouZero (Zha et al., ICML 2021) was the first program to beat all earlier AIs at
Dou Dizhu, a Chinese shedding game close to Big Two, with a strikingly simple
recipe that suits huge action spaces (every legal combination is an action):

1. **Play** many games of the current network against itself. At each decision,
   the network scores every legal move; play the best (with probability
   ε = 0.05, a random legal move instead, to keep exploring).
2. **Label** every (position, move) pair with the final score of the player who
   made it. This *Monte Carlo return* needs no value bootstrapping.
3. **Fit** the network to those labels (mean squared error), then repeat.

The network therefore learns `Q(position, move)` = the expected final score if
you play `move` here and everyone continues as the current network would.

Two additions here:

* A second output head predicts the **probability of going out first**
  (cross-entropy loss), so study mode can show win rates as the solver does.
* Training samples from the last 4 iterations are kept (a small replay window)
  to smooth updates.

Implementation: `python/poker/rl/dmc.py`; one iteration plays 1,024 games
(~45,000 decisions), then trains one epoch over the replay window.

## 3. What the network sees

Features are 0/1 values computed in Rust (`crates/big2/src/features.rs`), from the
acting player's viewpoint, with seats relative to them (next, across, previous):

* your hand; the cards you have not seen; the cards each seat has played;
* each opponent's card count;
* the play to beat and who made it; whether you lead; passes so far this trick;
* the last 12 actions (cards and who played or passed).

Each candidate move is encoded as its cards, its kind, and the hand you would have
left after it. A test checks that two positions differing only in hidden cards
encode identically — the network cannot cheat.

## 4. Network

```
state (1,103) ─ 512 ─ 256 ─┐
                            ├─ 256 ─ 128 ─ 2   (score, win logit)
move (61 + 52) ─ 128 ──────┘
```

The state tower runs once per decision; only the small head runs per legal move,
so scoring 100 candidates costs little more than scoring one. ~0.8 M parameters.

## 5. Network + search

`python/poker/rl/search.py` combines the network with the Phase 1 recipe:

1. The network ranks the legal moves; keep the best 6.
2. Sample 32 deals of the hidden cards.
3. For each deal and candidate, play the game to the end with **the network
   playing every seat** (instead of the greedy bot).
4. Pick the candidate with the best average result.

All 192 playouts of a decision run in lockstep, one batched network call per step.

## 6. Commands

```sh
poker rl train --run-dir runs/dmc/v1 --minutes 90         # train (logs + checkpoints in run dir)
poker rl train --resume runs/dmc/v1/latest.pt ...          # continue
poker big2 tournament dmc:checkpoint=models/big2-dmc.pt ismcts:iterations=1000 \
    dmc:checkpoint=models/big2-dmc.pt ismcts:iterations=1000 --deals 100 --workers 6
python experiments/phase3_rl.py --checkpoint models/big2-dmc.pt
```

Agent specs: `dmc:checkpoint=PATH[,objective=win]` (network alone) and
`dmcsearch:checkpoint=PATH[,determinizations=32,top=6]` (network + search).

The web UI offers both as opponents ("RL network", "RL network + search") and
shows the network's pick next to the solver's in study mode, when
`models/big2-dmc.pt` exists (override with `POKER_RL_CHECKPOINT`). The network
was trained under the default rules, so these options are disabled for other
rule variants.

## 7. Limits

* **Compute.** DouZero trained for days on four GPUs with dozens of actor
  processes. This project trains for about 1.5 hours on a laptop CPU, i.e. a few
  million games instead of billions of decisions. Results show where the curve
  is at that budget, not how far it can go.
* **Self-play only.** The network learns to beat copies of itself; nothing
  guarantees it is hard to exploit by a different style.
* **One rule set.** A change of rules needs a new network.

## References

* Zha et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
  Learning.* ICML 2021.
* Yang et al. *PerfectDou: Dominating DouDizhu with Perfect Information
  Distillation.* NeurIPS 2022.
* Charlesworth. *Application of Self-Play Reinforcement Learning to a Four-Player
  Game of Imperfect Information* (Big Two, PPO). arXiv:1808.10442, 2018.
* Tesauro, Galperin. *On-line Policy Improvement using Monte-Carlo Search.*
  NeurIPS 1996.
