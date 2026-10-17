# Phase 3 results — learning vs search

**Date:** 2026-10-17 · **Code:** `6831c06` · **Machine:** Apple M1 (8 cores, 8 GB), CPU only
**Reproduce:** training commands in [rl.md](../rl.md);
`python experiments/phase3_rl.py round-robin --deals 100 --workers 7` and
`python experiments/phase3_rl.py agreement --positions 300` (raw JSON in `runs/phase3/`).
**Rules:** the defaults of the time, i.e. last-card duty **off** (the default is on since 2026-10-18).

> **Update:** after 10 more hours of training, the DMC and distilled networks tie
> the solver; see [long-training.md](long-training.md).

## Summary

1. **Self-play learning (DMC, PPO) did not reach the solver's strength in 1–1.5
   hours.** The best self-play network loses about 2 points per game to the
   ISMCTS solver head-to-head. Both learned quickly at first and then improved
   slowly.
2. **Distilling the solver into the network worked best.** Trained on 730,000
   solver-labelled decisions (1 hour of data generation), the network nearly
   matches its teacher: the solver is ahead by only +0.5 ± 1.0 points per game,
   at **1.1 ms per move instead of 14 ms**. It picks the solver's preferred move
   70% of the time, the same as the solver itself at a lower budget.
3. **Network + search ties the solver** (−0.2 ± 0.9 head-to-head) and beats every
   other agent, but takes ~190 ms per move.
4. **Fine-tuning the distilled network with self-play made it worse** (it drifted
   from +5 to about +3 vs greedy and lost to the network it started from).
5. **"Beats the greedy bot by more" ≠ "stronger".** The distilled network scores
   +6.9 against greedy bots versus +3.5 for the DMC network, yet head-to-head the
   gap is smaller (+1.4 ± 1.3). Agents that learned from the solver inherit its
   built-in model of opponents as greedy bots, which is worth extra points against
   actual greedy bots specifically. Head-to-head play is the honest test.

## 1. Training runs

Evaluation every ~30 s: the greedy network in one seat vs three greedy bots, 256
duplicate deals (±0.7–0.9). Values are the latest evaluation at each time.

| Run | 5 min | 15 min | 30 min | 45 min | 60 min | 90 min | Data |
|---|---:|---:|---:|---:|---:|---:|---|
| DMC, lr 3e-4 | +2.0 | +2.5 | +2.6 | +3.1 | +3.6 | +4.0 | 811k self-play games |
| DMC, lr 1e-3 | +2.1 | +2.8 | +2.9 | +2.9 | +3.1 | +2.9 | 739k self-play games |
| PPO | +1.9 | +2.1 | +2.6 | +3.0 | +3.1 | — | 374k self-play games |
| Distillation | +1.8 | +4.4 | +4.5 | +4.7 | +5.5 | +6.5 (75 min) | 600k solver decisions (of 730k generated) |
| Distilled → DMC fine-tune | +3.2 | +2.6 | +3.0 | +3.6 | +3.0 | — | 647k self-play games |

Notes:

* Up to four jobs shared the CPU, so minutes are not strictly comparable across
  runs; game counts are.
* The distillation trainer crashed after 33 minutes (it read a data file while
  it was being written — since fixed) and was restarted from its last
  checkpoint; the table joins both parts.
* For the higher learning rate, progress stalled after ~15 minutes and became
  noisier: 3e-4 is the better choice.

## 2. Round-robin (head-to-head, 100 duplicate deals per match)

Each match puts two agents in alternating seats (A B A B), every deal played four
times with seats rotated. Cell = the row agent's mean score per game against the
column agent (95% CI about ±1.0–1.5).

|  | ISMCTS | Dist.+search | PIMC | Distilled | DMC | Dist.+DMC | PPO |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ISMCTS-1000** | — | −0.18 | +1.10 | +0.54 | +2.22 | +1.97 | +2.76 |
| **Distilled + search** | +0.18 | — | +1.01 | +1.78 | +2.23 | +0.93 | +1.70 |
| **PIMC-40** | −1.10 | −1.01 | — | +0.38 | +2.04 | +1.70 | +2.54 |
| **Distilled** | −0.54 | −1.78 | −0.38 | — | +1.44 | +1.51 | +0.79 |
| **DMC** | −2.22 | −2.23 | −2.04 | −1.44 | — | +0.59 | +0.94 |
| **Distilled + DMC** | −1.97 | −0.93 | −1.70 | −1.51 | −0.59 | — | +0.17 |
| **PPO** | −2.76 | −1.70 | −2.54 | −0.79 | −0.94 | −0.17 | — |

### Ranking

| Rank | Agent | Mean vs the field | vs 3 greedy bots | ms / move |
|---:|---|---:|---:|---:|
| 1 | ISMCTS, 1,000 simulations | +1.40 | +8.55 ± 1.70 | 13.9 |
| 2 | Distilled network + rollout search | +1.31 | +8.34 ± 1.59 | 188 |
| 3 | PIMC, 40 sampled deals | +0.76 | +8.14 ± 1.48 | 1.9 |
| 4 | Distilled network | +0.17 | +6.88 ± 1.58 | 1.1 |
| 5 | DMC self-play network | −1.07 | +3.52 ± 1.50 | 1.1 |
| 6 | Distilled → DMC fine-tuned | −1.09 | +3.28 ± 1.52 | 1.1 |
| 7 | PPO self-play network | −1.49 | +3.65 ± 1.50 | 1.0 |

(ms / move measured while 7 games ran in parallel; single-game times are lower.)

## 3. Agreement with a high-effort solver (300 positions)

Positions sampled from greedy-bot games (decisions with at least two legal moves).
Reference move: ISMCTS with 20,000 simulations. "Cost" = how much worse the
chosen move is than the best, in expected points, by PIMC with 400 sampled deals
(every legal move evaluated on the same deals).

| Agent | Same move as reference | Mean cost (points) | ms / move |
|---|---:|---:|---:|
| Distilled network | **70.0%** | 0.37 | 0.3 |
| ISMCTS, 1,000 | 69.7% | 0.29 | 8.6 |
| PIMC, 40 | 60.7% | **0.23** | 1.6 |
| Distilled + search | 59.0% | 0.68 | 54 |
| greedy bot | 56.7% | 1.00 | <0.1 |
| PPO | 54.7% | 0.85 | 0.3 |
| Distilled + DMC | 54.7% | 0.75 | 0.3 |
| DMC | 54.3% | 0.74 | 0.5 |

This measure is **not neutral**: both the reference and the cost estimate play
out games with greedy-bot rollouts, so they reward agents that share that model
of the opponents. Distilled + search disagrees with them more often yet ties
ISMCTS in actual play; it simply assumes the opponents play like the network.

## Interpretation

* **Why self-play was slow.** DMC learns from one noisy number per decision (the
  final score of one game, standard deviation ~20 points); its training loss
  stayed ~20× higher than distillation's. DouZero trained on billions of
  decisions across many GPUs; 1.5 CPU-hours is far too little for self-play to
  catch up with a search that recomputes every position from scratch.
  PPO had the same problem plus higher variance from sampling its own moves.
* **Why distillation worked.** Each solver-labelled decision provides values for
  several moves, each averaged over hundreds of simulated games: a far cleaner
  signal. The network becomes a fast copy of the solver — including its blind
  spots (the greedy-opponent model).
* **Why fine-tuning hurt.** Self-play targets reflect how *the network* plays
  from each position, which (early on) is worse than how the solver plays. The
  values regress toward those weaker targets faster than self-play can improve
  the policy. Longer training, a lower learning rate, or mixing solver and
  self-play data would be needed.
* **Search + network.** Replacing the greedy rollouts with the network made the
  search as strong as ISMCTS but not stronger, at ~13× the time. The network's
  own model of the opponents is not better enough yet to pay for itself.

## What would likely help next

1. **Iterate distillation** (true expert iteration): use network + search as the
   expert, generate new data, distill again. Each round's expert models
   opponents with the latest network rather than the greedy bot.
2. **Opponent modelling in sampling:** weight sampled deals by how likely each
   opponent's past actions are under the network (Phase 1 used uniform deals).
3. **Longer self-play** from the distilled network with a mix of solver-labelled
   and self-play data.
4. **Evaluate against humans** — the web UI logs every game in `runs/games/`.
