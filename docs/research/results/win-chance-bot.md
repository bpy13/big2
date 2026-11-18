# The win-chance bot against `expert` (plan Task 34)

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, Task 34. Brian (2026-11-11): "train a new model
that only optimizes changes of winning (instead of EV) and have it compete against our best bot".

**Status 2026-11-18: done.** Jobs 193 (labels), 209 (networks) and 210 (match, habits) ended; numbers below are copied from `runs/win_chance/table.md` and `runs/round<N>_win/f_calibration/table.md`.

## Result in one paragraph

The win-chance bot goes out first more often than `expert`, by about 3.4 points of share
(27.5 % vs 24.2 % per seat in `1+3`, 26.8 % vs 23.2 % in `ABAB`; both intervals exclude 0). It pays
heavily for it in points: −19.2 points per deal paired in `1+3` and −18.4 in `ABAB`, since it ends
the deals it loses holding many cards. It plays the way Brian expected, passing almost twice as
often as `expert` (39 % vs 21 % of the turns it could play) and keeping more 2s and five-card
hands when an opponent is down to 3 cards. Under the scoring the app uses (tiered, cards left
cost points) it is far weaker than `expert`, so it does not ship and gets no bot-notes line.

## What changed, and only that

Every bot so far maximises expected points (tiered scoring: cards left cost points). The
win-chance bot runs the same pipeline as today's best network, r3 (`build-phase3.md` §1), with one
switch, `--objective win` in `experiments/expert_iteration.py`:

| Step | r3 (expected points) | win-chance |
|---|---|---|
| Round 0 labels | ISMCTS-1000, 2,740,296 decisions, objective score | same, objective **win** |
| Round 0 network | fresh weights, loss weights 1 : 1 : 1, value 0.5 | same; value fits **P(go out first)** |
| Rounds 1–3 labels | ISMCTS-5000, 800,000 decisions each, objective score | same, objective **win** |
| Rounds 1–3 networks | from the previous round, labels of rounds 1..N, value weight 100 on the tiered score | same; value fits **P(go out first)** |
| Search in play | PUCT-2000, `c_puct` 2.5, λ = 0.5, objective score | same, objective **win** |

Under objective win a search counts a simulation as 1 when the seat goes out first, 0 otherwise.
The network's leaf value is class 0 of the actor's cards-left distribution. Its value loss is the
visit-weighted squared error of that probability against the search's win rate for each candidate
(`DistillV2Config.value_target = "win"`, `python/big2/rl/distill.py`). The other heads are unchanged.

Orchestrator's calls (2026-11-17; Brian may change them): (1) the win pipeline has its own round 0,
so no tiered label reaches it; (2) λ = 0.5 and `c_puct` 2.5 are fixed at r3's values, and the
per-round screens and A9 checks are skipped. Those checks only decided whether to stop, and this
task trains to r3's full budget.

## Match

`experiments/win_chance.py final`: the win-chance bot vs `expert` (ISMCTS-10,000 + exact endgame),
10,000 duplicate deals (each 4×, seats rotated), lineups `1+3` and `ABAB`, tiered scoring,
last-card duty on, no four-card plays. Reported per bot: share of deals won and points per deal,
each with a 95 % bootstrap interval over deals, plus paired differences.

Result (job 210, `runs/win_chance/results.json`). "Mean per game" is points per deal for one seat.

Win-chance bot `puct:net=runs/net_r3_win/pvnet-r3.pknn,simulations=2000,c_puct=2.5,leaf_network_weight=0.5,objective=win` vs `expert` `ismcts:iterations=10000,infoset_endgame=12,infoset_max_cards=10`; duplicate deals, tiered scoring, last-card duty on, no four-card plays.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| win-chance vs expert | 1+3 | 10,000 | -14.43 [-14.70, -14.16] | 27.5% [27.2%, 27.8%] | +4.81 | 24.2% | -19.24 [-19.59, -18.89] | 0.103 / 0.059 |
| win-chance vs expert | ABAB | 10,000 | -9.18 [-9.36, -9.01] | 26.8% [26.5%, 27.0%] | +9.18 | 23.2% | -18.36 [-18.72, -18.01] | 0.153 / 0.086 |

| Lineup | win-chance deals won | expert deals won | difference (95 %) |
|---|---:|---:|---:|
| 1+3 | 27.5% [27.2%, 27.8%] | 24.2% [24.1%, 24.3%] | +3.3% [+2.9%, +3.8%] |
| ABAB | 26.8% [26.5%, 27.0%] | 23.2% [23.0%, 23.5%] | +3.5% [+3.1%, +4.0%] |


## Habits

`experiments/win_chance.py habits`: 1,000 deals in `ABAB`, each played with both seatings. For each
bot: how often it passes when it holds a legal play, and, at the first moment an opponent has
3 cards or fewer, how many 2s it holds and how many five-card hands its best decomposition of the
rest uses. Brian's expectation (2026-11-11): "It should pass greedily to optimize late game dominance".

Result (job 210, `runs/win_chance/habits/`):

| Bot | passes when it could play | 2s held when an opponent has ≤ 3 cards | five-card hands held then |
|---|---:|---:|---:|
| win-chance | 39.2% [38.5%, 39.9%] (20,946 decisions) | 0.71 [0.69, 0.74] | 0.41 [0.39, 0.43] (3,703 moments) |
| expert | 21.3% [20.7%, 21.9%] (19,174 decisions) | 0.63 [0.61, 0.65] | 0.30 [0.28, 0.31] (3,628 moments) |

Brian's expectation holds: the win-only target taught the bot to pass much more and to keep
control cards for the end, with no hand-written rule. The cost shows in the match: when it does
not go out first it is caught with those cards, and tiered scoring punishes that.

## Calibration of the win-chance networks

Played move vs the deal's real result (held-out shards, `runs/round<N>_win/f_calibration/table.md`):

| Network | Decisions | Mean predicted P(out first) | Real share out first | Correlation | Prior top-1 |
|---|---:|---:|---:|---:|---:|
| r0_win | 55,716 | 29.4 % | 29.9 % | 0.702 | 65.0 % |
| r1_win | 16,361 | 39.2 % | 29.6 % | 0.718 | 69.0 % |
| r2_win | 15,387 | 39.2 % | 30.0 % | 0.737 | 69.0 % |
| r3_win | 16,061 | 38.8 % | 29.8 % | 0.732 | 69.9 % |

r0 is well calibrated. From r1 on the network over-states its chance of going out by about
9 points across the middle bins (for example 45 % predicted, 27 % real in r3's [0.40, 0.50) bin).
Rounds 1–3 fit the ISMCTS-5000 search's win rate per candidate, which runs above the real result;
the order of moves (correlation 0.73, prior top-1 70 %) is still learned. Training stopped on
"no better validation" in every round (40, 35, 75 and 65 minutes on the GPU).

## Jobs

| Job | Lane | Stage | Output (big2-workers worktree) |
|---|---|---|---|
| 193 | cpu | labels, rounds 0–3 | `runs/round<N>_win/label/progress.json` |
| 209 (was 194) | gpu | networks r0–r3, calibration, export | `runs/net_r<N>_win/progress.json`, `runs/round<N>_win/f_calibration/` |
| 210 (was 195) | cpu | final match and habits | `runs/win_chance/progress.json`, `results.json`, `table.md` |

All three run `experiments/win_chance_job.ps1 -Stage <stage>` and resume when added again.
