# Build run, phase 2: results

Plan: [`docs/plans/2026-10-29-big2-bot-build.md`](../plans/2026-10-29-big2-bot-build.md), phase 2.
Script: [`experiments/build_phase2.py`](../../experiments/build_phase2.py) (one subcommand per
section). Raw results live in `runs/p2_matches/` (git-ignored): per match and lineup, blocks of
100 deals, then `results.json` and `table.md` per subcommand.

Match format unless a section says otherwise: duplicate deals (seeds 0…N−1, each deal 4×, seats
rotated), last-card duty on, `tiered` scoring, no four-card plays (B22), both lineups (`1+3`:
the candidate in one seat; `ABAB`: alternate seats). Mean points per game for the candidate with
a 95 % bootstrap interval over deals, win rate beside it, and the paired difference per deal.

Network: round 0, network v2 (Task 12), `runs/net_r0/best.pt`, exported as
`runs/net_r0/pvnet-r0.pknn` (1.17 MB). Its validation losses on 55,046 held-out decisions:
actor 1.576, seats 1.818, prior 1.259 (top-1 68.5 %), score 0.0083 (RMS 4.56 points).

Status: the job ran 2026-10-29 14:43–19:47 (8 workers, headless, 303 minutes). The tables
below are copied from `runs/p2_matches/*/table.md`.

**Summary.** A9 fails: PUCT-2000 with the round-0 network ties ISMCTS-1000 in `1+3` and loses
in `ABAB`, and it loses clearly at equal wall time. A10 fails: the distribution head loses to
Task 7's mean-score network under `win_loss` and in `ABAB` under tiered. `c_puct = 2.5` is the
best of the three values tried, all negative. The plan's fallback (Task 15 notes, proposal §10
row 2) follows in §6.

Reading the tables: "paired diff" is the candidate's mean minus the opponent's mean per deal
(in `ABAB` that is twice the candidate's mean, since the game is zero-sum). In `1+3` a neutral
win rate is 25 % per seat; in `ABAB` too.

## 1. PUCT-2000 vs ISMCTS-1000 (a, A9)

1,000 deals. Fixed simulation counts, so CPU load changes the seconds per move only.
Bar A9: ≥ +1.0 with the interval above 0.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PUCT-2000 vs ISMCTS-1000 | 1+3 | 1,000 | -0.09 [-0.55, +0.39] | 17.0% [16.0%, 17.9%] | +0.03 | 27.7% | -0.12 [-0.78, +0.56] | 0.143 / 0.008 |
| PUCT-2000 vs ISMCTS-1000 | ABAB | 1,000 | -0.89 [-1.22, -0.57] | 18.8% [18.0%, 19.5%] | +0.89 | 31.2% | -1.78 [-2.42, -1.13] | 0.169 / 0.010 |

**Verdict: A9 fails in both lineups.** One PUCT seat among three ISMCTS seats ties
(−0.12, interval covers 0); two PUCT seats against two ISMCTS seats lose by 1.78 points per
deal. PUCT wins far fewer deals (17–19 % against 25 % neutral) while its mean score stays close
in `1+3`: it loses fewer cards when it loses, and goes out first less often. A search whose
leaf value is the network's expected tiered score trades wins for smaller losses; ISMCTS's
greedy rollouts end deals and value going out. PUCT-2000 takes 0.14–0.17 s per move against
ISMCTS-1000's 0.008–0.010 s (about 90 µs per simulation against 8–10 µs per iteration, under
load), so it is 15–20× dearer per move as well.

## 2. PUCT vs ISMCTS at equal wall time (b)

0.5 s per move each, one thread, no simulation cap; a forced move (one legal action) plays at
once for both sides. 1,000 deals. **This match ran under outside CPU load**: a 13 cards probe and
then a mahjong match job, plus other projects, shared the 20 logical CPUs; `load.json` in
`runs/p2_matches/b_equal_time/` records the load before the start. Both sides take turns on the
same worker, so load slows both alike, but the absolute simulation counts are lower than on an
idle machine. A single call at the opening on this PC reached 29,312 ISMCTS iterations and 6,064
PUCT simulations in 0.5 s.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PUCT vs ISMCTS, 0.5 s per move | 1+3 | 1,000 | -1.36 [-1.78, -0.94] | 16.8% [16.0%, 17.7%] | +0.45 | 27.7% | -1.81 [-2.36, -1.25] | 0.252 / 0.276 |
| PUCT vs ISMCTS, 0.5 s per move | ABAB | 1,000 | -1.22 [-1.61, -0.85] | 18.6% [17.8%, 19.4%] | +1.22 | 31.4% | -2.44 [-3.21, -1.67] | 0.250 / 0.274 |

**Verdict: the second half of A9 fails.** At the same time per move PUCT loses by 1.8–2.4
points per deal in both lineups. ISMCTS runs about 5× more iterations than PUCT runs
simulations in the same time (29,312 against 6,064 at the opening), and the extra iterations
are worth more than the network's guidance. The mean seconds per move are below 0.5 because
forced moves play at once. The `1+3` interval excludes the (a) result: with more time, ISMCTS
gains more than PUCT does.

## 3. Distribution head vs mean-score head (c, A10)

The v2 network played greedily without search (`pvnet:checkpoint=runs/net_r0/best.pt`: the move
with the best expected score for the actor from the four cards-left distributions, under the
game's own scoring) against Task 7's mean-score network
(`qnet:path=models/big2-qnet-duty.pknn,temperature=0`). 500 deals under `tiered` and 500 under
`win_loss`. Bar A10: loses ≤ 0.5 under tiered and wins under win_loss. Under `win_loss` the
winner scores +3 and each loser −1.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 head vs Task 7 qnet, tiered | 1+3 | 500 | +0.38 [-0.32, +1.05] | 18.0% [16.8%, 19.4%] | -0.13 | 27.3% | +0.50 [-0.43, +1.43] | 0.001 / 0.000 |
| v2 head vs Task 7 qnet, tiered | ABAB | 500 | -0.67 [-1.27, -0.09] | 18.5% [17.3%, 19.7%] | +0.67 | 31.5% | -1.34 [-2.57, -0.13] | 0.001 / 0.001 |
| v2 head vs Task 7 qnet, win_loss | 1+3 | 500 | -0.05 [-0.11, +0.01] | 23.7% [22.2%, 25.2%] | +0.02 | 25.4% | -0.07 [-0.15, +0.01] | 0.001 / 0.000 |
| v2 head vs Task 7 qnet, win_loss | ABAB | 500 | -0.11 [-0.15, -0.06] | 22.4% [21.1%, 23.5%] | +0.11 | 27.7% | -0.21 [-0.30, -0.12] | 0.001 / 0.001 |

**Verdict: A10 fails.** Under tiered the distribution head ties in `1+3` (+0.50, interval
covers 0) but loses 0.67 per game in `ABAB`, past the 0.5 allowance. Under `win_loss` it does
not win: a tie in `1+3` and a clear loss in `ABAB` (−0.21 [−0.30, −0.12]). The same pattern as
§1 shows here without search: the v2 network wins 18 % of tiered deals against the qnet's
27–32 %. So the distributions learned to keep cards low rather than to go out. A likely reason
(docs/rl.md §3.3): the cards-left heads learn from one move per decision and one outcome per
deal, while the mean-score network learns every candidate's search value directly.

## 4. `c_puct` in {1.0, 1.5, 2.5} (d)

PUCT-2000 against ISMCTS-1000 on the first 300 deals; 1.5 (the default) reuses the first 300
deals of section 1.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c_puct = 1.0 | 1+3 | 300 | -1.13 [-2.10, -0.18] | 15.2% [13.5%, 16.9%] | +0.38 | 28.3% | -1.51 [-2.71, -0.23] | 0.167 / 0.010 |
| c_puct = 1.0 | ABAB | 300 | -1.13 [-1.80, -0.45] | 17.9% [16.6%, 19.2%] | +1.13 | 32.1% | -2.25 [-3.52, -1.04] | 0.154 / 0.009 |
| c_puct = 1.5 | 1+3 | 300 | -0.74 [-1.67, +0.17] | 16.7% [14.9%, 18.5%] | +0.25 | 27.8% | -0.98 [-2.21, +0.23] | 0.091 / 0.005 |
| c_puct = 1.5 | ABAB | 300 | -1.04 [-1.60, -0.48] | 19.6% [18.4%, 20.9%] | +1.04 | 30.4% | -2.07 [-3.20, -0.95] | 0.179 / 0.010 |
| c_puct = 2.5 | 1+3 | 300 | -0.96 [-1.86, -0.08] | 18.6% [17.0%, 20.2%] | +0.32 | 27.1% | -1.28 [-2.47, -0.10] | 0.170 / 0.010 |
| c_puct = 2.5 | ABAB | 300 | -0.69 [-1.29, -0.14] | 21.0% [19.6%, 22.3%] | +0.69 | 29.0% | -1.39 [-2.54, -0.22] | 0.162 / 0.009 |

**Verdict: keep `c_puct = 2.5`** (best sum over both lineups: −1.65 against −1.78 for 1.5 and
−2.26 for 1.0). The gaps are well inside the ±1.2 intervals, so the choice is weak; a larger
constant helps a little and raises the win rate; §6.1 checks whether the prior is too sharp.
Every value loses to ISMCTS-1000. The first 300 deals were worse for PUCT than the full 1,000
(1.5 in `1+3`: −0.98 here against −0.12 in §1): a 300-deal result moves by about ±1 point.

## 5. WebAssembly timing (Task 16)

(pending: not started while other jobs load the CPU)

## 6. Fallback: calibration and a rollout in the leaf (proposal §10 row 2)

(in progress)
