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
row 2) is in §6: the prior is well calibrated, the value is over-spread; a leaf value
`0.5·network + 0.5·rollout` lifts PUCT by about 1.6–2.0 points to a tie with ISMCTS-1000
(+0.30 and +0.61, intervals covering 0, 300 deals), still short of A9 and 19× dearer per move.

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

Proposal §10 row 2: when PUCT only ties ISMCTS, check the prior's calibration and the leaf
value first, then try a value plus a short rollout. §6.1 is the check, §6.2 the match.

### 6.1 Calibration of the round-0 network (f)

`experiments/build_phase2.py calibration` (GPU, 1 minute), on the 101 shards that training held
out: 55,046 decisions and 276,502 candidates from ISMCTS-1000 self-play (Task 11). Losses agree
with Task 12's: prior 1.259 (top-1 68.5 %), actor 1.576, seats 1.818. Raw output:
`runs/p2_matches/f_calibration/`.

**Prior against the search's visit shares** (decisions with more than one candidate; "most
visited": how often a move in the bin was the search's choice):

| Prior bin | Candidates | Mean prior | Mean visit share | Most visited |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 99,114 | 0.058 | 0.057 | 2.1 % |
| [0.1, 0.2) | 79,947 | 0.144 | 0.139 | 11.7 % |
| [0.2, 0.3) | 40,342 | 0.245 | 0.240 | 24.0 % |
| [0.3, 0.4) | 22,416 | 0.345 | 0.345 | 38.8 % |
| [0.4, 0.5) | 13,485 | 0.446 | 0.457 | 53.9 % |
| [0.5, 0.6) | 8,400 | 0.546 | 0.564 | 74.1 % |
| [0.6, 0.7) | 5,184 | 0.647 | 0.683 | 90.5 % |
| [0.7, 0.8) | 3,753 | 0.747 | 0.785 | 96.8 % |
| [0.8, 0.9) | 2,467 | 0.844 | 0.864 | 98.8 % |
| [0.9, 1.0) | 1,394 | 0.942 | 0.955 | 99.9 % |

Expected calibration error 0.005. Mean entropy 1.28 nats for the prior, 1.19 for the visit
shares: the prior is slightly *flatter* than the search, not sharper. **The prior is well
calibrated**; it is not why PUCT loses.

**Predicted tiered score against the search's mean score** (262,595 candidates simulated ≥ 20
times; ten equal-count bins of the prediction):

| Predicted bin | Mean predicted | Mean search score |
|---|---:|---:|
| 1 (lowest) | −21.17 | −18.27 |
| 2 | −6.91 | −7.02 |
| 3 | −3.19 | −3.59 |
| 4 | −0.85 | −1.31 |
| 5 | +1.08 | +0.57 |
| 6 | +2.97 | +2.20 |
| 7 | +4.99 | +3.94 |
| 8 | +7.46 | +5.93 |
| 9 | +11.21 | +9.04 |
| 10 (highest) | +22.97 | +19.19 |

Correlation 0.92, RMS 4.82 points, bias +0.79 points, slope of the search's score on the
prediction 0.86 (spread 12.2 predicted against 11.3 searched). **The value is good but
over-spread and optimistic**: a move predicted +11 is worth +9 to the search, one predicted −21
is worth −18. Inside PUCT the leaf values are therefore about 15 % too extreme and about
0.8 points too high, by the search's own measure.

**Played move against the deal's real result**: bias +0.38 points, correlation 0.67. The
chance of going out first is calibrated below 50 % and over-confident above it:

| Predicted P(out first) | Decisions | Mean predicted | Real share |
|---|---:|---:|---:|
| [0.0, 0.1) | 16,491 | 4.7 % | 5.6 % |
| [0.2, 0.3) | 6,524 | 24.6 % | 26.0 % |
| [0.4, 0.5) | 3,908 | 44.8 % | 44.6 % |
| [0.6, 0.7) | 2,514 | 64.8 % | 59.6 % |
| [0.7, 0.8) | 2,063 | 74.8 % | 68.3 % |
| [0.8, 0.9) | 1,878 | 84.9 % | 78.0 % |
| [0.9, 1.0) | 3,550 | 97.4 % | 94.3 % |

(Every bin is in `runs/p2_matches/f_calibration/table.md`; the omitted rows follow the same
pattern.) Overall predicted 31.2 %, real 30.9 %.

**Verdict.** Calibration does not explain the gap on its own. The prior is as good as a
distilled prior gets at 68.5 % top-1. The value is the weaker part: over-spread, slightly
optimistic, and over-confident about going out once a seat is ahead. These are all measured on
ISMCTS self-play positions; PUCT visits other positions too. The value is also a copy of
ISMCTS-1000's own estimate (RMS 4.8 points from it), so a search on it cannot see further than
ISMCTS-1000 does: the rollout mix in §6.2 adds real play-outs back into the leaf.

### 6.2 Leaf value `λ · network + (1 − λ) · rollout` (e)

`experiments/build_phase2.py rollout-leaf`: PUCT-2000, `c_puct = 2.5` (§4's best), against
ISMCTS-1000 on deals 0…299, both lineups. The leaf rollout is the greedy policy with epsilon
0.1 that ISMCTS-1000 itself plays, to the end of the deal (`rollout_moves = 0`); Task 5's
structured rollout did not beat it (A4). λ = 0 is the control: the network's prior orders the
tries, rollouts alone value them. Job 2026-10-30 01:34–03:07, 6 workers, under outside load
(and limited to logical CPUs 4–19 from 01:40), so the seconds per move are high for both sides.
Raw output: `runs/p2_fallback/e_rollout_leaf/`.

| Match | Lineup | Deals | Candidate mean per game (95% CI) | Candidate win rate (95% CI) | Opponent mean | Opponent win rate | Paired diff (95% CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| λ = 0.5 | 1+3 | 300 | +0.22 [-0.59, +1.09] | 22.1% [20.4%, 23.7%] | -0.07 | 26.0% | +0.30 [-0.84, +1.41] | 0.226 / 0.012 |
| λ = 0.5 | ABAB | 300 | +0.31 [-0.27, +0.85] | 23.2% [22.1%, 24.3%] | -0.31 | 26.8% | +0.61 [-0.56, +1.73] | 0.225 / 0.013 |
| λ = 0.0 | 1+3 | 300 | -0.49 [-1.34, +0.35] | 23.4% [21.8%, 25.2%] | +0.16 | 25.5% | -0.65 [-1.75, +0.42] | 0.506 / 0.028 |
| λ = 0.0 | ABAB | 300 | +0.31 [-0.16, +0.76] | 25.3% [24.2%, 26.3%] | -0.31 | 24.7% | +0.62 [-0.28, +1.57] | 0.458 / 0.025 |

For comparison, the network-only leaf (λ = 1) with the same `c_puct` on the same 300 deals (§4):
−1.28 [−2.47, −0.10] in `1+3`, −1.39 [−2.54, −0.22] in `ABAB`.

**Verdict: the fallback fails the bar, but the rollout mix helps.** Neither λ passes the A9
screen (paired diff ≥ +1.0, interval above 0) in either lineup; every interval covers 0, so
PUCT with a rollout ties ISMCTS-1000. Against the network-only leaf on the same deals, λ = 0.5
gains about +1.6 (`1+3`) and +2.0 (`ABAB`) points, and the win rate climbs from 17–19 % to
22–23 %: mixing real play-outs into the leaf removes most of the network value's damage
(§6.1: over-spread and optimistic). λ = 0.5 is at least as good as λ = 0 (`1+3` +0.30 against
−0.65; `ABAB` equal), so the network value adds a little on top of rollouts, not much. The cost
stays high: PUCT-2000 with rollouts takes about 19× the time per move of ISMCTS-1000, so at
equal time it would lose (§2). Proposal §10 row 2 now says: one more round of expert iteration,
then ship ISMCTS as `expert` if PUCT still only ties. The next step is in `docs/open-items.md`.
