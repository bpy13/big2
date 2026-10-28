# Build run, phase 1: results

Plan: [`docs/plans/2026-10-29-big2-bot-build.md`](../plans/2026-10-29-big2-bot-build.md), phase 1.
Script: [`experiments/build_phase1.py`](../../experiments/build_phase1.py) (one subcommand per
section). Raw results live in `runs/` (git-ignored).

Match format unless a section says otherwise: duplicate deals (each deal 4×, seats rotated),
last-card duty on, `tiered` scoring, no four-card plays (B22); mean points per deal with a 95 %
interval over deals, win rate beside it.

## 1. Structured rollout inside ISMCTS (Task 5)

**Planner speed (Task 4).** `cargo bench -p big2-big2 --bench engine`, row `min_plays_13_cards`:
**2.44 µs** per 13-card hand (interval 2.41–2.47 µs); target < 50 µs.

**Question.** Does ISMCTS play better when its rollouts keep the hand's plan? The bar (A4) is
a gain of at least +1.0 point per game at 300 deals.

**The rollout.** `big2_big2::bots::StructuredGreedy { epsilon, plan_depth }`. A play *keeps
the plan* when `min_plays(hand − play) + 1 == min_plays(hand)`, i.e. it is a group of one of
the hand's best decompositions. Following, it plays the weakest beating play that keeps the
plan; leading, the first play holding the weakest card (Greedy's order) that keeps it. When
no checked play keeps the plan it plays as `Greedy`. `plan_depth = 8` plays are checked per
decision; `epsilon = 0.1` as today's rollout.

**A variant that passes loses badly.** The first version passed instead of breaking a pair,
triple or five-card group to beat a single. As a bot it lost **−7.6 ± 0.5** points per game
to `Greedy` (2,000 duplicate deals, A B A B, Rust test harness). The shipped version, which
breaks like `Greedy` when it must, wins **+1.07 ± 0.23** (the same deals; the unit test
`structured_beats_greedy_head_to_head` keeps it). This matches the Patwa-smart finding in §3:
passing whenever every play breaks structure is a losing rule.

**Cost.** One ISMCTS-1000 call at the opening takes 38 ms with the structured rollout against
15 ms with the greedy one (one thread, this PC, other jobs running).

**Method.** `python experiments/build_phase1.py structured-rollout --deals 300 --workers 14`,
the same as `big2 tournament "ismcts:iterations=1000,rollout=structured" ismcts:iterations=1000
"ismcts:iterations=1000,rollout=structured" ismcts:iterations=1000 --deals 300 --workers 14
--out runs/p1_structured/results.json`. Agent specs: `ismcts:rollout=structured` and
`greedy:structured=1`.

**Results** (`runs/p1_structured/results.json`, 2026-10-29, 6 min on 14 workers). 300
duplicate deals, A B A B, seed 0, last-card duty on, tiered scoring:

| Agent | Games | Mean score per game (95% CI) | Win rate (95% CI) | s per move |
|---|---:|---:|---:|---:|
| `ismcts:iterations=1000,rollout=structured` | 2,400 | +0.07 ± 0.58 | 25.9 % ± 1.1 % | 0.130 |
| `ismcts:iterations=1000` (greedy rollout) | 2,400 | −0.07 ± 0.58 | 24.1 % ± 1.1 % | 0.048 |

Paired difference per deal (structured seat − greedy-rollout seat): +0.14 ± 1.17.

**Verdict (A4): fails the +1.0 bar.** The structured rollout gains +0.07 ± 0.58 points per
game, a shortfall of 0.93; the interval covers 0, so the gain is indistinguishable from none.
It also costs 2.7× the time per move. Following proposal §10 (row 1), the planner stays for
reason keys and candidate lists only, and round 0's relabelling (Task 11) uses today's greedy
rollout. A better bot (`StructuredGreedy` beats `Greedy` by about 1 point) did not make a
better rollout policy at equal iterations.

## 2. Re-baseline under duty-on, and σ (Task 6)

Method: `python experiments/build_phase1.py rebaseline --workers 14` (headless job
`runs/p1_rebaseline/run.ps1`).

Candidate `ismcts:iterations=1000` (greedy rollout: Task 5's structured rollout did not pay). 1,000 duplicate deals per tournament (each deal 4×, seats rotated), deal seeds 0…N−1, last-card duty on, tiered scoring, no four-card plays. Mean score per game, 95 % interval over deals. Seconds per move measured under load (other CPU jobs ran beside it); results do not depend on load (fixed iterations).

| Tournament | Agent | Games | Mean score (95% CI) | Win rate (95% CI) | s per move (under load) |
|---|---|---:|---:|---:|---:|
| vs 3 greedy (1 + 3) | `ismcts:iterations=1000` | 4,000 | +7.40 ± 0.45 | 50.1% ± 1.4% | 0.049 |
| vs 3 greedy (1 + 3) | `greedy` | 12,000 | -2.47 ± 0.15 | 16.6% ± 0.5% | 0.000 |
| vs DMC-10h (A B A B) | `ismcts:iterations=1000` | 8,000 | +0.17 ± 0.37 | 29.4% ± 0.7% | 0.072 |
| vs DMC-10h (A B A B) | `dmc:checkpoint=models/big2-dmc-selfplay-10h.pt` (trained with duty off) | 8,000 | -0.17 ± 0.37 | 20.6% ± 0.7% | 0.005 |
| self-play (all four seats) | `ismcts:iterations=1000 (seat 0 only)` | 4,000 | -0.86 ± 1.47 | 23.4% ± 2.1% | 0.075 |

σ of one seat's score in one game (ISMCTS self-play, 16,000 seat-games): **27.45 points**. Of that, the spread between rotations of the same deal and seat (same cards, new search seeds) is 14.76 points.

## 3. B1: Patwa's bots and scoring (Task 8)

**Question.** How does today's ISMCTS-1000 do under the protocol of Patwa (2026)
([survey §2.4](../research/survey.md))? His best agent (PPO, entropy 0.05) won 64.8 % of games
against three greedy bots and 43.5 % against three smart bots, 1,000 games per cell, one seed,
no intervals.

**Method.** `python experiments/build_phase1.py b1 --games 10000 --seeds 1,2,3 --workers 14`
(about 40 minutes on 14 workers beside other jobs). Patwa's protocol as far as his text gives
it:

- One candidate seat against three copies of a bot; each game is a fresh deal with the four
  seats shuffled at random (`run_tournament(..., seat_random=True)`, CLI `--seat-random`), not
  our duplicate rotation.
- `sum_of_others` scoring (Charlesworth's and Patwa's): the winner gets the sum of the other
  three players' cards left, each loser minus their own count (`score_as="sum_of_others"`, CLI
  `--score-as sum_of_others`). Raw counts, no tiers. Win rate is the share of games the
  candidate went out first; par is 25 %.
- Our rules, not his: last-card duty on, no four-card plays (B22). His rule set is not fully
  stated; Charlesworth's allows four-card plays.
- 10,000 games per cell and seed, 3 seeds; the interval is a binomial 95 % interval over games.

**The opponents are approximations.** Patwa published no code. `big2.agents.patwa`
re-implements his two one-line descriptions (B14: read, never copied):

- `patwa-greedy`: the lowest non-pass action by (type, rank key), ignoring hand structure.
- `patwa-smart`: a hand-written scorer. It plays out its hand when it can, then rewards cards
  shed and penalises 2s spent early, new orphan singles, broken pairs, triples and five-card
  hands, and high cards. The weights are ours. The first version (passing whenever every play
  broke structure) lost badly to `patwa-greedy`, which contradicts Patwa's numbers. The
  shipped weights came from a small grid: 400 duplicate deals A B A B against `patwa-greedy`,
  +1.6 ± 0.25 points per game and 35 % wins. So our smart bot is stronger than our greedy bot,
  as his are; how close either is to his is unknown.

**Results.**

Candidate `ismcts:iterations=1000`, 10,000 seat-random games per seed, seeds 1,2,3, `sum_of_others` scoring, last-card duty on, no four-card plays.

| Opponents (×3) | Seed | Games | Win rate (95% CI) | Mean score (95% CI) | Patwa's PPO win rate |
|---|---|---:|---:|---:|---:|
| patwa-greedy | 1 | 10,000 | 69.2% ± 0.9% | +8.97 ± 0.19 | |
| patwa-greedy | 2 | 10,000 | 69.0% ± 0.9% | +8.93 ± 0.19 | |
| patwa-greedy | 3 | 10,000 | 68.9% ± 0.9% | +8.80 ± 0.19 | |
| **patwa-greedy** | **pooled** | 30,000 | **69.0% ± 0.5%** | +8.90 ± 0.11 | 64.8% |
| patwa-smart | 1 | 10,000 | 31.1% ± 0.9% | +2.12 ± 0.21 | |
| patwa-smart | 2 | 10,000 | 30.9% ± 0.9% | +2.10 ± 0.21 | |
| patwa-smart | 3 | 10,000 | 31.6% ± 0.9% | +2.23 ± 0.21 | |
| **patwa-smart** | **pooled** | 30,000 | **31.2% ± 0.5%** | +2.15 ± 0.12 | 43.5% |

**Verdict.** Against three greedy bots ISMCTS-1000 wins 69.0 % ± 0.5 %, above Patwa's 64.8 %.
Against three smart bots it wins 31.2 % ± 0.5 %, below his 43.5 %, though above par (25 %)
with a positive score. Our smart bot is our own tuned approximation, so that row is not a
like-for-like comparison with his.

## 4. Ladder v0 (Task 9)

**Question.** Do five candidate levels order strictly, with adjacent gaps of at least 1.5
points per game over 500 deals and intervals that exclude each other (A8)? Ladder v0 does not
change the app's levels; Task 10 proposes the switch and Task 29 retunes.

**Levels** (weakest first): L0 `random`; L1 `greedy:structured=1,epsilon=0.1`
(`StructuredGreedy`); L2 `qnet:temperature=1.0` (the distilled duty-on network,
`models/big2-qnet-duty.pknn`, Task 7's export); L3 `ismcts:iterations=1000`; L4
`ismcts:iterations=5000`.

**The `qnet` spec.** The Rust `QNetPolicy` is not exposed through the Python extension, and
the extension cannot be rebuilt while jobs run. `big2.agents.qnet` therefore reads the
`.pknn` file in NumPy and repeats the Rust forward pass on the weights the Rust reader sees
(int8 groups and f16 rounded the same way), and Rust's sampling rule: a move with probability
proportional to `exp(score / temperature)`, temperature in points. A test checks it against
PyTorch on an f32 export (within 0.001 points).

**Harness.** `big2.eval.harness` (CLI `big2 match <cand> <opp> --deals N --lineups both`): one
candidate against one opponent on duplicate deals, lineups "1+3" and "A B A B". Each report has
`candidate, opponent, lineup, deals, mean, ci95, win_rate, win_ci95, seconds_per_move`, plus
the opponent's mean and the paired difference per deal (candidate minus opponent). Intervals are
95 % percentile bootstrap intervals over deals (2,000 resamples), written `[low, high]`.

**Method.** `python experiments/build_phase1.py ladder --deals 500 --lineups both --workers 14`
(headless job `runs/p1_ladder/run.ps1`; it waits for the Task 6 and Task 8 jobs to finish).
Every pair of levels plays the same 500 deals (seeds 0…499) in both lineups, the stronger
level as candidate. The ladder is judged on the A B A B lineup:

- the **gap** between adjacent levels is the upper level's mean score per game against the
  lower one; it meets A8 when it is ≥ 1.5 with its interval above 0;
- each level's **standing** is its mean score over its four A B A B matches (per deal, then a
  bootstrap over deals); adjacent standings' intervals must not overlap;
- the ordering is strict when the standings sort L0 < … < L4 and every gap's interval is above 0.

The 1+3 results are in `runs/p1_ladder/results.json` beside them.

**Results.**

500 duplicate deals per pair (deal seeds 0…499, each deal 4×, seats rotated), lineup 1+3, ABAB, last-card duty on, tiered scoring. Points per game; 95 % bootstrap intervals over deals.

**Levels**: L0 `random`; L1 `greedy:structured=1,epsilon=0.1`; L2 `qnet:temperature=1.0`; L3 `ismcts:iterations=1000`; L4 `ismcts:iterations=5000`.

Head to head (ABAB): the row level's mean score per game against the column level.

| | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| L0 | – | -9.82 | -11.39 | -13.12 | -14.03 |
| L1 | +9.82 [+9.24, +10.41] | – | -1.80 | -3.01 | -3.65 |
| L2 | +11.39 [+10.83, +12.00] | +1.80 [+1.32, +2.29] | – | -1.64 | -2.14 |
| L3 | +13.12 [+12.52, +13.71] | +3.01 [+2.50, +3.55] | +1.64 [+1.23, +2.07] | – | -0.55 |
| L4 | +14.03 [+13.35, +14.72] | +3.65 [+3.10, +4.20] | +2.14 [+1.69, +2.57] | +0.55 [+0.16, +0.94] | – |

Round-robin standing (mean score per game over the level's four matches):

| Level | Agent | Mean (95% CI) | s per move (under load) |
|---|---|---:|---:|
| L4 | `ismcts:iterations=5000` | +5.09 [+4.76, +5.42] | 0.2133 |
| L3 | `ismcts:iterations=1000` | +4.31 [+4.01, +4.61] | 0.0384 |
| L2 | `qnet:temperature=1.0` | +2.35 [+2.09, +2.61] | 0.0009 |
| L1 | `greedy:structured=1,epsilon=0.1` | +0.34 [-0.01, +0.68] | 0.0000 |
| L0 | `random` | -12.09 [-12.59, -11.60] | 0.0000 |

Adjacent gaps (upper level's mean score per game against the lower, ABAB); bar A8: ≥ 1.5 with the interval above 0, and the two standings' intervals apart:

| Lower | Upper | Gap (95% CI) | Paired diff (95% CI) | Upper's win rate | Bar |
|---|---|---:|---:|---:|---|
| `random` | `greedy:structured=1,epsilon=0.1` | +9.82 [+9.24, +10.41] | +19.64 [+18.46, +20.81] | 45.1% | meets |
| `greedy:structured=1,epsilon=0.1` | `qnet:temperature=1.0` | +1.80 [+1.32, +2.29] | +3.60 [+2.64, +4.58] | 30.4% | meets |
| `qnet:temperature=1.0` | `ismcts:iterations=1000` | +1.64 [+1.23, +2.07] | +3.28 [+2.41, +4.14] | 28.3% | meets |
| `ismcts:iterations=1000` | `ismcts:iterations=5000` | +0.55 [+0.16, +0.94] | +1.09 [+0.35, +1.87] | 24.9% | **short** |

Ordering strict: yes. A8 fails.
