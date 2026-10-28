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

To be filled by Task 6 (`build_phase1.py rebaseline`).

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

**Results.** _Pending: the run was still going at hand-back (check
`runs/p1_b1/progress.json`; the table lands in `runs/p1_b1/table.md`)._

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

**Results.** _Pending: the job was waiting for the CPU at hand-back (check
`runs/p1_ladder/progress.json`; the table lands in `runs/p1_ladder/table.md`)._
