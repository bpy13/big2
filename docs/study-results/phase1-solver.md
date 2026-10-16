# Phase 1 results — the search solver

**Date:** 2026-10-16 · **Code:** `925bc2f` · **Machine:** Apple M1 (8 cores)
**Reproduce:** `python experiments/phase1_solver.py --deals 100 --workers 7`
(raw JSON in `runs/phase1/`) and
`cargo run --release -p poker-search --example arena -- <deals> <iters> <c> [endgame] [nodes]`.

All tables use **duplicate deals**: each deal is played four times with the lineup
rotated, so every agent plays every hand. Confidence intervals (95%) are over
deals. Scores use the default tiered pairwise scoring; a score of 0 means "as
strong as the table average". "vs greedy" means one candidate against three greedy
bots.

## Summary

1. **The solver is far stronger than the baselines.** ISMCTS with 1,000
   simulations per move (≈16 ms) averages **+8.6 points per game** against three
   greedy bots and goes out first **48%** of the time (25% = equal strength).
2. **More thinking time barely helps.** 4× the simulations gains +0.6 ± 0.9
   points head-to-head (not significant). The search has hit the limits of its
   *model* — uniform guesses about hidden cards and a simple greedy rollout
   policy — rather than its budget. Better models are Phase 3's job.
3. **ISMCTS beats the simpler PIMC on win rate** (28.1% vs 21.9%, significant)
   but only marginally on score (+1.1 ± 1.1), while PIMC is ~8× faster.
4. **Maximizing win chance is the wrong goal under this scoring.** A win-seeking
   solver goes out first more often (26.8% vs 23.2%) but loses **7.6 points per
   game** to the score-seeking one, because it ignores how many cards it is left
   with when it does not win.
5. **The exact endgame solver did not help** against greedy opponents and costs
   2–6× the time; it is off by default.
6. Exploration constant (0.1–0.7) and rollout noise (ε 0–0.3) do not matter
   within measurement precision.

## 1. Strength ladder (one candidate vs three greedy bots, 100 deals)

| Candidate | Mean score (95% CI) | Win rate (95% CI) | ms / move |
|---|---:|---:|---:|
| `ismcts:iterations=1000` | **+8.55 ± 1.70** | **48.0% ± 4.5%** | 16.2 |
| `pimc:iterations=40` | +8.14 ± 1.48 | 43.0% ± 4.2% | 2.3 |
| `greedy:epsilon=0.1` | −0.94 ± 0.83 | 24.0% ± 2.2% | <0.1 |
| `random` | −16.36 ± 2.21 | 2.0% ± 1.3% | <0.1 |

## 2. Head-to-head (two copies of each agent in alternating seats, 100 deals)

| Match-up | Mean score of first (95% CI) | Win rates | ms / move |
|---|---:|---:|---:|
| ISMCTS 4,000 vs ISMCTS 1,000 | +0.62 ± 0.85 | 26.6% vs 23.4% | 64.7 vs 15.3 |
| ISMCTS 1,000 vs PIMC 40 | +1.10 ± 1.14 | 28.1% vs 21.9% | 16.6 vs 2.0 |
| ISMCTS rollout ε=0.0 vs ε=0.1 | +0.24 ± 0.72 | 24.8% vs 25.2% | 14.4 vs 15.1 |
| ISMCTS rollout ε=0.1 vs ε=0.3 | +0.41 ± 0.76 | 24.4% vs 25.6% | 15.3 vs 16.4 |
| ISMCTS objective=score vs objective=win | **+7.56 ± 1.55** | 23.2% vs 26.8% | 15.7 vs 15.7 |

(In a two-agent lineup the second agent's score is the negative of the first's.)

## 3. Budget and exploration vs greedy (Rust arena)

Simulations per move (exploration 0.25, same 100 deals):

| Simulations | Mean score (95% CI) | Win rate |
|---:|---:|---:|
| 250 | +7.35 ± 1.90 | 43.5% |
| 1,000 | +8.16 ± 1.89 | 46.5% |
| 4,000 | +8.98 ± 1.82 | 46.8% |

Exploration constant `c` (1,000 simulations, same 150 deals):

| c | 0.1 | 0.25 | 0.5 | 0.7 | 1.0 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| Mean score | +7.92 | **+8.28** | +7.67 | +7.85 | +6.79 | +6.65 |
| ± | 1.59 | 1.58 | 1.53 | 1.57 | 1.52 | 1.50 |

Default set to 0.25 (best observed; 0.1–0.7 statistically tied).

## 4. Exact endgame vs greedy (1,000 simulations, same 30 deals)

| Endgame setting | Mean score (95% CI) | Win rate | Run time |
|---|---:|---:|---:|
| off | +9.60 ± 3.03 | 53.3% | 29 s |
| ≤12 cards, 200 nodes | +9.54 ± 3.03 | 52.5% | 53 s |
| ≤12 cards, 2,000 nodes | +9.68 ± 3.02 | 54.2% | 167 s |
| ≤16 cards, 200 nodes | +9.52 ± 3.04 | 52.5% | 95 s |

A threshold of 8 cards never triggered: the deal ends when *one* player is out,
so the four hands rarely total that few cards.

## Interpretation and open questions

* **Why the plateau?** Against greedy opponents, the rollout policy *is* the
  opponent model, and it is already exact; more simulations only refine an
  estimate that is already good enough to pick the move. Against stronger
  opponents, the uniform hidden-card sampling and greedy rollouts become the
  bottleneck. Both are what learned networks improve: a policy network as a
  rollout/prior, a value network to cut rollouts short, and a model of opponents
  to weight sampled deals.
* **Why no endgame gain?** Max^n assumes every player plays perfectly *with
  full information*, while the actual opponents are greedy; the greedy rollout
  models them better. The exact solver should be re-tested against strong
  opponents (ISMCTS or RL agents) in Phase 3.
* **Scoring shapes strategy.** Under tiered pairwise scoring, reducing your card
  count when you will not win (especially getting below 10 and 8 cards, where the
  multiplier drops) is worth a lot. Study mode should show both win rate and
  expected score, and say which one the recommendation optimizes.
* **Sample sizes.** 100 duplicate deals (400 games) resolve differences of about
  ±1–2 points. Smaller effects (e.g. exploration constant) need ~1,000 deals.
