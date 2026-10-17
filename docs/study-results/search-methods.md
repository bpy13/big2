# New search methods vs the Phase 1 solver

**Date:** 2026-10-17 · **Code:** `f104dcd` · **Machine:** Apple M1, shared with four
training jobs (times are inflated and only comparable within a row)
**Reproduce:** `python experiments/search_methods.py --variants belief belief-soft value value-mix belief+value oracle --deals 200 --workers 2`,
`... --vs-greedy --variants belief --deals 300`, `... --belief-accuracy --positions 100`
(raw JSON in `runs/search_methods*/`).

## Summary

1. **Neither new method beats the solver.** Opponent-aware sampling ties it;
   learned leaf values tie it when mixed with rollouts and lose ~1 point per
   game on their own. Combining the two loses 1.5 points.
2. **Hidden information matters, but the history reveals little of it.** A
   cheating search that sees the true deal wins by +2.95 points per game. Yet the
   opponents' actions raise the probability given to each hidden card's true
   holder only from 0.37 to 0.39–0.40 (solver opponents) or 0.47 (greedy-bot
   opponents, modelled exactly).
3. Together with Phase 1 (4× more simulations: +0.6), this points away from the
   search itself: the solver's limit is now mostly **its model of how the hand
   will be played out** (greedy rollouts) — and a value network trained in 45
   minutes is not yet a better model.

## 1. Head-to-head vs ISMCTS-1000 (200 duplicate deals each)

Variant in seats A, baseline (ISMCTS, 1,000 simulations, uniform sampling,
greedy rollouts) in seats B, A B A B, every deal played 4× with seats rotated.

| Variant | Score vs baseline | Win rate (variant / baseline) | ms / move (variant / baseline) |
|---|---:|---:|---:|
| Opponent-aware sampling, ε = 0.2 | −0.10 ± 0.62 | 23.6% / 26.4% | 54 / 44 |
| Opponent-aware sampling, ε = 0.5 | −0.02 ± 0.58 | 23.5% / 26.5% | 52 / 42 |
| Value network leaves | −0.96 ± 0.74 | 24.0% / 26.0% | 31 / 19 |
| Value network + rollout (½ each) | +0.09 ± 0.62 | 25.0% / 25.0% | 130 / 18 * |
| Opponent-aware + value network | −1.53 ± 0.76 | 24.3% / 25.7% | 120 / 18 * |
| *Oracle (sees the true deal — cheats)* | *+2.95 ± 0.85* | *24.6% / 25.4%* | *16 / 19* |

\* Measured before a speed fix to the network code (`86eefd8`), which made
network evaluation ~3× faster; the value-network row was re-run after it
(the first run gave −0.94 ± 0.75 at 115 ms).

Notes:

* Opponent-aware sampling won slightly *fewer* deals but lost fewer points when
  it lost, so the score came out even.
* The value network was trained on 248,000 self-play games of the distilled
  network (`poker rl value-train`, 45 min, 2 cores) and explains 45% of the
  variance of final scores on held-out positions. Its errors are systematic
  (the same position is always misjudged the same way), so unlike rollout
  noise they do not average out over 1,000 simulations.

## 2. The mechanism where its model is exact: vs three greedy bots (300 deals)

| Agent | Score vs 3 greedy bots |
|---|---:|
| ISMCTS-1000 | +7.69 ± 0.81 |
| ISMCTS-1000 + opponent-aware sampling (ε = 0.2) | +7.88 ± 0.81 |

Same deals for both; the difference (+0.2) is within noise even though the
opponent model is exactly right here.

## 3. How much do the samplers know? (100 positions per game type)

Average probability that a sampled deal puts each hidden card in its true
holder's hand (uniform ≈ the holder's share of the hidden cards):

| Sampler | Greedy-bot games | Solver (ISMCTS-300) games |
|---|---:|---:|
| Uniform | 0.363 | 0.368 |
| Opponent-aware, ε = 0.2 (tempered, ESS ≥ 25%) | 0.407 | 0.387 |
| Opponent-aware, ε = 0.5 | 0.409 | 0.388 |
| Opponent-aware, ε = 0.2, untempered | 0.468 | 0.400 |
| Oracle | 1.000 | 1.000 |

Even with the exact model and no tempering, the history moves the guess only a
sixth of the way from uniform to certain. Against realistic (solver) opponents
the greedy-bot model is a poor fit, and the gain is small.

## What would likely help next

* **A better playout model, with time to train it.** The value network is the
  right shape of idea (it replaced greedy rollouts and nearly broke even); it
  needs much more data, a stronger policy to imitate, and probably more capacity.
  The GPU option (`--device cuda`) makes this practical.
* **A better opponent model for sampling:** the distilled network's move
  probabilities instead of the greedy bot. Section 3 bounds the possible gain
  (the history is not very informative), so this is lower priority.
