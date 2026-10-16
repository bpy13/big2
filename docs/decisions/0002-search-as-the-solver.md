# 0002 — Decision-time search is the "solver"; learning is the comparison

**Status:** accepted (2026-10-16)

## Context

"Solver" usually means CFR-family equilibrium computation (PioSOLVER, Libratus,
Pluribus). Its guarantee — converging to an unexploitable strategy — holds only for
two-player zero-sum games. Big Two has four players, and its information sets
(your exact remaining hand plus the play history) almost never repeat, so tabular
CFR does not learn; it needs heavy abstraction to work at all.

The strongest published programs for Big Two's close relatives (Dou Dizhu, Guandan)
are deep RL agents (DouZero 2021, PerfectDou 2022). The strongest systems overall
combine a learned network with search (AlphaZero, ReBeL, Student of Games).

## Decision

* **Phase 1 solver:** information-set Monte Carlo tree search (ISMCTS) over
  sampled hidden hands, with an exact endgame solver when few cards remain. It
  needs no training, gets stronger with more thinking time, and directly produces
  per-move win rates and expected scores for study mode.
* **Phase 3:** learning methods (DouZero-style Deep Monte Carlo first, PPO second,
  Deep CFR as a stretch goal) evaluated head-to-head against the Phase 1 solver,
  then combined with it (network-guided search).

## Consequences

* The solver's per-move numbers are *estimates* with confidence intervals, not
  equilibrium values. The UI and docs say so.
* ISMCTS shares PIMC's known weakness ("strategy fusion": it partly plans as if it
  could see hidden cards). Measuring how much this costs against learned agents is
  one of the study's questions.
