# The Big Two solver

This document explains what the solver computes, how, and how far to trust its
numbers. Measured results are in
[study-results/phase1-solver.md](study-results/phase1-solver.md).

## 1. What "solving" can mean here

In poker, a *solver* (PioSOLVER, GTO Wizard, Libratus) computes an approximate
**Nash equilibrium** with counterfactual regret minimization (CFR). In two-player
zero-sum games an equilibrium strategy cannot lose in expectation, whatever the
opponent does. That guarantee is why "GTO" means something in heads-up poker.

Big Two breaks both preconditions:

* **Four players.** With more than two players an equilibrium no longer guarantees
  anything: other players' choices (even mistakes) can shift value between you and
  a third player. Pluribus (six-player Hold'em, 2019) still worked well in
  practice, but without a guarantee.
* **Huge, non-repeating information sets.** What a player knows is their exact
  remaining hand plus the whole play history. Tabular CFR needs the same
  information set to come up many times; here it almost never does. Poker solvers
  get around this by grouping similar hands into buckets; Big Two's hand structure
  (which pairs, straights and flushes you can still make) does not bucket well.

So this project's solver is a **decision-time search** (ADR
[0002](decisions/0002-search-as-the-solver.md)): it does not precompute a
strategy; it analyzes the position in front of it by simulation. This is how strong
programs for other hidden-card games (Bridge, Skat, Hearts) work, and its output —
a win rate and expected score per move — is what study mode needs.

## 2. How it works

### 2.1 Determinization

The solver cannot see opponents' cards. It *samples* them: deal the unseen cards
at random among the opponents, keeping each opponent's card count (and the fact
that, before the first play, the player to move holds 3♦). Each sample is a
**determinization** — a guess at the full deal that is consistent with everything
the searching player knows.

Sampling is currently **uniform**. It ignores soft clues, e.g. "this player passed
on a 7, so they probably don't hold many cards above 7". Learning those clues is a
Phase 3 topic (weighting samples with a learned model of opponents).

### 2.2 ISMCTS (the main solver)

Information-set Monte Carlo tree search (Cowling, Powley & Whitehouse, 2012) runs
many iterations; each one:

1. **Samples** a determinization.
2. **Walks down a tree of public actions** (plays and passes). At each node it
   only considers actions that are legal in the current sample, and picks with the
   UCB1 rule: average result for the player to move + an exploration bonus
   `c·√(ln availability / visits)`.
3. **Expands** one new action.
4. **Finishes the game** with a fast rollout policy (the greedy bot with 10%
   random moves).
5. **Records the result** at every node on the path, from the point of view of
   the player who chose that node's action.

Because one tree is shared by all samples, each node's average is over the hands
the opponents might hold — the quantity a player actually cares about. Opponent
nodes are chosen with the opponents' own interests (multi-player UCT), so the
search assumes everyone plays well for themselves.

**Recommended move:** the most-simulated action ("robust child"). The UCB rule
gives more simulations to actions that look better, so visit counts are a stable
ranking signal.

### 2.3 PIMC (baseline)

Perfect-information Monte Carlo: sample N deals, try every legal action in each,
finish each game with the rollout policy, average. No tree. Every action gets the
same number of simulations, so its numbers are easy to interpret, but it wastes
time on bad actions and never plans beyond the first move.

### 2.4 Exact endgame (optional)

Inside a sample all cards are known, so the rest of the game can be solved exactly
by **max^n** search (each player maximizes their own result), with a transposition
table. ISMCTS can use it instead of a rollout once few cards remain
(`endgame_threshold`, `endgame_max_nodes`). It is **off by default**: in our
measurements it did not improve results and cost 2–6× the time (see results).

## 3. Reading the numbers

For each move the solver reports:

| Field | Meaning |
|---|---|
| Win rate | Fraction of simulated games in which you went out first |
| Expected score | Average final score (pairwise settlement, rules §7) |
| ± | 95% confidence interval from the number of simulations |
| Simulations | How many simulated games started with this move |

Caveats, in order of importance:

1. **Model error, not just sampling error.** The ± only covers simulation noise.
   The estimates assume opponents play like the rollout policy and that hidden
   cards are uniformly distributed. Against much stronger or weaker opponents,
   the true numbers differ.
2. **Strategy fusion.** Each sample is played out as if the cards were known, so
   the search slightly overrates moves that only work if you guess the hidden
   cards right. ISMCTS reduces this compared with PIMC but does not remove it.
3. **Weak moves get few simulations** in ISMCTS, so their intervals are wider.
   Compare moves #1–#3 by their intervals, not just the point estimates.
4. **Score vs. win.** The default objective maximizes expected score, which is how
   the game is scored. With `objective="win"` the solver maximizes the chance of
   going out first; the two can disagree (e.g. dumping high cards to cut a
   13-card penalty versus holding them to win tricks).

## 4. Parameters

| Parameter | Default | Notes |
|---|---|---|
| `iterations` | 10,000 (CLI analyze: 20,000) | More = tighter intervals and somewhat stronger play |
| `exploration` | 0.25 | 0.1–0.7 performed the same; ≥1.0 slightly worse |
| `threads` | 1 | Independent trees, results summed |
| `objective` | `score` | or `win` |
| `rollout_epsilon` | 0.1 | Random-move rate of the rollout policy |
| `endgame_threshold` | 0 (off) | Cards in all hands at which to solve exactly |
| `endgame_max_nodes` | 20,000 | Budget per exact solve; falls back to a rollout |
| `time_limit` | none | Seconds; stops early |

## 5. Usage

```sh
poker big2 analyze --seat 0 --hand "3d 4d 4h 5h 6d 7d 7s 8d 8s Jd Jh Kc 2s"
poker big2 analyze --seat 2 --hand "..." --history "0:3d 4d 5h 6s 7c; 1:pass; 2:..."
```

```python
from poker._core import analyze
from poker.games.big2 import Rules, pretty

state = Rules().new_deal(seed=1)
result = analyze(state, iterations=20_000, threads=4)
for m in result["moves"][:3]:
    print(pretty(m["cards"]), m["win_rate"], m["mean_score"], m["score_ci95"])
```

## References

* Cowling, Powley, Whitehouse. *Information Set Monte Carlo Tree Search.* IEEE
  TCIAIG 4(2), 2012.
* Long, Sturtevant, Buro, Furtak. *Understanding the Success of Perfect Information
  Monte Carlo Sampling in Game Tree Search.* AAAI 2010.
* Ginsberg. *GIB: Imperfect Information in a Computationally Challenging Game.*
  JAIR 14, 2001.
* Luckhardt, Irani. *An Algorithmic Solution of N-Person Games.* AAAI 1986 (max^n).
* Brown, Sandholm. *Superhuman AI for multiplayer poker* (Pluribus). Science, 2019.
* Zha et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
  Learning.* ICML 2021.
