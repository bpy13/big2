# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

**Which agreement measure should the A21 grader bar use?**
- What it is: Task 27's re-grade gives 90.2 % same grade across seeds (strict) and 97.6 % compatible (counting `unclear` as agreeing). The bar is 95 %.
- Example: a move graded `mistake` once and `unclear` the second time counts as a disagreement under strict, an agreement under compatible.
- Why it matters: Task 27 stays unticked until the bar is defined; Tasks 28 and 31 wait on it. Details in `docs/open-items.md`.
- Options: (a) compatible agreement; (b) strict agreement with a larger deal budget (93.8 % at 4,800 deals, still short). Default: (a).

**Which search do `hard` and `expert` ship in the app?**
- What it is: Task 29's ladder run: with `hard` = ISMCTS-2,000 and `expert` = ISMCTS-10,000, expert vs hard is -0.03; `hard` = ISMCTS-500 gives a strict order with a top gap of +1.02 (bar 1.5).
- Example: `hard` at 500 simulations is weaker but visibly below `expert`.
- Why it matters: A23 (strict ordering, gaps of 1.5) cannot be met as configured; Task 29 stays unticked. Details in `docs/open-items.md`.
- Options: (a) `hard` = ISMCTS-500, `expert` = ISMCTS-10,000 + exact endgame, record the short gap; (b) keep the app's current budgets. Default: (a).

## Answered

