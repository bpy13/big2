# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

## Answered

**Keep the hour-24 self-play network for `expert`, or keep the old one?** (Task 36)
- What it is: Task 36's self-play from r3 ran 24 GPU-hours. The network alone (no search) beats the app's `expert` (ISMCTS-10,000) by +3.62 / +3.25 a deal (1+3 / ABAB) at hour 24, +3.39 [+2.96, +3.84] over r3. Hours 8-24 are level head to head (all ranges cover 0). Details in `docs/study-results/selfplay-luck.md`.
- Example: hour 24 vs hour 20 +0.07 / +0.40, vs hour 8 +0.25 / +0.38 (2,000 deals per lineup).
- Why it matters: the network inside the new expert (Tasks 38-41, 45) and every grade built on it.
- Options: (a) hour 24 for the expert; (b) keep the old ISMCTS-10,000 `expert`. Default: (a).
- Brian (2026-11-27): "agree, use hour 24 as the expert" (decision 0007). Recorded by Task 36, which the question closes.

**May a big2 worker add the phase 3 entry to hk-games `docs/streams/bot-notes.md`?** (Task 25)
- What it is: Task 25's last step is an entry in the hk-games checkout. The permission system refused my edit there (outside the big2 worktree). The finished text is `docs/bot-notes-phase3-entry.md` in big2.
- Example: the entry says no model file ships, lists `RulesMask`, `NetSampler`, `infoset_endgame`, what `quality` carries, and the B2/B3 results in three lines.
- Why it matters: Task 25 stays "(waiting: inbox)" until the entry is in bot-notes (A24). Everything else in Task 25 is done and pushed.
- Options: (a) you or the orchestrator paste the file into bot-notes.md, newest first; (b) allow workers to commit that one file in hk-games. Default: (a).
- Brian (2026-11-17): "yes" (to "Big2: may I add the finished notes entry to the games project? ... I can add it from my side. Recommend: yes."). The orchestrator added the entry to hk-games `docs/streams/bot-notes.md` (hk-games 877b6f9).

**Should big2 stop after training round 3 and ship the search bot as `expert`?** (Task 17) r3
misses A9 in the ABAB lineup (+0.66, bar +1.0).
Brian (2026-11-11): "For now ship the search bot as expert and queue the round 4 task as Milestone 2"
**Which agreement measure should the A21 grader bar use?**
- What it is: Task 27's re-grade gives 90.2 % same grade across seeds (strict) and 97.6 % compatible (counting `unclear` as agreeing). The bar is 95 %.
- Example: a move graded `mistake` once and `unclear` the second time counts as a disagreement under strict, an agreement under compatible.
- Why it matters: Task 27 stays unticked until the bar is defined; Tasks 28 and 31 wait on it. Details in `docs/open-items.md`.
- Options: (a) compatible agreement; (b) strict agreement with a larger deal budget (93.8 % at 4,800 deals, still short). Default: (a).
- Orchestrator (2026-11-10): default (a) taken; Brian may change it.

**Which search do `hard` and `expert` ship in the app?**
- What it is: Task 29's ladder run: with `hard` = ISMCTS-2,000 and `expert` = ISMCTS-10,000, expert vs hard is -0.03; `hard` = ISMCTS-500 gives a strict order with a top gap of +1.02 (bar 1.5).
- Example: `hard` at 500 simulations is weaker but visibly below `expert`.
- Why it matters: A23 (strict ordering, gaps of 1.5) cannot be met as configured; Task 29 stays unticked. Details in `docs/open-items.md`.
- Options: (a) `hard` = ISMCTS-500, `expert` = ISMCTS-10,000 + exact endgame, record the short gap; (b) keep the app's current budgets. Default: (a).
- Orchestrator (2026-11-10): default (a) taken; Brian may change it.
