# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

**Grade words, accuracy and the luck-removed rating: do these settings go into the app?** (Task 43)
- What it is: how the app grades one move, scores one game, and rates a player. (1) A move's loss
  is the points the expert's search says it gave away. Trivial moves (only one legal move, or
  every move worth the same within noise) get no word and count in nothing; that is about half
  of all moves. (2) Words by loss in points: great up to 0.5, good up to 2.5, inaccuracy up to
  7, mistake up to 10.5, blunder above (today: 0.2 / 0.69 / 2.75 / 6.86). (3) A game's accuracy
  (0-100) is the average of `100 x exp(-loss / 10.1)` over its non-trivial moves, so a loss of 7
  scores 50; the average points lost (APL) is shown beside it. (4) The rating is points won
  minus the luck of the cards: the luck of a deal is what the expert's network scores when it
  plays all four seats of that deal (16 sampled games). Random play is 0, the raw network 3000,
  and the network with search lands at 3047. A player's rating moves after each deal by
  `K x 102 x (points minus luck minus what the table's ratings predict)`, K = 1/(deals + 21)
  until it reaches 1/400. Details: `docs/study-results/grading-template.md`,
  `runs/p6_grading/proposal.json`.
- Example: in Task 42's games the raw network (the best bot without search) loses more than 2.75
  points on 10 % of its non-trivial moves, so today's lines call a tenth of the expert's own moves
  mistakes; under the new lines it plays 75 % great, 14 % good, 8 % inaccuracy, 3 % mistake or
  blunder. Where the six bots sit on the new scale: random 0, easy (greedy) 1989, medium
  (Q-network) 2305, hard (ISMCTS-500) 2425, old expert (ISMCTS-10,000) 2712, raw hour-24 network
  3000; new expert (network + PUCT-2,000) 3047 [2981, 3111]. A player who has played 100 deals
  has a rating known to about +-420; after 400 deals +-210; after 2,000 deals +-95 (the cards
  decide about a third of a deal's score; the luck term removes 35 % of the variance, the most the
  method can reach is 37 %).
- Why it matters: every grade word players see, the accuracy number on each game, and the rating
  shown on the ladder and used to match puzzles; hk-games Tasks E2 and E3 build from this file.
- Options: (a) take the proposal as it stands (lines 0.5 / 2.5 / 7 / 10.5, accuracy as above,
  rating with the raw network at 3000 and the new expert at 3047, show the APL-based estimate
  only as "performance this game"); (b) same, but pin the new expert at 3000 (then the raw network
  is 2954, old expert 2670, medium 2270); (c) keep today's lines (0.2 / 0.69 / 2.75 / 6.86) and
  take only the trivial rule, accuracy and rating. Default: (a).
- Card sent 2026-11-28 (orchestrator).

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
