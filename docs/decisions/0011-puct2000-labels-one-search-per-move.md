# 0011: PUCT-2,000 labels: one search per move, belief network with 2,048 candidates

Date: 2026-11-28

Orchestrator's call, 2026-11-28; Brian may change it.

Under decision 0010, Task 42's reference is PUCT at 2,000 simulations with the belief network and the exact endgame. A single PUCT search gives its visits by prior. A low-prior move, which is often the bot's mistake or a puzzle's wrong answer, gets few visits or none, so its loss and its grade word would be noise. So each move to grade (prior at least 1 %, the top 8 by prior, the move played, and for puzzles every answer) gets its own PUCT-2,000 search with the root fixed to that move (`analyze(..., root_move=<cards>)`, `big2_search::puct_from_move`). Every move uses the same seed. EV and SE come from each search's mean score and its 95 % range. The loss SE is unpaired, `hypot(se_best, se_move)`.

Two settings followed from measurement:
- **Belief candidates 2,048, not the default 256.** On B3-02 (deal 5574300040, turn 41), the loss of Q♥ against J♣ over 6 seeds had an SD of 2.63 points at 256 candidates, 1.21 at 2,048 and 1.08 at 8,192. A search costs 0.10, 0.17 and 0.23 s. The sampler tempers the network down to its 25 % effective-sample floor at every size.
- **The exact endgame now weighs deals by the belief network.** Before this, `analyze(belief="net", infoset_endgame=...)` solved every hidden deal with the same weight, including deals the rules exclude. Now each deal weighs `exp(log weight)` under the network (exact Bayes, no tempering) and rules-excluded deals weigh 0. Phase 3 defaults stay: at most 12 hidden cards, at most 10 cards in all hands, 1 M positions.

Note: a search's reported SE (about 0.5 a move on B3-02) is smaller than the spread between seeds (about 1.2 on the loss). Task 42's repeatability check (5 % of decisions labelled again with a new seed) measures the real spread.

Orchestrator (2026-11-28), to the question "one root search or one root-restricted search per option": "(a), one root-restricted PUCT-2,000 search per option, same seed, SE from each option's ci".
