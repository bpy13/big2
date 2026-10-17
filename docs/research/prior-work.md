# Prior work on Big Two and similar games, and how this project compares

**Date:** 2026-10-18. Literature found by web search on that date; numbers are
quoted from the papers (their own setups, not re-run here). Our numbers are from
[study-results/](../study-results/).

## Summary

1. **Published Big Two AI is thin and evaluated weakly.** Four works: one PPO
   self-play agent (Charlesworth 2018), one MCTS/ISMCTS system (Chen & Lu 2022),
   one DMC speed-up paper (2024) and one controlled comparison of PPO against
   value-based methods (Patwa, ICML 2026). All measure strength against random or
   rule-based bots or a few human amateurs; none plays a search solver head-to-head,
   and only Charlesworth reports a confidence interval. Our evaluation (duplicate
   deals, seat rotation, 95% CIs, head-to-head against ISMCTS) is stricter than
   any of them.
2. **The state of the art lives in the sister game Dou Dizhu** (also a
   shedding game with combinations). The line there is DouZero (DMC, 2021) →
   DouZero+ (opponent model) → PerfectDou (PPO with a critic that sees all
   hands, 2022) → **Ataraxos (Nature 2026): self-play network + belief network +
   test-time search, which beat both PerfectDou and DouZero.**
3. **Our methods match the established ones but miss the components that made
   the leaders win:**
   * our search uses a weak rollout/opponent model (the greedy bot) and no
     learned policy prior or learned belief;
   * our PPO lacks the full-information critic, GAE, reward shaping and the
     entropy level that the papers found necessary;
   * our compute is ~1,000× below the Dou Dizhu systems.
4. **The 2026 Big Two paper's "PPO beats Monte Carlo Q" does not contradict our
   result.** Its Monte Carlo Q baseline has no replay buffer and a 10× smaller
   learning rate than DouZero-style DMC; ours is the DouZero recipe. It does show
   which PPO settings matter (entropy 0.05, GAE, LR schedule), which we did not use.

**Most promising improvements, in order** (details in §5):

| # | Improvement | Evidence | Cost here |
|---|---|---|---|
| 1 | Stronger playout/opponent policy for the solver ("structured greedy", minimum-combination planning) | PerfectDou's and ODMC's use of minimum-play-count planning; our own Phase 1 finding that more simulations barely help | ~1 day, CPU |
| 2 | Network-guided search: distilled network as move prior and leaf value inside ISMCTS, learned belief for sampling, then iterate (expert iteration) | Ataraxos, AlphaZero/ExIt; our belief sampler and value net already exist | days; GPU helps |
| 3 | PPO done properly: full-information critic, GAE, entropy 0.05, LR decay, min-steps reward shaping | PerfectDou ablation (vanilla PPO 0.509 WP vs ~0.73+ with these), Patwa entropy ablation (+4.7–6.4 pts win rate) | ~1–2 days + long training |
| 4 | Opponent-hand prediction as network input | DouZero+ (peasant gains), ODMC opponent model | ~1 day |
| 5 | Action filtering to plausible combinations | ODMC (MCS), DanZero+ (top-k DMC candidates for PPO) | ~1 day |

## 1. Big Two

| Work | Method | Compute | Evaluation | Result |
|---|---|---|---|---|
| Charlesworth 2018 (arXiv:1808.10442) | PPO self-play; fixed list of 1,695 actions with masking; 412 hand-crafted features; 512-256 MLP | 150 M steps ≈ 3 M games, 4-core PC + GPU, ~2 days | 664 games vs 7 human amateurs | Humans averaged −0.96 ± 0.38 per game; AI won 22.5% of games (scores: winner gets others' card counts) |
| Chen & Lu 2022 (IEEE Access 10:40661) "Big2AI" | Card-superiority analysis, dynamic weights, learned game features, multi-opponent move prediction with MCTS/ISMCTS | — | — (paper not accessible; abstract only) | — |
| *Applied Soft Computing* 158, 2024 "Improved learning efficiency of DMC…" | DMC + an **opponent model** predicting hidden cards + **ODMC**: Minimum Combination Search filters suboptimal actions | — | Dou Dizhu and Big 2 | Comparable strength to DMC with less training time (abstract) |
| Patwa 2026 (ICML, arXiv:2605.28863) | Controlled comparison: PPO vs Monte Carlo Q vs SARSA vs Q-learning, same network (card embeddings + self-attention, legal moves scored by dot product) | 320 k games (5,000 × 64), 6-core laptop, 7–13 h per run | 1,000 games vs 3 random / greedy / "smart" bots; single seed, no CIs | Win rate vs 3 greedy: **PPO 64.8%** (entropy 0.05), PPO 58.2% (entropy 0), MC-Q 50.2%, Q-learning 49.8%, SARSA 44.2% |

Notes on Patwa 2026, the closest to our setup:

* **PPO settings that worked:** lr 3e-5 with warm-up and cosine decay, clip 0.2,
  γ 0.99, GAE λ 0.95, 4 epochs, minibatch 256, value coefficient 0.5, gradient
  clip 0.5, **entropy 0.05** (0.10 slightly worse; 0 let the policy pick its top
  move with 90%+ probability "in ambiguous but strategically important" spots).
* **Its Monte Carlo Q is not DouZero's DMC:** ε-greedy decaying 0.5 → 0, one
  update per batch, **no replay buffer**, lr 3e-5. DouZero (and our DMC) learn
  from a replay of many recent games at lr ~1e-4–3e-4. So its ranking says little
  about DMC as we implement it.
* **Self-play against the current policy beat** training against past checkpoints
  or a fixed heuristic bot, for both PPO and MC-Q.

## 2. Dou Dizhu, Guandan and other shedding games

| Work | Key idea | Compute | Result |
|---|---|---|---|
| Whitehouse, Powley & Cowling 2011 | Determinized MCTS and ISMCTS for Dou Di Zhu | — | No significant difference between the two |
| DouZero (Zha et al., ICML 2021) | **DMC**: Q(s, a) with action features, LSTM over history, Monte Carlo returns, many actors | billions of samples, multi-GPU, days | Beat all earlier Dou Dizhu programs |
| DouZero+ (2022, arXiv:2204.02558) | **Opponent model** (predicts next player's cards, fed to the Q-network); **coach network** skips lopsided deals | — | Top of Botzone (400+ agents); gains mainly for peasants |
| PerfectDou (NeurIPS 2022, arXiv:2203.16406) | **PPO + GAE with a critic given all hands** ("perfect-training-imperfect-execution"); reward shaping from **minimum steps to play out** (dynamic programming); 621 abstract actions; entropy 0.1 | 880 CPU cores + 8 GPUs, 2.5e9 steps | 0.543 win rate vs DouZero; 1e9 steps beat DouZero's 1e10. **Ablation: vanilla PPO 0.509 WP ("fails"), no full-info critic 0.717, no shaping 0.738 WP but loses points** |
| AlphaDou (2024, arXiv:2407.10279) | End-to-end including bidding, no hand-crafted abstractions | — | — |
| DanZero / DanZero+ (Guandan, arXiv:2210.17087, 2312.02561) | DMC (80 CPU actors, 30 days); then **PPO choosing among the DMC model's top-k moves** (k = 2 best) | <1 day extra for PPO | 90.1% → 92.7% vs strongest baseline; 71/100 vs skilled humans. Plain PPO was impractical with 5,000+ legal moves |
| **Ataraxos (Nature 2026)** | Self-play **policy-value network** + separate **belief network** for hidden information + **test-time search** that samples hidden states and runs depth-limited rollouts; "dynamically damped" self-play | 16 H100 × 1 week + 4 H100 × 4 days (belief), 163 M games | Superhuman Stratego; Hanabi SOTA; **beat PerfectDou and DouZero in Dou Dizhu**, significant |

## 3. General imperfect-information methods (for context)

* **ReBeL** (2020) and **Student of Games** (2023): search + learning with
  game-theoretic guarantees in two-player zero-sum games (poker, Scotland Yard).
  Big Two is four-player and not zero-sum between pairs, so the guarantees do
  not carry over; the cost (belief over all hidden-card assignments) is large.
* **SePoT** (IJCAI 2024, arXiv:2312.15220): look-ahead search on top of a
  policy-gradient network using a critic over "policy transformations";
  significant gains in small games, needs tractable belief reconstruction.
* **ISMCTS** (Cowling, Powley & Whitehouse 2012) — what our solver is — remains
  the standard training-free baseline; its known weaknesses are strategy fusion
  and its dependence on the rollout policy and the sampling of hidden cards.

## 4. How this project compares

| Component | Here | Best practice found | Gap |
|---|---|---|---|
| Rules engine | Rust, rule variants, differential-tested | Comparable (most use Python) | none; faster than most |
| Training-free search | ISMCTS, greedy rollouts, uniform or greedy-model belief sampling | Big2AI: ISMCTS + opponent move prediction; Ataraxos: learned belief + network-guided search | **rollout/opponent model and belief are rule-based** |
| Self-play value learning | DMC as in DouZero (replay, action features, history of 12 moves) | DouZero, DanZero | compute (~2.7–5.6 M games vs billions); no opponent model input |
| Policy gradient | PPO, Monte Carlo advantage (no GAE), entropy 0.01, constant lr 3e-4, critic sees own hand only | PerfectDou: full-info critic, GAE, shaping, entropy 0.1; Patwa: entropy 0.05, GAE, lr decay | **all four missing**, matching PerfectDou's failed "vanilla PPO" ablation |
| Search distillation | ISMCTS-labelled data → Q-network (expert iteration, one round) | AlphaZero/ExIt iterate it; Ataraxos combines net + search | one round only; network not used inside search |
| Network + search | `dmcsearch` (network plays out sampled deals), value-net leaves | Ataraxos | no learned belief; no policy prior in tree |
| Evaluation | Duplicate deals, seat rotation, CIs, vs solver head-to-head | Literature mostly vs heuristic bots, no CIs | ahead; but no human evaluation and different scoring from the literature |

**Can our strength be compared with the papers?** Only loosely.

| Agent vs 3 greedy bots (win rate) | Win rate |
|---|---:|
| Patwa PPO (entropy 0.05) | 64.8% |
| Our ISMCTS-1000 | 48.0% ± 4.5% |
| Our distilled network (+10 h) | 47.9% |
| Our DMC (+10 h) | 39.2% |
| Our PPO (+10 h) | 34.1% |

Differences that make this table unreliable:

* Scoring: the papers reward the winner with the other players' card counts.
  We use tiered pairwise settlement, and our agents maximize expected score, which
  Phase 1 showed costs win rate (score-seeking ISMCTS won 23.2% vs 26.8%
  head-to-head against its win-seeking twin, while scoring +7.6 more per deal).
* "Greedy" differs: theirs plays its weakest legal non-pass action; ours leads the
  largest combination containing its weakest card.

A fair comparison needs both our agents and a re-implementation of their
setup under the same scoring and bots (§5, item 6).

## 5. Improvements, with what the literature says about each

1. **Stronger playout and opponent policy for the solver.** ISMCTS's estimates are
   only as good as the policy that plays out the simulated games, and ours breaks
   pairs and five-card hands to beat singles. PerfectDou and ODMC both use a
   *minimum number of plays to empty the hand* computation; the same planner gives
   a "structured greedy" that keeps combinations together. Measure: structured vs
   old greedy, and ISMCTS with each, head-to-head. Cheap; benefits every component
   built on the solver (hints, distillation labels, the opponents' model).
2. **Network-guided search with a learned belief (Ataraxos-style).** Put the
   distilled network inside ISMCTS as a move prior (PUCT) and as the playout
   policy or leaf value, sample hidden hands from a learned belief network instead
   of uniformly, then distill the stronger search again (expert iteration).
   We have the parts: belief-weighted sampling (`belief=greedy`), a value network
   (R² 0.45) and `dmcsearch`. Our earlier attempts failed because the value network
   was too weak and the belief model was the greedy bot; both are learnable.
3. **PPO as the leaders do it.** Full-information critic (our `encode_full`
   already exists), GAE λ ≈ 0.95, entropy ≈ 0.05 with decay, LR decay, and
   optionally a shaping reward from the change in minimum plays to empty the hand
   (computed by the planner from item 1). This tests whether PPO's weak showing
   here was the implementation.
4. **Opponent-hand prediction as input** (DouZero+, ODMC): a small network trained
   on self-play to predict who holds each unseen card; its output is added to the
   Q-network's features and can double as the belief for search (item 2).
5. **Action filtering** (ODMC's Minimum Combination Search, DanZero+'s top-k):
   restrict learning and search to moves consistent with good hand decompositions.
   Mostly a speed-up; also helps PPO, whose softmax spans every legal move.
6. **Comparable evaluation.** Add the literature's scoring and greedy definition
   as options and report our agents under them; add a stronger hand-written bot
   (like Patwa's "smart"); collect human games from the UI (`runs/games/`).
7. **Compute.** Every leading system trained 10–1,000× longer. On this laptop,
   the cheapest win is items 1–2 (better search), not longer self-play. A GPU
   (`--device cuda`) speeds the learning step most.

## Sources

* Charlesworth, H. *Application of Self-Play Reinforcement Learning to a
  Four-Player Game of Imperfect Information.* arXiv:1808.10442, 2018.
  <https://arxiv.org/abs/1808.10442>
* Chen, L.-W. & Lu, Y.-R. *Challenging Artificial Intelligence With Multiopponent
  and Multimovement Prediction for the Card Game Big2.* IEEE Access 10:40661–40676,
  2022.
* *Improved learning efficiency of deep Monte-Carlo for complex imperfect-information
  card games.* Applied Soft Computing 158, 2024.
  <https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193>
* Patwa, A. *Self-Play Reinforcement Learning under Imperfect Information in Big 2.*
  ICML 2026, arXiv:2605.28863. <https://arxiv.org/abs/2605.28863>
* Whitehouse, D., Powley, E. & Cowling, P. *Determinization and Information Set
  Monte Carlo Tree Search for the Card Game Dou Di Zhu.* IEEE CIG 2011.
* Cowling, P., Powley, E. & Whitehouse, D. *Information Set Monte Carlo Tree
  Search.* IEEE TCIAIG, 2012.
* Zha, D. et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
  Learning.* ICML 2021.
* *DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided
  Learning.* arXiv:2204.02558. <https://arxiv.org/abs/2204.02558>
* Yang, G. et al. *PerfectDou: Dominating DouDizhu with Perfect Information
  Distillation.* NeurIPS 2022, arXiv:2203.16406. <https://arxiv.org/abs/2203.16406>
* *AlphaDou: High-Performance End-to-End Doudizhu AI Integrating Bidding.*
  arXiv:2407.10279. <https://arxiv.org/abs/2407.10279>
* *DanZero: Mastering GuanDan Game with Reinforcement Learning.* arXiv:2210.17087;
  *DanZero+: Dominating the GuanDan Game through Reinforcement Learning.*
  arXiv:2312.02561.
* *Scalable decision-making for games of imperfect information* (Ataraxos).
  Nature, 2026. <https://www.nature.com/articles/s41586-026-11036-y>
* Brown, N. et al. *Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games* (ReBeL). arXiv:2007.13544.
* Schmid, M. et al. *Student of Games.* Science Advances, 2023, arXiv:2112.03178.
* *Look-ahead Search on Top of Policy Networks in Imperfect Information Games*
  (SePoT). IJCAI 2024, arXiv:2312.15220.
