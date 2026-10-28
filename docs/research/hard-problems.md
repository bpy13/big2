# Big Two (鋤大D): what makes a bot hard, what is known, and our ideas

**Date:** 2026-10-28. Companion to `survey.md` (same folder), which holds the citations in full.
Each section states the problem, what the literature and our own studies know, and at least
three ideas of our own marked **(our idea)**, each with an expected cost and a test. Costs are
for one person with helper agents on the local box (RTX 4060 Ti 16 GB, i7-12700K, 32 GB).
"Points" means our tiered pairwise score per game (`big2\docs\rules\big2.md` §7); a 100-deal
duplicate match resolves about ±1.5 points, 1,000 deals about ±0.5 (`phase1-solver.md`).

Teaching ideas are marked **(teaching)**. They serve D1 (every hint carries a reason) and D42
(the engine returns keys and numbers, never text).

## Summary of the ideas we rate highest

| # | Idea | Problem | Cost | First test |
|---|---|---|---|---|
| 1 | Decomposition-first move abstraction with a min-plays planner, used by the rollout policy, the network's candidate list and the reason keys (§4) | action space, lead control, teaching | 1 week | ISMCTS with the structured rollout vs today's ISMCTS, 300 duplicate deals |
| 2 | Expert iteration with the network inside the search (PUCT prior + value leaves), three rounds, distilled to a `.pknn` for the phone (§3, §7) | belief, phone budget | 2–3 weeks GPU | each round vs the previous expert, 1,000 deals |
| 3 | Full-information critic + oracle guiding for self-play, VRPO-style advantage (§1, §3) | four players, hidden hands | 1 week + 2–4 GPU-days | self-play net vs DMC-10h and ISMCTS-1000 |
| 4 | Cards-left distribution head conditioned on the scoring variant, for threshold-aware play and threshold reasons (§2) | scoring variants | 3 days | score vs the mean-score head under tiered and win_loss |
| 5 | Exact information-set endgame when the unseen cards are few, with "forced" proofs for review (§6) | endgames, teaching | 1 week | agreement with 100k-sim ISMCTS and time per call on wasm |
| 6 | Value-lost grading with counterfactual reason keys from the solver's own move table (§8) | teaching | 1 week | human readers rate reasons on 50 positions |

## 1. Four players, no equilibrium, non-zero-sum between pairs

**Why it is hard.** With four players there is no strategy that cannot lose in expectation;
a third player's choice can move points between the other two (ADR 0002). Pairwise settlement
makes this concrete: the sum of scores is zero, but any pair's sum is not. A player who cannot
win still decides who wins by what they pass on. Self-play finds one convention; humans at a
Hong Kong table follow another (feeding the player on your right, blocking the leader).

**What is known.**
- Pluribus shows multi-player search works empirically without a guarantee (Brown & Sandholm
  2019). Multi-player UCT with each player maximising their own score is what our ISMCTS does.
- Dou Dizhu's leaders sidestep this: the two peasants share a payoff, so the game is two-sided.
  Ataraxos even evaluates it as heads-up (one agent plays both peasants). None of them faces
  three independent opponents.
- Patwa 2026 found current-policy self-play beat checkpoint self-play and fixed-opponent
  training in Big Two (64.8% vs 61.6% vs 55.4% vs greedy; one seed). Self-play against
  yourself converges on a convention that may not be the human one.
- Our Phase 3: a network that learned from the solver inherits the solver's greedy opponent
  model and over-scores against greedy bots; only head-to-head play is honest.

**Ideas.**
- **Population self-play with style tags (our idea).** Train against a pool: current net,
  ISMCTS, structured greedy, a "feeder" that dumps high cards early, a "blocker" that always
  beats the leader when it can. Give the network a one-hot "opponent style" input during
  training, and at play time feed either "unknown" or a style inferred from the history.
  Cost: 3 days plus 2 GPU-days. Test: a net trained with the pool vs a net trained by pure
  self-play, each against the pool and against each other, 1,000 duplicate deals; the pool
  net should lose less to every style.
- **All-seat value heads (our idea).** Predict the final score of every seat, not only the
  actor's. The actor's choice still maximises its own score, but the other three predictions
  give the search a cheap kingmaking check and give teaching the key
  `reason.helps_seat` with `{seat, delta}` ("this pass gives the player on your left the
  lead"). Cost: 2 days; the full-information training data already contains every seat's
  result. Test: head accuracy (R²) per seat on held-out self-play; and a 300-position review
  where a human checks whether "helps seat X" statements are true.
- **Best-response probe as the exploitability number (our idea).** Freeze a candidate bot,
  train a DMC agent for 10 GPU-hours against three copies, report its gain. Use the number
  as the internal "unexploitable enough" metric (bar B3b in `survey.md`). Cost: 10 GPU-hours
  per candidate, automated. Test: the probe should gain more against the greedy bot than
  against ISMCTS-1000, which validates the probe before we trust it on new bots.
- **Variant-aware convention check (our idea, teaching).** Log the human games from the app
  and compute, per position type, how often humans agree with the expert bot. Positions where
  skilled humans disagree systematically mark a convention gap, not a mistake. Cost: 2 days
  once the app logs games. Test: the disagreement rate should fall as human Elo rises; if it
  does not, the bot's convention is the odd one.

## 2. Scoring variants, and score versus win rate

**Why it is hard.** Tiered scoring multiplies the penalty at 8, 10 and 13 cards. A player with
10 cards who sheds two cards halves their penalty. The right move depends on the variant: under
`win_loss` only going out matters. Our Phase 1 measured the gap: a win-seeking solver scores
7.6 points per game less than a score-seeking one. Every published Big Two agent trains on
"winner gets the sum", which rewards none of the thresholds.

**What is known.**
- AlphaDou keeps two heads (win probability and expected points), prunes to moves within 5% of
  the best expected points, then picks the highest win probability. That is a two-objective
  rule we can copy.
- PerfectDou shapes the reward with the change in minimum plays to empty the hand; its
  ablation says shaping costs little win rate and gains points.
- Our networks take the variant at training time only; the web UI offers a network only for
  the rules it was trained under (`rl.md` §4).

**Ideas.**
- **Cards-left distribution head (our idea).** Instead of a mean score, predict the probability
  of finishing with 0, 1, …, 13 cards for the actor (14 outputs), plus the same for each
  opponent (all-seat heads from §1). Any scoring variant is then a dot product with a payoff
  vector, so **one network serves every variant** and the solver can show "chance to finish
  under 8 cards" as a number. Cost: 3 days (head, loss, export). Test: under tiered, win_loss
  and original, the distribution net vs the mean-score net, 1,000 duplicate deals each; and
  calibration of the predicted distribution on held-out games.
- **Threshold-aware endgame objective (our idea).** When the actor cannot win within the
  minimum-plays bound (§4) but can cross a tier boundary, switch the search objective to
  "expected penalty" with the tier table, and emit `reason.cross_tier` with
  `{from: 10, to: 7}`. Cost: 2 days inside the existing `objective` switch. Test: value lost
  on 300 positions where the actor holds 8–12 cards and the leader is at 1–3 cards.
- **Variant as input, not as a model (our idea).** Encode the scoring, straights, flush order,
  pass rule and last-card-duty variants as inputs to one network, trained on a mixture. This
  removes the "one network per rule set" limit in `rl.md` §4 for the phone download. Cost:
  2 days plus one training run per mixture. Test: the mixed net vs a per-variant net under each
  variant, 500 deals; accept if the loss is under 0.5 points.
- **Score-or-win explanations (our idea, teaching).** Every hint already shows win rate and
  expected score. Add the key `reason.score_vs_win` with `{best_for_score, best_for_win}`
  when the two disagree, so a learner sees why the hint sacrifices win chance. Cost: 1 day.
  Test: count how often the two disagree in 1,000 positions; if under 5%, show it only in
  review.

## 3. Hidden hands and belief

**Why it is hard.** 39 unseen cards sit in three hands; the number of deals consistent with the
counts is C(39,13)·C(26,13) ≈ 8 × 10^16 at the start. Players may pass with a winning card, so
few actions are hard constraints. Our measurement (`search-methods.md`): a cheating search gains
+2.95 points, but even an exact opponent model moves the per-card belief only from 0.37 to
0.47, and against solver opponents to 0.40.

**What is known.**
- Ataraxos trains an autoregressive belief transformer (3.3 M params, 8 H100-hours in Dou
  Dizhu) on self-play snapshots and samples hidden hands from it for search. It does not ablate
  the belief net in Dou Dizhu.
- DouZero+ predicts the next player's per-rank counts and feeds them to the Q-network; gains
  mainly for peasants (who coordinate). Luo & Tan 2024 report an opponent model for Big 2
  (abstract only).
- PerfectDou's full-information critic learns the value as if hands were visible, then plays
  without them; OADMCDou goes further with oracle guiding (train with the hands visible, fade
  them out), +0.104 ADP over DouZero at a small budget.
- Goodman's re-determinizing ISMCTS stops the search's opponent nodes from using the searcher's
  own cards; the leak is worst when the opponent model is strong.
- Bertram et al. 2024: for value targets, three sampled deals per position across many
  positions beat many samples per position.

**Ideas.**
- **Hard-constraint belief from the rules (our idea).** Under `last_card_duty` and
  `locked_out`, and from "did not play" facts, some deals have probability zero. More subtly,
  a player who led a single while holding only three cards tells us the other two are not a
  pair with it (otherwise they would have led a triple) — true only under a policy
  assumption, but a strong one. Build a small set of such rules, apply them as hard filters
  in the sampler before any learned weighting, and measure the share of deals they reject.
  Cost: 3 days. Test: per-card belief accuracy on 1,000 positions (the `belief-accuracy`
  experiment exists), and ISMCTS with the filter vs without, 300 deals.
- **Belief from the current bot, retrained each expert-iteration round (our idea).** Train a
  small network to predict the holder of each unseen card from the history, on self-play of
  the current best bot (as Ataraxos), and use it as the sampler in the next round's search.
  Report its log-likelihood every round. Cost: 2 days plus 2 GPU-hours per round. Test:
  log-likelihood against the uniform and greedy-model samplers, and search gain at 300
  deals; stop investing if the gain stays under 0.5 points, which our measurements predict.
- **Oracle-guided self-play, then a full-information critic (our idea, borrowed and combined).**
  Start DMC or PPO with opponents' hands visible, fade them out over the first third of
  training (OADMCDou), keep the critic full-information for the rest (PerfectDou). Our
  `encode_full` already exists. Cost: 1 week plus 2–4 GPU-days. Test: the self-play net vs
  DMC-10h and ISMCTS-1000, 1,000 duplicate deals.
- **"Highest card still out" and "safe card" keys (our idea, teaching).** From the belief,
  compute for each of the actor's cards the probability that no opponent can beat it as a
  single, and the same for pairs. Emit `reason.safe_single` `{card, p}` and
  `reason.beatable_by` `{card, seat, p}`. These are exact under the uniform belief and
  improve with any learned belief. Cost: 2 days. Test: 50 positions reviewed by a strong
  player for whether the key names the right card.

## 4. The action space and hand decomposition

**Why it is hard.** A 13-card hand can hold hundreds of legal five-card plays; the real decision
is which decomposition of the hand into plays to aim for. Charlesworth used a fixed list of
1,695 actions; Patwa scores each legal action; our network scores each move with its
"hand left after it". None of them plans the decomposition.

**What is known.**
- PerfectDou computes the minimum number of plays to empty a hand by dynamic programming and
  uses it for reward shaping. Luo & Tan's ODMC uses a minimum-combination search to filter
  actions. DanZero+ lets PPO choose only among the DMC model's top two moves; k = 3 and 5 were
  worse.
- Our greedy rollout breaks pairs and five-card hands to beat singles; the `prior-work.md`
  improvement list rates a "structured greedy" as the cheapest win.

**Ideas.**
- **Decomposition-first abstraction (our idea).** Enumerate the actor's minimal-play
  decompositions (usually under 20 for 13 cards; the DP that counts them is the min-plays
  planner). A move is "play group G of decomposition D". Candidates are the union over the
  top few decompositions, which cuts the move list to a dozen and labels each move with the
  structure it keeps. Use the same list for the rollout policy, the network's candidate set
  and the PUCT prior. Cost: 1 week (Rust planner, tests against brute force). Test: ISMCTS
  with the structured rollout vs today's ISMCTS, 300 duplicate deals; expect +1 to +2 points
  from the `prior-work.md` reasoning, then verify.
- **Two-level move: decomposition then timing (our idea).** Let the network first score
  decompositions (what the hand aims to become), then score "which group now" given the
  trick. Both heads share the state tower. The first head changes rarely within a deal, so
  the phone can cache it. Cost: 4 days. Test: agreement with the 100k-sim reference and
  ms per move on wasm, vs the flat move head.
- **Keep-structure reason keys (our idea, teaching).** Every candidate carries the
  decomposition it keeps; the hint emits `reason.keeps` `{combo: "full_house"}` and
  `reason.breaks` `{combo: "straight", cost}` where cost is the solver's value difference.
  Cost: 2 days on top of the planner. Test: readers rate 50 hints with and without the key.
- **Min-plays race display (our idea, teaching).** Show each seat's minimum plays to go out
  (ours exactly, theirs as the belief's expectation) as `numbers.plays_to_out`. "You need 4
  leads; the player on your right needs 2" explains most lead-control decisions. Cost: 1 day
  after the planner. Test: correlation between the race gap and the solver's win rate; show it
  only if the correlation is strong.

## 5. Control of the lead (the free-playing right)

**Why it is hard.** Only the leader can play a five-card hand of their choice; everyone else
must beat the current play or pass. Winning a trick with a high card costs that card; passing
now to keep the 2♠ for later is a tempo trade. Chen & Lu 2025 make "when to seek the
free-playing right" an explicit decision (abstract only). No one reports what it is worth.

**What is known.**
- Our ISMCTS plans several tricks ahead inside each sampled deal, so it already values the
  lead implicitly; the greedy rollout does not.
- Patwa's "smart" bot keeps 2s and avoids orphan singles by hand-written penalties, and
  it is his hardest opponent (PPO 43.5% vs it).

**Ideas.**
- **Lead value as a learned quantity (our idea).** Train the all-seat value head (§1) on
  positions tagged with "who leads next" and report `numbers.lead_value` = the score
  difference between the actor leading the next trick and the opponent leading. It becomes a
  feature for the search's tree policy and a teaching number. Cost: 2 days. Test: the
  distribution of lead value across positions; and a drill set where the only difference is
  who holds the lead, graded by the solver.
- **Pass-and-keep candidate (our idea).** Add "pass, intending to win the next trick with
  card X" as an explicit candidate in the move list, so the search credits a pass with the
  plan behind it instead of treating pass as "no move". The engine still plays a plain pass;
  the plan is a reason key `reason.pass_to_keep` `{card}`. Cost: 2 days. Test: agreement of
  hints with the 100k-sim reference on the 20% of positions where pass is best.
- **Tempo drills (our idea, teaching).** Generate positions from solver games where the best
  move passes with a winning card available and the value lost by playing is over 2 points;
  tag them `skill.when_to_pass`. Rate them by the value gap. Cost: 1 day once review grading
  exists. Test: human solve rate should fall with the rating.

## 6. Endgames

**Why it is hard.** The deal ends when one player goes out, so "few cards" means few in the
leader's hand, not in all hands; our 8-card threshold never fired. Max^n assumes every player
plays perfectly with full information and did not beat greedy rollouts against greedy
opponents (`phase1-solver.md` §4). Yet endgames are where humans make most counted mistakes
(cannot be verified until the app logs games).

**What is known.**
- Our exact solver exists (`endgame_threshold`, `endgame_max_nodes`) with a transposition table.
- Bertram et al. 2024 says few sampled deals per position suffice for value targets, which
  bounds the cost of an information-set endgame.

**Ideas.**
- **Information-set exact endgame (our idea).** When the number of unseen cards is small
  (say ≤ 12) enumerate every consistent deal (C(12,4)·C(8,4) = 34,650 at the worst split),
  solve each by max^n, and average by the belief. The result is an exact expected score per
  move under "opponents play perfectly with full information", with no sampling noise. Cost:
  1 week with a wasm time check. Test: agreement with 100k-sim ISMCTS on 300 late positions,
  and time per call; set the threshold where the call fits 1 s on the phone.
- **Forced-outcome proofs for review (our idea, teaching).** Within the exact endgame, when
  every consistent deal gives the same winner for a move, emit `reason.forced_win`
  `{plays: n}` or `reason.forced_loss`. Review shows "you had a forced win in 2 plays".
  Cost: 2 days on top of the exact endgame. Test: count forced positions in 1,000 solver games;
  check 20 proofs by hand.
- **Endgame vs strong opponents, re-tested (our idea, cheap).** The Phase 1 "no gain" result
  was against greedy bots. Re-run the exact endgame inside ISMCTS against ISMCTS and the
  DMC-10h network, 300 duplicate deals. Cost: hours. Test: that match; keep the endgame off
  if it still ties.

## 7. The phone budget

**Why it is hard.** One WebAssembly thread, 0.5–2 s per hint, less for `bot_action`, no
PyTorch, no `Instant`, models loaded from bytes (`engine-interface.md` §1). The app stream
estimates ISMCTS at 20,000 sims ≈ 0.5 s on a phone and says 5,000 sims lose little. Ataraxos'
search (200 full rollouts through a 2 M-parameter transformer) is roughly 8,000 forward passes
per move, which is 8–30 s on one wasm thread (own estimate in `survey.md` §5).

**What is known.**
- Our distilled MLP (0.8 M params) runs at ~1 ms in PyTorch; the Rust `crates/nn` already
  evaluates `.pknn` value nets with no Python. `tract` runs ONNX in wasm if we need more.
- DanZero+ shows a network plus a tiny choice (top-2) can beat the network alone; AlphaDou
  shows two heads and 5% pruning; both are millisecond-class.

**Ideas.**
- **PUCT with the distilled policy-value net, 500–2,000 simulations (our idea, from AlphaZero
  and Ataraxos).** Put the network inside ISMCTS as the prior and the leaf value, no rollouts.
  Per simulation one forward pass (~0.2 ms in Rust for the MLP, own estimate), so 2,000 sims
  fit 0.5 s. Cost: 1 week (Rust PUCT, model export of the policy head). Test: vs ISMCTS-5000
  with greedy rollouts at equal wall time on one thread, 1,000 duplicate deals.
- **Anytime hints (our idea).** Return the best move after 100 ms, then refine until the
  budget ends; `budget_used_ms` already exists in the interface. The app can show the first
  answer at once and update the numbers. Cost: 1 day. Test: measure how often the move changes
  after 100 ms (should be under 15%); if more, the early answer is misleading.
- **Level ladder by budget, not by model (our idea).** random → ε-greedy structured greedy
  (easy) → distilled net with temperature (medium) → PUCT 300 sims (hard) → PUCT 2,000 sims
  plus the exact endgame (expert). One model file, five levels. Cost: 2 days. Test: a
  round-robin of the five levels, 500 deals, must order them strictly with gaps of at least
  1.5 points.
- **Server fallback flag per call (our idea).** The `hint` result carries
  `numbers.sims` and the engine exposes a "quality" number (sims or exact), so the app can
  decide later to route expert review to a server (as holdem does, D13) without an interface
  change. Cost: hours. Test: none beyond the schema.

## 8. Teaching: reasons, value-lost grading, drills, human-like levels

**Why it is hard.** A solver says "play 7♠ 7♥", not why. D1 bans hints without reasons; D42
bans text in the engine. Reasons must be computable, true in the position, and few enough to
read. Grading by value lost needs a reference value that is stable across runs on a phone.

**What is known.**
- `engine-interface.md` §5 fixes the shape: `reasons: [{key, args}]`, `loss`, grades
  best/good/inaccuracy/mistake/blunder.
- Our Phase 3 "cost" measure (PIMC-400 on the same deals for every move) is a value-lost
  grader already; it is biased toward the greedy opponent model.
- The Guandan commentary paper pairs an RL agent with an LLM for prose; the LLM4CardGame
  result shows LLMs can imitate a card AI but not beat it.

**Ideas.**
- **Counterfactual reason keys from the solver's move table (our idea).** The solver already
  values every candidate. For the chosen move, compute the three largest value drops among
  alternatives and name what differs: `reason.keeps` / `reason.breaks` (§4),
  `reason.pass_to_keep` (§5), `reason.cross_tier` (§2), `reason.safe_single` (§3),
  `reason.helps_seat` (§1). Each key is derived from a feature difference between two
  candidates plus the value gap, so it is true by construction. Cost: 1 week for the first
  eight keys. Test: 50 positions read by two strong players; each reason rated "true and
  useful", "true but obvious", or "wrong"; ship keys with under 10% "wrong".
- **Stable value-lost grading with paired deals (our idea).** Grade a played move by
  evaluating every legal move on the same sampled deals (as our `agreement` experiment does),
  so the sampling noise cancels in the difference. Store the seed, so review is reproducible on
  the phone. Thresholds in points: good < 0.5, inaccuracy < 1.5, mistake < 4, blunder ≥ 4,
  to be calibrated. Cost: 3 days. Test: re-grade the same game twice with different seeds;
  grades must agree on over 95% of decisions.
- **Mistake taxonomy by clustering (our idea, teaching).** Cluster graded mistakes from
  solver games and later human games by the reason keys attached. The clusters become the
  `skill.*` list and give each drill its skill tag automatically. Cost: 2 days after grading.
  Test: the clusters should match Frame 7 of the Miro curriculum for at least half the
  skills; the rest are candidates for new drills.
- **Drill difficulty from solver disagreement (our idea).** Rate a drill by two numbers: the
  value gap between the best and second-best move (small gap = hard) and the share of bot
  levels that get it right (low = hard). Convert to an Elo-like rating by fitting to human
  solve rates once the app has them. Cost: 2 days. Test: the Spearman correlation between the
  rating and human solve rate over the first 1,000 drill attempts.
- **Human-like weaker levels by temperature plus an error model (our idea).** Sample the
  distilled network's moves with a temperature per level, and add human-style errors
  (break a pair to beat a single; hold the 2♠ too long) at a rate fitted to the human game
  logs. Cost: 2 days plus the logs. Test: a strong player plays ten deals at each level and
  ranks them blind; the ranking must match the level order.

## 9. Open questions for Brian

1. Which scoring variant is the default for Elo and drills: tiered (our default), original, or
   win_loss? Default: tiered, with the distribution head (§2) so the rest are cheap.
2. Do Hong Kong tables you know allow four-card plays (two pairs, four of a kind alone)?
   Charlesworth's rules do; ours do not. Default: no, keep our rules; add a variant only for
   the B2 benchmark.
3. May we run Charlesworth's checkpoint as a benchmark opponent? The repo shows no licence
   file. Default: run it locally for measurement only, never ship or link it, and record that
   in decisions.
4. Do you have IEEE access for Chen & Lu 2025 (the "free-playing right" paper)? Default: we
   design lead control from our own measurements (§5).
5. Expert level on the phone or on a server? Default: phone-first with PUCT 2,000 sims (§7)
   and the server flag kept open.
