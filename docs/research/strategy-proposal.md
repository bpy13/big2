# Big Two (鋤大D) strategy proposal

**Date:** 2026-10-28. Task 7 of `docs/plans/2026-10-28-hk-games-research.md`. Written from
[survey.md](survey.md), [hard-problems.md](hard-problems.md), the cross-game report
[teaching-and-deployment.md](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md), the big2 repo's
own results (`big2\docs\study-results\`), and the hk-games decisions, engine interface and bot
brief. It follows the shape of
[holdem's proposal](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/holdem/docs/research/strategy-proposal.md). The
[Dou Dizhu proposal](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/doudizhu/docs/research/strategy-proposal.md) was written in
parallel; §8 and §9 align the shared toolkit with its §8 and its Q8.

**How to read it.** Every claim about a published method links to the report section that
cites the paper. **(estimate)** marks a number I scaled from a published figure or from our
own laptop runs; nothing in this file was run. **(our idea)** marks ideas from our own reports.
**(judgment)** marks a choice of mine that Brian can overturn. Brian decides the questions in
§11; training details (network widths, learning rates) belong to the build plan.

**Terms.** *Simulation*: one sampled deal played to the end inside the search. *Duplicate
deal*: one deal played four times with the lineup rotated through all four seats, so every
agent plays every hand. *Points*: our tiered pairwise score per deal (`big2\docs\rules\big2.md`
§7). *Expert iteration*: train a network on a search's answers, put the network inside the
search, and repeat. *PUCT*: tree search where a network's move probabilities steer which moves
get simulations, as in AlphaZero.

## 1. Summary (one screen)

**What we build.** One Big Two brain in four layers, all in Rust, all on the phone.
(1) A *search* that samples the hidden hands and plays them out: today's ISMCTS, upgraded with
a hand-decomposition planner and later with a network inside it. (2) A *network* distilled
from that search, so the search's judgment is available in a millisecond and as the search's
own playout model. (3) A *belief* layer: hard rules first ("this player cannot hold that card"),
a small learned model only if it measurably pays. (4) A *teaching* layer that reads the
search's own move table: reasons as keys, grades by value lost, drills by ladder disagreement.

**The method in one line.** Expert iteration with the network inside the search, shipped as
one model file and five time budgets. Round 0 is today's solver; each round distils the
current search into the network and then puts the network back in as the prior and the leaf
value. We stop when a round no longer beats the previous one on 1,000 duplicate deals.

**Why this and not self-play RL.** Our own measurements decide it: after 10 hours of self-play
the best network only *ties* the search ([long-training.md](../study-results/long-training.md)),
while the search's own limit is its greedy playout model, not its budget
([phase1-solver.md](../study-results/phase1-solver.md) §3,
[search-methods.md](../study-results/search-methods.md)). The network is what
replaces that playout model. The Dou Dizhu leaders agree on the shape: every 2024–2026 leader
is a learned network plus search at play time ([survey §3.1](survey.md#31-the-leaders-read-in-full)).
And every round has a yardstick (the previous round), which pure self-play lacks until the end.

**What is new (our idea).** A decomposition-first move list that serves the search, the
network and the reason keys at once; a cards-left distribution head so one network serves every
scoring variant; value heads for all four seats, which give a kingmaking check and the key
`reason.helps_seat`; an exact information-set endgame with forced-win proofs; reason keys built
from value differences in the search's own move table, so they are true by construction; one
model with five budgets as the level ladder (§4).

**Cost.** Phases 1–4 need about 3 GPU-days in total and no run over 12 hours except the
best-response probes (10 GPU-hours each, §6). The optional research phase (§3, phase 5) adds
6–9 GPU-days in runs of ≤ 3 days, each announced.

**Measured by.** Six bars from [survey §6](survey.md#6-what-state-of-the-art-for-big-two-can-mean-and-the-bars-we-propose):
Patwa's bots under his scoring (B1), Charlesworth's network under his rules (B2), our own
ladder with a best-response probe (B3), the Ataraxos-class method bar (B4), humans in the app
(B5) and the phone budget (B6). Duplicate deals, every seat, 95% intervals, 1,000 deals for a
±0.5-point claim (§5).

## 2. The recommended path, round by round

### 2.1 What exists and what it measured

The big2 repo already holds most of layer 1 and the first version of layer 2
([big2 README](../../README.md)): a Rust rules engine with variants, ISMCTS and PIMC with
duplicate tournaments and intervals, an exact max^n endgame, a belief-weighted sampler, a value
network loader (`.pknn`) in Rust, and PyTorch trainers for DMC, PPO and distillation. The
measured facts that shape the plan, all with last-card duty **off** and on an M1 laptop:

| Fact | Number | Source |
|---|---|---|
| ISMCTS-1000 vs three greedy bots | +8.6 ± 1.7 points, 14 ms per move | [phase1 §1](../study-results/phase1-solver.md) |
| 4× more simulations | +0.6 ± 0.9 (not significant) | phase1 §2 |
| Score-seeking vs win-seeking search | +7.6 ± 1.6 | phase1 §2 |
| Distilled network vs the search it copies | −0.47 ± 0.76, 1 ms per move | [long-training](../study-results/long-training.md) |
| DMC self-play, 10 h, vs the search | −0.17 ± 0.82 | long-training |
| Network playing out the search's samples vs the search | −0.18 ± 0.9, 13× slower | [phase3 §2](../study-results/phase3-rl.md) |
| A search that sees the true deal | +2.95 ± 0.85 | [search-methods §1](../study-results/search-methods.md) |
| What the history tells the belief, per card | 0.37 → 0.40 | search-methods §3 |
| Exact endgame vs greedy opponents | no gain, 2–6× the time | phase1 §4 |

Reading: the search is strong and cheap; its ceiling is the greedy playout model; hidden
cards are worth about 3 points but the history reveals little; the network copies the search
but does not pass it. So the lever is a better playout model inside the search, and expert
iteration is the loop that builds one from the search itself.

### 2.2 Round 0: ship the four calls from what exists (phase 1)

Deliver `bot_action`, `hint`, `review` and `grade` with reason keys from the current ISMCTS,
made WebAssembly-safe, before any training. The app stream already replaced `Instant` with
`web-time` (`bot-notes.md`, 2026-10-28); what remains ours is a single-thread path past
`rayon`, a time budget inside the search, and a `.pknn` export of the distilled network's
move head (today only the value net exports; the Q-net is PyTorch-only, `bot-stream.md` §4).

Two cheap upgrades go in here because everything later builds on them:

- **The min-plays planner and the structured rollout (our idea, with precedent).** A
  dynamic-programming count of the fewest plays that empty a hand, as PerfectDou and ODMC use
  it ([survey §4.2](survey.md#42-general-methods-that-transfer)), plus the enumeration of the
  minimal decompositions ([hard-problems §4](hard-problems.md#4-the-action-space-and-hand-decomposition)).
  The rollout policy stops breaking pairs and straights to beat singles. Expected +1 to +2
  points for the search (`prior-work.md` §5, to be verified).
- **Decomposition-first move list (our idea).** Candidates are "play group G of decomposition
  D" over the top few decompositions, usually a dozen moves instead of hundreds, each labelled
  with the structure it keeps. The same list feeds the search, the network's candidate set and
  the keys `reason.plan.keeps` / `reason.plan.breaks` (hard-problems §4).

### 2.3 Round 1: the network goes inside the search (phase 2)

1. Relabel: run the structured ISMCTS on self-play games and store every candidate's mean
   score, win rate and simulation count, as the distillation generator does today
   ([rl.md §3.3](../rl.md)).
2. Retrain the network with two new heads:
   - **Cards-left distribution head (our idea).** Fourteen probabilities (finish with 0 … 13
     cards) for the actor, instead of one mean score. Any scoring variant is then a dot product
     with its payoff vector, so **one network serves tiered, original and win_loss**, and the
     hint can show "chance to finish under 8 cards" as a number
     ([hard-problems §2](hard-problems.md#2-scoring-variants-and-score-versus-win-rate)).
   - **All-seat heads (our idea).** The same distribution for each opponent. The search gets a
     cheap kingmaking check; teaching gets `reason.helps_seat` `{seat, delta}`
     ([hard-problems §1](hard-problems.md#1-four-players-no-equilibrium-non-zero-sum-between-pairs)).
   - A **policy prior** over the decomposition-first candidates, trained on the search's
     visit counts.
3. **PUCT with the network inside (our idea, from AlphaZero and Ataraxos).** The prior steers
   the tree; the leaf value is the network's distribution, no rollouts; 300–2,000 simulations.
   One forward pass per simulation, about 0.2 ms in Rust for the 0.8 M-parameter MLP
   **(estimate)**, so 2,000 simulations fit 0.5 s on one thread
   ([hard-problems §7](hard-problems.md#7-the-phone-budget)).
4. Test: PUCT-2000 vs ISMCTS-1000, 1,000 duplicate deals, and PUCT vs ISMCTS-5000 at equal
   wall time on one thread. Pass: ≥ +1.0 point with the interval excluding zero.

Why this round should pass where the Phase 3 "network + search" tied: that search used the
network only as a playout policy for 32 sampled deals and 6 candidates, at 190 ms. PUCT uses
the network at every node and spends the budget on the moves the prior likes; this is what
expert iteration needs to compound ([survey §4.2](survey.md#42-general-methods-that-transfer),
Anthony et al. 2017).

### 2.4 Rounds 2–3: iterate, add belief where it pays, exact endgames (phase 3)

- Label with PUCT-2000 as the expert, retrain, put the new network back in. Each round must
  beat the previous expert on 1,000 duplicate deals or the loop stops.
- **Hard-constraint belief from the rules (our idea).** Deals that contradict the history
  (last-card duty, locked-out passes, "led a single with three cards so the other two are not
  a pair with it") get weight zero before any learned weighting
  ([hard-problems §3](hard-problems.md#3-hidden-hands-and-belief)). Cheap, exact, and each
  rule is a reason key.
- **Learned belief per round, kept only if it pays (our idea).** A small network predicts the
  holder of each unseen card from the history, trained on the current expert's self-play (as
  Ataraxos does, [survey §3.1](survey.md#31-the-leaders-read-in-full)). Our measurement
  predicts under 0.5 points of gain; we report its log-likelihood every round and stop
  investing if the search gain stays under 0.5 points at 300 deals. Its teaching value
  (`reason.control.safe_single`, `reason.control.highest_out`) stays even then.
- **Information-set exact endgame (our idea).** When ≤ 12 cards are unseen, enumerate every
  consistent deal, solve each by max^n, and average by the belief; no sampling noise, and the
  key `reason.counterfactual.forced` `{result, plays}` when every deal agrees
  ([hard-problems §6](hard-problems.md#6-endgames)). Re-tested against strong opponents, not
  greedy ones, because Phase 1's "no gain" was against greedy bots.

### 2.5 Teaching depth (phase 4)

Reasons, grades, drills and human-like levels, built on the search's move table and the shared
teaching crate (§8). Detail in §4 and in
[teaching-and-deployment §1–4](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#1-reasons-from-solver-output-to-keys).

### 2.6 Research phase, optional (phase 5): the B4 bar

Self-play with a full-information critic and oracle guiding, VRPO-style advantages, population
opponents with style tags, and a reduced Ataraxos search on top, compared with the expert-
iteration network under B4 ([survey §6](survey.md#6-what-state-of-the-art-for-big-two-can-mean-and-the-bars-we-propose)).
This is where "at least the state of the art" is honestly tested. It waits until phases 1–4
ship and the GPU is free (Q13).

## 3. Alternatives considered, and why they lose

| Alternative | What it would be | Why not as the main line | Kept as |
|---|---|---|---|
| **A. Self-play RL at scale** (PerfectDou / VRPO recipe: PPO with a full-information critic, oracle guiding, then Ataraxos-style search) | One network trained by self-play for weeks, search added at the end | On our hardware a VRPO-class policy is 2–4 GPU-weeks **(estimate, [survey §5](survey.md#5-comparison-table-fit-for-our-hardware-and-for-a-phone))**, over the 3-day cap per run, and the GPU serves five games. Our own 10-hour DMC only tied the search; our PPO was weakest ([long-training](../study-results/long-training.md)). No yardstick until the end. Self-play finds one convention; Hong Kong tables follow another ([hard-problems §1](hard-problems.md#1-four-players-no-equilibrium-non-zero-sum-between-pairs)). DanZero+ found plain PPO over thousands of moves impractical ([survey §3.2](survey.md#32-guandan)) | Phase 5 (the B4 bar); the full-information critic and oracle guiding as the trainer for phase 5's net; a 10-GPU-hour DMC run as the best-response probe (§5) |
| **B. Search only, better hand-made models** (the Big2AI / Big2MDP line: ISMCTS plus opponent move prediction and hand-made evaluation, [survey §2.2](survey.md#22-chen--lu-2022-and-2025-abstract-only)) | No network; structured rollouts, rules belief, exact endgame | The search plateaued at its playout model: 4× simulations gave +0.6, the greedy-model belief gave 0.0 ([phase1 §2](../study-results/phase1-solver.md), [search-methods §1](../study-results/search-methods.md)). Hints then inherit the greedy opponent model, which biases grades ([phase3 §3](../study-results/phase3-rl.md)). The two Taiwanese papers give no numbers we can check | Exactly phase 1: it ships first, it is round 0's teacher, and it is the fallback if round 1 fails |
| **C. Ataraxos re-implemented as published** (damped self-play net, 3.3 M-parameter belief transformer, 200 full rollouts per move) | The only published stack that beats every Dou Dizhu leader, +0.107 from its search ([survey §3.1](survey.md#31-the-leaders-read-in-full)) | 200 rollouts × ~40 plies × a 2 M-parameter transformer is 8–30 s per move on one WebAssembly thread **(estimate, [survey §5](survey.md#5-comparison-table-fit-for-our-hardware-and-for-a-phone))**; 1–2 GPU-weeks to train; no weights to compare against; the belief part is predicted to gain little in Big Two (0.37 → 0.40) | The shape of our PUCT (network inside search) and the belief-per-round idea; a reduced version (20–50 rollouts, small net) as phase 5's search |
| **D. Equilibrium search** (ReBeL, Student of Games, CFR) | Public-belief-state search with guarantees | Two-player zero-sum only; four independent players and 39 hidden cards in three hands break both the guarantee and the belief size ([survey §4.2](survey.md#42-general-methods-that-transfer)); unchanged from `prior-work.md` | Nothing |
| **E. LLM agents** | Fine-tuned 7–9 B models reach 0.85 win rate from DouZero data ([survey §3.3](survey.md#33-llm-based-play-full-text-for-the-first)) | Never beats its teacher; seconds per move on a GPU; not on a phone; D42 forbids text from the engine | A server-side rewriter of keys into prose, later, if ever |

**(judgment)** The recommended path is B first, then the network from A's family placed
inside B's search, with C's shape at a phone-sized budget. It is the only order in which every
step is checked against the step before it, and in which the app has working bots within
weeks.

## 4. The phone: what runs where, with budgets

Rule: everything for Big Two runs on the phone (D4, D12; [teaching §5.3](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#53-what-runs-where-per-game)).
No server tier at the first release. The `hint` result carries a `quality` number (simulations
used, or "exact") so the app can route expert review to a server later without a schema change
(our idea, hard-problems §7; request to the app stream through `bot-notes.md`).

**Known blockers and their state.** `Instant::now()` panicked in WebAssembly; the app stream
replaced it with `web-time` on 2026-10-28 (`bot-notes.md`). `rayon` compiles for wasm32 but
has no threads; we keep a path with `threads = 1` that never builds a pool. The Q-network runs
only in PyTorch; we export its state tower and heads to `.pknn` (v2, int8) and evaluate in
`crates/nn`. The exact endgame needs a node cap tuned for one slow thread.

**Level ladder by budget, not by model (our idea).** One model file, five levels:

| Level | Mechanism | `bot_action` time **(estimate)** | Why it plays like a human of that level |
|---|---|---|---|
| `random` | random legal move | < 1 ms | the floor the interface requires |
| `easy` | structured greedy with ε plus the error model (hold the 2♠ too long; break a pair to beat a single) | < 1 ms | named mistakes, each a reason key the review can show |
| `medium` | distilled network sampled at a temperature, thinner error model | ~5 ms | plays "by feel", mostly right, sometimes greedy |
| `hard` | PUCT, 300 simulations | 60–100 ms | a club player who thinks a little |
| `expert` | PUCT, 2,000 simulations, plus the exact endgame | 0.4–0.6 s | the reference the reviews grade against |

Big2's networks come from distillation, so there are no "beginner" checkpoints as in a
self-play run; the error model replaces the cross-game report's checkpoint ladder here
([teaching §4.2](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#42-stage-a-no-human-data-recommendation-for-the-first-release)).
Stage B (a skill-conditioned human policy from the app's own games) applies unchanged once logs
exist.

**Call budgets.**

| Call | What runs | Budget | Note |
|---|---|---|---|
| `bot_action` | the level's mechanism above | the level's time; `expert` fills `budget_ms` | anytime: the best move so far is always ready |
| `hint` | PUCT anytime: first answer after 100 ms, refine until the budget ends | 0.5–2 s | **Anytime hints (our idea)**: `budget_used_ms` already exists; the move must change after 100 ms in under 15% of positions, else the early answer misleads |
| `review` | paired grading of every decision on the same sampled deals, computed during the bots' turns and chunked to `budget_ms` | per chunk | seed stored, so a review repeats on the phone |
| `grade` | as `hint` on the drill position | as `hint` | |

**Planning numbers.** The bot brief says a hint takes about 0.5 s at 20,000 simulations on a
phone; the survey says 5,000 simulations in 0.5 s. The M1 measured 14 ms per 1,000
simulations on one thread, and WebAssembly on a mid-range phone runs 3–5× slower
**(estimate)**, so 5,000 simulations ≈ 0.3–0.4 s. I plan with **5,000 simulations per 0.5 s
for ISMCTS and 2,000 for PUCT**; bar B6 replaces both numbers with measurements on the
benchmark phone (Q8). Model size: 0.8 M parameters ≈ 0.8 MB in int8.

## 5. How strength is measured

**Match format (every bar).** Duplicate deals: each deal played four times with the lineup
rotated through all four seats. Two lineups: "1 of ours + 3 others" and "2 + 2 alternating"
(A B A B). Score = the candidate's mean points per deal; 95% interval over deals; win rate
reported beside it, never alone (score-seeking play wins fewer deals and scores more,
[phase1 §2](../study-results/phase1-solver.md)). Every bar is re-run under the
current defaults (last-card duty **on**), because every number so far was measured with duty
off ([survey §8](survey.md#8-where-the-claims-are-weak)).

**Sample sizes (estimate, scaling our Phase 1 intervals by 1/√N).**

| Deals | Games | Half-width | Use |
|---|---|---|---|
| 100 | 400 | ±1.5 points | smoke tests only |
| 300 | 1,200 | ±0.9 | go/no-go on a cheap idea (structured rollout, belief) |
| 1,000 | 4,000 | ±0.5 | every "beats the previous round" claim; the ladder order |
| 10,000 | 40,000 | ±0.15 | the final numbers against the bars; B1's 3 seeds |

A claim about a *method* (not a checkpoint) uses 3 training seeds and reports the spread.
Two independently seeded copies also play each other (cross-play), so "works only against
itself" shows up.

**The bars** (from [survey §6](survey.md#6-what-state-of-the-art-for-big-two-can-mean-and-the-bars-we-propose); pass conditions as written there):

| Bar | What | Phase | Pass |
|---|---|---|---|
| **B1** Patwa's bots, Patwa's scoring | add `scoring=sum_of_others` and his *greedy* and *smart* bots; 10,000 games per cell, seat randomised, 3 seeds | 1 | win rate vs 3 greedy > 64.8% and vs 3 smart > 43.5%, interval excluding his number |
| **B2** Charlesworth's network | port his checkpoint; his rules (four-card plays) and scoring; 1,000 duplicate deals, both lineups | 3 | mean score > 0 with interval excluding 0 in both lineups; licence checked first (Q4) |
| **B3a** Internal ladder | vs ISMCTS-1000 and vs the DMC-10h network, 1,000 duplicate deals, tiered scoring | 2, 3 | ≥ +1.0 point vs ISMCTS-1000, interval excluding 0 |
| **B3b** Best-response probe (our idea) | train a DMC agent for 10 GPU-hours against three frozen copies of the candidate; report its gain | 3 | the probe gains less against the candidate than against ISMCTS-1000 and DMC-10h; validated first by gaining more against greedy than against ISMCTS |
| **B4** Ataraxos-class method bar | network-only vs network+search vs ISMCTS-1000, 10,000 duplicate deals | 5 | search adds a measurable gain over its own network; the stack beats ISMCTS-1000 by more than B3a |
| **B5** Humans in the app | 500 human games at `expert`; humans' mean score with a standard error, as Charlesworth did | after release | humans average below 0 with the interval excluding 0 |
| **B6** Phone budget | one WebAssembly thread on the benchmark phone: agreement with a 100,000-simulation reference and value lost per decision at 0.5 s and 2 s | 1, 2 | at 2 s: agreement ≥ 69.7% (the laptop ISMCTS-1000 figure) and mean value lost ≤ 0.3 points |

**What we claim, and when.** After phase 3: meaning A (beats every published Big Two agent we
can run, B1 + B2) and meaning C (strongest in our ladder, hard to exploit, B3). After phase 5:
meaning B (method parity with the Dou Dizhu leaders, B4). B5 is the number the product cares
about. Nobody has published a Big Two result with duplicate deals and intervals against another
agent ([survey §6](survey.md#6-what-state-of-the-art-for-big-two-can-mean-and-the-bars-we-propose)),
so our evaluation is stricter than the field's from day one.

**Grading and teaching bars** (phase 1 and 4; from [hard-problems §8](hard-problems.md#8-teaching-reasons-value-lost-grading-drills-human-like-levels) and [teaching §1.5, §2.4, §3.4](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#15-how-to-check-a-reason-is-true)):
the same game re-graded with new seeds agrees on ≥ 95% of decisions; the phone grader matches
a 10× search on ≥ 90% of decisions; the ladder orders strictly with gaps ≥ 1.5 points over
500 deals; 50 hints audited by a strong player with ≤ 10% "wrong" (Q10); drill solve rate
falls with the rating (Spearman over the first 1,000 attempts).

## 6. Training and evaluation time on the local box

Hardware: RTX 4060 Ti 16 GB, i7-12700K (20 threads), 32 GB RAM. One GPU run ≤ 3 days; runs
over 12 hours are announced; the GPU is shared with four other games, so every line states
GPU-days. All numbers are **(estimate)**; the basis is in the last column. The first benchmark
in phase 1 replaces them.

| Step | CPU | GPU | Basis |
|---|---|---|---|
| 1.1 Min-plays planner, decomposition list, structured rollout | ~1 week of work; tests in minutes | 0 | brute-force checks on 10,000 random hands |
| 1.2 Re-baseline under duty-on: ISMCTS vs greedy, vs DMC-10h, 1,000 duplicate deals | 10–20 min on 20 threads | 0 | 4,000 games × ~45 decisions × ≤ 4 searching seats × 15 ms |
| 1.3 `.pknn` export of the distilled network (state tower + heads), int8 | days of work; minutes to run | 0 | the value-net export exists |
| 1.4 B1: Patwa's bots and scoring, 10,000 games × 3 seeds | minutes | 0 | scripted bots, no search |
| 2.1 Relabel round 0: 5 M decisions with the structured ISMCTS-1000 | 1.5–3 h on 20 workers | 0 | Phase 3: 730 k decisions per hour on 4 M1 workers |
| 2.2 Train the network with the distribution, all-seat and prior heads on 5 M decisions | — | 2–4 GPU-hours | 291 min on an M1 CPU for 800 k decisions; a 4060 Ti is ≥ 10× on this MLP |
| 2.3 Rust PUCT with the network inside | ~1 week of work | 0 | |
| 2.4 Round 1 labelling: 4 M decisions with PUCT-500 | 5–6 h on 20 threads | 0 | 0.1 s per decision at 0.2 ms per forward pass |
| 2.5 Retrain, evaluate 1,000 deals with PUCT-2000 in every seat | 2 h | 2–4 GPU-hours | 180 k decisions × 0.4 s |
| 3.1 Rounds 2–3: as 2.4–2.5, twice | ~16 h | 4–8 GPU-hours | |
| 3.2 Belief network per round: 1 M positions from the expert's self-play, small net | 1 h | ~2 GPU-hours | Ataraxos: 32 H100-hours for a 3.3 M transformer; ours is ≤ 0.5 M parameters |
| 3.3 Information-set exact endgame | ~1 week of work | 0 | ≤ 34,650 deals × max^n with a node cap |
| 3.4 B3b best-response probe, per candidate (3 candidates) | — | **10 GPU-hours each** | DMC closed −2.2 → −0.2 in 10 M1-CPU-hours; a GPU at 1,024 envs runs several times more games |
| 3.5 B2: Charlesworth port and 1,000 duplicate deals | 2–3 days of work; 1 h to run | 0 | 412 features, 1,695-action head, NumPy loader |
| 3.6 10,000-deal final matches, PUCT-2000 in every seat | ~9 h per opponent | 0 | 1.8 M decisions × 0.4 s |
| 4 Teaching: keys, grader, drill miner, error model | 2–3 weeks of work | 0 | hours of CPU per thousand drills |
| 5 Self-play net with full-information critic and oracle guiding, VRPO-style advantage, population opponents | 20 threads feed the learner | **2–3 GPU-days per run, 2–3 runs** | Big Two is smaller than Dou Dizhu (1,695 actions vs 27,472, no bombs); we accept a reduced scale per the 3-day cap; [survey §5](survey.md#5-comparison-table-fit-for-our-hardware-and-for-a-phone) says 2–4 weeks at paper scale, and the [Dou Dizhu proposal §5](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/doudizhu/docs/research/strategy-proposal.md#5-training-time-on-the-local-hardware) counts 20–28 GPU-days for a full VRPO-level run there. Big2's phase 5 reuses Dou Dizhu's proven trainer, so its runs are short and fewer |
| 5 Reduced Ataraxos search, B4 at 10,000 deals | ~1 day | 0 | 20–50 rollouts per move |

Totals **(estimate)**: phases 1–4 use about 1 GPU-day of training plus 1.3 GPU-days of probes,
**≈ 3 GPU-days with margin**, and no single run over 12 hours except the probes (10 h, kept
under the line). CPU work is the larger share and runs beside other games' GPU jobs. Phase 5
asks for **6–9 GPU-days** in runs of 2–3 days, each announced (Q7). RAM is never a limit: the
largest buffer is 5 M decisions × ~1.2 k features as bytes ≈ 6 GB.

## 7. Phases with acceptance criteria

The app needs `bot_action`, `hint`, `review` and `grade` with reason keys first
(`bot-stream.md` §4), so phase 1 delivers them from the existing solver and the training
starts only after that. Targets are my defaults **(judgment)**; Brian sets them.

### Phase 1 – The four calls on the phone, from the existing solver (3–4 weeks)

Delivers: WebAssembly-safe ISMCTS with a single-thread path and an anytime budget; the
min-plays planner, decomposition list and structured rollout; the distilled network exported
to `.pknn` v2; the five-level ladder v0 (random / structured greedy + error model / distilled
net / ISMCTS-1000 / ISMCTS-5000 + exact endgame); the first eight reason keys
(`reason.plan.keeps`, `reason.plan.breaks`, `reason.control.pass_to_keep`,
`reason.value.score_vs_win`, `reason.tempo.plays_to_out`, `reason.control.safe_single`,
`reason.value.gap`, `reason.counterfactual.outcomes`); paired-deal grading with stored seeds;
B1; the benchmark harness on the benchmark phone (B6 for ISMCTS); the re-baseline under
duty-on.

Done when: all four calls return within `budget_ms` × 1.2 on the benchmark phone (p99); the
structured rollout beats today's rollout inside ISMCTS by ≥ +1.0 point at 300 deals, or we
record that it does not and keep the planner for the keys only; the ladder orders strictly
with gaps ≥ 1.5 points over 500 duplicate deals; re-grading with new seeds agrees on ≥ 95% of
decisions; 50 hints audited with ≤ 10% wrong; B1 passes or the shortfall is recorded with its
interval; the keys are listed in `bot-notes.md`.

### Phase 2 – The network inside the search (3–4 weeks, ≤ 0.5 GPU-day)

Delivers: the relabelled round-0 data; the network with the cards-left distribution, all-seat
and prior heads; Rust PUCT with the network as prior and leaf value; the ladder v1 (`hard` =
PUCT-300, `expert` = PUCT-2000); B6 for PUCT; `reason.helps_seat`; `numbers.plays_to_out` and
`numbers.lead_value`.

Done when: PUCT-2000 beats ISMCTS-1000 by ≥ +1.0 point over 1,000 duplicate deals with the
interval excluding 0, and beats ISMCTS-5000 at equal wall time on one thread; the distribution
head loses ≤ 0.5 points to the mean-score head under tiered scoring and wins under win_loss
(500 deals each); the mixed-variant network loses ≤ 0.5 points to a per-variant network under
each variant; B6 at 2 s passes; `expert` on the phone agrees with the laptop PUCT-2000 on
≥ 90% of 300 positions.

### Phase 3 – Iterate, belief, exact endgames, external bars (4–6 weeks, ≤ 2.5 GPU-days)

Delivers: rounds 2 and 3; the rules-based hard-constraint belief; the learned belief per round
with its log-likelihood report; the information-set exact endgame with forced proofs; B2; B3a
and B3b; cross-play of two seeds.

Done when: each round beats the previous expert at 1,000 deals with the interval excluding 0
(or the loop stops and the last winner ships); the hard-constraint filter rejects a reported
share of deals and does not lose points at 300 deals; the exact endgame agrees with a
100,000-simulation reference on ≥ 95% of 300 late positions and fits 1 s on the phone at its
threshold; B2 passes in both lineups; B3a ≥ +1.0 vs ISMCTS-1000; the probe gains less against
the final network than against ISMCTS-1000 and DMC-10h; two seeds do not lose to each other.

### Phase 4 – Teaching depth (3–4 weeks, no GPU)

Delivers: the full key set under the shared taxonomy; counterfactual reasons in review ("same
deals, other move"); the luck ledger; σ-scaled grade thresholds with an "unclear" grade; the
drill miner by ladder disagreement with seed ratings; the mistake clustering against Frame 7
of the Miro curriculum; the error model fitted first by hand; the convention-gap check once
human logs exist.

Done when: the luck ledger's terms sum to the outcome on every reviewed deal; drill tag
precision ≥ 80% on 50 audited drills; drill solve rate falls with rating over the first 1,000
attempts; a strong player ranks ten deals per level blind in the right order; every hint shows
at least one reason.

### Phase 5 – Research: the B4 bar (optional, after Brian's go; 6–9 GPU-days)

Delivers: the self-play trainer (full-information critic, oracle guiding, VRPO-style advantage,
population opponents with style tags); a reduced Ataraxos search; B4 on 10,000 deals.

Done when: the self-play network's strength against the phase-3 network is reported with an
interval either way; the search adds a measurable gain over its own network; the stronger of
the two stacks becomes `expert` only if B6 still passes.

## 8. What is shared with Dou Dizhu

Both are shedding games with the same combination grammar, hidden hands and a search-plus-
network frontier. The cross-game report fixes the toolkit
([teaching §6](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#6-cross-game-reuse)); Dou Dizhu's
hard-problems report marks the same pieces "shared with big2". Big2 is the first game (D4), so
it builds the first version of each and Dou Dizhu reuses it:

| Shared crate | Big2 builds | Dou Dizhu reuses | Stays per game |
|---|---|---|---|
| `shedding-core` | the combination grammar as data, the min-plays planner, the decomposition enumerator, the rules-masked belief sampler, ISMCTS and PUCT with anytime stopping | all of it, plus bombs, airplanes, kickers and the prefix-decomposed action encoding | rules, scoring, five-card ordering vs kicker rules |
| `belief-core` | per-card holder posterior, hard rule masks, carried samples | the same over two hidden hands | which rules apply |
| `eval-harness` | duplicate deals over all seatings, paired differences, bootstrap intervals, the best-response probe, cross-play, one report format | the same with role-averaged score and deal-strength strata | the bars |
| `teach-core` | reason families and typed args, paired grader with σ thresholds and "unclear", luck ledger, drill miner and seed rating, ladder calibration, error-model framework | the same | the keys, the skill mapping, the error list |
| `nn-core` | `.pknn` v2 int8, wasm SIMD inference, loaders from bytes | the same | the heads |
| `train-core` (Python) | distillation and the expert-iteration loop; PPO with a full-information critic in phase 5 | PPO with the full-information critic and VRPO-style advantage as its main line | the objectives |

The two games take **different main lines on purpose**: Dou Dizhu has public benchmark
opponents with weights (DouZero, PerfectDou) and a self-play recipe that is proven against
them, so self-play RL is its honest route; Big Two has no public yardstick, so expert iteration
with its own ladder is. The trainer and the search are shared; the order in which each game
leans on them is not. Target from the cross-game report: Dou Dizhu's engine, search and bots
come up in under two weeks once big2's crates exist.

The [Dou Dizhu proposal §8](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/doudizhu/docs/research/strategy-proposal.md#8-what-is-shared-with-big2)
says the same from its side and adds one order rule (its Q8): the engine and teaching crates
come from big2 first, and the PPO trainer in `train-core` is proved in Dou Dizhu first, because
only Dou Dizhu has public opponents to prove it against. This proposal accepts that rule: big2's
phases 1–4 need no PPO trainer, and phase 5 reuses Dou Dizhu's (Q13). Two details it gives that
big2 takes: the prefix-decomposed action encoding (type, rank, kickers) as the shared encoding
even though Big Two's five-card hands have no kickers, and the "two-net search" (a big net at the
root, a small student net in rollouts) if big2's network ever outgrows one MLP.

## 9. Conflicts between the reports, and what I chose

| Conflict | Sources | What I chose and why |
|---|---|---|
| Phone budget: 20,000 simulations in 0.5 s (bot brief) vs 5,000 (survey) | `bot-stream.md` §4; [survey §5](survey.md#5-comparison-table-fit-for-our-hardware-and-for-a-phone) | 5,000 for planning; the M1 figure times a 3–5× WebAssembly slowdown supports it; B6 decides |
| Grade thresholds: 0.5 / 1.5 / 4 points (big2) vs σ-scaled 0.2 / 0.7 / 1.8 / 4.5 with σ ≈ 6–8 (cross-game) | [hard-problems §8](hard-problems.md#8-teaching-reasons-value-lost-grading-drills-human-like-levels); [teaching §2.2](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#22-recommendation-sigma-scaled-thresholds-paired-loss-an-unclear-grade) | The σ rule (one rule for five games), but σ ≈ 6–8 looks wrong: Phase 3 measured a ~20-point standard deviation of one game's score ([phase3](../study-results/phase3-rl.md), Interpretation). With σ = 20 the rule gives 0.5 / 2 / 5 / 12. Phase 1 measures σ in expert self-play under duty-on; until then the lines are 0.5 / 1.5 / 4 (Q9) |
| Level ladder: early checkpoints for `easy`/`medium` (cross-game) vs budget ladder with an error model (big2) | [teaching §4.2](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#42-stage-a-no-human-data-recommendation-for-the-first-release); [hard-problems §7](hard-problems.md#7-the-phone-budget) | Budget ladder with the error model: distillation has no beginner-like checkpoints, and the error model names its mistakes as keys. Checkpoint levels stay the plan for Dou Dizhu's self-play nets |
| Expert iteration cost: 2–3 weeks of GPU (big2 hard-problems) vs my ~1 GPU-day | [hard-problems summary](hard-problems.md#summary-of-the-ideas-we-rate-highest); §6 here | My table: the expert search runs on the CPU threads and the MLP trains in hours; the GPU is light. If round 1 needs a bigger network or far more data, the cost rises toward the report's number; the plan announces that before it happens |
| "Expert on the phone or a server" | hard-problems §9 Q5; [teaching §5.3](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#53-what-runs-where-per-game) | Phone only, with the `quality` number as the door (Q6) |
| Belief: Ataraxos-style learned belief (survey) vs "worth under a point" (our measurement) | [survey §3.1](survey.md#31-the-leaders-read-in-full); [search-methods §3](../study-results/search-methods.md) | Rules first, learned belief per round with a stop rule at 0.5 points; its teaching keys ship regardless |
| Scoring for benchmarks: "winner gets the sum" (every paper) vs tiered pairwise (ours) | [survey §2](survey.md#2-big-two-the-five-published-works); rules §7 | Train and rank under tiered; the distribution head makes the papers' scoring a payoff vector for B1 and B2, not a second model |
| Four-card plays (Charlesworth) vs none (ours) | survey §2.1; rules §3 | A rule variant used only for B2 (Q3) |
| Which game proves the self-play trainer first: carved out of big2 (cross-game report) vs proved in Dou Dizhu (its proposal) | [teaching §6](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md#6-cross-game-reuse); [Dou Dizhu proposal §8, Q8](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/doudizhu/docs/research/strategy-proposal.md#8-what-is-shared-with-big2) | Dou Dizhu proves it; big2's main line needs no self-play trainer before phase 5, and phase 5 reuses it (Q13) |
| GPU cost of a VRPO-level run: 1–2 weeks (big2 survey) vs 20–28 GPU-days (Dou Dizhu proposal) | [survey §5](survey.md#5-comparison-table-fit-for-our-hardware-and-for-a-phone); [Dou Dizhu proposal §5](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/doudizhu/docs/research/strategy-proposal.md#5-training-time-on-the-local-hardware) | The Dou Dizhu count is the more careful one; it is why big2's phase 5 is a reduced-scale run on a proven trainer, not a full re-run |

## 10. Risks, and what would make us change course

| Risk | Signal | Change of course |
|---|---|---|
| The structured rollout does not help the search | ≤ 0 at 300 deals | Keep the planner for keys and candidates; round 1 relabels with today's rollout |
| PUCT with the network inside only ties ISMCTS (as the Phase 3 search did) | B3a < +1.0 after round 1 | Check the prior's calibration and the leaf value first; try value + short rollout mix; if still a tie after one more round, ship ISMCTS-5000 as `expert` and move the GPU to phase 5 |
| Expert iteration stalls after round 1 | round 2 ≤ round 1 at 1,000 deals | Stop the loop; the last winner ships; the gain then has to come from the belief or phase 5 |
| The network inherits and amplifies one convention | the probe gains a lot; cross-play seeds beat each other | Population opponents with style tags move up from phase 5 into round 2 |
| Phone slower than estimated | B6 fails at 2 s | int8 SIMD first, then fewer simulations with the anytime search; native bindings only after a measured failure (D12); a server tier for `expert` review as the last step (Q6) |
| The exact endgame is too slow in WebAssembly | > 1 s at the threshold | lower the threshold; node cap; fall back to PUCT leaves |
| The grader's reference carries the playout model's bias | phone grades disagree with a 10× search on > 10% of decisions | re-grade a sample with the strongest expert; report the bias; the σ rule's "unclear" band widens |
| Charlesworth's checkpoint has no licence | no answer from the author | B2 is run locally, never shipped or linked, and reported as such (Q4); or dropped |
| GPU contention with four other games | a queued run slips a phase | phases 1 and 4 need no GPU; reorder so big2's CPU work runs while another game trains |
| Human conventions differ from the bot's | the convention-gap check shows a disagreement that does not fall with Elo | treat those positions as "convention", not "mistake", in review; consider the HK-style population for `expert` |

## 11. Questions for Brian

Each stands on its own, with a default you can accept with "ok".

1. **The overall path.** Accept "search first, distil, put the network inside the search,
   repeat" (§2) with alternatives A–E held as described in §3? This fixes the order of the
   build plan. *Default: yes.*

2. **Scoring default for Elo and drills.** Tiered pairwise (our default), original, or
   win_loss? The distribution head makes the others cheap, but one must be the yardstick for
   Elo and for every bar. *Default: tiered.*

3. **Four-card plays.** Charlesworth's rules allow two pairs or four of a kind alone as a play;
   ours do not. Do Hong Kong tables you know allow them? *Default: no; add the variant only for
   the B2 benchmark.*

4. **Charlesworth's checkpoint.** His repo shows no licence file. May we run it locally as a
   benchmark opponent, never shipping or linking it, and record that in decisions? *Default:
   yes, run locally for measurement only.*

5. **IEEE access.** Chen & Lu 2025 (the "free-playing right" paper) is paywalled. If you have
   access, its lead-control rule is worth reading before we design ours. *Default: we design
   lead control from our own measurements (`numbers.lead_value`, `reason.control.pass_to_keep`).*

6. **Expert on the phone.** All five levels and all four calls run on the phone; no server for
   Big Two at the first release; the `quality` number keeps a server door open. *Default: yes.*

7. **GPU budget.** Phases 1–4 need about 3 GPU-days, no run over 12 hours except three
   10-hour probes. Phase 5 needs 6–9 GPU-days in announced runs of 2–3 days. Accept the
   phase 1–4 budget now and decide phase 5 later? *Default: yes.*

8. **Benchmark phone.** Every phone number here is an estimate. Which Android phone is the
   reference for B6? *Default: your phone plus one mid-range model you name.*

9. **Grade thresholds.** Start with good < 0.5, inaccuracy < 1.5, mistake < 4, blunder ≥ 4
   points, then switch to the σ rule once σ is measured, and tune after your first 20 reviewed
   deals. *Default: yes.*

10. **Reason audits.** Each release needs 50 positions where a strong player rates each reason
    "right", "irrelevant" or "wrong". *Default: you rate 50 per release; club players later.*

11. **Ship phase 1 early.** Phase 1's ISMCTS bots and hints go to the app as soon as they pass,
    weeks before the network search exists; the app's stand-in today is a random move.
    *Default: yes, ship phase 1 to the app and upgrade `hard` and `expert` in place later.*

12. **Hong Kong conventions.** Self-play bots play for their own score. Tables you know may
    feed the player on the right or block the leader by habit. Should the bots follow such a
    convention, or should review flag where humans and the bot disagree? *Default: bots play
    for their own score; review flags convention gaps; we revisit once the app logs games.*

13. **Phase 5 (the B4 research bar).** Go after phases 1–4 ship, or defer until Dou Dizhu's
    self-play trainer exists and can be reused (the Dou Dizhu proposal's Q8 asks the same from
    its side)? *Default: defer; reuse Dou Dizhu's trainer.*

## Sources

Report sections cited above: [survey.md](survey.md) §2–§8; [hard-problems.md](hard-problems.md)
§1–§9; [teaching-and-deployment.md](file:///C:/Users/brian/OneDrive/Documents/workspace/repo/bot-stream/docs/research/teaching-and-deployment.md) §1–§7;
big2 `README.md`, `docs/solver.md`, `docs/rl.md`, `docs/research/prior-work.md`,
`docs/study-results/{phase1-solver,phase3-rl,search-methods,long-training}.md`,
`docs/rules/big2.md`; hk-games `docs/decisions.md` (D1, D4, D12, D13, D42, D51, D52),
`docs/engine-interface.md`, `docs/streams/bot-stream.md` §4, `docs/streams/bot-notes.md`;
bot-stream `docs/decisions.md` (B5); holdem `docs/research/strategy-proposal.md` (the model).
Published work is cited through the survey's source list.
