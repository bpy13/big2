# Build run phase 3: iterate, belief, exact endgames, external bars

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, phase 3 (Tasks 17–25). Match format as in
`build-phase2.md`: duplicate deals (each deal 4×, seats rotated), last-card duty on, tiered
scoring, no four-card plays, lineups `1+3` and `ABAB`, 95 % bootstrap intervals over deals.

Status 2026-11-01: the three rounds of Task 17 are done (§1). A9 is not met in both lineups: r3
vs ISMCTS-1000 is +1.06 (1+3, passes) and +0.66 (ABAB, fails). r3 ties its teacher ISMCTS-5000,
so the teacher stays, and by the default in `docs/open-items.md` ISMCTS ships as `expert` until
Brian answers. Task 18 done (A13 met, §2). Task 20 done (§4; agreement missed at 94.2 %, time met
at 10 cards in play). Tasks 22–24 run as one job (§6–§8). Task 21 ran on 2026-11-05 (§5).

## 1. Expert iteration (Task 17)

`experiments/expert_iteration.py round 1`. What changed from round 0, and why
(`docs/open-items.md`, 2026-10-30):

- **Teacher: ISMCTS-5000** (greedy rollout) instead of ISMCTS-1000. ISMCTS-5000 beat
  ISMCTS-1000 by +1.09 [+0.35, +1.87] paired (500 deals, `build-phase1.md` ladder v0).
  PUCT λ = 0.5 only tied ISMCTS-1000 (`build-phase2.md` §6.2), costs about 4× ISMCTS-5000's
  time per move, and would feed round 0's own value errors back into the labels.
- **Value on every candidate's search score.** Round 0 weighted the score loss 0.5 against
  three cross-entropies; at a validation score loss of 0.0083 it was about 0.4 % of the
  total, so the distributions learnt mainly from the played move's single outcome. Round 1:
  score 100, prior 1, actor and seats distributions 0.25 each; weights start from round 0's.
  Check: the calibration slope of `build-phase2.md` §6.1 near 1 (round 0: 0.86).
- **Matches:** PUCT-2000 (`c_puct` 2.5) with the new network at λ = 1 and λ = 0.5 against
  ISMCTS-1000, 300 deals each; the better λ at 1,000 deals (A9); round 1 vs round 0 at
  1,000 deals only if A9 passes (A12).

### Round 1 results (2026-10-30)

Source: `runs/round1/results.json`, `runs/round1/table.md`; calibration
`runs/round1/f_calibration/results.json`; training `runs/net_r1/results.json`.

- **Labels:** 807,251 decisions (4.1 M candidates, 1,542 shards) in 128 min on 8 workers.
- **Training:** from round 0's weights; stopped after 60 min (no better validation loss in 6
  evaluations); best at 30 min: validation prior top-1 74.8 %, score RMS 2.32 points.
- **Calibration** on the 31 held-out shards (16,506 decisions): slope of the search's score on
  the prediction **1.03** (round 0: 0.86), bias +0.11, RMS 2.53, correlation 0.977. Prior vs
  visit share: expected calibration error 0.006. The value is no longer over-spread.

| Match | Lineup | Deals | Candidate mean (95 % CI) | Candidate win rate | Paired diff (95 % CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|
| r1 PUCT-2000 λ = 1 vs ISMCTS-1000 | 1+3 | 300 | +0.26 [−0.59, +1.06] | 22.8 % | +0.35 [−0.75, +1.50] | 0.169 / 0.010 |
| r1 PUCT-2000 λ = 1 vs ISMCTS-1000 | ABAB | 300 | +0.11 [−0.46, +0.68] | 23.5 % | +0.22 [−0.91, +1.39] | 0.179 / 0.011 |
| r1 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | 1+3 | 300 | +0.45 [−0.42, +1.28] | 24.0 % | +0.59 [−0.51, +1.75] | 0.332 / 0.019 |
| r1 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | ABAB | 300 | +0.29 [−0.26, +0.84] | 24.1 % | +0.58 [−0.52, +1.74] | 0.294 / 0.017 |
| **r1 λ = 0.5 vs ISMCTS-1000 (A9)** | 1+3 | 1,000 | +0.35 [−0.13, +0.83] | 24.0 % | **+0.46 [−0.18, +1.13]** | 0.280 / 0.016 |
| **r1 λ = 0.5 vs ISMCTS-1000 (A9)** | ABAB | 1,000 | +0.08 [−0.25, +0.40] | 24.4 % | **+0.16 [−0.47, +0.81]** | 0.236 / 0.014 |

**Verdict round 1: A9 fails** in both lineups (bar ≥ +1.0 with the interval above 0); A12 was
not run. It is up from round 0 (λ = 1, 1,000 deals: −0.12 / −1.78, `build-phase2.md` §1), and
λ = 0.5 again beats λ = 1 on the screen. PUCT costs about 17× ISMCTS-1000's time per move; A9's
equal-time part was not measured in round 1. Teacher rule (Brian, 2026-10-30): "use teacher unless distilled
model wins significantly". r1 does not beat ISMCTS-1000, so it cannot beat the ISMCTS-5000
teacher; round 2 labels with ISMCTS-5000 again.

### Round 2 (2026-10-31)

Same teacher (ISMCTS-5000, greedy rollout, 800,000 decisions); network from r1's weights,
trained on round-1 and round-2 labels together (`DistillV2Config.extra_data_dirs`; each folder
holds out its own every 50th shard, so round 1's held-out shards stay unseen); same loss
weights; calibration on round 2's held-out shards; PUCT-2000 r2 at λ = 0.5 and 1 vs
ISMCTS-1000 (300 deals), the better at 1,000 deals (A9); r2 vs r1 (A12) only if A9 passes.
Job: `runs\b25_round2\run.ps1` (20:58 → 01:49).

Source: `runs/round2/results.json`, `runs/round2/table.md`; calibration
`runs/round2/f_calibration/results.json`; training `runs/net_r2/results.json`.

- **Labels:** 805,394 decisions (4.1 M candidates, 1,541 shards) in 126 min on 8 workers.
- **Training:** from r1's weights on 1.61 M decisions (rounds 1 and 2); stopped after 90 min (no
  better validation loss in 6 evaluations); best at 60 min: validation prior top-1 76.2 %
  (r1: 74.8 %), score RMS 2.17 points (r1: 2.32).
- **Calibration** on round 2's 31 held-out shards (16,158 decisions): slope **1.05** (r1: 1.03),
  bias +0.04, RMS 2.44, correlation 0.980; prior vs visit share: expected calibration error 0.007.

| Match | Lineup | Deals | Candidate mean (95 % CI) | Candidate win rate | Paired diff (95 % CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|
| r2 PUCT-2000 λ = 1 vs ISMCTS-1000 | 1+3 | 300 | +0.60 [−0.22, +1.42] | 23.7 % | +0.80 [−0.25, +1.91] | 0.182 / 0.011 |
| r2 PUCT-2000 λ = 1 vs ISMCTS-1000 | ABAB | 300 | +0.28 [−0.32, +0.85] | 23.4 % | +0.55 [−0.66, +1.79] | 0.161 / 0.010 |
| r2 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | 1+3 | 300 | +0.54 [−0.29, +1.31] | 25.8 % | +0.72 [−0.31, +1.85] | 0.172 / 0.010 |
| r2 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | ABAB | 300 | +0.41 [−0.12, +0.91] | 25.2 % | +0.81 [−0.26, +1.86] | 0.176 / 0.010 |
| **r2 λ = 0.5 vs ISMCTS-1000 (A9)** | 1+3 | 1,000 | +0.57 [+0.13, +1.03] | 24.7 % | **+0.76 [+0.13, +1.37]** | 0.167 / 0.009 |
| **r2 λ = 0.5 vs ISMCTS-1000 (A9)** | ABAB | 1,000 | +0.43 [+0.13, +0.74] | 24.5 % | **+0.86 [+0.29, +1.50]** | 0.177 / 0.010 |

**Verdict round 2: A9 fails** on its +1.0 bar, but for the first time r2 beats ISMCTS-1000
**significantly** in both lineups (intervals above 0). A12 was not run. The script's verdict line
("ISMCTS ships as expert") is the A9 stop rule; round 3 still runs, because the plan approves
rounds 1–3. PUCT still costs about 18× ISMCTS-1000's time per move.

Teacher check: no direct r2-vs-ISMCTS-5000 match exists. By Brian's rule the teacher stays
ISMCTS-5000 for round 3. Round 3 adds that match (`expert_iteration.py --stages teacher`).

### Trend over the rounds (A9 cell, 1,000 deals, paired diff vs ISMCTS-1000)

| Round | Teacher | Trained on | λ | 1+3 | ABAB | Calibration slope |
|---|---|---|---:|---:|---:|---:|
| r0 | ISMCTS-1000 | round-0 labels | 1 | −0.12 | −1.78 | 0.86 |
| r1 | ISMCTS-5000 | round 1 | 0.5 | +0.46 [−0.18, +1.13] | +0.16 [−0.47, +0.81] | 1.03 |
| r2 | ISMCTS-5000 | rounds 1–2 | 0.5 | **+0.76 [+0.13, +1.37]** | **+0.86 [+0.29, +1.50]** | 1.05 |
| r3 | ISMCTS-5000 | rounds 1–3 | 0.5 | **+1.06 [+0.46, +1.67]** | **+0.66 [+0.08, +1.28]** | 1.06 |

The mean of the two lineups runs −0.95 → +0.31 → +0.81 → +0.86. Round 1 and round 2 each gained
points; round 3 gained almost nothing on that mean, because 1+3 went up and ABAB went down, and
both moves are well inside the ± 0.6 of one cell. The gain has levelled off at about +0.85,
close to what the teacher itself gets against ISMCTS-1000 (+1.09 [+0.35, +1.87], `build-phase1.md`
ladder v0). The direct match below says the same thing: r3 has caught up with its teacher and
does not pass it.

### Round 3 (2026-11-01)

The last round the plan approves. Same teacher ISMCTS-5000; r3 from r2's weights on round-1, 2
and 3 labels; same loss weights; calibration on round 3's held-out shards; λ screen; A9 at 1,000
deals; r3 vs r2 (A12) only if A9 passes; then **r3 vs ISMCTS-5000 at 1,000 deals**, so Brian's
rule could be applied to the teacher itself. Job `runs\b25_round3\run.ps1`. A Windows Update
restart on 31 Oct stopped the job during labelling and cost 19 h; relaunched at 23:56 it resumed
from its shards, and it ended at 04:07 on 1 Nov.

Source: `runs/round3/results.json`, `runs/round3/table.md`, `runs/round3/vs_teacher/table.md`;
calibration `runs/round3/f_calibration/results.json`; training `runs/net_r3/results.json`.

- **Labels:** 807,378 decisions (4.1 M candidates, 1,548 shards), ISMCTS-5000, 8 workers.
- **Training:** from r2's weights on 2.42 M decisions (rounds 1–3); stopped after 35 min (no better
  validation loss in 6 evaluations); best at 5 min: validation prior top-1 76.3 % (r2: 76.2 %),
  score RMS 2.14 points (r2: 2.17). So r3 is r2 with a small polish on the extra labels.
- **Calibration** on round 3's 31 held-out shards (16,265 decisions): slope **1.06** (r2: 1.05),
  bias −0.16, RMS 2.36, correlation 0.980; prior vs visit share: expected calibration error 0.008.

| Match | Lineup | Deals | Candidate mean (95 % CI) | Candidate win rate | Paired diff (95 % CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|
| r3 PUCT-2000 λ = 1 vs ISMCTS-1000 | 1+3 | 300 | +0.96 [+0.14, +1.81] | 23.2 % | +1.28 [+0.21, +2.40] | 0.186 / 0.011 |
| r3 PUCT-2000 λ = 1 vs ISMCTS-1000 | ABAB | 300 | +0.38 [−0.21, +0.94] | 23.4 % | +0.76 [−0.40, +1.87] | 0.170 / 0.010 |
| r3 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | 1+3 | 300 | +0.91 [+0.06, +1.78] | 24.2 % | +1.21 [+0.13, +2.32] | 0.189 / 0.010 |
| r3 PUCT-2000 λ = 0.5 vs ISMCTS-1000 | ABAB | 300 | +0.62 [+0.09, +1.14] | 24.8 % | +1.25 [+0.20, +2.31] | 0.199 / 0.011 |
| **r3 λ = 0.5 vs ISMCTS-1000 (A9)** | 1+3 | 1,000 | +0.80 [+0.35, +1.28] | 23.9 % | **+1.06 [+0.46, +1.67]** | 0.193 / 0.011 |
| **r3 λ = 0.5 vs ISMCTS-1000 (A9)** | ABAB | 1,000 | +0.33 [+0.03, +0.63] | 24.2 % | **+0.66 [+0.08, +1.28]** | 0.203 / 0.011 |
| **r3 λ = 0.5 vs ISMCTS-5000 (teacher)** | 1+3 | 1,000 | +0.03 [−0.37, +0.43] | 23.4 % | **+0.04 [−0.52, +0.57]** | 0.185 / 0.055 |
| **r3 λ = 0.5 vs ISMCTS-5000 (teacher)** | ABAB | 1,000 | −0.00 [−0.29, +0.31] | 23.9 % | **−0.00 [−0.62, +0.58]** | 0.147 / 0.043 |

**Verdict round 3: A9 is not met in both lineups.** 1+3 passes (+1.06), ABAB fails (+0.66); both
intervals sit above 0. A12 was not run. The stop rule therefore applies, and no shipped network
exists.

**Teacher rule** (Brian, 2026-10-30): "use teacher unless distilled model wins significantly". r3
vs ISMCTS-5000 is +0.04 and −0.00, with both intervals across 0. The teacher stays. r3 ties
ISMCTS-5000 at about 3.4× its time per move (0.19 s against 0.05 s), and needs a 1.2 MB network
on the phone.

**What happens next.** The plan's three rounds are done. The default in `docs/open-items.md`
stands until Brian answers: stop after round 3, and ISMCTS ships as `expert`. A round 4 is not
approved. It would need a stronger teacher to help: a network that has caught its teacher learns
little more from it.

## 2. Hard-constraint belief from the rules (Task 18)

`RulesMask` (`crates/search/src/belief.rs`): uniform deals, minus those under which an
action already taken would have been illegal. In Big Two only the last-card duty ties an
action to the hidden cards (a single played while the next player held one card was the
player's highest), so the check is one backward pass over the history
(`State::history_consistent`, tested equal to a full replay). After 1,000 rejections in a
row it returns the last draw and counts a fallback. `analyze(belief="rules")` for ISMCTS
and PUCT; agent spec option `belief=rules`.

Smoke run (2 self-play games, ISMCTS-100): 13 % of decisions rejected at least one draw;
those rejected 38 % of draws on average (largest 57 %); no fallbacks.

**Full run (2026-10-30),** `experiments/build_phase3.py rules-mask`; source
`runs/p3_matches/a_rules_mask/` (`table.md`, `results.json`). Rejection share over 1,095
decisions of 40 ISMCTS-300 self-play games (`belief=rules`): **mean 0.5 % of draws per
decision**; 1.0 % of decisions reject at least one draw, and those reject 53.9 % on average
(largest 91.2 %); no fallbacks after 1,000 rejections.

Match: PUCT-2000 (round-0 network, `c_puct` 2.5, λ = 0.5) with `belief=rules` against the same
without it, 300 duplicate deals, duty on, tiered.

| Match | Lineup | Deals | Candidate mean (95 % CI) | Candidate win rate | Paired diff (95 % CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|
| PUCT-2000 rules mask vs uniform | 1+3 | 300 | +0.15 [−0.31, +0.64] | 25.6 % | +0.20 [−0.40, +0.82] | 0.196 / 0.197 |
| PUCT-2000 rules mask vs uniform | ABAB | 300 | +0.04 [−0.24, +0.35] | 25.1 % | +0.08 [−0.47, +0.65] | 0.205 / 0.205 |

**Verdict A13: met** (the mask loses nothing: both intervals include 0, means slightly
positive; no measurable cost per move). The mask rarely bites, because only the last-card duty
ties an action to hidden cards; it is kept on as a correctness fix, not a strength gain.

## 3. Learned belief (Task 19)

Status 2026-11-10: code done, round 1 job queued (hkq job 46); numbers below when it ends.

- **Network** (`python/big2/rl/belief.py`): state features (what the Q-network sees) → 320 → 256
  → `52 × 3` logits, added to the hand-size baseline (seat `s` holds a hidden card with
  probability `cards_s / hidden cards`); 475,548 parameters. Loss: log-likelihood of the true
  holder of each hidden card. Exported as `.pknn` (`big2-belief`), read by
  `big2_big2::belief_net::BeliefNet`; the Rust output equals PyTorch's to 1e-4
  (`python/tests/test_belief.py`).
- **Sampler** (`big2_search::NetSampler`): draw 256 uniform deals, drop those the rules
  exclude, weight the rest by the product of the net's per-card holder probabilities, temper
  as `Belief` does (effective sample size ≥ 25 %), draw from them. `analyze(belief="net",
  belief_net=<.pknn>)` for ISMCTS and PUCT.
- **Data deviation:** the plan trains on PUCT self-play. PUCT costs about 0.2 s per move, so
  1 M positions would take about 50 CPU-hours; the positions come from self-play of round 1's
  network by its prior (5 % random moves), 256 games at a time, about 32 CPU-minutes per
  million. Hidden cards are the true hands at each decision. Every 20th shard is held out.
- **Smoke run** (20,000 positions, 2 epochs, 10,000 held out): log-likelihood per hidden card
  −1.0550 against −1.0555 for the hand-size baseline (too little data to say anything).
- **Rule (A14):** keep the sampler in the ladder only if PUCT-2000 with it beats PUCT-2000
  with the rules mask by ≥ +0.5 paired with the interval above 0; else "teaching only".
  Round 1 only.

## 4. Information-set exact endgame (Task 20)

`crates/search/src/infoset_endgame.rs`: when at most `threshold` (12) cards are hidden and
at most `max_cards` (10) are in all hands together, every deal of the hidden cards is listed
(`Game::for_each_deal`), solved by max^n (`Endgame::solve_actions_within`, one node budget for
all deals), and each action averaged, weighted by the sampler (0 for deals the rules mask
excludes). `Forced { winner, plays }` when every deal gives the same winner after the best
move. Hooked into ISMCTS and PUCT at the root (`infoset_endgame` setting); when the budget
runs out, the normal search runs (`exact: false`).

**Cost follows the cards in play, not the hidden cards.** A probe on positions from greedy
self-play (≤ 12 hidden, 5 M-node cap, one thread, CPU under outside load):

| Cards in all hands | Positions | Solved | Mean s (solved) | Max s (solved) |
|---:|---:|---:|---:|---:|
| 9 | 1 | 1 | 0.006 | 0.006 |
| 10 | 2 | 2 | 0.317 | 0.633 |
| 11 | 4 | 4 | 0.744 | 1.917 |
| 12 | 3 | 2 | 0.157 | 0.168 |
| 13 | 5 | 4 | 0.997 | 2.623 |
| 14 | 4 | 1 | 0.426 | 0.426 |
| 15–16 | 9 | 2 | 0.56 | 0.639 |
| 17–20 | 16 | 0 | – | – |

Plain max^n prunes nothing, at about 1 M positions per second here. The default `max_cards`
is therefore 10; the queued run (`build_phase3.py endgame`) measures agreement with
ISMCTS-100,000 (A15) and times by cards in play 8–14, then plays PUCT-2000 with and without it.
A faster per-deal solver is in `docs/open-items.md`.

**Full run (2026-10-30),** `experiments/build_phase3.py endgame`; source
`runs/p3_matches/b_endgame/` (`table.md`, `results.json`). 300 late positions (≤ 12 cards hidden
from the player to move, two or more legal moves) from PUCT-300 self-play; node budget 20 M per
call; 60 positions not solved within it.

Agreement with ISMCTS-100,000 on the 240 solved positions:

| Measure | Value | Bar | Verdict |
|---|---|---|---|
| Same move | 85.8 % | – | – |
| A move of the same exact value | 94.2 % | ≥ 95 % (A15) | not met (by 0.8 points) |
| ISMCTS's move: mean loss by the exact values | 0.03 points | – | – |
| Forced results | 94 of 240 | – | – |

Native time per exact call (one thread; 8 positions solved in parallel, CPU 80.6 % busy before
the start):

| Hidden cards ≤ | Positions | Not solved | Mean s | 95th pct s | Max s |
|---:|---:|---:|---:|---:|---:|
| 8 | 164 | 4 | 2.625 | 21.524 | 51.418 |
| 10 | 238 | 25 | 7.514 | 44.479 | 51.418 |
| 12 | 300 | 60 | 11.939 | 46.703 | 51.466 |

| Cards in play ≤ | Positions | Not solved | Mean s | 95th pct s | Max s |
|---:|---:|---:|---:|---:|---:|
| 8 | 41 | 0 | 0.003 | 0.008 | 0.020 |
| 10 | 86 | 0 | 0.045 | 0.169 | 0.956 |
| 12 | 177 | 0 | 1.944 | 14.239 | 49.096 |
| 14 | 300 | 60 | 11.939 | 46.703 | 51.466 |

The plan's thresholds 8 / 10 / 12 (hidden cards) are all over 1 s natively at the 95th
percentile, so the limit that matters is cards in play: **10 cards in play** (with ≤ 12 hidden)
is the largest with every position solved and 95 % of calls under 1 s natively. That is the
default (`infoset_max_cards = 10`).

**Node WebAssembly timing (2026-11-01).** The same 300 positions, the same 20 M node budget, the
app's build flags, 4 threads side by side (`experiments/wasm_timing.mjs endgame`, source
`runs/wasm_timing/endgame.json`). A 13 cards probe ran beside it on about 4 cores. The native
table above ran on a busier CPU, so the two tables do not compare exactly. The same 60 positions
stay unsolved.

| Hidden cards ≤ | Positions | Not solved | Mean s | 95th pct s | Max s |
|---:|---:|---:|---:|---:|---:|
| 8 | 164 | 4 | 1.358 | 6.915 | 24.177 |
| 10 | 238 | 25 | 3.855 | 23.174 | 24.908 |
| 12 | 300 | 60 | 6.358 | 24.084 | 27.657 |

| Cards in play ≤ | Positions | Not solved | Mean s | 95th pct s | Max s |
|---:|---:|---:|---:|---:|---:|
| 8 | 41 | 0 | 0.003 | 0.010 | 0.019 |
| 10 | 86 | 0 | 0.025 | 0.100 | 0.361 |
| 12 | 177 | 0 | 0.981 | 7.045 | 23.290 |
| 14 | 300 | 60 | 6.358 | 24.084 | 27.657 |

As natively, no hidden-card threshold stays under 1 s. **10 cards in play is the largest limit
under 1 s in Node WebAssembly:** every position solved, 95th percentile 0.10 s, slowest 0.36 s.
That meets A15's time part at the default `infoset_max_cards = 10`.

Match: PUCT-2000 (round-0 network, λ = 0.5) with the endgame (`infoset_endgame=12,
infoset_max_cards=10`) against the same without it, 300 duplicate deals, PUCT-2000 opponents:

| Match | Lineup | Deals | Candidate mean (95 % CI) | Candidate win rate | Paired diff (95 % CI) | s per move cand / opp |
|---|---|---:|---:|---:|---:|---:|
| PUCT-2000 + exact endgame | 1+3 | 300 | +0.14 [−0.32, +0.63] | 25.5 % | +0.19 [−0.40, +0.82] | 0.181 / 0.181 |
| PUCT-2000 + exact endgame | ABAB | 300 | +0.03 [−0.25, +0.35] | 25.1 % | +0.07 [−0.49, +0.65] | 0.173 / 0.172 |

**Verdict:** the endgame costs nothing per move and gains nothing measurable at 300 deals (it
only fires in the last few tricks). The two matches of §2 and §4 play the same deals with the
same seeds: 579 of their 600 deal records give the candidate identical scores (checked
2026-10-30). So the rules mask and the endgame each change play in only a few deals, and the
shared +0.2 / +0.1 is the same noise in both, not an effect of either feature. A15 is narrowly missed on agreement (94.2 % against 95 %)
and met on time (0.36 s at most in Node WebAssembly at 10 cards in play);
the 60 unsolved positions are all above 10 cards in play, where the hook does not fire. Its use
is teaching: exact results and `forced` proofs (`reason.counterfactual.forced`).

## 5. Charlesworth's network (Task 21)

Run 2026-11-05 after Brian's yes ("local only"). Weights: `modelParameters136500` and
`actionIndices.pkl` from `github.com/henrycharlesworth/big2_PPOalgorithm` at commit
`eea4b04f544fcbf3ca29f117858363b466873f0e` (default branch `master`); the repo has no licence
file, so they stay in `runs/external/charlesworth/` (gitignored), loaded by a restricted
unpickler (numpy arrays only), never committed or shipped (B14). The forward pass is our own
NumPy code (`python/big2/agents/external/charlesworth.py`); its 412-feature layout was checked
against his source (b45d709: five mismatches fixed; win rate against three greedy bots went
from 29.7 % to 47.0 %). Rules: his four-card plays (hidden `four_card_plays` flag, 95480b7)
and his scoring (`sum_of_others`). Queue job 12, 27 minutes on 8 threads.

| Lineup | Deals | `expert` mean | `expert` win rate | Charlesworth mean | Charlesworth win rate | Paired diff expert − Charlesworth | s per move expert / his |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1+3 | 1,000 | +0.05 ± 0.20 | 24.7 % ± 1.1 % | −0.02 ± 0.07 | 25.1 % ± 0.4 % | +0.06 ± 0.26 | 0.089 / 0.0005 |
| ABAB | 1,000 | +0.02 ± 0.15 | 24.6 % ± 0.8 % | −0.02 ± 0.15 | 25.4 % ± 0.8 % | +0.04 ± 0.30 | 0.095 / 0.0006 |

`expert` = `ismcts:iterations=10000,infoset_endgame=12,infoset_max_cards=10`. Both intervals
include 0: **no difference between `expert` and Charlesworth's network is detected at 1,000
deals** (A16, which asks for `expert` ahead with the interval excluding 0, is missed), while it
thinks about 170 times longer per move. Our inference, not tested: his rules (four-card plays)
are new to `expert`'s rollout policy, and his network was trained on exactly these rules.
Raw: `runs/b2_charlesworth/b2_charlesworth/table.md`, `results.json`.

## 6. Best-response probe (Task 22)

Queued in `runs\b25_tasks22_24\run.ps1`; results land here when it ends. Code:
`big2 rl train --opponents <spec>` (`FrozenOpponents`, `collect_vs_frozen` in
`python/big2/rl/dmc.py`), `experiments/build_phase3.py probe` and `probe-eval`.

What differs from the plan, and why:

- **The third candidate is `expert`** (ISMCTS-10,000 + exact endgame), not the final PUCT
  network: no network passed A9, and ISMCTS ships by default.
- **Every probe starts from the same distilled network** (`models/big2-distilled-duty.pt`), not
  from random weights. A search candidate gives few games: a smoke run reached about 150 games
  per minute against `expert`, so about 88,000 games in 10 h. A learner from scratch would not
  learn the game in that many. The gain is end minus start, 1,000 deals, 1 + 3.
- **Equal games as well as equal hours.** ISMCTS-1000 gives about 10× the games of `expert` per
  hour, and the DMC-10h probe far more, so a smaller gain against `expert` could only mean fewer
  games. `probe-eval` therefore also plays each probe at the snapshot with the games of the
  slowest probe. A17 is read on that column too.

### 6.1 From-scratch probe against `expert` (Task 33)

Status 2026-11-10: code done, job queued (see board); numbers below when it ends.

Task 22's probes start from the distilled network, so they may share the teacher's blind spots.
This one starts from fresh weights (same net shape as `models/big2-distilled-duty.pt`), three
frozen `expert` seats, 10 h on the GPU lane. Code: `experiments/build_phase3.py probe --probes
scratch_expert` (no `init`) and `probe-scratch-eval` (the probe's end vs three `expert`, 1,000
deals, 1 + 3; the mean score is the gain, the match being zero-sum). Job
`runs\probe_scratch_expert\run.ps1`; check `runs\probe_scratch_expert\progress.json`.
Caveat from the note above: about 88,000 games in 10 h may be too few to learn the game from
nothing, so a gain near or below 0 is possible and would say so.

## 7. Cross-play of two seeds (Task 23)

Queued in the same job. The plan's second seed reuses round N's command. The labels exist, so
seed B trains on rounds 1–3's labels from **fresh weights** with seed 1 (at most 3 h on the GPU,
`experiments/build_phase3.py seeds-train`): a second chain from r2's weights would share r3's
start. Then PUCT-2000 λ = 0.5 r3 vs seed B, 1,000 deals, both lineups (A18: the interval covers
0). No network ships, so this measures how repeatable the teaching network is. No shipped bot
depends on it.

## 8. Final 10,000-deal matches (Task 24)

Queued in the same job. The plan named PUCT-2000 (final). **This measures the bot that ships by
default instead:** `expert` (ISMCTS-10,000 + exact endgame) vs ISMCTS-1000, 10,000 deals, both
lineups, 8 workers. The match against Charlesworth ran separately (Task 21, §5). PUCT-2000 r3 at 10,000 deals
was not queued; at its 0.19 s per move it would take about 3× as long, and it is worth running
only if Brian picks PUCT.

## 9. Summary (Task 25)

Filled in as the job ends. Numbers so far, each with its source:

| Bar | Result | Source | Verdict |
|---|---|---|---|
| A9 | r3 vs ISMCTS-1000: +1.06 (1+3), +0.66 (ABAB) | `runs/round3/results.json` | not met |
| A11 | PUCT-2000: 0.83 s app build, 0.28 s SIMD; move change 3.0 % | `runs/wasm_timing/puct.json`, `puct_simd.json` | time not met in the app build; move change met |
| A12 | not run (A9 failed) | `runs/round3/results.json` | stop rule applied |
| A13 | mask loses nothing | `runs/p3_matches/a_rules_mask/results.json` | met |
| A14 | Task 19 not started | – | open |
| A15 | 94.2 % agreement; 0.36 s at most in Node at 10 cards | `runs/p3_matches/b_endgame/results.json`, `runs/wasm_timing/endgame.json` | agreement missed by 0.8; time met |
| A16 | +0.06 ± 0.26 (1+3), +0.04 ± 0.30 (ABAB) vs Charlesworth, his rules | `runs/b2_charlesworth/b2_charlesworth/results.json` | missed: intervals include 0 |
| A17, A18, A19 | queued | `runs/probe_eval`, `runs/seed_crossplay`, `runs/final_matches` | pending |
| Teacher rule | r3 vs ISMCTS-5000: +0.04, −0.00 | `runs/round3/vs_teacher/results.json` | teacher stays |
