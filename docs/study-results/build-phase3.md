# Build run phase 3: iterate, belief, exact endgames, external bars

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, phase 3 (Tasks 17–25). Match format as in
`build-phase2.md`: duplicate deals (each deal 4×, seats rotated), last-card duty on, tiered
scoring, no four-card plays, lineups `1+3` and `ABAB`, 95 % bootstrap intervals over deals.

Status 2026-10-31: rounds 1 and 2 of Task 17 done (A9 fails both; r2 beats ISMCTS-1000
significantly, +0.76 / +0.86, but under the +1.0 bar; §1); round 3 runs
(`runs\b25_round3\run.ps1`, plan Task 17 status). Task 18 done (A13 met, §2). Task 20's runs
done (§4; A15 missed at 94.2 %, Node WebAssembly timing waits for Task 16's harness).

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

Each round gains about +0.3 to +0.7 points. If round 3 gains as much again, it would reach
about +1.0 to +1.2: right at the A9 bar. The A9 intervals are about ± 0.6 wide at 1,000 deals.

### Round 3 (running 2026-10-31)

The last round the plan approves. Same teacher ISMCTS-5000 (above); r3 from r2's weights on
round-1, 2 and 3 labels (2.4 M decisions); calibration; λ screen; A9 at 1,000 deals; r3 vs r2
(A12) only if A9 passes; then **r3 vs ISMCTS-5000 at 1,000 deals** (`runs/round3/vs_teacher/`),
so the next session can apply Brian's rule to the teacher itself. Job
`runs\b25_round3\run.ps1` (plan Task 17 status has the check and resume commands).

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

Not started (after Task 17's round 1).

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
default (`infoset_max_cards = 10`). **Node WebAssembly timing is not measured yet:** it waits for
Task 16's harness.

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
shared +0.2 / +0.1 is the same noise in both, not an effect of either feature. A15 is narrowly missed on agreement (94.2 % against 95 %);
the 60 unsolved positions are all above 10 cards in play, where the hook does not fire. Its use
is teaching: exact results and `forced` proofs (`reason.counterfactual.forced`).

## 5.–8. Tasks 21–24

Not started.
