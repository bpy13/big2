# Self-play pilot with the luck taken out (plan Task 36)

Status 2026-11-27: done. 24 GPU-hours of training (76.6 M games) ended at hour 24; every
4-hour checkpoint was measured against the app's `expert` (Task 36b) and against its older
selves (Task 36c). Brian kept the hour-24 network for the expert (decision 0007).

Brian, 2026-11-26: "try the self-play luck removal approach with big2 to utilize the currently
idle GPU".

## Method

Trainer `python/big2/rl/selfplay.py` (a new trainer: `dmc.py` learns move values, not a
policy, and `ppo.py` has neither duplicate deals nor a start network to pull towards). Driver
`experiments/selfplay_luck.py`, queue script `experiments/selfplay_luck_job.ps1`.

- **Start network: round 3** (`runs/net_r3/best.pt`). `expert` is ISMCTS-10,000 with no network
  inside, so there is no "network inside expert" to start from. r3 is the best network
  (inside PUCT-2000 it ties its teacher ISMCTS-5000, build-phase3.md §1) and has a prior
  head, which a policy-gradient method needs. Its weights were copied read-only from the
  standalone checkout (orchestrator, 2026-11-26).
- **Policy**: r3's prior head, a softmax over each decision's legal moves; moves are sampled.
  PPO-clip (clip 0.2, AdamW lr 3e-5, 2 epochs, minibatch 2,048 decisions, grad-norm 1,
  entropy 0), as in both earlier runs.
- **Luck removed (duplicate)**: each deal is played 8 times: its four hands rotated over the
  seats, each rotation twice. A seat's advantage is its score minus the mean score of the same
  hand in the other games of that deal where it faced the same lineup (leave-one-out).
  256 deals = 2,048 games per update.
- **Oracle critic**: a separate value network sees every hand (`full_features`) and learns each
  decision's score; it never plays and is not exported. The baseline is the duplicate mean,
  the critic, or their average, whichever left the least return variance in the smoke run
  (mahjong found its critic nearly useless, R² 0.06; Dou Dizhu's duplicate baseline left 0.62
  of the variance).
- **KL pull**: constant `pull · KL(start ‖ policy)` per decision. Smoke run: 10 minutes each at
  0.02, 0.1, 0.5 (Dou Dizhu's grid); taken: the smallest whose end network vs the start (2,000
  duplicate deals per lineup) has a 95 % range reaching 0 or above in both lineups.
- **Frozen opponents**: half the deals are played learner / frozen / learner / frozen, the
  frozen seats being the start network or one of the last four hourly snapshots (Dou Dizhu's
  lesson: gains against its own copies did not transfer). Rotating by two seats keeps that
  lineup, so the duplicate baseline still compares like with like.
- **Distributions kept**: the cards-left heads keep learning the real outcome of the move
  played (weight 0.1), so a new network can still serve as PUCT's leaf value.
- **Budget and checks**: gpu lane, 4-hour jobs up to 24 GPU-h, a snapshot every hour.
  Each 4-hour snapshot (playing its prior's top move, no search) and the start r3 play the
  app's `expert` (below) on the same 1,000 deals (seed 7), both lineups, 95 % bootstrap ranges
  over deals; the gain over r3 is paired deal by deal. The last checkpoint (hour 24, or the
  one where the run stopped) and r3 are measured again at 2,000 deals for the record.
  **Stop rule**: unless hour 8's gain over r3 against `expert` (mean of both lineups) has its
  whole 95 % range above 0, the run stops (`runs/selfplay_luck/stop.json`; the later train
  jobs then exit at once). Until 2026-11-27 the checks were against r3 and ISMCTS-1000
  (4,000 deals); Brian: "it needs to beat our strongest?" and "it probably should only be
  matched against expert for real benchmark purposes".

## The benchmark: `expert` (Task 36b)

- **How it is loaded**: `ismcts:iterations=10000` through `big2._core.analyze`, whose defaults
  are the app's (`crates/app-api/src/solver.rs`: ISMCTS-10,000, exploration 0.25, greedy
  rollout ε 0.1, score objective, one thread, no exact endgame, no network). The app also caps
  a move's wall-clock time; the match does not, so every move gets its full 10,000
  simulations. (The phase-3 studies' `expert` added an exact endgame,
  `infoset_endgame=12`; the app does not.)
- **Is `expert` the strongest bot we have?** Yes, as far as anything has been measured, though
  it is not much stronger than cheaper searches. It beats ISMCTS-1000 by +0.92 a deal (10,000
  deals, build-phase3.md §8), but ties ISMCTS-2,000 (−0.03 ± 0.38, build-phase4.md §3).
  The best network search, r3 inside PUCT-2000, ties ISMCTS-5000 (+0.04 / −0.00,
  build-phase3.md §1) at about 0.19 s a move against `expert`'s 0.09 s. Nothing in the repo
  has been measured above `expert`.
- **Cost and range**: 25 CPU-s a deal (1+3) and 16 CPU-s (ABAB) against `expert`, so about
  11 CPU-h a checkpoint for 1,000 deals in both lineups (1.4 h on 8 threads), with a range of
  about ±0.66 per lineup and about ±0.3 on the paired gain over r3. The plan's target was ±0.5
  within 3 CPU-h. That needs about 1,750 deals per lineup (about 18 CPU-h), so the two cannot
  both be met. The orchestrator chose 1,000 deals (2026-11-27); the ranges actually got are in
  the table.
- **The checkpoint with search**: not measured. No network-guided ISMCTS runs at the app's
  speed: the network search (PUCT-2000) takes about 2× `expert`'s time a move, and it needs
  the checkpoint exported to `.pknn`.

## Smoke run

Job 343, 2026-11-26 (`runs/selfplay_luck/choice.json`). 10 minutes from r3 at each pull, seed 0,
duplicate baseline; each end network plays r3 (prior's top move, no search) on 2,000 duplicate
deals per lineup, paired diff per deal with 95 % range.

| Pull | Updates | Games | KL at end | vs r3 1+3 | vs r3 ABAB |
|---:|---:|---:|---:|---:|---:|
| **0.02** | 322 | 659,456 | 0.280 | +3.64 [+3.24, +4.07] | +3.04 [+2.50, +3.62] |
| 0.1 | 309 | 632,832 | 0.084 | +3.33 [+2.93, +3.72] | +2.82 [+2.25, +3.39] |
| 0.5 | 291 | 595,968 | 0.011 | +1.83 [+1.54, +2.14] | +1.68 [+1.25, +2.14] |

- Pull taken: **0.02**, the smallest on the grid; no setting got worse than r3, so the rule
  (smallest pull whose range reaches 0 or above in both lineups) picks it.
- Baseline taken: **mix** (duplicate mean and oracle critic averaged, oracle weight 0.5). Share
  of the return variance left (second half of the smoke logs): duplicate 0.63, critic 0.54,
  mix 0.47. Unlike mahjong (critic R² 0.06), Big Two's critic removes about as much as the
  duplicate, and the two together more. At the end of training: duplicate 0.46, critic 0.38,
  mix 0.33.

## Results against `expert`

Jobs 356 (r3 and hour 8, stop rule), 357 (hour 4), 359, 361, 363 (hours 12-20) and 399
(hour 24, the re-run of failed job 365, Task 47), 2026-11-27. The prior's top move, no
search, against three `expert` seats (1+3) or two (ABAB), 1,000 deals per lineup, seed 7;
paired diff per deal (the checkpoint's seat minus `expert`'s on the same cards) with 95 %
bootstrap range; the gain over r3 is paired deal by deal, mean of both lineups.

| Hour | Deals | vs `expert` 1+3 | vs `expert` ABAB | Gain over r3 vs `expert` (both lineups, paired) |
|---:|---:|---:|---:|---:|
| 0 | 1,000 | -1.93 [-2.53, -1.34] | -1.99 [-2.67, -1.32] | (start r3) |
| 4 | 1,000 | +3.21 [+2.61, +3.84] | +2.59 [+1.85, +3.37] | +3.07 [+2.68, +3.49] |
| 8 | 1,000 | +3.47 [+2.86, +4.09] | +3.05 [+2.29, +3.87] | +3.29 [+2.87, +3.72] |
| 12 | 1,000 | +3.33 [+2.70, +3.98] | +2.85 [+2.04, +3.69] | +3.18 [+2.76, +3.65] |
| 16 | 1,000 | +3.85 [+3.22, +4.55] | +3.19 [+2.33, +4.01] | +3.47 [+2.99, +3.94] |
| 20 | 1,000 | +3.69 [+3.07, +4.38] | +2.90 [+2.14, +3.72] | +3.33 [+2.88, +3.80] |
| 24 | 1,000 | +3.62 [+3.02, +4.24] | +3.25 [+2.47, +4.06] | +3.39 [+2.96, +3.84] |

- Ranges got: about ±0.6 a deal (1+3) and ±0.7-0.8 (ABAB) per lineup, ±0.4 on the gain over r3.
- r3 alone loses to `expert` by about 2 points a deal; after 4 GPU-hours of self-play the
  network alone (no search) beats `expert` by about 3, and hour 8 a little more (+0.2 over
  hour 4, inside the range). Stop rule passed at hour 8 (gain +3.29 [+2.87, +3.72]), so
  training goes on (job 358).
- This agrees with the superseded check below (gain +3.05 against ISMCTS-1000 at hour 4).
- Hours 12 to 24 stay level with hour 8: every one is +3.2 to +3.5 over r3, and every pair of
  hours lies inside the others' ranges. Two thirds of the final gain (and more) came in the
  first 4 hours, as in mahjong.
- The 2,000-deal final match of the last checkpoint and r3 (jobs 366, 400) was dropped on
  Brian's word (2026-11-27): "we already know it's much stronger than expert, we need to find
  out whether this improves over the previous versions".

Training (from `runs/selfplay_luck/log.jsonl`): KL(r3 ‖ policy) per decision and the policy's
entropy at each checkpoint. The pull held the network about the same distance from r3 all run.

| Hour | Updates | Games | KL to r3 | Entropy |
|---:|---:|---:|---:|---:|
| 4 | 7,683 | 15.7 M | 0.350 | 0.326 |
| 8 | 14,218 | 29.1 M | 0.362 | 0.303 |
| 12 | 19,837 | 40.6 M | 0.371 | 0.312 |
| 16 | 25,846 | 52.9 M | 0.383 | 0.302 |
| 20 | 32,325 | 66.2 M | 0.380 | 0.308 |
| 24 | 37,382 | 76.6 M | 0.376 | 0.307 |

## Without luck removal (Task 36d)

Same 10-minute smoke setting from r3 (pull 0.02, seed 0, same frozen-opponent mix). Arms: `dup` =
Task 36's smoke network (duplicate rotations, leave-one-out baseline, no oracle critic); `plain` =
no duplicate, no oracle critic (advantage = score minus batch mean, divided by its sd); `oracle` =
no duplicate, oracle critic only. Measured vs r3 at 2,000 deals per lineup and vs `expert` at
1,000 deals (jobs 372, 373); paired, 95 % ranges. Games differ because the arms ran the same
10 minutes.

| Arm | Games | KL at end | vs r3 1+3 | vs r3 ABAB | vs `expert` 1+3 | vs `expert` ABAB | Gain over r3 vs `expert` (both) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dup | 659,456 | 0.280 | +3.64 [+3.24, +4.07] | +3.04 [+2.50, +3.62] | +2.01 [+1.41, +2.60] | +1.26 [+0.50, +2.02] | +2.29 [+1.91, +2.69] |
| plain | 466,944 | 0.280 | +2.79 [+2.40, +3.17] | +1.55 [+1.02, +2.12] | +1.20 [+0.63, +1.81] | +0.41 [-0.35, +1.17] | +1.78 [+1.40, +2.17] |
| oracle | 468,992 | 0.285 | +3.65 [+3.25, +4.11] | +2.93 [+2.37, +3.50] | +2.04 [+1.47, +2.65] | +1.24 [+0.51, +1.95] | +2.30 [+1.91, +2.72] |

- Verdict 1: luck removal accounts for part of the gain, not most: with none of it the network
  still gains +1.78 [+1.40, +2.17] over r3 against `expert`, versus +2.29 [+1.91, +2.69] with
  the duplicate, so about three quarters of the gain remains (the ranges overlap a little).
- Verdict 2: the oracle critic alone (no duplicate) matches the duplicate arm (+2.30 vs +2.29),
  so the variance reduction is what helps, and the duplicate adds nothing on top of the critic.

## Head to head against older selves (Task 36c)

Jobs 367-370 and 401 (hour 24, the re-run of failed job 371, Task 47). Networks only (each
plays its prior's top move), 2,000 deals per lineup (the first 1,000 are the `expert` matches'
deals), seed 7; paired diff per deal, the first-named hour's seat minus the other's, with 95 %
range. Hour 24 vs hour 16 (Task 48, `h2h --hour 24 --against 16`): both ranges include 0.

| Hour | Against | Deals | 1+3 | ABAB |
|---:|---|---:|---:|---:|
| 8 | hour 4 (previous) | 2,000 | +0.10 [-0.24, +0.46] | +0.43 [-0.09, +0.93] |
| 12 | hour 8 (previous) | 2,000 | -0.07 [-0.42, +0.28] | -0.10 [-0.62, +0.38] |
| 16 | hour 8 (first above expert) | 2,000 | +0.13 [-0.23, +0.49] | +0.11 [-0.38, +0.63] |
| 16 | hour 12 (previous) | 2,000 | -0.10 [-0.45, +0.26] | -0.05 [-0.52, +0.44] |
| 20 | hour 8 (first above expert) | 2,000 | -0.10 [-0.46, +0.26] | -0.24 [-0.74, +0.27] |
| 20 | hour 16 (previous) | 2,000 | -0.10 [-0.46, +0.25] | -0.00 [-0.49, +0.48] |
| 24 | hour 8 (first above expert) | 2,000 | +0.25 [-0.09, +0.59] | +0.38 [-0.13, +0.89] |
| 24 | hour 16 (Task 48) | 2,000 | +0.08 [-0.26, +0.42] | +0.13 [-0.34, +0.60] |
| 24 | hour 20 (previous) | 2,000 | +0.07 [-0.26, +0.42] | +0.40 [-0.09, +0.89] |

- No head-to-head after hour 4 is clear of luck at 2,000 deals: training plateaued at about
  hour 8, within ±0.4 a deal. Hour 24 is the only checkpoint leaning positive against both its
  predecessor and hour 8.

## Verdict

- Self-play with the luck taken out works for Big Two: from r3 (-1.9 a deal against `expert`)
  the network alone, with no search, beats the app's `expert` (ISMCTS-10,000) by about +3.4 a
  deal in both lineups, a gain of +3.39 [+2.96, +3.84] over r3. Most of it came in the first
  4 GPU-hours; hours 8 to 24 are level within ±0.4.
- Unlike Dou Dizhu, the gain against its own start carried over to the outside benchmark. The
  start was 2 points below that benchmark (Dou Dizhu's was level with it), and half the seats
  were frozen opponents.
- Luck removal explains part of the gain, not most of it (Task 36d): about three quarters
  remains with none of it, and the oracle critic alone matches the duplicate.
- Kept: Brian (2026-11-27) chose the hour-24 network as the network inside the new expert
  (decision 0007, Tasks 38-41, 45). The shipped app's `expert` is unchanged by this task.
- This task started no second long run (that needs Brian's word).

## First measure (superseded): against r3 and ISMCTS-1000

4,000 deals per lineup, seed 7, job 345. The prior's top move with no search, paired diff per
deal with 95 % range. `expert` itself beats ISMCTS-1000 by about +0.9 a deal.

| Hour | vs start 1+3 | vs start ABAB | vs ISMCTS-1000 1+3 | vs ISMCTS-1000 ABAB | Gain vs ISMCTS-1000 over start (both lineups) |
|---:|---:|---:|---:|---:|---:|
| 0 | (start) | (start) | -0.42 [-0.75, -0.09] | -0.65 [-0.97, -0.31] | (start) |
| 4 | +4.72 [+4.41, +5.05] | +4.25 [+3.82, +4.67] | +4.59 [+4.23, +4.97] | +4.05 [+3.68, +4.44] | +3.05 [+2.83, +3.26] |
