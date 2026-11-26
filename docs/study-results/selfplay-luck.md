# Self-play pilot with the luck taken out (plan Task 36)

Status 2026-11-27: training to hour 8 done; from 2026-11-27 every checkpoint is measured against
the app's `expert` only (Task 36b; jobs in the plan's Tasks 36 and 36b).
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

(pending: `runs/selfplay_luck/choice.json`)

## Results against `expert`

(pending: `runs/selfplay_luck/table_expert.md`, jobs in the plan's Task 36b)

## First measure (superseded): against r3 and ISMCTS-1000

4,000 deals per lineup, seed 7, job 345. The prior's top move with no search, paired diff per
deal with 95 % range. `expert` itself beats ISMCTS-1000 by about +0.9 a deal.

| Hour | vs start 1+3 | vs start ABAB | vs ISMCTS-1000 1+3 | vs ISMCTS-1000 ABAB | Gain vs ISMCTS-1000 over start (both lineups) |
|---:|---:|---:|---:|---:|---:|
| 0 | (start) | (start) | -0.42 [-0.75, -0.09] | -0.65 [-0.97, -0.31] | (start) |
| 4 | +4.72 [+4.41, +5.05] | +4.25 [+3.82, +4.67] | +4.59 [+4.23, +4.97] | +4.05 [+3.68, +4.44] | +3.05 [+2.83, +3.26] |
