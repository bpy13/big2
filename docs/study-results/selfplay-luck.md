# Self-play pilot with the luck taken out (plan Task 36)

Status 2026-11-26: built and queued; numbers pending (jobs listed in the plan's Task 36).
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
  Each 4-hour snapshot (playing its prior's top move, no search) plays the start network and
  ISMCTS-1000 (never trained against) on the same 4,000 deals (seed 7), both lineups, 95 %
  bootstrap ranges over deals; the gain over the start against ISMCTS-1000 is paired deal by
  deal with the start's own match. **Stop rule**: unless hour 8's gain (mean of both lineups)
  has its whole 95 % range above 0, the run stops (`runs/selfplay_luck/stop.json`; the later
  train jobs then exit at once).

## Smoke run

(pending: `runs/selfplay_luck/choice.json`)

## Results

(pending: `runs/selfplay_luck/table.md`)
