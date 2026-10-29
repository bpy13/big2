# Build run, phase 2: results

Plan: [`docs/plans/2026-10-29-big2-bot-build.md`](../plans/2026-10-29-big2-bot-build.md), phase 2.
Script: [`experiments/build_phase2.py`](../../experiments/build_phase2.py) (one subcommand per
section). Raw results live in `runs/p2_matches/` (git-ignored): per match and lineup, blocks of
100 deals, then `results.json` and `table.md` per subcommand.

Match format unless a section says otherwise: duplicate deals (seeds 0…N−1, each deal 4×, seats
rotated), last-card duty on, `tiered` scoring, no four-card plays (B22), both lineups (`1+3`:
the candidate in one seat; `ABAB`: alternate seats). Mean points per game for the candidate with
a 95 % bootstrap interval over deals, win rate beside it, and the paired difference per deal.

Network: round 0, network v2 (Task 12), `runs/net_r0/best.pt`, exported as
`runs/net_r0/pvnet-r0.pknn` (1.17 MB). Its validation losses on 55,046 held-out decisions:
actor 1.576, seats 1.818, prior 1.259 (top-1 68.5 %), score 0.0083 (RMS 4.56 points).

Status: the job started 2026-10-29 14:43 (8 workers, headless). The tables below are filled in
from `runs/p2_matches/*/table.md` once it ends.

## 1. PUCT-2000 vs ISMCTS-1000 (a, A9)

1,000 deals. Fixed simulation counts, so CPU load changes the seconds per move only.
Bar A9: ≥ +1.0 with the interval above 0.

(pending)

## 2. PUCT vs ISMCTS at equal wall time (b)

0.5 s per move each, one thread, no simulation cap; a forced move (one legal action) plays at
once for both sides. 1,000 deals. **This match ran under outside CPU load**: a 13 cards probe and
then a mahjong match job, plus other projects, shared the 20 logical CPUs; `load.json` in
`runs/p2_matches/b_equal_time/` records the load before the start. Both sides take turns on the
same worker, so load slows both alike, but the absolute simulation counts are lower than on an
idle machine. A single call at the opening on this PC reached 29,312 ISMCTS iterations and 6,064
PUCT simulations in 0.5 s.

(pending)

## 3. Distribution head vs mean-score head (c, A10)

The v2 network played greedily without search (`pvnet:checkpoint=runs/net_r0/best.pt`: the move
with the best expected score for the actor from the four cards-left distributions, under the
game's own scoring) against Task 7's mean-score network
(`qnet:path=models/big2-qnet-duty.pknn,temperature=0`). 500 deals under `tiered` and 500 under
`win_loss`. Bar A10: loses ≤ 0.5 under tiered and wins under win_loss.

(pending)

## 4. `c_puct` in {1.0, 1.5, 2.5} (d)

PUCT-2000 against ISMCTS-1000 on the first 300 deals; 1.5 (the default) reuses the first 300
deals of section 1.

(pending)

## 5. WebAssembly timing (Task 16)

(pending)
