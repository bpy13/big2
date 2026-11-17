# The win-chance bot against `expert` (plan Task 34)

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, Task 34. Brian (2026-11-11): "train a new model
that only optimizes changes of winning (instead of EV) and have it compete against our best bot".

**Status 2026-11-17: code done, queue jobs waiting.** Numbers below are filled in when the jobs end.

## What changed, and only that

Every bot so far maximises expected points (tiered scoring: cards left cost points). The
win-chance bot runs the same pipeline as today's best network, r3 (`build-phase3.md` §1), with one
switch, `--objective win` in `experiments/expert_iteration.py`:

| Step | r3 (expected points) | win-chance |
|---|---|---|
| Round 0 labels | ISMCTS-1000, 2,740,296 decisions, objective score | same, objective **win** |
| Round 0 network | fresh weights, loss weights 1 : 1 : 1, value 0.5 | same; value fits **P(go out first)** |
| Rounds 1–3 labels | ISMCTS-5000, 800,000 decisions each, objective score | same, objective **win** |
| Rounds 1–3 networks | from the previous round, labels of rounds 1..N, value weight 100 on the tiered score | same; value fits **P(go out first)** |
| Search in play | PUCT-2000, `c_puct` 2.5, λ = 0.5, objective score | same, objective **win** |

Under objective win a search counts a simulation as 1 when the seat goes out first, 0 otherwise.
The network's leaf value is class 0 of the actor's cards-left distribution. Its value loss is the
visit-weighted squared error of that probability against the search's win rate for each candidate
(`DistillV2Config.value_target = "win"`, `python/big2/rl/distill.py`). The other heads are unchanged.

Orchestrator's calls (2026-11-17; Brian may change them): (1) the win pipeline has its own round 0,
so no tiered label reaches it; (2) λ = 0.5 and `c_puct` 2.5 are fixed at r3's values, and the
per-round screens and A9 checks are skipped. Those checks only decided whether to stop, and this
task trains to r3's full budget.

## Match

`experiments/win_chance.py final`: the win-chance bot vs `expert` (ISMCTS-10,000 + exact endgame),
10,000 duplicate deals (each 4×, seats rotated), lineups `1+3` and `ABAB`, tiered scoring,
last-card duty on, no four-card plays. Reported per bot: share of deals won and points per deal,
each with a 95 % bootstrap interval over deals, plus paired differences.

Pending (queue job 195).

## Habits

`experiments/win_chance.py habits`: 1,000 deals in `ABAB`, each played with both seatings. For each
bot: how often it passes when it holds a legal play, and, at the first moment an opponent has
3 cards or fewer, how many 2s it holds and how many five-card hands its best decomposition of the
rest uses. Brian's expectation (2026-11-11): "It should pass greedily to optimize late game dominance".

Pending (queue job 195).

## Jobs

| Job | Lane | Stage | Output (big2-workers worktree) |
|---|---|---|---|
| 193 | cpu | labels, rounds 0–3 | `runs/round<N>_win/label/progress.json` |
| 194 | gpu | networks r0–r3, calibration, export | `runs/net_r<N>_win/progress.json`, `runs/round<N>_win/f_calibration/` |
| 195 | cpu | final match and habits | `runs/win_chance/progress.json`, `results.json`, `table.md` |

All three run `experiments/win_chance_job.ps1 -Stage <stage>` and resume when added again.
