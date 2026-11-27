# 0006 — The depth and budget study runs as planned (about 125 CPU-h), with four stated assumptions

**Status:** accepted by default (2026-11-27, orchestrator; Brian may change it)

## Context

Plan Task 39 compares search settings on top of the self-play network (Task 37's `.pknn`)
and estimated about 65 CPU-h. Measured on this PC, `k 3, n 64, depth full` costs 0.2-0.7 s a
move native, so the screen (18 settings, 300 deals, two lineups) is about 32 CPU-h, the
finalists at 2,000 deals about 37 CPU-h and the five finalists against `expert`
(ISMCTS-10,000, 1,000 deals, both lineups; 41 CPU-s a deal, Task 36b) about 57 CPU-h: about
125 CPU-h, about 8 h wall on the cpu lane's 2 x 8 threads. Slot 1 asked whether to trim the
`expert` matches (500 deals, or only the knee and the best setting).

## Decision

Proceed as planned: every finalist against `expert` at 1,000 deals in both lineups. Brian
wants the benchmark against `expert` only, and CPU time costs nothing. Four points of method
are accepted and stated in the report (`docs/study-results/expert-search-budget.md`):

1. The raw network is the same `.pknn`'s top prior move played in Rust (`netsearch` with
   `k = 1, n = 1, depth = 0`, about 0.15 ms a move), not the PyTorch checkpoint.
2. PUCT runs with `c_puct` 2.5, no rollout, `belief=rules` (phase 3's settings). PUCT-200
   never changed the raw network's move on 40 positions, so its gain is 0 by construction.
3. hk-games has no phone timing yet (`docs/bot-speed.md`: a real phone's table comes with
   Task P2), so phone time = 3 x Node WebAssembly, stated as assumed.
4. `big2._core` with `net_search` is built in a frozen snapshot worktree
   (`.worktrees/big2-p6budget`); the queue jobs' `PYTHONPATH` points there, so this worktree's
   `_core` stays as the Task 36 jobs loaded it.

## Consequences

* Queue jobs 391-397 run the study; the knee question goes to the inbox when they end.
* If Brian wants the phone factor measured, the Node figures in the report convert directly.
