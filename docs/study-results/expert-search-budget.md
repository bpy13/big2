# Expert search budget (Phase 6, Tasks 37-40)

## 0. The self-play network in Rust (Task 37)

Network: `runs/selfplay_luck/hour_24.pt` (Task 36, hour 24; `qnet_v2`, state 512/256, action 128,
head 256), exported with `big2 rl export` to `runs/p6_net/selfplay-h24.pknn` (mixed int8/f16,
1.17 MB, gitignored, not shipped). The self-play trainer's checkpoint keys (`arch`, `model`,
`net_config`, plus `train_config`, `info`) match r3's, so `export.py` needed no mapping.

**Agreement with PyTorch** (`python/tests/test_export.py::test_selfplay_h24_network_in_rust_matches_pytorch`,
1,000 positions from greedy-prior self-play, seed 0, 5.3 legal moves on average):

| measure | result | bar |
|---|---|---|
| Rust prior top move = PyTorch's | 99.8 % | >= 99 % |
| value outputs (each seat's expected cards left), RMS difference | 0.0081 | <= 0.01 |
| per-seat cards-left distribution, L1 mean / p99 / max | 0.0024 / 0.0116 / 0.113 | (info) |

**Speed of one forward pass** (all legal moves of one position, prior and cards-left heads),
1,400 positions (seed 7, after 100 warm-up positions), one thread, this machine, release build:

| where | median ms | 95th percentile ms |
|---|---|---|
| native (`puct_timing` example, step `forward`) | 0.059 | 0.126 |
| Node 22 WebAssembly (`experiments/forward_timing.mjs`) | 0.246 | 0.553 |

WebAssembly is about 4x slower than native. Sources: `runs/p6_net/forward_native.json`, step
`forward 1500 7` of `crates/search-wasm`.

## 1. The depth and budget study: method (Task 39)

Status 2026-11-27 (slot 1): code built and queued; the table, curve and knee arrive when queue
jobs 409-415 end (the first jobs 391-397 and 402-407 were cancelled before any ran, while Brian
picked the expert network; decision 0007 picked hour 24, the network this study already used) (`runs/p6_budget/report.md`, `curve.svg`, `curve.png`). Decision
[0006](../decisions/0006-task-39-budget-study-method.md) records the orchestrator's call to run
the study as planned (about 125 CPU-h) with the assumptions below.

Driver `experiments/expert_search_budget.py` (stages `screen`, `timing`, `finalists`, `confirm`,
`report`), queue script `experiments/expert_search_budget_job.ps1`; timing steps `netsearch` and
`netpuct` in `crates/search-wasm` (the same code natively and in Node WebAssembly, as Task 37's
`forward`).

- **Network**: `runs/p6_net/selfplay-h24.pknn` (§0). **Raw network** = its prior's top move,
  played in Rust from the same file (`netsearch` with `k = 1, n = 1, depth = 0`: one candidate,
  so no play-out can change the move; about 0.15 ms a move natively).
- **Settings (18)**: play-outs (`crates/search/src/netsearch.rs`, rules mask on) from
  `k 3, n 64, depth full`, one knob at a time: `k` in {2, 3, 5, 8}; `n` in {16, 32, 64, 128, 256};
  `depth` in {0, 4, 8, 16, full} plies; `value_mix` 0.5 at depths 8 and 16 (orchestrator,
  2026-11-27; `value_mix` 0 equals `full`). PUCT with the network at {200, 500, 1,000, 2,000}
  simulations, `c_puct` 2.5, no rollout, `belief=rules` (phase 3's settings). A quick check on 40
  positions from the network's own games: PUCT-200 never changed the raw network's move,
  PUCT-2,000 changed 2, `k 3 n 64 full` 13, `k 3 n 64 depth 0` 19; PUCT-200's gain is therefore
  0 by construction.
- **Matches**: every setting against the raw network on the same deals (seed 7, Task 36b's
  deals), lineups `1+3` and `ABAB`, duplicate seats, tiered scoring, last-card duty on, no
  four-card plays. **Gain** = the setting's score minus the raw network's, paired by deal,
  averaged over the two lineups; 95 % bootstrap range over deals. Screen 300 deals per lineup
  (jobs 410, 411; the one-at-a-time grid halves). **Finalists** (job 412): the efficient
  frontier of (native median ms, gain) — no cheaper setting gains at least as much — taking the
  five with the highest gain (filled from the next-best gains if the frontier is short).
  **Confirm** (job 413): finalists at 2,000 deals against the raw network (the screen's 300 are
  reused) and 1,000 deals against the app's `expert` (`ismcts:iterations=10000`, as Task 36b);
  the raw network's own `expert` result is Task 36b's hour-24 row (job 365, same deals, the same
  network in PyTorch) when it exists.
- **Time**: one thread, 300 positions (after 30 warm-up) drawn one per game from games the
  network plays against itself (`network_positions`, seed 7): native (`puct_timing` example,
  job 409, quiet lane) for every setting; Node 20 WebAssembly (`runs/p6_budget/pkg`,
  `experiments/search_budget_timing.mjs`, job 414, quiet lane) for the finalists. **Phone** =
  3 x Node, assumed: hk-games has no phone timing (`docs/bot-speed.md`: a real phone comes with
  Task P2; its browser worker read 123 ms median for the ISMCTS-10,000 `expert` where Node read
  113 ms, so browser and Node are alike). Non-finalists get a phone estimate from native x the
  finalists' median Node/native factor.
- **Knee rule** (orchestrator's default): among settings whose phone 95th percentile is at most
  500 ms for a bot move, the cheapest (native median) whose gain is within 0.2 points a deal of
  the best measured gain or whose 95 % range overlaps the best's. The same rule at 600 ms gives
  the `grade` setting. Gain per doubling of time: the least-squares slope of gain against
  log2(native ms) along the frontier, plus each frontier step's slope.
- **Build note**: `big2._core` with `net_search` was built in the frozen snapshot worktree
  `.worktrees/big2-p6budget` (`maturin build`, the wheel's `_core.pyd` copied in); the jobs'
  `PYTHONPATH` points there, so this worktree's `_core` stays as the Task 36 jobs loaded it.
  First probe on a loaded PC (one GPU and one cpu job running): `k 3 n 64 full` 220 ms median,
  2.4 s p95 a move; `depth 0` 80 / 200 ms, of which the network's 193 forward passes are about
  12 ms — the rest is the rules-mask determinizer (rejection sampling), which sets a floor
  under every shallow setting.

## Exploration weight (Task 52)

Question (Brian, 2026-11-28): on his deal 23 (hk-games `deal_logs` id 23, seat 0's first
decision, copied to `experiments/data/hkgames_deal23.json`), PUCT picks the flush
3s 6s 7s 8s Js although his straight 3h 4c 5c 6s 2d scores higher. Is `c_puct` 2.5 too large
for utilities that are scores divided by the largest possible score (tiered: 52 x 3 = 156)?
Script: `experiments/puct_exploration.py` (`root`, `hand`), hour-24 network, `belief=rules`,
one thread.

**(1) The root at 32,000 simulations (c_puct 2.5, seed 301).** The reading holds. Q is
mean score / 156, the prior term is c·P·√(N+1)/(1+n) at the end of the search:

| Move | Visits | Share | Mean score | Q | Prior P | c·P·√N/(1+n) | Q + term |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3s 6s 7s 8s Js | 18,790 | 58.7% | -0.39 | -0.0025 | 0.633 | 0.0151 | +0.0126 |
| 3h 4c 5c 6s 2d | 5,289 | 16.5% | +0.37 | +0.0024 | 0.121 | 0.0102 | +0.0126 |
| 7s 8s 9h Td Js | 3,733 | 11.7% | +0.34 | +0.0022 | 0.087 | 0.0104 | +0.0126 |
| 7s 8s 9h Tc Js | 1,589 | 5.0% | -0.82 | -0.0053 | 0.063 | 0.0178 | +0.0126 |

A 0.76-point lead is 0.005 in Q, smaller than the prior term's spread (0.008); the four moves
sit level at Q + term = 0.0126, so the visits are where the prior puts them, and "most visited"
returns the network's favourite.

**(2) The hand: c_puct x simulations, 10 seeds each (1000-1009).** Picked, visit share and mean
score (points) of the three most visited moves:

| c_puct | Sims | Picked (x/10) | Visit share | Mean score |
|---:|---:|---|---|---|
| 0.1 | 2,000 | flush 8, straight-2d 1, 3s 4c 5c 6s 2d 1 | 66 % / 14 % / 11 % | +0.33 / -0.37 / -1.44 |
| 0.25 | 2,000 | flush 10 | 65 % / 15 % / 5 % | +0.19 / +0.20 / -0.64 |
| 0.5 | 2,000 | flush 10 | 64 % / 16 % / 5 % | +0.18 / +0.48 / +0.48 |
| 1 | 2,000 | flush 10 | 68 % / 13 % / 6 % | +0.36 / +0.13 / -1.18 |
| 2.5 | 2,000 | flush 10 | 65 % / 13 % / 8 % | +0.27 / +0.57 / -1.26 |
| 0.1 | 8,000 | flush 7, straight-2d 2, 3s 4c 5c 6s 2d 1 | 60 % / 25 % / 9 % | +0.07 / -0.00 / -1.05 |
| 0.25 | 8,000 | flush 8, 7s-Js(Td) 1, 3s 4c 5c 6s 2d 1 | 61 % / 18 % / 7 % | -0.03 / +0.03 / -1.16 |
| 0.5 | 8,000 | flush 10 | 67 % / 17 % / 4 % | +0.03 / +0.16 / -0.04 |
| 1 | 8,000 | flush 10 | 66 % / 15 % / 5 % | +0.04 / +0.23 / -1.26 |
| 2.5 | 8,000 | flush 10 | 65 % / 14 % / 7 % | -0.06 / +0.28 / -1.30 |

(flush = 3s 6s 7s 8s Js; straight-2d = Brian's 3h 4c 5c 6s 2d; the first two columns of share
and score are those two moves, the third is the next most visited.) No weight makes the search
pick the straight. At 2,000-8,000 simulations the gap between the two (0-0.4 points here) is
inside the noise: one simulation's score has an SD of about 9 points, so 5,000 visits give a
95 % range of about ±0.25 points per move. Lowering c_puct moves no visits to the straight;
at 0.1 the pick starts to wander (2-3 of 10 runs pick other moves, one a -1.4 move). So on this
hand the prior term does dominate at 2.5, but a smaller weight would not change the answer at
these budgets either: the flush and the straight are close to level under the search's own
estimate.

**(3) The match: c_puct 0.25 and 0.5 at 2,000 simulations against the raw network** (the
lowest weights that stay stable on the hand; same measure as Task 39's finalists: 2,000 deals,
1+3 and ABAB, paired, seed 7). Queue jobs 429 (c 0.25) and 430 (c 0.5), cpu lane, 8 threads;
results in `runs/p6_budget/cpuct/puct2000-c0.25/results_2000.json` and `-c0.5`.

| Setting | Gain over raw network [95 %] | 1+3 | ABAB |
|---|---|---|---|
| puct2000, c 2.5 (Task 39) | +0.34 [+0.10, +0.57] | +0.26 [+0.05, +0.47] | +0.41 [+0.07, +0.75] |
| puct2000, c 0.5 | -0.84 [-1.19, -0.49] | -0.65 [-1.00, -0.29] | -1.04 [-1.52, -0.53] |
| puct2000, c 0.25 | -1.35 [-1.72, -0.98] | -1.22 [-1.63, -0.81] | -1.48 [-1.99, -1.01] |

Both smaller weights lose to the raw network, and the loss grows as the weight drops: c 0.5 is
1.2 points a deal below c 2.5, c 0.25 is 1.7 below, both well outside the 95 % ranges, in both
lineups. With little exploration the search locks onto noisy early scores (one simulation's SD
is about 9 points) and drops the network's prior, which is the better guide at this budget.
So the reading in (1) is right about the mechanism, but leaning on the prior is what makes
c 2.5 the best of the weights tried; the weight stays at 2.5. No weight won, so Task 51's 4,000-16,000 runs need not be
redone. (Weights above 2.5 were not tried.)
