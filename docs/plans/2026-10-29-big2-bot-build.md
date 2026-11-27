# Plan: big2 bot build run (B25)

**Date:** 2026-10-29. **Repo:** `C:\Users\brian\OneDrive\Documents\workspace\repo\big2` (standalone
checkout, branch `main`). **Scope:** item 2 of `bot-stream\docs\plans\HANDOFF.md`: move the staged
reports, then phases 1–4 of `docs/research/strategy-proposal.md` (after Task 1 moves it there).
Phase 5 (the B4 research bar) is not in this plan.

Written for someone who has not seen the chat. Read first: `CLAUDE.md`, `docs/solver.md`,
`docs/rl.md`, `docs/research/strategy-proposal.md` (§2, §4–§7), `crates/app-api/README.md`,
`bot-stream\docs\decisions.md`, and `hk-games\docs\streams\bot-notes.md`.

## Goal

One Big Two brain in four layers, all Rust, all on the phone: a search (ISMCTS now, PUCT with a
network inside later), a network distilled from that search, a belief layer (rules first), and a
teaching layer (reason keys, grades, drills, levels). Measured against today's ISMCTS expert and
Charlesworth's network, with duplicate deals and 95 % intervals.

## Approach

Expert iteration. Round 0 is today's ISMCTS with a structured rollout. Each round labels
self-play decisions with the current search, trains the network on them, and puts the network
back inside the search as prior and leaf value (PUCT). A round ships only if it beats the previous
round on 1,000 duplicate deals. Phase 4 builds the teaching layer on the search's own move table.

## Constraints (copied from `bot-stream\docs\decisions.md`)

- **B5 Compute and outside code as in holdem:** local PC only; no commercial solvers; open
  source agents (e.g. DouZero, Mortal, RLCard) may be read and run as benchmarks, never
  linked or shipped.
- **B7 App first.** Mahjong and big2 ship `bot_action`, `hint`, `review`, `grade` with reason
  keys from today's bots before training; 13 cards and Dou Dizhu do the same once their rules
  engines exist.
- **B14 Outside code as local test opponents only** (Charlesworth's big2 network, the 13
  cards repos without licence, PerfectDou and AlphaDou through WSL2); never copied or
  shipped.
- **B18 Scope.** Dou Dizhu up to release 2a (beat DouZero, approach PerfectDou; no 2b, no
  full belief net). Mahjong, big2 and 13 cards follow their proposals (mahjong plus B15).
  Holdem follows the fix-and-retrain plan once Brian restarts it.
- **B19 Compute on this PC only.** No cloud, no new hardware for now.
- **B20 `quality` and `interval` fields** are requested from the app stream.
- **B22 Big2:** no four-card plays. Bots play for their own score; the only table obligation
  is the last-card duty (block when the next player holds one card), already the rules'
  default `last_card_duty = on`. Scoring for Elo, drills and training: `tiered`, as
  originally planned.

Rules of this run (from HANDOFF.md and the hk-games CLAUDE.md):
- **Never edit `crates/app-api/`.** The app stream owns it. Where a new bot or key must reach the
  app, the task says what to write in `hk-games\docs\streams\bot-notes.md`; commit that file alone
  in the hk-games checkout. If a public engine item used by `crates/app-api/src/` changes shape,
  do not change it: add a new item beside it and ask the app stream in bot-notes to switch.
- **Never stage `web/package-lock.json` or `models/big2-distilled-duty.pt`** (Brian's). Commit per
  task with `git add <named files>`; never `git add -A`.
- **Keep a path that never needs a rayon thread pool.** Every search entry point runs on the
  calling thread when `threads == 1` (Task 2 makes this explicit). The app calls with
  `threads = 1` in WebAssembly.
- Pull first: `git -C <repo> pull --rebase --autostash` (Brian's two files stay unstaged).
- Fable helpers only on tasks marked `(fable)`; opus otherwise (B1 amended). At most 4 helpers.

## Test and lint commands (run 2026-10-29, all from `C:\Users\brian\OneDrive\Documents\workspace\repo\big2`)

| Command | Time | State today |
|---|---|---|
| `cargo test --workspace` | 12 s | pass (89 tests, 1 ignored) |
| `C:\Users\brian\miniconda3\envs\big2\python.exe -m pytest -q` | 47 s | 196 passed, 1 skipped |
| `...\python.exe -m ruff check python experiments` + `ruff format --check` + `mypy python/big2` | 2 s | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | 4 s (cached; ~2 min cold) | pass |
| `cargo fmt --all --check` | 1 s | **fails** in `crates/app-api/src/lib.rs` (pyo3 block, app stream's file). Not ours to fix: Task 10 reports it in bot-notes. Until then check our crates only: `cargo fmt -p big2-cards -p big2-games -p big2-nn -p big2-big2 -p big2-search -p big2-pybind -p big2-teach --check` |
| `make test-web`, `make test-e2e` | not run | `web/` is outside this plan |

Python runs from the `big2` conda env: `C:\Users\brian\miniconda3\envs\big2\python.exe`. The `big2`
CLI is `...\envs\big2\Scripts\big2.exe`. After any Rust change under `crates/pybind`, `crates/big2`
or `crates/search`, rebuild the extension: `...\python.exe -m pip install -e ".[dev,rl]"` from the
repo root (not while a training job runs: CLAUDE.md).

## Environment facts found on 2026-10-29

- PyTorch in the `big2` env is `2.14.1+cpu`; `torch.cuda.is_available()` is **False**. Task 3
  installs the CUDA build. GPU driver 610.88.
- Two `stock_advisor` Python jobs have been running since 2026-10-28 (`backfill_news.py`,
  `news_backtest_cli`) and use CPU heavily. Before any CPU job here, run
  `Get-Process python | Select-Object Id,CPU,StartTime` and use `--workers 14` (not 20) while
  they run. The GPU showed 1.6 GB used and 64 % load from another process at planning time;
  check `nvidia-smi` before every GPU job and wait if load is over 20 %.
- `models/big2-dmc-selfplay-10h.pt` exists on disk (gitignored). It is the "DMC-10h" opponent.
  `models/big2-distilled-duty.pt` is the duty-on distilled network (Brian's file, read-only here).
- `wasm-pack` is at `C:\Users\brian\.cargo\bin\wasm-pack.exe`; target `wasm32-unknown-unknown` is
  installed; Node is v22. WebAssembly timings here are measured in Node (the app stream's method);
  phone timings (bar B6) come from the app stream on the benchmark phones (B12).

## Long-job conventions (every task marked **[long]** or **[GPU]**)

- Launch headless and detached so it outlives this session:
  ```powershell
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = 'conhost.exe --headless powershell.exe -NoProfile -File "C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\<job>\run.ps1"';
    CurrentDirectory = 'C:\Users\brian\OneDrive\Documents\workspace\repo\big2' }
  ```
  `run.ps1` activates nothing: it calls `C:\Users\brian\miniconda3\envs\big2\python.exe` by absolute
  path and writes stdout to `runs\<job>\console.log`.
- Every job writes `runs\<job>\progress.json` at least every 30 minutes (what is done, what is
  next) and `runs\<job>\results.json` at the end. **Check command:**
  `Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\<job>\progress.json`.
- Every job resumes from its own checkpoint or shard list; the task names its resume command.
  Trainers already take `--resume`; `distill-gen` appends shards, so rerunning it resumes.
- `runs/` is gitignored. Results that matter go into `docs/study-results/build-*.md` as tables.
- Announce in the hand-back any job over 12 hours before starting it (none planned; the probes are
  10 h each).
- Do not run `pip install -e .` while a job runs (the job holds the old `_core`).

## Acceptance criteria (measurable on this PC)

Match format everywhere: duplicate deals (each deal 4×, seats rotated), **last-card duty on,
tiered scoring, no four-card plays** (B22), two lineups ("1 + 3" and "A B A B"), mean points per
deal with a 95 % interval over deals, win rate beside it. Sizes: 300 deals for go/no-go on a cheap
idea (±0.9), 1,000 deals for every "beats" claim (±0.5), 10,000 deals for final numbers (±0.15).

| # | Criterion | Task |
|---|---|---|
| A1 | The three reports live in `docs/research/` with working links | 1 |
| A2 | `PimcConfig` has `time_limit`; `ismcts`/`pimc`/PUCT with `threads = 1` never touch rayon; a test proves the single-thread path gives the same result as before | 2 |
| A3 | The min-plays planner agrees with brute force on 10,000 random hands; the decomposition list has ≤ 20 candidates on 95 % of 13-card hands | 4 |
| A4 | Structured rollout inside ISMCTS-1000 vs today's rollout: ≥ +1.0 at 300 deals, or the shortfall is recorded and the planner stays for keys only | 5 |
| A5 | Re-baseline under duty-on: ISMCTS-1000 vs 3 greedy and vs DMC-10h, 1,000 deals each, intervals recorded; σ of one deal's score in ISMCTS self-play measured | 6 |
| A6 | Distilled Q-network runs in Rust (`.pknn` v2): ≥ 99 % move agreement with PyTorch on 1,000 positions; < 1 ms per decision native | 7 |
| A7 | B1: win rate vs 3 Patwa-greedy and vs 3 Patwa-smart under `sum_of_others` scoring, 10,000 games × 3 seeds, interval recorded against 64.8 % and 43.5 % | 8 |
| A8 | Ladder v0 orders strictly, gaps ≥ 1.5 points over 500 deals, with intervals excluding each other | 9 |
| A9 | PUCT-2000 beats ISMCTS-1000 by ≥ +1.0 (interval excluding 0) at 1,000 deals, and beats ISMCTS at equal wall time on one thread | 15 |
| A10 | Distribution head loses ≤ 0.5 to the mean-score head under tiered and wins under `win_loss` (500 deals each) | 15 |
| A11 | PUCT-2000 in Node WebAssembly: ≤ 0.6 s per `bot_action`-sized call at the opening position; move changes after 100 ms in < 15 % of 300 positions | 16 |
| A12 | Each expert-iteration round beats the previous at 1,000 deals, or the loop stops and the last winner ships; result table in docs | 17 |
| A13 | Hard-constraint belief rejects a reported share of deals and loses nothing at 300 deals | 18 |
| A14 | Learned belief: log-likelihood per round reported; kept only if ≥ +0.5 at 300 deals | 19 |
| A15 | Information-set exact endgame agrees with a 100,000-simulation ISMCTS on ≥ 95 % of 300 late positions; ≤ 1 s in Node WebAssembly at its threshold | 20 |
| A16 | B2: mean score > 0 with interval excluding 0 vs Charlesworth's network in both lineups, 1,000 deals, under his rules | 21 |
| A17 | B3b: the 10-GPU-hour probe gains less against the final network than against ISMCTS-1000 and DMC-10h; validated by gaining more against greedy than against ISMCTS | 22 |
| A18 | Two training seeds do not lose to each other (interval covers 0 at 1,000 deals) | 23 |
| A19 | 10,000-deal final matches vs ISMCTS-1000 and vs Charlesworth recorded | 24 |
| A20 | Every reason key is a unit-tested fact; 50-position audit file produced for Brian | 26 |
| A21 | Re-grading the same deal with new seeds agrees on ≥ 95 % of decisions; luck-ledger terms sum to the outcome on every reviewed deal (exact to 1e-9) | 27 |
| A22 | Drill file in the engine-interface `Drill` shape, ≥ 300 drills, tag audit file of 50 for Brian | 28 |
| A23 | Error model and ladder v2: five levels order strictly with gaps ≥ 1.5 at 500 deals | 29 |
| A24 | bot-notes carries every new public item, key and request; docs synced | 10, 16, 25, 31 |

Totals **(estimate)**: GPU ≈ 51 GPU-hours ≈ **2.1 GPU-days** (budget 3). CPU long jobs ≈ 55
CPU-hours ≈ 2.3 days on 14 workers; they run beside the GPU jobs. Per-task times are in the task
headers.

---

## Phase 0 – Reports (opus)

- [x] **Task 1: Move the three staged reports into `docs/research/`** `[parallel-ok]`
  - Files: create `docs/research/strategy-proposal.md`, `docs/research/survey.md`,
    `docs/research/hard-problems.md` (copies of
    `C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\staging\big2\*.md`); modify
    `README.md` (documentation table: three rows); modify `docs/research/prior-work.md` (one
    "see also" line at the top).
  - Done when: `rg -n '\]\((\.\./)+' docs/research/*.md` lists only links that resolve. Check with a
    script in the scratchpad that opens every relative link target from `docs/research/` and prints
    `MISSING <path>` for each miss; expected output: no `MISSING` lines. Links to
    `big2\docs\...` become `../...`; links to `bot-stream\docs\research\teaching-and-deployment.md`
    and to holdem/doudizhu proposals become absolute `file:///C:/Users/brian/OneDrive/Documents/workspace/repo/...`
    links (those repos are siblings, not inside this repo). `git status` shows only the four
    changed/added paths staged.
  - Notes: do not delete the staging copies; the coordinator removes them in bot-stream. Keep the
    files' text unchanged apart from links. Commit: "docs: research reports from the bot stream
    (survey, hard problems, strategy proposal)".

## Phase 1 – The four calls from the existing solver: what is still ours (opus; Task 4 fable)

The app stream has built `crates/app-api` (bot_action, hint, review, grade, WebAssembly, pyo3).
What remains ours: the engine-side upgrades the proposal's phase 1 lists, the re-baseline, B1,
the Rust runtime for the distilled network, and the eval harness. Everything the app must switch
to goes through bot-notes (Task 10).

- [x] **Task 2: `PimcConfig.time_limit` and an explicit single-thread path** `[parallel-ok]`
  - Files: modify `crates/search/src/pimc.rs` (add `pub time_limit: Option<Duration>`, default
    `None`; stop between determinizations when past the deadline; count what ran), modify
    `crates/search/src/ismcts.rs` and `pimc.rs` (when `cfg.threads <= 1`, call `search_tree` /
    the per-thread body directly on the calling thread; `into_par_iter` only for `threads >= 2`),
    test in `crates/search/src/pimc.rs` and `ismcts.rs`.
  - Done when: `cargo test -p big2-search` passes with two new tests: (a) `pimc` with
    `time_limit = Some(50 ms)` and `determinizations = 1_000_000` returns in under 200 ms with
    `simulations > 0`; (b) `ismcts` and `pimc` with `threads = 1` give byte-identical `Analysis`
    (same visits and sums) before and after the change on seed 7 (record the expected numbers in
    the test from a run on the parent commit). `cargo test -p big2-app-api` still passes (the app
    crate constructs `PimcConfig` with `..Default::default()`, so the new field is additive).
  - Notes: this answers the app stream's request of 2026-10-28. The single-thread path matters
    because rayon's global pool must never be built in WebAssembly. Do not add a `rayon` feature
    flag in this task; a feature flag is an open item if the app stream asks.

- [x] **Task 3: PyTorch with CUDA in the `big2` env** `[parallel-ok]` **[~10 min download]**
  (done 2026-10-29 by the coordinator: torch 2.14.1+cu130 from the `cu130` index; CUDA True on
  the RTX 4060 Ti; `test_rl.py` 18 passed in 105 s)
  - Files: none in the repo (environment only); `docs/rl.md` §6 unchanged unless the CUDA tag
    differs from `cu128`.
  - Done when: from the repo root, `C:\Users\brian\miniconda3\envs\big2\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/<tag>`
    with the tag pytorch.org lists for driver 610.88, then
    `...\python.exe -c "import torch;print(torch.cuda.is_available())"` prints `True`, and
    `...\python.exe -m pytest python/tests/test_rl.py -q` passes (it runs the trainers on every
    device present).
  - Notes: `make` is not on the PATH outside the env, so run the pip command directly. Do not
    rebuild the extension while any `big2` job runs (none should at this point).

- [x] **Task 4: Min-plays planner and decomposition enumerator** `(fable)` `[parallel-ok]`
  - Files: create `crates/big2/src/plan.rs` (`min_plays(hand, rules) -> u32`;
    `decompositions(hand, rules, max: usize) -> Vec<Decomposition>` where a `Decomposition` is a
    list of `Combo`s covering the hand with the fewest plays, ordered by a fixed tiebreak (fewest
    singles, then highest kept five-card hand first); `candidates(state, decomps) -> Vec<Candidate>`
    where a `Candidate` is a legal action plus the index of the decomposition it keeps and the
    combo it breaks, if any); modify `crates/big2/src/lib.rs` (`pub mod plan`); test module in
    `plan.rs`; modify `crates/pybind/src/lib.rs` (`min_plays(hand)` and `decompositions(hand)` for
    Python) and `python/big2/_core.pyi`.
  - Done when: `cargo test -p big2-big2 plan` passes: (a) `min_plays` equals a brute-force
    minimum-set-cover over all legal combos on 10,000 random hands of 1–13 cards under default
    rules (brute force may use a bitmask DP over subsets; cap at 13 cards); (b) rules §3 examples
    (a full house, a straight, a flush) are decomposed as expected; (c) on 1,000 random 13-card
    hands, `decompositions(hand, 8)` has ≤ 20 entries for ≥ 95 % of hands and `candidates` has
    ≤ 24 entries at the opening position. `cargo bench -p big2-big2 --bench engine` gains a
    `min_plays` row; record its time in `docs/study-results/build-phase1.md` (target < 50 µs
    per 13-card hand).
  - Notes: the DP is over per-rank suit sets, as `movegen.rs` builds combos. Both `tiered` and
    `original` scoring count cards, so "fewest plays" is the right first objective; break ties as
    above and say so in rustdoc. This module becomes the `shedding-core` seed (proposal §8); keep
    it free of `State` apart from `candidates`.

- [x] **Task 5: Structured rollout policy, and its test inside ISMCTS** **[long: ~20 min CPU]**
  (done 2026-10-29: A4 fails, +0.07 ± 0.58 at 300 deals; planner kept for keys and candidates;
  Task 11 uses the greedy rollout)
  - Files: create `crates/big2/src/bots/structured.rs` (`StructuredGreedy { epsilon, plan_depth }`
    implementing `Policy<State>`: never break a pair, triple or five-card group of the best
    decomposition to beat a single unless the planner says the hand's min-plays does not rise;
    otherwise as `Greedy`); move `bots.rs` to `bots/mod.rs` keeping every public path
    (`big2_big2::bots::Greedy` must still resolve: the app crate imports it); modify
    `python/big2/agents/__init__.py` (`greedy:structured=1` and `ismcts:rollout=structured`);
    modify `crates/pybind/src/lib.rs` for the `rollout` option; create
    `experiments/build_phase1.py` subcommand `structured-rollout`; create
    `docs/study-results/build-phase1.md` §1.
  - Done when: `cargo test -p big2-big2 structured` passes (a hand with a pair and a lone higher
    single beats a low single with the lone single, not the pair's card); then
    `big2 tournament "ismcts:iterations=1000,rollout=structured" ismcts:iterations=1000 "ismcts:iterations=1000,rollout=structured" ismcts:iterations=1000 --deals 300 --workers 14 --out runs/p1_structured/results.json`
    runs in ~15–20 min and the structured side's mean score and 95 % interval are in
    `build-phase1.md` §1 with the verdict against the +1.0 bar (A4). If it fails the bar, say so
    there and keep the planner for keys and candidates only (proposal §10).
  - Notes: duty on and tiered are the CLI defaults; keep them. `epsilon = 0.1` as today. Run
    `Get-Process python` first (environment facts).

- [x] **Task 6: Re-baseline under duty-on, and σ** **[long: ~45 min CPU]** `[parallel-ok after 5]`
  (done: vs 3 greedy +7.40 ± 0.45; vs DMC-10h +0.17 ± 0.37; σ = 27.45 points per seat-game)
  - Files: create `experiments/build_phase1.py` subcommand `rebaseline`; modify
    `docs/study-results/build-phase1.md` §2.
  - Done when: three tournaments finish with results in `runs/p1_rebaseline/*.json` and a table in
    `build-phase1.md` §2: (a) ISMCTS-1000 (best rollout from Task 5) vs 3 greedy, 1,000 deals;
    (b) ISMCTS-1000 vs `dmc:checkpoint=models/big2-dmc-selfplay-10h.pt` in the A B A B lineup,
    1,000 deals; (c) ISMCTS-1000 in all four seats, 1,000 deals, reporting the standard deviation
    σ of one seat's score per deal (this σ feeds the grade thresholds, proposal §9). Each table row
    has mean ± 95 % and win rate.
  - Notes: workers 14. The DMC-10h network was trained with duty **off**; note that beside its row.
    Note: one tournament of 1,000 deals with four searching seats is 4,000 games × ~45 decisions ×
    15 ms ≈ 45 min on 14 workers; (a) is faster.

- [x] **Task 7: The distilled Q-network in Rust (`.pknn` v2)** `[parallel-ok]`
  (done af571df: 99.9 % same move, 0.030 points RMS, 1.16 MB, 33 µs per decision; int8 only on
  the first state layer (grouped scales + bias correction), f16 elsewhere: per-row int8 gave 93 % / 0.75)
  - Files: modify `crates/nn/src/lib.rs` (format v2: a `graph` of named towers so one file holds
    the state tower, the move tower and the head; int8 weights with per-row scales, f32 bias;
    `from_bytes`; v1 still loads); create `crates/big2/src/qnet.rs` (`QNet::from_bytes`,
    `score_moves(state, legal) -> Vec<(score, win_logit)>`, and `impl Policy<State>` with a
    temperature, `QNetPolicy { temperature }`); create `python/big2/rl/export.py`
    (`export_qnet_pknn(checkpoint, out)`), CLI `big2 rl export --checkpoint ... --out ...`;
    tests in `qnet.rs` and `python/tests/test_export.py`; modify `docs/rl.md` §4 (the export).
  - Done when: `big2 rl export --checkpoint models/big2-distilled-duty.pt --out runs/p1_qnet/distilled-duty.pknn`
    writes a file under 1.2 MB; `pytest python/tests/test_export.py` passes: on 1,000 positions
    from seeded games, the Rust `score_moves` picks the same move as PyTorch on ≥ 99 % and the
    scores differ by < 0.05 points RMS; `cargo test -p big2-big2 qnet` passes (loads a tiny
    synthetic v2 file, checks the hidden-card invariance test from `features.rs` still holds);
    `cargo bench` gains a `qnet_decision` row < 1 ms (record it).
  - Notes: `models/big2-distilled-duty.pt` is Brian's: read it, never write or stage it. The
    exported `.pknn` for shipping lives at `models/big2-qnet-duty.pknn` later (Task 9 decides and
    adds the gitignore exception). Keep `crates/nn` free of big2 types.

- [x] **Task 8: B1 — Patwa's bots and scoring** `[parallel-ok]` **[~15 min CPU]**
  (done: 69.0 % ± 0.5 % vs 3 patwa-greedy (his 64.8 %); 31.2 % ± 0.5 % vs 3 patwa-smart (his
  43.5 %; our smart bot is a tuned approximation))
  - Files: create `python/big2/agents/patwa.py` (`PatwaGreedy`, `PatwaSmart` as described in
    `docs/research/survey.md` §2.4: keep 2s, avoid orphan singles by hand-written penalties);
    modify `python/big2/agents/__init__.py` (specs `patwa-greedy`, `patwa-smart`); modify
    `python/big2/eval/tournament.py` (option `score_as="sum_of_others"`: winner takes the sum of
    the others' cards left, computed from penalties after the deal, no engine change; and a
    `seat_random=True` mode for Patwa's protocol); modify `python/big2/cli/main.py` (`--score-as`,
    `--seat-random`); tests in `python/tests/test_agents.py` and `test_tournament.py`; create
    `experiments/build_phase1.py` subcommand `b1`; `build-phase1.md` §3.
  - Done when: `pytest python/tests/test_agents.py python/tests/test_tournament.py` passes (a
    hand-made deal where `sum_of_others` and tiered disagree is asserted); `b1` runs ISMCTS-1000
    vs 3 patwa-greedy and vs 3 patwa-smart, 10,000 games per cell, 3 seeds, and writes the win rate
    with a 95 % interval per cell to `build-phase1.md` §3 against 64.8 % and 43.5 %.
  - Notes: Patwa's bots are re-implemented from his text (B14 allows reading; nothing is copied).
    Say in the doc that his agents are approximations.

- [x] **Task 9: Eval harness and ladder v0** **[long: ~30 min CPU]** (after 5, 7)
  (done: order strict; A8 fails only at ISMCTS-1000 → 5000, gap +0.55 [+0.16, +0.94] < 1.5;
  Task 29 retunes)
  - Files: create `python/big2/eval/harness.py` (both lineups, bootstrap 95 % intervals over
    deals, paired differences, one JSON report shape `{"candidate", "opponent", "lineup", "deals",
    "mean", "ci95", "win_rate", "win_ci95", "seconds_per_move"}`), CLI `big2 match <cand> <opp>
    --deals N --lineups both`; create `experiments/build_phase1.py` subcommand `ladder`; modify
    `.gitignore` (`!/models/big2-qnet-duty.pknn`) and add that file from Task 7's export;
    `build-phase1.md` §4.
  - Done when: `pytest python/tests/test_harness.py` passes (a two-seat synthetic result gives the
    expected paired mean and an interval that shrinks with √N); `ladder` runs the five levels
    random / `greedy:structured=1,epsilon=0.1` / `qnet:temperature=1.0` / ISMCTS-1000 /
    ISMCTS-5000 in a round-robin of 500 deals and writes the ordering, the gaps and their intervals
    to `build-phase1.md` §4 against the 1.5-point bar (A8).
  - Notes: this is the `eval-harness` seed (proposal §8). If a gap is under 1.5, report it; Task 29
    retunes the ladder. Ladder v0 does not change the app's levels: Task 10 proposes the switch.

- [x] **Task 10: bot-notes entry for phase 1** (after 2–9)
  (done: hk-games 23c381e (items, level switch, numbers, fmt) and 5342a50 (time_limit). The B20
  reminder was dropped: the app stream accepted `quality`/`interval` on 2026-10-29.)
  - Files: modify `C:\Users\brian\OneDrive\Documents\workspace\repo\hk-games\docs\streams\bot-notes.md`
    only (newest entry first; commit it alone in that checkout with `git add docs/streams/bot-notes.md`);
    modify `docs/study-results/build-phase1.md` (summary table at the top).
  - Done when: the entry lists: (1) `PimcConfig.time_limit` done (their request), with the commit;
    (2) new public items and that no existing item changed shape: `big2_big2::plan`,
    `bots::StructuredGreedy`, `qnet::{QNet, QNetPolicy}`, `big2_nn` v2; (3) the proposal to
    switch levels: `easy` → `StructuredGreedy { epsilon: 0.1 }`, `medium` → `QNetPolicy` from
    `models/big2-qnet-duty.pknn` (bytes handed to the engine), `hard`/`expert` unchanged until
    phase 2; (4) the measured numbers (A4, A5, A8) in three lines; (5) `cargo fmt --all --check`
    fails in `crates/app-api/src/lib.rs` (theirs); (6) B20 reminder: `quality` and `interval`
    fields still await their answer. `git -C hk-games log -1 --stat` shows only bot-notes.md.
  - Notes: the app stream edits the hk-games checkout; pull with `--rebase` before editing.

## Phase 2 – The network inside the search (opus; Task 14 fable)

- [x] **Task 11: Relabel round 0 with the structured ISMCTS** **[long: ~3 h CPU, headless]**
  (done bd179ec; job finished 2026-10-29 11:35: greedy rollout (Task 5), 2,740,296 searched
  decisions, 13,800,253 candidate rows, 5,030 shards in `runs/round0/data`. "≥ 5 M" read as
  candidate rows; rerun `runs\round0\run.ps1` to add decisions if Task 12 needs more.)
  - Files: modify `python/big2/rl/distill.py` (`distill-gen` stores per decision: the candidate
    list from `plan::candidates`, each candidate's visits, mean score, win rate; the final cards
    left of all four seats relative to the actor; who leads the next trick); modify
    `crates/pybind/src/lib.rs` if the generator needs the candidate list from Rust; test
    `python/tests/test_rl.py::test_distill_gen_round0_fields`; create `runs/round0/run.ps1`.
  - Done when: the test passes on a 2-game run (shards hold the new arrays with the right shapes);
    then the job runs headless: `big2 rl distill-gen --out-dir runs/round0/data --workers 14 --iterations 1000 --rollout structured --minutes 200`
    and `progress.json` reports ≥ 5 M decisions (`sum of shard rows`). Check:
    `(Get-ChildItem runs\round0\data\*.npz | Measure-Object).Count` and `progress.json`. Resume:
    rerun the same command (shards append).
  - Notes: duty on, tiered, default rules. Use the rollout that won Task 5. 25 % of the time the
    generator plays a random top-3 move (today's behaviour), keep that.

- [x] **Task 12: Network v2: cards-left distribution, all-seat and prior heads** **[GPU: 2–4 h]**
  (code dfef495, `pytest -k v2` passes. Adds a 0.5-weight auxiliary loss: tiered score from the
  distributions vs the search's mean score per candidate, since cards-left targets exist only for
  the played move. Done 2026-10-29 14:10: patience stop after 50 min (62,398 steps; no better
  validation loss in 6 evaluations); `best.pt` is step 21,049 at 20 min. Validation loss per head
  on 55,046 held-out decisions: actor 1.576, seats 1.818, prior 1.259, score 0.0083 (total 4.657);
  prior top-1 68.5 %, score RMS 4.56 points. `test_export.py -k v2` on the final `best.pt`: prior
  argmax 99.9 %, distributions L1 mean 0.0013 (99th percentile 0.0068) on 1,000 positions. Export:
  `runs/net_r0/pvnet-r0.pknn` (1.17 MB). Extension rebuilt (`pip install -e ".[dev,rl]"`).)
  - Files: modify `python/big2/rl/model.py` (`QNetV2`: shared state tower; per candidate a
    14-way cards-left distribution for the actor and for each of the three opponents (4 × 14),
    plus a policy logit; the mean-score output is `distribution · payoff(scoring)` computed in the
    model for any `scoring`); modify `python/big2/rl/distill.py` (`distill-train --model v2`:
    cross-entropy for the distributions and the prior from visit shares, weighted by visits);
    tests in `python/tests/test_rl.py`; `docs/rl.md` §2 and §3.3; create `runs/net_r0/run.ps1`.
  - Done when: `pytest python/tests/test_rl.py -q -k v2` passes (shapes; the tiered payoff vector
    reproduces rules §7.1 exactly for 0–13 cards: `[0,1,…,7,16,18,30,33,36,52]` as penalties, then
    the pairwise settlement); the headless job
    `big2 rl distill-train --model v2 --data-dir runs/round0/data --run-dir runs/net_r0 --device cuda --minutes 180`
    ends with `results.json` holding validation loss per head and `best.pt`. Check:
    `progress.json`. Resume: `--resume runs/net_r0/latest.pt`.
  - Notes: ≈ 0.8–1.0 M parameters; keep the state tower 512–256 as today so Task 7's export path
    stays. Three heads' loss weights 1 : 1 : 1 (open question 3 for Brian lists the alternative).

- [x] **Task 13: Policy-value network in Rust** `[parallel-ok with 12 once the v2 shapes are fixed]`
  (done: on `runs/net_r0/best.pt` at 5 min of training, prior argmax 99.9 %, distributions L1 mean
  0.0016 (99th percentile 0.007) on 1,000 positions; `pvnet_evaluate_opening` 39 µs native, recorded in
  `docs/rl.md` §4.1. Rerun `pytest python/tests/test_export.py -k v2` on the final `best.pt`.
  `lead_value` rebuilds positions with `State::from_position` (no history); an additive `State`
  method would keep the history.)
  - Files: create `crates/big2/src/pvnet.rs` (`PvNet::from_bytes`, `evaluate(state, candidates)
    -> PvOutput { prior: Vec<f32>, dist: [[f32; 14]; 4], score: f32 }`, `score_for(scoring)`,
    `plays_to_out` passthrough from the planner, `lead_value` = score if actor leads next minus
    score if not, from two evaluations); modify `python/big2/rl/export.py` (v2 export);
    tests; `cargo bench` row `pvnet_evaluate`.
  - Done when: `pytest python/tests/test_export.py -k v2` passes (Rust vs PyTorch: prior argmax
    agrees ≥ 99 %, distributions within 0.01 L1 on 1,000 positions); `cargo bench` row shows
    < 0.3 ms per evaluation at the opening (record it: the PUCT budget rests on this number).
  - Notes: the file goes to `models/big2-pvnet-r0.pknn` with a gitignore exception once Task 15
    picks a winner; until then under `runs/`.

- [x] **Task 14: PUCT with the network inside** `(fable)` (after 13)
  (done 2026-10-29: `big2_search::puct` (`puct`, `puct_with`, `PuctConfig`, `PuctAnalysis { analysis,
  first_answer }`, generic `PuctPolicy`), trait `big2_games::PriorValue` implemented by `PvNet`;
  `analyze(algorithm="puct", net=…, c_puct=…, anytime_first_ms=…)` and spec `puct:net=…,simulations=…`
  (pybind not rebuilt: the trainer runs). Scoring comes from the state's rules, not a `scoring_payoff`
  field (the search crate knows no game). Quick check with the 15-minute `best.pt`
  (`runs/p2_puct/pvnet-r0-early.pknn`), 60 deals vs 3 greedy, one thread under a CPU job: PUCT-300
  +0.6 ± 5.0 at 15 ms per move; PUCT-2000 −1.0 ± 5.8 at 100 ms per move. Task 15 measures properly.)
  - Files: create `crates/search/src/puct.rs` (`PuctConfig { simulations, time_limit, c_puct,
    threads, seed, objective, scoring_payoff, endgame_threshold, endgame_max_nodes,
    anytime_first_ms }`; information-set tree over public actions as `ismcts.rs`, but the
    expansion uses the network's prior over `candidates`, no rollouts: the leaf value is the
    network's per-seat score; multi-player: each node's value is for the player who chose it;
    returns `Analysis` plus `first_answer: Option<Action>` taken at `anytime_first_ms`); modify
    `crates/search/src/lib.rs` (export); `crates/big2/src/bots/puct.rs` or a `PuctPolicy` in
    `pvnet.rs` implementing `Policy<State>`; pybind `analyze(..., method="puct", net=...)`;
    Python agent spec `puct:simulations=2000,net=PATH`; tests.
  - Done when: `cargo test -p big2-search puct` passes: (a) with `threads = 1` and a seed, two runs
    are identical; (b) the hidden-card invariance test from `ismcts.rs` holds for PUCT; (c) on a
    position with one immediately winning move, PUCT-100 picks it; (d) `time_limit = 50 ms` returns
    in < 120 ms. `cargo test -p big2-app-api` still passes (nothing of theirs changed).
  - Notes: the leaf value comes from the network's distribution dotted with the payoff vector of
    the game's scoring, so one model serves `tiered`, `original` and `win_loss`. Keep the
    `threads == 1` path on the calling thread (Task 2's rule). `c_puct` default 1.5; Task 15 tries
    1.0 and 2.5 on 300 deals and keeps the best.

- [x] **Task 15: Phase 2 matches** **[long: ~4 h CPU, headless]** (after 14) (Brian 2026-11-11, D95: a recorded miss counts as done for milestone 1)
  (**results in, bars failed; fallback running.** Job finished 2026-10-29 19:47; tables and
  verdicts in `build-phase2.md` §1–§4. A9 fails: PUCT-2000 vs ISMCTS-1000 at 1,000 deals −0.12
  [−0.78, +0.56] in `1+3`, −1.78 [−2.42, −1.13] in `ABAB`; at 0.5 s per move −1.81 and −2.44.
  A10 fails: distribution head vs Task 7 qnet −1.34 (`ABAB`, tiered), −0.21 (`ABAB`, win_loss).
  `c_puct` 2.5 best, all negative. Fallback (proposal §10 row 2): calibration check and
  `leaf = λ·network + (1−λ)·rollout` at 300 deals, `build-phase2.md` §6. Calibration done
  2026-10-30 (§6.1): prior calibrated (ECE 0.005), value over-spread (slope 0.86) and +0.8
  optimistic. `PuctConfig { leaf_network_weight, rollout_moves }` and `puct_with_rollout`
  added; λ = 1 bit-identical to before (golden-fingerprint test). Fallback match done
  2026-10-30 03:07 (`runs\p2_fallback`, §6.2): PUCT-2000, c_puct 2.5, λ = 0.5 vs ISMCTS-1000 at
  300 deals +0.30 [−0.84, +1.41] `1+3`, +0.61 [−0.56, +1.73] `ABAB` (λ = 1: −1.28, −1.39);
  λ = 0 −0.65, +0.62. Fallback fails the bar too: a tie. Next step (one more round, then
  ISMCTS ships as `expert` if PUCT still ties) in `docs/open-items.md`. Task 15 stays open
  only for A9; Task 16's WebAssembly timing waits for a free CPU.)
  (job started 2026-10-29 14:43, headless, 8 workers (13 cards and mahjong jobs share the CPU),
  `runs\p2_matches\run.ps1` = `experiments/build_phase2.py all --workers 8`, order a, d, c, b;
  expected end ≈ 19:00 (b, 0.5 s per move, is ~3.5 h of it). The script opts out of Windows power
  throttling: a detached headless launch otherwise gave each worker ~0.15 of a core. Check:
  `Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\p2_matches\progress.json`
  (moves every 100-deal block; `runs\p2_matches\results.json` appears at the end). Resume: rerun
  `run.ps1` (finished subcommands skipped, saved blocks read back). Then copy the four
  `runs\p2_matches\*\table.md` into `build-phase2.md` §1–§4 with verdicts. New agent
  `pvnet:checkpoint=…` (v2 network played greedily, game's scoring) for (c); search agents now
  play a forced move without searching (same seeds as before).)
  - Files: create `experiments/build_phase2.py` (subcommands `puct-vs-ismcts`, `equal-time`,
    `heads`); create `docs/study-results/build-phase2.md`; `runs/p2_matches/run.ps1`.
  - Done when: `build-phase2.md` holds, each with mean ± 95 % and win rate in both lineups:
    (a) PUCT-2000 vs ISMCTS-1000, 1,000 deals (A9 bar: ≥ +1.0, interval excluding 0);
    (b) PUCT vs ISMCTS at equal wall time on one thread (0.5 s per move each), 1,000 deals;
    (c) the v2 network's distribution head vs the Task 7 mean-score network as `qnet` policies,
    500 deals under tiered and 500 under `win_loss` (A10); (d) `c_puct` in {1.0, 1.5, 2.5} on 300
    deals each. Check: `progress.json`. Resume: the script skips a subcommand whose `results.json`
    exists.
  - Notes: if (a) fails the bar, follow proposal §10 row 2: check the prior's calibration (log
    loss on held-out round-0 data) and try `leaf = 0.5·network + 0.5·short rollout`; record both.

- [x] **Task 16: Ladder v1, WebAssembly timing, bot-notes for phase 2** (after 15)
  (Bot-notes entry posted 2026-11-01. Not ticked: A11's time part is unmet in the app build, and
  `pvnet::teaching_numbers` does not exist; build it only if a PUCT level is proposed.)
  (**Timing done 2026-11-01** (`build-phase2.md` §5, r3's network, a 13 cards probe beside it):
  PUCT-2000 λ 0.5 at the opening 0.83 s with the app's build flags (A11 bar 0.6 s: not met), 0.28
  s with SIMD; PUCT-300 0.14 s and 0.05 s; move changes after 100 ms in 3.0 % of 300 positions
  (bar < 15 %: met). Harness: `crates/search-wasm` (example `puct_timing`),
  `experiments/wasm_timing.mjs`. **Not ticked:** the bot-notes entry is drafted, not posted, and
  says PUCT does not ship: the done-when's `hard` → PUCT-300 proposal and `models/big2-pvnet-r0.pknn`
  no longer apply. Tick when the entry is in bot-notes.)
  - Files: create `crates/search/examples/puct_timing.rs` (times PUCT-300 and PUCT-2000 at the
    opening position and the anytime move-change share over 300 positions) plus
    `experiments/wasm_timing.mjs` that runs it in Node after
    `wasm-pack build crates/search --example`-style packaging (a tiny `crates/search-wasm` crate
    with `wasm-bindgen` is acceptable if `wasm-pack` cannot build an example directly); modify
    `crates/big2/src/bots/ladder.rs` later (Task 29) not here; `build-phase2.md` §5; bot-notes
    entry. If the app stream has already wired PUCT into `crates/app-api`, time through their
    Node build instead (read-only use of their crate).
  - Done when: `build-phase2.md` §5 records PUCT-2000 and PUCT-300 time per call in Node
    WebAssembly at the opening position (A11 bar: ≤ 0.6 s for 2,000) and the share of 300
    positions where the move changes after `anytime_first_ms = 100` (bar: < 15 %). The bot-notes
    entry lists: new items `big2_search::puct`, `PvNet`, `PuctPolicy`, the model file
    `models/big2-pvnet-r0.pknn` and its size; the proposal `hard` → PUCT-300, `expert` → PUCT-2000;
    `numbers.plays_to_out`, `numbers.lead_value` and `reason.helps_seat {seat, delta}` as functions
    the engine offers (`pvnet::teaching_numbers(state, analysis)`); the measured A9–A11 numbers;
    the request for phone timings on the benchmark phones (B6).
  - Notes: phone measurements are the app stream's; ask, do not wait.

## Phase 3 – Iterate, belief, exact endgames, external bars (opus; Tasks 19, 20 fable)

- [x] **Task 17: Expert-iteration rounds 1–3** **[long: per round ~6 h CPU labelling + 2–4 GPU-h + 2 h CPU eval; headless]** (closed 2026-11-11 by Brian: "For now ship the search bot as expert and queue the round 4 task as Milestone 2"; ISMCTS ships as `expert`, D97 in hk-games; round 4 is Task 17b)
  (**All three rounds done 2026-11-01; not ticked: A9 is not met in both lineups, so no network
  ships** (`build-phase3.md` §1). r3 PUCT-2000 λ 0.5 vs ISMCTS-1000 at 1,000 deals +1.06
  [+0.46, +1.67] 1+3 (passes), +0.66 [+0.08, +1.28] ABAB (fails); r3 vs the teacher ISMCTS-5000
  +0.04 [−0.52, +0.57], −0.00 [−0.62, +0.58], so the teacher stays; calibration slope 1.06; A12
  not run. The done-when names a shipped `models/big2-pvnet-final.pknn`; none exists. The open
  items' default stands until Brian answers: stop after round 3, ISMCTS ships as `expert`. A
  round 4 is not approved. History follows.)
  (**Round 1 done 2026-10-30, A9 fails** (`build-phase3.md` §1): r1 PUCT-2000 λ 0.5 vs
  ISMCTS-1000 at 1,000 deals +0.46 [−0.18, +1.13] 1+3, +0.16 [−0.47, +0.81] ABAB (round 0:
  −0.12 / −1.78); calibration slope 1.03; A12 not run. **Round 2 done 2026-10-31, A9 fails
  the +1.0 bar but is significant** (`build-phase3.md` §1): r2 λ 0.5 vs ISMCTS-1000 at 1,000
  deals +0.76 [+0.13, +1.37] 1+3, +0.86 [+0.29, +1.50] ABAB; calibration slope 1.05; A12 not
  run; no r2-vs-ISMCTS-5000 number, so the teacher stays. **Round 3 running** (the last
  approved round): teacher ISMCTS-5000; r3 from r2's weights on round-1 + 2 + 3 labels;
  calibration; λ screen; A9 at 1,000 deals; A12 (r3 vs r2) only if A9 passes; then r3 vs
  ISMCTS-5000 at 1,000 deals (`expert_iteration.py --stages teacher`, `runs\round3\vs_teacher`)
  for Brian's teacher rule. Job `runs\b25_round3\run.ps1`, launched 2026-10-31 02:50 detached,
  PowerShell PID 2632 (conhost 18520), CPUs 4–19, Normal priority; labelling at ~6,200
  decisions/min on 8 workers after 9 min (other games' jobs share the CPU). Expected end
  ≈ 09:00–10:00 on 2026-10-31 (≈ 12:00 if A12 runs). Check: `Get-Content
  C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\b25_round3\progress.json` and
  `run.log` (details `runs\round3\label\progress.json`, `runs\net_r3\progress.json`,
  `runs\round3\progress.json`, `runs\round3\vs_teacher\progress.json`). Resume: relaunch the
  same `run.ps1` detached (command in its header); every step skips finished work. Then copy
  `runs\round3\table.md` and `runs\round3\vs_teacher\table.md` into §1 and apply the stop
  rule (open items). Round-2 history follows.)
  (**Round 2** (rounds 1–3
  approved): same teacher ISMCTS-5000 (Brian 2026-10-30: "use teacher unless distilled model
  wins significantly"; r1 does not beat it); r2 from r1's weights on round-1 + round-2 labels
  (`DistillV2Config.extra_data_dirs`, `expert_iteration.py --extra-data`, default rounds
  1..N−1; test `test_distill_v2_trains_on_several_folders`); calibration; λ 0.5 / 1 screen at
  300 deals, the better at 1,000 (A9); r2 vs r1 (A12) only if A9 passes. Plus two Task 29
  ladder probes (hard ISMCTS-500 / 1,000) in the training window. Job
  `runs\b25_round2\run.ps1`, launched 20:58 detached, PowerShell PID 33548 (conhost 33200),
  CPUs 4–19, Normal priority; labelling at ~4,800 decisions/min on 8 workers. Expected end
  ≈ 03:00–04:00 on 2026-10-31 (≈ 06:00 if A12 runs). Check: `Get-Content
  C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\b25_round2\progress.json` and
  `run.log` (details `runs\round2\label\progress.json`, `runs\net_r2\progress.json`,
  `runs\round2\progress.json`). Resume: relaunch the same `run.ps1` detached (command in its
  header); every step skips finished work. Then copy `runs\round2\table.md` into §1.
  Round-1 history follows.)
  (**round 1 prepared and queued 2026-10-30.** Per `docs/open-items.md`: teacher ISMCTS-5000
  (beat ISMCTS-1000 +1.09 [+0.35, +1.87] in ladder v0; PUCT λ 0.5 only tied it at ~4× the
  cost), 800,000 decisions (12 h cap; ~107 ms per decision per worker measured under load);
  `train_v2` gains `init` (round 0's weights) and round 1 trains with score weight 100, prior
  1, distributions 0.25 each (value on every candidate's search score); calibration slope check
  (`build_phase2.py calibration --data-dir`); PUCT-2000 c 2.5 at λ 1 and 0.5 vs ISMCTS-1000,
  300 deals each, the better λ at 1,000 (A9), r1 vs r0 at 1,000 only if A9 passes (A12).
  `experiments/expert_iteration.py round 1`; tiny end-to-end run (tag `_tiny`) passed and was
  deleted. **Queued job** `runs\b25_round1\run.ps1`, launched 04:24, PowerShell PID 29832
  (conhost 9176), CPUs 4–19, Normal priority: waits for the mahjong Task 11 job (PID 10588),
  then labels on 8 workers (~4–6 h with the 13 cards job beside it), then trains r1 on the GPU
  while the CPU plays Tasks 18 and 20 (`build_phase3.py all`, ~4 h), then round 1's matches
  (~3.5 h; ~8.5 h if A12 runs). Expected end ≈ 12–19 h after the mahjong job ends (≈ 05:00–
  12:00 on 2026-10-31 if it ends at 17:00). Check: `Get-Content
  C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\b25_round1\progress.json` (and
  `run.log`; details in `runs\round1\label\progress.json`, `runs\net_r1\progress.json`,
  `runs\p3_matches\progress.json`, `runs\round1\progress.json`). Resume: relaunch the same
  `run.ps1` detached (command in its header); every step skips finished work. Then copy
  `runs\round1\table.md` into `build-phase3.md` §1 and the two `runs\p3_matches\*\table.md` into
  §2 and §4.)
  - Files: create `experiments/expert_iteration.py` (`round N`: label 4 M decisions with
    PUCT-500 using net r(N−1) on 14 workers → train net rN with `--resume` from r(N−1) → match
    PUCT-2000(rN) vs PUCT-2000(r(N−1)), 1,000 deals, both lineups → write
    `runs/round{N}/results.json` and a row in `docs/study-results/build-phase3.md` §1; stop rule:
    if the interval does not exclude 0 on the positive side, mark "stopped" and exit 0);
    `runs/round{N}/run.ps1`.
  - Done when: `build-phase3.md` §1 has one row per round run (at most three) with the verdict, and
    names the shipped network (`models/big2-pvnet-final.pknn`, gitignore exception). Check:
    `Get-Content runs\round1\progress.json`. Resume: `expert_iteration.py round 1 --resume`
    (skips labelling if the shard count is reached, resumes training from `latest.pt`, skips the
    match if `results.json` exists).
  - Notes: 25 % random top-3 moves in labelling stay. GPU per round 2–4 h; check `nvidia-smi` first.
    If round 1 loses to round 0, do not run rounds 2–3; record it (proposal §10 row 3).

- [x] **Task 18: Hard-constraint belief from the rules** `[parallel-ok]`
  (**done 2026-10-30.** `RulesMask` in `belief.rs` (rejection sampler,
  counters, `Sampler::weight`), `Rewind::history_consistent` (State: one backward pass over
  the history, tested equal to a full replay); `analyze(belief="rules")` for ISMCTS and PUCT
  with `belief_rejection_share`/`belief_fallbacks`; spec `belief=rules`. `cargo test -p
  big2-search rules_mask` passes (3 tests). Full run (`runs/p3_matches/a_rules_mask/`,
  `build-phase3.md` §2): mean rejection share 0.5 % of draws per decision (1.0 % of decisions
  reject any); PUCT-2000 with vs without, 300 deals: +0.20 [−0.40, +0.82] 1+3, +0.08
  [−0.47, +0.65] ABAB. A13 met (loses nothing). Play changes in only a few deals.)
  - Files: modify `crates/search/src/belief.rs` (`RulesMask`: a `Sampler` that rejects a sampled
    deal if replaying the history under it makes any observed action illegal: last-card duty
    singles, a `locked_out` pass, a play of cards the deal gives to someone else; counts
    rejections); pybind option `belief=rules`; agent spec option; tests; `build-phase3.md` §2.
  - Done when: `cargo test -p big2-search rules_mask` passes (a constructed history where seat 1
    played a single below their highest card while seat 2 held one card is rejected for every deal
    that gives seat 1 a higher card); a 300-deal match PUCT-2000 with `belief=rules` vs without,
    both lineups, is in `build-phase3.md` §2 with the mean rejection share per decision (A13: the
    interval includes or exceeds 0).
  - Notes: `belief.rs` already rejects illegal plays inside `log_likelihood`; this task makes the
    hard mask a sampler of its own, cheap enough for the phone (no greedy model). Each rule is a
    future reason key (`reason.control.safe_single`, Task 26).

- [x] **Task 19: Learned belief per round, kept only if it pays** `(fable)` **[GPU: ~2 h per round; CPU 1 h]** `[parallel-ok]` (beside earlier phases: job 46 already ran; recording its numbers needs no Task 15 or 17 result) (**done 2026-11-10:** round 1 recorded in `build-phase3.md` §3; A14 missed, teaching only)
  (**code done 2026-11-10 (slot 2); job running, not ticked.** Python `python/big2/rl/belief.py`
  (`big2 rl belief-gen`, `belief-train --export`; 476 k parameters; the net corrects the
  hand-size baseline), Rust `big2_big2::belief_net::BeliefNet`, `big2_search::NetSampler` +
  `DealPrior`, `analyze(belief="net", belief_net=…)` for ISMCTS and PUCT, match command
  `experiments/build_phase3.py belief`; tests pass (3 + 2 Rust, 4 Python). Data comes from
  self-play of round 1's network by its prior, not PUCT self-play (PUCT: ~0.2 s per move,
  1 M positions would be ~50 CPU-hours). **Round 1 only** (the plan says once per round at most).
  Queue job 46 (`hkq.ps1 status`; `hkq.ps1 log 46`; `runs\belief\run.log` in the slot-2
  worktree `.worktrees\.worktrees\big2-2`): generate 1 M positions, train, export
  `runs\belief\belief-r1.pknn`, 300-deal match PUCT-2000 net vs rules mask →
  `runs\belief_r1\results.json`. Then: copy the numbers into `build-phase3.md` §3 (held-out
  log-likelihood from `runs\belief\v1\results.json`), apply the A14 rule, tick.)
  - Files: create `python/big2/rl/belief.py` (small net ≤ 0.5 M parameters: from the actor's view
    predict, for each unseen card, which of the three opponents holds it; trained on 1 M positions
    from PUCT self-play of the current round; `big2 rl belief-train`); `.pknn` export; create
    `crates/search/src/belief_net.rs` (`NetSampler`: draw candidates uniformly, weight by the
    net's log-likelihood, tempered as `Belief` does today); tests; `build-phase3.md` §3.
  - Done when: `build-phase3.md` §3 reports, per round trained: held-out log-likelihood per card
    vs the uniform baseline (today's measurement was 0.37 → 0.40), and a 300-deal match PUCT-2000
    with `NetSampler` vs `RulesMask`. Rule: keep the sampler in the ladder only if the gain is
    ≥ +0.5 with the interval excluding 0 (A14); otherwise record "teaching only".
  - Notes: the net's per-card posterior feeds `reason.control.highest_out` even when strength
    gains nothing. Run after Task 17's round 1 exists; once per round at most.

- [x] **Task 20: Information-set exact endgame with forced proofs** `(fable)` `[parallel-ok]`
  (**done 2026-11-01:** everything in the done-when is recorded (`build-phase3.md` §4). Node
  WebAssembly: no hidden-card threshold stays under 1 s; 10 cards in play does (95th percentile
  0.10 s, slowest 0.36 s), so `max_cards` 10 stays. A15: time met, agreement missed at 94.2 %
  against 95 %. History follows.)
  (**code done 2026-10-30, full run queued.** `infoset_endgame.rs` (`InfosetConfig { threshold
  12, max_cards 10, max_nodes 1 M }`, `InfosetResult { exact, forced, deals, excluded }`,
  `Forced { winner, plays }`), `Game::hidden_count`/`for_each_deal` (State enumerates every
  split), hook at the root of ISMCTS and PUCT (`infoset_endgame` setting; golden fingerprints
  unchanged), `analyze(infoset_endgame=…)` returns `exact`/`forced`. `cargo test -p
  big2-search infoset` passes (6 tests incl. the hand-provable forced win and the 13-hidden
  refusal). Finding: cost follows cards in play, not hidden cards (§4 probe; open item), hence
  `max_cards`. **Full run done 2026-10-30** (`runs/p3_matches/b_endgame/`, `build-phase3.md`
  §4): 240 of 300 positions solved within 20 M nodes; same move as ISMCTS-100,000 85.8 %,
  same exact value 94.2 % (A15 bar 95 %: missed); native 95th percentile under 1 s only up to
  10 cards in play (hidden-card thresholds 8/10/12 are all over 1 s), so `max_cards` 10 stays;
  300-deal match with vs without: +0.19 [−0.40, +0.82] 1+3, +0.07 [−0.49, +0.65] ABAB.)
  - Files: create `crates/search/src/infoset_endgame.rs` (when unseen cards ≤ `threshold`
    (default 12): enumerate every consistent deal (≤ 34,650), solve each by `Endgame::solve_actions`
    under a shared node cap, average per action by the sampler's weights; `forced: Option<Forced
    { result, plays }>` when every deal agrees on the winner for the best move); hook into PUCT and
    ISMCTS as the leaf when the threshold is met (`cfg.infoset_endgame: bool`); pybind option;
    tests; `build-phase3.md` §4.
  - Done when: `cargo test -p big2-search infoset` passes (a 3-card endgame where the forced win
    is provable by hand is reported as `forced`; a position with 13 unseen cards is not solved);
    `experiments/build_phase3.py endgame` writes to §4: agreement with ISMCTS-100,000 on 300 late
    positions sampled from PUCT self-play (A15 bar ≥ 95 %), time per call native and in Node
    WebAssembly at thresholds 8, 10, 12 (pick the largest under 1 s and record it), and a 300-deal
    match PUCT-2000 with vs without the endgame against PUCT-2000 opponents (not greedy ones).
  - Notes: the node cap is per call, shared across deals; when it is hit, fall back to the network
    leaf and say so in the result (`exact: false`). This is `quality: {"exact": true}` later.

- [x] **Task 21: B2 — Charlesworth's network as a local opponent** **[~1 h CPU]** (needs open question 1)
  (**Done 2026-11-05** after Brian's yes: no difference detected, paired diff +0.06 ± 0.26
  (1+3) and +0.04 ± 0.30 (ABAB), 1,000 deals each, so A16 is missed; `build-phase3.md` §5.)
  (Earlier: **Skipped until Brian answers 2026-11-01:** open question 1 has no answer in `docs/open-items.md`
  or `bot-stream\docs\decisions.md` beyond B14's general rule; question 1 of the open items'
  2026-11-01 item asks it.)
  - Files: modify `crates/big2/src/rules.rs` (`four_card_plays: bool`, default `false`, not in
    `names!`, so the app's `meta()` does not list it); modify `combo.rs`/`movegen.rs` (two pairs
    and four of a kind as four-card plays when the flag is on; ordering per Charlesworth's rules as
    read in `survey.md` §2.1); differential test rows; create `python/big2/agents/external/charlesworth.py`
    (NumPy forward pass of his released checkpoint: 412 features, 1,695-action head, our own code
    from his published description; weights under `runs/external/charlesworth/`, gitignored, never
    committed); create `experiments/build_phase3.py b2`; `build-phase3.md` §5.
  - Done when: `cargo test -p big2-big2` and `pytest python/tests/test_differential.py` pass with
    the flag off (unchanged behaviour: the differential test still agrees on 5,000 games) and new
    unit tests with the flag on; `b2` plays PUCT-2000 (final net) vs Charlesworth under his rules
    and his scoring (via `--score-as`), 1,000 deals, both lineups, and §5 records mean ± 95 %
    (A16) and the licence state.
  - Notes: `RuleSet` is constructed in `crates/app-api` via `RuleSet::default()` plus field sets
    (shapes.rs:157), so an added field with a default breaks nothing; verify with
    `cargo test -p big2-app-api`. The network trained without four-card plays plays under them
    here; report that handicap. If Brian answers "no" to open question 1, skip the Python agent and
    record B2 as "not run".

- [x] **Task 22: B3b — the best-response probe** **[GPU: 3 × 10 h, three separate headless jobs]** (after 17)
  (**Code done and queued 2026-11-01** in the one job of Task 24's status: `--opponents`,
  `test_dmc_frozen_opponents`, `build_phase3.py probe`, `probe-eval`. Third candidate is
  `expert`, not the network; every probe starts from the distilled duty network; the evaluation
  also compares at equal games (`build-phase3.md` §6).)
  - Files: modify `python/big2/rl/dmc.py` (`--opponents <spec>`: three frozen seats play the given
    agent spec; the learner holds one seat, rotated per game); CLI flag; tests
    (`test_rl.py::test_dmc_frozen_opponents` on a 1-minute run); `experiments/build_phase3.py probe`;
    `build-phase3.md` §6; `runs/probe_{ismcts,dmc10h,final}/run.ps1`.
  - Done when: §6 holds, for each of the three candidates (ISMCTS-1000, DMC-10h, the final PUCT
    network as a `qnet`-style policy), the probe's final gain against the candidate at 1,000 deals
    with intervals, plus the validation run (a 2-hour probe gains more against 3 greedy than
    against 3 ISMCTS-1000). A17: the gain against the final network is the smallest. Check:
    `progress.json` per job. Resume: `--resume runs/probe_<x>/latest.pt`.
  - Notes: 10 GPU-hours each; run them one at a time; check `nvidia-smi` and the GPU queue
    (HANDOFF: big2 is first) before each. Announce each in the hand-back. The probe plays against
    a Rust policy for ISMCTS; for the network candidates it plays against the exported `.pknn`
    through the pybind, so the frozen seats stay on the CPU.

- [x] **Task 23: Cross-play of two seeds** **[GPU: 2–4 h; CPU 2 h]** (after 17)
  (**Queued 2026-11-01** in the one job of Task 24's status: seed B from fresh weights on rounds
  1–3's labels, then PUCT-2000 r3 vs seed B, `build_phase3.py seeds-train`, `seeds-match`,
  `build-phase3.md` §7.)
  - Files: `experiments/expert_iteration.py round 1 --seed 1 --tag seedB` (reuse); `build-phase3.md` §7.
  - Done when: §7 records PUCT-2000(seed 0) vs PUCT-2000(seed 1), 1,000 deals, both lineups; A18:
    the interval covers 0. If it does not, record it as a risk (proposal §10 row 4).

- [x] **Task 24: Final 10,000-deal matches** **[long: ~9 h CPU each, headless]** (after 17, 21)
  (**Queued 2026-11-01:** `expert` (ISMCTS-10,000 + exact endgame) vs ISMCTS-1000, the bot that
  ships by default, not PUCT-2000 (`build-phase3.md` §8); Charlesworth ran in Task 21 (§5).
  One job for Tasks 24, 23 and 22: `runs\b25_tasks22_24\run.ps1`, launched 2026-11-01 06:05
  detached, PowerShell PID 26572 (conhost 23152), CPUs 4–19, Normal priority; 8 workers at about
  92 % of a core each. Order: Task 24 with seed B's training beside it, Task 23's match, the five
  probes (34 h), the probe evaluation. Expected end ≈ 02:00–08:00 on 2026-11-03. Check:
  `Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\big2\runs\b25_tasks22_24\progress.json`
  and `run.log` beside it (details: `runs\final_matches`, `runs\net_seedB`, `runs\seed_crossplay`,
  `runs\probe_<name>`, `runs\probe_eval`, `progress.json` in each). Resume, also after a Windows
  Update restart: relaunch the same `run.ps1` detached (command in its header); every step skips
  finished work. Don't edit `experiments/build_phase3.py` or `python/big2/rl/dmc.py` while it
  runs: each step reads them when it starts. Then copy the three `table.md` files into
  `build-phase3.md` §6–§9.)
  - Files: `experiments/build_phase3.py final`; `build-phase3.md` §8; `runs/final_matches/run.ps1`.
  - Done when: §8 holds PUCT-2000(final) vs ISMCTS-1000 and vs Charlesworth (if run), 10,000
    deals each, both lineups, mean ± 95 % and win rate (A19). Check: `progress.json`. Resume: the
    script resumes from the last saved deal index in `progress.json`.

- [x] **Task 25: Phase 3 docs and bot-notes** (after 17–24) (Brian 2026-11-17: "yes"; the orchestrator added the entry to hk-games `docs/streams/bot-notes.md`, hk-games 877b6f9; every other Task 25 file was done 2026-11-17: check and tick)
  (**Started 2026-11-01:** `docs/solver.md` §6.3–6.5, `docs/rl.md` §3.3, `docs/architecture.md`,
  and a summary table with sources (`build-phase3.md` §9) cover what is known. Left: the job's
  results in that table, the README findings table, and the bot-notes entry.)
  - Files: `docs/study-results/build-phase3.md` summary table; `docs/solver.md` §6 (PUCT, rules
    mask, information-set endgame); `docs/rl.md` (v2 net, expert iteration loop, belief net);
    `docs/architecture.md` (new modules); `README.md` findings table; hk-games bot-notes entry.
  - Done when: every number in the summary table has a source `runs/<job>/results.json` named
    beside it; bot-notes lists the shipped model file name and size, the new public items
    (`RulesMask`, `NetSampler`, `infoset_endgame`), what `quality` would carry (`{"samples": n}` or
    `{"exact": true}`) once they accept B20, and the B2/B3 results in three lines.

- [x] **Task 32: Rollout best response against the shipped `expert`** `[parallel-ok]` **[long: 1–2 CPU-days, job queue cpu lane]** (beside earlier phases: it tests the bot that ships now and needs no Task 15 or 17 result) (job 88 ended 2026-11-11 11:54 with exit 0, a 4.8 h run, checked by the orchestrator: a full run, 120 of 120 deals in runs\rollout_br_full (progress.json), so the estimate below was too high; full run, cpu lane; est. ~37 h wall on 8 threads (60 deals x 2 lineups, 8 rollouts, rollout iterations 10,000, ~17,700 CPU-s per deal pair from the pilot); output runs
ollout_br_full in the big2-workers worktree (needs its copied pythonbig2_core.pyd); resume: rerun runs
ollout_br
un.ps1 through the queue, raise --deals in it to add deals (same seeds, same deals.jsonl); a worker then records pilot and run in §7a and ticks)
  (Job 47 is a pilot, not the run: it took 17 minutes, not 1–2 CPU-days, and its rollouts
  played ISMCTS-100, not `expert`'s ISMCTS-10,000. Pilot results, 40 deals each: 1+3 −1.35 ± 3.75,
  ABAB −0.59 ± 1.92, 2.6 s per searched decision; files in
  `.worktrees\.worktrees\big2-2\runs\rollout_br\`. Orchestrator 2026-11-10: rerun with
  `--rollout-iterations` as close to 10,000 as the 2-CPU-day budget allows, timed from the
  pilot, with a new `--out` folder; record the pilot and the run in §7a, then tick.)
  (Added 2026-11-10 on Brian's word "Yes, add all three", from the strength-check research: no
  outside test of `expert` exists, and Charlesworth's network only ties it, A16.)
  - What: one seat plays a best response found by rollouts. At each decision it tries every legal
    move and plays each out many times, with the other three seats played by `expert`'s own code
    (ISMCTS-10,000 + exact endgame). It keeps the move with the best mean. No training.
  - Files: create `experiments/rollout_br.py` and a test that runs it on a few deals; add a
    section to `docs/study-results/build-phase3.md`; `runs/rollout_br/run.ps1` (job queue,
    `hkq.ps1 add -Lane cpu`, reads `$env:HKQ_THREADS`, resumes from `progress.json`).
  - Done when: the section records the attacker's mean score per deal against three `expert`,
    with a 95 % interval, in both lineups, and the deal count. Choose the deal count so the run
    fits 2 CPU-days. No bar is set: a gain whose interval excludes 0 goes to `docs/inbox.md` as
    a question for Brian.

- [x] **Task 33: Best-response probe trained from scratch** `[parallel-ok]` **[GPU: 10 h, job queue gpu lane]** (beside earlier phases: it tests the bot that ships now and needs no Task 15 or 17 result) (ready 2026-11-14: queue job 69 finished 11:19, exit 0; training 600 min, 98,176 games, last learner score -3.86 (`runs\probe_scratch_expert\progress.json`); the 1,000-deal match is in `runs\probe_scratch\` of `C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-workers`; a worker records the numbers in §6.1 and ticks)
  (Added 2026-11-10 on Brian's word "Yes, add all three". Task 22's probes start from the
  distilled network, so they may share the teacher's blind spots.)
  - What: Task 22's probe (`dmc.py --opponents`, `build_phase3.py probe`), but from fresh
    weights, with three `expert` seats frozen. Reuse Task 22's code; add only a from-scratch flag
    if the code lacks one.
  - Files: `runs/probe_scratch_expert/run.ps1` (job queue, `jobq.ps1 add -Lane gpu`); the probe
    section (§6) of `docs/study-results/build-phase3.md`.
  - Done when: §6 records the probe's gain against `expert` at 1,000 deals with a 95 % interval,
    beside Task 22's distilled-start probes. Check: `progress.json`. Resume: `--resume
    runs/probe_scratch_expert/latest.pt`. A gain whose interval excludes 0 goes to the inbox.

## Phase 4 – Teaching depth (opus; Task 27 fable)

The teaching code lives in a new crate we own, `crates/teach` (package `big2-teach`), so the app
stream can call it from `crates/app-api` without us touching their crate.

- [x] **Task 26: `big2-teach`: reason keys from the move table** `[parallel-ok]`
  (done 2026-10-30, opus: `crates/teach` is a workspace member; `MoveTable` (from the paired
  evaluator or any `Analysis`), `reasons_for`/`reasons` with the 11 keys, rules in rustdoc and
  `crates/teach/README.md`; `reason.helps_seat` (the proposal's spelling, not `reason.value.*`)
  needs the gain's paired interval to exclude 0; `reason.plan.keeps` only names a group the
  chosen play leaves whole. `cargo test -p big2-teach reasons` 25 tests (fire / must-not per
  key); clippy clean. Audit: `build_phase4.py audit` → `runs/p4_audit/hints-50.json`, 50 entries
  (counts in `build-phase4.md` §0). The audit runs the native `big2-teach` program, so no
  extension rebuild was needed.)
  - Files: create `crates/teach/` (`Cargo.toml`, `src/lib.rs`, `src/reasons.rs`): `Reason { key:
    &'static str, args: serde_json::Value }`, `reasons_for(state, analysis, candidates, chosen)
    -> Vec<Reason>` with the proposal's keys: `reason.plan.keeps {combo}`, `reason.plan.breaks
    {combo, cost}`, `reason.control.pass_to_keep {card}`, `reason.value.score_vs_win
    {score_delta, win_delta}`, `reason.tempo.plays_to_out {mine, theirs: [3]}`,
    `reason.control.safe_single {card}` (no unseen card beats it), `reason.control.highest_out
    {card}`, `reason.value.gap {points}`, `reason.counterfactual.outcomes {best, played,
    deals}`, `reason.helps_seat {seat, delta}`, `reason.counterfactual.forced {result, plays}`;
    each key's rule in rustdoc and in `crates/teach/README.md` (the table the app stream copies
    into their README); add to workspace members; tests; `experiments/build_phase4.py audit`
    writes `runs/p4_audit/hints-50.json` (50 positions, hint, reasons, for Brian's audit, B11).
  - Done when: `cargo test -p big2-teach` passes with one test per key that constructs a position
    where the key must fire and one where it must not; `cargo clippy -p big2-teach --all-targets
    -- -D warnings` clean; the audit file exists with 50 entries (A20).
  - Notes: keys are true by construction: each is a feature difference between two candidates plus
    the solver's value gap, never a guess (hard-problems §8). Keep the app's existing keys
    (`reason.keeps_control` etc.) untouched; ours are additions. serde_json is already a workspace
    dependency through app-api; add it to `crates/teach` directly.

- [x] **Task 27: Paired grader, σ thresholds with "unclear", luck ledger** `(fable)` `[parallel-ok]` (beside earlier phases: orchestrator's call 2026-11-10, since Tasks 26 and 30 of this phase were built while phase 3 ran) (inbox default taken: A21 uses compatible agreement; orchestrator's call 2026-11-10, Brian may change it)
  (**code done 2026-10-30 by opus (not fable: B25 helper brief); repeatability bar not met.**
  `grade.rs` (`evaluate_paired`, `grade_from_table`, `grade_paired`/`grade_paired_table` with
  doubling while `unclear` up to `max_deals`, `Thresholds { from_sigma, APP }`, σ 27.45 →
  0.69 / 2.75 / 6.86), `ledger.rs` (`luck_ledger`, `Ledger::from_tables`, exact by
  construction). `cargo test -p big2-teach -- grade ledger` passes (11 tests: ledger sums to
  1e-9 on 200 random deals, a forced move grades `best`). `build_phase4.py regrade`
  (`build-phase4.md` §1, 52 s): same grade with a new seed 90.2 % (A21 bar 95 %: not met;
  compatible 97.6 %), `unclear` 13.4 %, phone vs 10× 87.2 % (bar 90 %: not met; compatible
  94.9 %), ledger error 4e-15 (met). Budget probe up to 4,800 deals: 93.8 % strict. Open item:
  which agreement measure A21 uses. Orchestrator's call 2026-11-10: the default, compatible agreement (97.6 %, met); a worker ticks it after recording this in decisions.md.)
  - Files: create `crates/teach/src/grade.rs` (`grade_paired(position, played, cfg) -> Graded {
    loss, interval, grade, seed }`: every legal move evaluated on the same sampled deals so noise
    cancels; thresholds as multiples of σ from Task 6 (`good < 0.025σ`, `inaccuracy < 0.1σ`,
    `mistake < 0.25σ`, `blunder ≥ 0.25σ`, written as a `Thresholds` struct with those defaults
    and a constructor from σ; `"unclear"` when the 95 % interval crosses a line); create
    `src/ledger.rs` (`luck_ledger(deal) -> Ledger { deal_strength, opponents_errors, own_errors,
    residual }` whose terms sum to the seat's final score); tests; `experiments/build_phase4.py
    regrade`; `docs/study-results/build-phase4.md` §1.
  - Done when: `cargo test -p big2-teach grade ledger` passes (ledger sums exactly on 200 random
    deals; a forced move grades `best`); `regrade` re-grades 50 self-play deals twice with
    different seeds and writes the agreement share to §1 (A21 bar ≥ 95 %) and the share of
    `"unclear"` grades; and the phone-sized grader (300 deals) agrees with a 10× grader on ≥ 90 %.
  - Notes: σ comes from Task 6 (expected ≈ 20 points; then the lines are 0.5 / 2 / 5 points). The
    app's current lines are 1 / 3 / 9 points; open question 3 asks Brian which to ship. Both
    tables are in `crates/teach/README.md`.

- [x] **Task 28: Drill miner, seed ratings, mistake clustering** **[long: ~3 h CPU]** (after 26, 27)
  (**code done 2026-10-30, tiny run only; the full run is pending.** Miner in Rust:
  `big2-teach mine` (self-play by easy, medium, hard, hard; paired table of 600 deals per
  decision; candidate = gap ≥ 2 with interval above 0, easy and medium wrong on ≥ half of 16
  samples, expert right; every graded mistake with the best move's keys; `raw.jsonl`,
  resumes from `progress.json`). Python: `big2/teach/drills.py` (Drill shape, rating formula,
  skill from keys, validation, audit sample), `big2/teach/cluster.py`, CLI `big2 teach
  mine-drills` / `big2 teach cluster`; `pytest python/tests/test_drills.py` 7 passed
  (incl. a native 4-deal run validated against the §6 shape). Positions come from ISMCTS
  self-play, not PUCT (open items). **Full run done 2026-10-30** (700 deals in 185 s;
  `build-phase4.md` §2): 502 drills, every §6 field present, 50-drill audit file, 55 mistake
  clusters beside the app's skill list. **Ticked 2026-11-10 (slot 3):** the seed formula spread 548 points (A22 needs ≥ 600), so the
  inbox default is applied (orchestrator's call 2026-11-10; Brian may change it): rank by share wrong, then
  by gap, stretched over 600–1,400 (`RATING_FORMULA = "bands"` in `big2/teach/drills.py`; `"seed"` restores the old formula; decision 0004). Re-run: 553 drills, ratings 600–1,400, `drills/big2.json` committed.)
  - Files: create `python/big2/teach/drills.py` (mine positions from PUCT self-play where the
    ladder disagrees: `easy` and `medium` wrong, `expert` right, value gap ≥ 2 points; rating =
    1000 + 400 × (share of levels wrong) − 100 × log2(gap); emit the engine-interface `Drill`
    shape with `source: "solver"`, `solver: {samples, gap}`, `skill` from the reason keys of the
    best move); `python/big2/teach/cluster.py` (group graded mistakes by their key sets; print the
    clusters as the `skill.*` candidates); CLI `big2 teach mine-drills`, `big2 teach cluster`;
    tests; output `drills/big2.json` (≥ 300 drills) and `runs/p4_audit/drills-50.json`;
    `build-phase4.md` §2.
  - Done when: `pytest python/tests/test_drills.py` passes (the output validates against the shape
    in `hk-games\docs\engine-interface.md` §6: every drill has `id`, `game`, `skill`, `prompt`,
    `position`, `seat`, `rating`, `answers`, `source`); `drills/big2.json` has ≥ 300 entries with
    ratings spread over at least 600 points; `cluster` output is in §2 next to the skill list in
    `hk-games\app\src\drills\big2.json` (A22). Hand the file over in Task 31's bot-notes entry.
  - Notes: the Miro "Frame 7" curriculum is read-only (D30) and not reachable from here; compare
    with the app's skill list instead and name the unmatched clusters as new-skill candidates.
    Human solve rates (Spearman) arrive only after release: open item, not a task.

- [x] **Task 29: Error model and ladder v2** **[long: ~45 min CPU]** (after 26) (beside earlier phases: orchestrator's call 2026-11-10, since Tasks 26 and 30 of this phase were built while phase 3 ran) (inbox default taken: hard = ISMCTS-500, expert = ISMCTS-10,000 + exact endgame, short top gap recorded; orchestrator's call 2026-11-10, Brian may change it)
  (**Ticked 2026-11-10 (slot 3): inbox default (a) applied: `hard` = ISMCTS-500 (`ladder.rs`, app preset, UI text, docs/ui.md); `expert` = ISMCTS-10,000 + exact endgame; top gap +1.02 recorded short of A23 bar 1.5. `cargo test -p big2-teach ladder` 2 passed.** Earlier: **code done 2026-10-30, tiny run only; the 500-deal round is pending.**
  `crates/big2/src/bots/error_model.rs` (`ErrorModel<P>`, `Mistake` with keys
  `mistake.hold_two_too_long`, `mistake.break_pair_for_single`, `mistake.miss_safe_single`,
  `choose_explained` returns the mistake). `cargo test -p big2-big2 error_model` 6 passed
  (each rate ± 1 % over 100,000 draws; 300 noisy deals, every move legal, every mistake
  fires). Ladder in `crates/teach/src/ladder.rs`, not `crates/big2` (`hard`/`expert` need
  `big2-search`, which `big2-big2` cannot depend on): `random`, `easy` (StructuredGreedy ε 0.1
  + rates 0.5), `medium` (Q-net T 1.0 + rates 0.15), `hard` ISMCTS-2,000, `expert`
  ISMCTS-10,000 + exact endgame (ISMCTS, not PUCT, per open items). `cargo test -p big2-teach
  ladder` 2 passed. `big2-teach ladder` / `build_phase4.py ladder`. Tiny run (8 deals, small
  searches): every upper level ahead of the one below (+3.97, +2.56, +1.41, +0.34), intervals
  about ± 3, so nothing is decided yet. **Full run done 2026-10-30** (500 deals per pair,
  777 s; `build-phase4.md` §3): easy vs random +6.56 ± 0.50, medium vs easy +3.15 ± 0.56,
  hard vs medium +3.08 ± 0.46, **expert vs hard −0.03 ± 0.38**. A23 not met at the top; the
  plan's knobs (temperature, rates) only reach easy and medium. **Probes done 2026-10-30**
  (round-2 job; `build-phase4.md` §3): `hard` = ISMCTS-1,000: hard vs medium +2.76 ± 0.46,
  expert vs hard +0.31 ± 0.40 (order not strict); `hard` = ISMCTS-500: +2.72 ± 0.49 and
  **+1.02 ± 0.40** (strict order, top gap short of 1.5). Which `hard`/`expert` ship is Brian's
  call (`docs/open-items.md`). **Not ticked.**)
  - Files: create `crates/big2/src/bots/error_model.rs` (`ErrorModel { hold_two_too_long,
    break_pair_for_single, miss_safe_single }` rates; wraps any `Policy<State>` and applies a
    named mistake with its rate, returning which key fired so review can show it); the five
    levels as named constructors in `crates/big2/src/bots/ladder.rs` (`random`, `easy =
    StructuredGreedy + ErrorModel(hand-set rates)`, `medium = QNetPolicy { temperature } + thin
    ErrorModel`, `hard = PUCT-300`, `expert = PUCT-2000 + infoset endgame`); `experiments/build_phase4.py ladder`;
    `build-phase4.md` §3.
  - Done when: `cargo test -p big2-big2 error_model ladder` passes (each mistake fires at its rate
    ± 1 % over 100,000 draws, and never produces an illegal move); the ladder round-robin of 500
    deals is in §3 with strict ordering and gaps ≥ 1.5 points (A23); if a gap is short, the
    task tunes `temperature` and the rates and records the final values in `ladder.rs` rustdoc.
  - Notes: the blind ranking by a strong player (proposal §7, phase 4) needs Brian's time: open
    question 5; the measured gaps are the evidence this plan can give.

- [x] **Task 30: Convention-gap check (scaffold)** `[parallel-ok]`
  (done 2026-10-30: `python/big2/teach/convention_gap.py` (key set per decision, 200-point
  rating bands, weighted slope of the disagreement rate per 100 points; tags `convention`
  (flat, rate ≥ 15 %), `skill` (falls ≥ 1 point per 100), `rare`, `too_few`), CLI
  `big2 teach convention-gap <log>`; `pytest python/tests/test_convention_gap.py` 4 passed
  (the synthetic log's flat set is `convention`, the falling one `skill`). Waits for real logs
  (B13).)
  - Files: create `python/big2/teach/convention_gap.py` (reads a game log in the hk-games review
    shape: list of `{position, played, best, loss, rating}`; groups disagreements by reason-key set
    and by player rating band; prints the sets whose disagreement rate does not fall with rating,
    tagged "convention"); test on a synthetic log; CLI `big2 teach convention-gap <log>`.
  - Done when: `pytest python/tests/test_convention_gap.py` passes (a synthetic log with one
    rating-independent disagreement and one rating-dependent one is split correctly).
  - Notes: real logs exist only after release with consent (B13); the script waits for them.

- [x] **Task 31: Phase 4 docs, handover, bot-notes** (after 26–30)
  (done 2026-11-17: summary in build-phase4.md, README synced, entry in hk-games bot-notes (387544e). Rust tests, fmt, clippy pass; the Python tests were NOT run: no `big2._core` is built in this worktree. Third audit file named in the plan has no source: two exist.)
  - Files: `docs/study-results/build-phase4.md` summary; `crates/teach/README.md` (keys, grades,
    thresholds, ledger, drills); `docs/architecture.md`, `README.md`, `docs/ui.md` if the web UI
    changed (it should not); hk-games bot-notes entry.
  - Done when: bot-notes lists: `big2-teach` and how to call `reasons_for`, `grade_paired`,
    `luck_ledger`; the full key table; the two threshold tables and which one Brian chose (or
    "pending, open question 3"); `drills/big2.json` as the handover (≥ 300 drills, shape checked)
    and where it lives; the ladder v2 constructors to switch the five levels to; the three audit
    files for Brian. Full tests and lint pass (the table at the top, minus the app crate's fmt).

---

## Phase 5 - A win-chance bot (milestone 2)

Task 17b was moved here from Phase 3 on 2026-11-17 by the orchestrator: it waits for milestone 2, and under the phase rule it held Tasks 22 to 25 and 31, whose job ended on 2026-11-03 (`runs\b25_tasks22_24\progress.json`: step done) and which only need their results written up.

- [ ] **Task 17b: Expert-iteration round 4** (waiting: milestone 2, Brian 2026-11-11: "For now ship the search bot as expert and queue the round 4 task as Milestone 2")
  - Why: r3 PUCT-2000 vs ISMCTS-1000 at 1,000 deals: +1.06 (1+3, passes A9) and +0.66 (ABAB, fails);
    r1 +0.46 / +0.16, r2 +0.76 / +0.86 (`docs/study-results/build-phase3.md`).
  - Files and done when: as Task 17 for one more round, labelled by ISMCTS-5000, network from r3.
  - Brian 2026-11-11 (hidden-card network, answer "(a)"): the round's hidden-card network also guesses each
    opponent's hand shape (five-card hands, pairs, triples, 2s and aces held) beside the per-card guess;
    the search weights imagined deals by both; compare against the card-only network on the same test.
    The round's labelling and opponent model are still to be settled with Brian before this task starts.

- [x] **Task 35: a reason for every move the coach can only price today** (opus) `[parallel-ok]`
  (done 2026-11-17: `contrast_reasons` in `crates/app-api/src/reasons.rs`, 10 cause keys and 2 merit keys added to `reasons`/`best_reasons` in review, grade and hint; 200 graded losses from `drills/big2.json`: 0 % before, 90.0 % now; all 8 coach examples get a cause; slowest call 3.9 ms. `cargo test --workspace --release` passes; Python tests 418 passed, 6 skipped (run against a locally unpacked `_core` wheel, shared env untouched); wasm-pack web build and the `python`-feature wheel build. Entry for hk-games bot-notes: `docs/bot-notes-reasons-entry.md` (the orchestrator pastes it). build-phase4.md §4.)
  - Why: hk-games D115 and `docs/decisions/0005-reasons-and-bot-notes.md` (Brian, 2026-11-17: "yes"). In the app's 10 Big2 coach examples (hk-games `docs/coach-voice-examples.md`), 8 headlines are the points sentence alone (examples 2-8 and 10), because `reasons` gives only `cards_left`, `lowest_beating`, `saves_high_card` and the points, none of which says why the best move is better.
  - What: for the played move and for the best move, `crates/app-api/src/reasons.rs` adds causes and merits that the search already knows or can compute cheaply, each a key with numbers the app can print. First list, for the 8 examples and for 200 graded decisions from stored games, what truly separates the best move from the played one; then cover the most common cases. Candidates to check, not a fixed list: the play gives up the lead when a control card was held; it breaks a five-card hand, a pair or a triple that was needed later; it spends a 2 or the top card of a kind too early; it leaves an opponent with few cards a free play; it keeps a single low card that cannot be shed later; passing here lets the next player go out; the best move keeps the lead to the end (exact endgame). A key only when the fact is true for this move and false for the other (played against best), so the app never states a cause both share.
  - Files: `crates/app-api/src/reasons.rs`, its tests, `crates/app-api/README.md` (each new key: meaning, numbers, an English sentence), `crates/teach` if the facts belong there, `docs/study-results/build-phase4.md` (a short section), `docs/bot-notes-reasons-entry.md` (the entry for hk-games bot-notes: new keys, their numbers, example sentences; the orchestrator pastes it).
  - Done when: on the 200 graded decisions that lost points, at least 80 % carry a cause that is true of the played move and not of the best move (state the share before and the share now); on the 8 examples each has one; no existing key changes meaning or shape; `cargo test` and the Python tests pass; the wasm and server modules build; the notes entry file exists.
  - Notes: no model training, CPU only, minutes per run. Keep move time inside the app's budget (`grade` 600 ms). The hk-games side is coach-voice plan Task 17, which the orchestrator releases when this task and thirteen-cards Task 6.1 are ticked.

- [x] **Task 34: Train a model that maximises the chance of going out first, and match it against the best bot** (opus) `[parallel-ok]` (Brian 2026-11-11: "Add new task for Big2, at milestone 2, train a new model that only optimizes changes of winning (instead of EV) and have it compete against our best bot"; Brian 2026-11-17: "big2 highest-winrate bot can start now, since we've not decided what to do with mahjong and doudizhu. It can come before poker") **[GPU: hours, job queue gpu lane]**
  - Why: every bot today maximises expected points (the cards-left penalty). Brian wants to see
    a bot whose only goal is to win the deal (go out first), and how it fares against `expert`.
  - What: the same training pipeline as today's best network, with the value target changed to
    1 when the seat goes out first and 0 otherwise; the search uses that value too. Train it to the
    same budget as the network it is compared with, so the objective is the only difference.
  - Files: the training and search code the current network uses (a config switch, not a copy),
    their tests, `docs/research/results/` (a new results file), `docs/bot-notes` if the stream uses it.
  - Done when: a duplicate match (10,000 deals, seats rotated, same deals for both) of the win-chance
    bot against `expert` reports both the share of deals each wins and the points per deal, each with
    a 95 % interval; the GPU training goes through `jobq.ps1` in the gpu lane.
  - Queue (2026-11-17): the gpu lane is free. Poker job 181 (cpu lane, river data, ends about 18 Nov 17:30) is running; Brian allows this task ahead of poker. Queue the CPU labelling through `jobq.ps1` (cpu lane, at most 8 threads). If it cannot start beside job 181, send the orchestrator a QUESTION; never cancel or reorder another project's job yourself. Long jobs: end the turn with the task marked waiting on the job number.
  - Queue order (orchestrator, 2026-11-17 14:36; Brian: "swap so big2 gets second priority", "swap so test runs get third priority"): this task's jobs were re-queued ahead of the Dou Dizhu test jobs, same script and arguments: label = job 193 (cpu, no wait), train = 194 (gpu, after 193), match = 195 (cpu, after 194). Jobs 187, 188, 190 are cancelled; read 193/194/195 wherever a line says otherwise.
  - Queue order, later the same day (15:35): the Dou Dizhu search jobs 201-208 go first (Brian's order: search first, big2 second), so train = job 209 (gpu, after 193) and match = job 210 (cpu, after 209); 194 and 195 are cancelled. Label job 193 is unchanged and running.
  - Expected play (Brian, 2026-11-11: "It should pass greedily to optimize late game dominance"): the bot should learn to pass often and keep its strongest cards to control the end of the deal. The win-only target is meant to teach this, not a hand-written rule. Report how often each bot passes when it could play, and how many of its 2s and five-card hands it still holds when an opponent is down to 3 cards or fewer, for both bots.
  - Orchestrator's calls (2026-11-17, Brian may change them): the win pipeline relabels its own round 0 (ISMCTS-1000, objective win, 2,740,296 decisions, fresh weights), so the objective is the only difference from r3; λ = 0.5 and `c_puct` 2.5 are fixed, with no per-round matches; only the final 10,000-deal match vs `expert` and a 1,000-deal habits pass are played.
  - Built 2026-11-17 (slot 1, 5d02d6d): `--objective win` in `experiments/expert_iteration.py` (rounds 0-3 into `runs/round<N>_win`, `runs/net_r<N>_win`); `generate(objective=)` and `DistillV2Config.value_target="win"` in `python/big2/rl/distill.py` (CLI `distill-gen --objective`, `distill-train --value-target`); `experiments/win_chance.py` (final match, habits, table); `experiments/win_chance_job.ps1 -Stage label|train|match`; test `test_rl.py::test_win_objective_labels_trains_and_plays`; results file `docs/research/results/win-chance-bot.md` (method written, numbers pending). Smoke run of the whole chain at tiny sizes passed.
  - Queue (all in the big2-workers worktree; they import big2 from it through PYTHONPATH, so do not rebuild its `_core` while one runs): job 193 cpu labels rounds 0-3 (~12 h on 8 threads), job 194 gpu trains r0-r3 (after 193), job 195 cpu final match + habits (after 194; ~16 h); the orchestrator set this order, Dou Dizhu jobs 196-198 follow. Check: `jobq.ps1 status`; the files `runs/round<N>_win/label/progress.json`, `runs/net_r<N>_win/progress.json`, `runs/win_chance/progress.json`. Jobs 184, 187, 188 and 190 (earlier copies) were cancelled by the orchestrator (184 after five minutes); its shards in `runs/round0_win/data` are valid and count toward the target. A failed job resumes when added again with the same script and args.
  - Then: copy `runs/win_chance/table.md` and the calibration lines (`runs/round<N>_win/f_calibration/`) into `docs/research/results/win-chance-bot.md`, add a bot-notes line only if the bot is to ship, and tick.
  - Done 2026-11-18 (slot 1): job 210 ended. Win-chance bot vs `expert`, 10,000 duplicate deals: deals won 27.5 % vs 24.2 % (`1+3`, diff +3.3 % [+2.9, +3.8]) and 26.8 % vs 23.2 % (`ABAB`, +3.5 % [+3.1, +4.0]); points per deal paired −19.24 [−19.59, −18.89] (`1+3`) and −18.36 [−18.72, −18.01] (`ABAB`). It passes 39 % vs 21 % when it could play and holds more 2s and five-card hands late. Does not ship (no bot-notes line). Results: `docs/research/results/win-chance-bot.md`.
- [ ] **Task 36: self-play pilot with the luck taken out (the mahjong method)** (opus) `[parallel-ok]` (checkpoint) (beside earlier phases) (waiting: queue jobs 356-366)
  - Why: Brian, 2026-11-26: "try the self-play luck removal approach with big2 to utilize the currently idle GPU".
  - What worked elsewhere, and what did not: mahjong lab Task 29 (`hk-mahjong-lab/docs/plans/2026-11-16-self-play-network.md`, setting `oracle`): a value head that sees every hidden hand, used only for the advantage; it went from -0.40 to +1.21 a hand above its hand-written benchmark bots in 24 GPU-h, two thirds of the gain by hour 4. Dou Dizhu Task 26/27 (`doudizhu` repo, `docs/study-results/self-play-gap.md`): from a start already level with its benchmark, self-play beat its own copies (+0.036 a deal) but not the outside benchmark (-0.025), and without the KL pull it got worse within an hour. Read both before designing.
  - Build: from the current best network (the one inside `expert`'s search, or say which and why), self-play with (1) the luck removed: each deal replayed with the seats rotated, the advantage measured against the other replays of the same deal (duplicate), plus the oracle value head as in mahjong if it fits the trainer (`python/big2/rl/dmc.py` or a new trainer; say which); (2) a small constant KL pull to the start network, weight picked by a 30-minute smoke run; (3) because of Dou Dizhu's lesson, a share of seats played by frozen opponents (earlier checkpoints and the start network), not only the latest copy.
  - Queue: gpu lane, training in 4-hour jobs (so Dou Dizhu Task 28's short checks can run between them), at most 24 GPU-h in all, a checkpoint every hour. Measuring (cpu lane or gpu, whichever is free): each 4-hour checkpoint against the start network and against a fixed outside opponent that self-play never trains against (ISMCTS-1000, or PUCT with the start network; same fixed deals each time, duplicate seating, 95 % ranges). Stop rule: if hour 8 is not better than the start against the outside opponent by more than its 95 % range, stop and report.
  - Done when: the smoke run trains without error; the queue holds the jobs (ids in this task); after the run or the stop, a table (start, each 4 h checkpoint; vs start and vs the outside opponent) in `docs/study-results/` and an inbox question to Brian: keep the new network for `expert` / keep the old one. Never start a second long run without his word.
  - Notes: CPUs 4-19 only; no sound; local only, no paid API; the shipped `expert` and the hk-games app are not changed by this task.
  - Built 2026-11-26 (slot 1): trainer `python/big2/rl/selfplay.py` (new: `dmc.py` learns move values, not a policy), driver `experiments/selfplay_luck.py`, queue script `experiments/selfplay_luck_job.ps1`, test `python/tests/test_selfplay.py`, method in `docs/study-results/selfplay-luck.md`. **Start network r3** (`runs/net_r3/best.pt`, copied read-only from the standalone checkout, orchestrator 2026-11-26): `expert` is ISMCTS-10,000 with no network inside; r3 is the best network and has the prior head a policy method needs. PPO on r3's prior; duplicate baseline (each deal 4 rotations x 2 copies, leave-one-out mean of the same hand against the same lineup); oracle critic (full features) logged beside it, and the smoke run picks duplicate / critic / mix by residual variance; constant pull `KL(start || policy)`, smoke picks 0.02 / 0.1 / 0.5 by Dou Dizhu's rule; half the deals learner/frozen/learner/frozen with the start or one of the last 4 hourly snapshots. Tiny CPU dry run of smoke -> train -> eval -> stop rule -> table passed.
  - Queue (2026-11-26, all from the big2-workers worktree; do not rebuild its `_core` while one runs): smoke 343 (gpu) -> `runs/selfplay_luck/choice.json`; train to hour 4/8/12/16/20/24 = 344, 346, 348, 350, 352, 354 (gpu); eval hour 4/8/12/16/20/24 = 345, 347, 349, 351, 353, 355 (cpu; hour 0 is measured with hour 4). Eval 347 applies the stop rule to `runs/selfplay_luck/stop.json`; after a stop, the later train jobs exit at once and the evals find no network. Check: `jobq.ps1 status`, `runs/selfplay_luck/progress.json`, `runs/selfplay_luck/table.md`.
  - Changed 2026-11-27 (Task 36b; Brian: "it probably should only be matched against expert for real benchmark purposes"): `expert` as the app runs it is the only benchmark. Evals 347/349/351/353/355 were cancelled before they ran (the queue then skipped 348-355); jobs 344-346 (hour 4 and 8 training, eval 345) ran as built. Requeued: eval hour 8 = 356 (cpu, 16 threads; r3 and hour 8 vs `expert`, 1,000 deals per lineup, then the stop rule: hour 8's gain over r3, both vs `expert`, paired by deal, whole 95 % range above 0), eval hour 4 = 357; train to hour 12/16/20/24 = 358, 360, 362, 364 (gpu; 358 waits on 356's stop rule); eval hour 12/16/20/24 = 359, 361, 363, 365 (cpu); final 366 (the last trained checkpoint and r3 at 2,000 deals). Check: `jobq.ps1 status`, `runs/selfplay_luck/table_expert.md`, `runs/selfplay_luck/stop.json`.
  - Then (a later run, once 366 has ended): copy `choice.json` (smoke) and `table_expert.md` into `docs/study-results/selfplay-luck.md`, write the inbox question (keep the new network for `expert` / keep the old one; compare both against `expert`), tick. Never start a second long run without Brian's word.

- [x] **Task 36b: the self-play checkpoints against the app's `expert`** (opus) `[parallel-ok]` (checkpoint)
  - Why: Brian, 2026-11-27, on Task 36's hour-4 figures (measured only against the start network r3 and ISMCTS-1000): "big2 I still don't understand which bot you've matched the new luck removal self-play against. Out best network? I thought that's the one with search?" and "what's the point of matching against these much weaker opponents? it needs to beat our strongest?".
  - What: match each Task 36 checkpoint against the app's `expert` exactly as the app runs it (ISMCTS-10,000, no network inside; say how you load it), on fixed deals, both lineups (1+3 and ABAB) as Task 36's eval, points a deal with 95 % range, and also the start network r3 against `expert` on the same deals as the baseline. Hour 4 now (its checkpoint exists); then add the same match to each later Task 36 eval (hour 8 onward) as extra cpu queue jobs after the existing eval job of that hour, or fold it into `experiments/selfplay_luck_job.ps1` without disturbing jobs already queued. Estimate the CPU-hours per match first (ISMCTS-10,000 is slow) and pick the deal count that gives a range of about +-0.5 a deal or better within about 3 CPU-h per checkpoint; say the range you got. Also measure the checkpoint WITH search if the code allows a network-guided ISMCTS at app speed; if not, say so in one line.
  - Done when: a table (start r3, h4, ... vs `expert`, both lineups, range) in `docs/study-results/selfplay-luck.md`; the queue holds the later-hour expert matches (ids in this task); Task 36's final inbox question compares against `expert`. Pushed. Do not change Task 36's training jobs, the stop rule or the shipped `expert`.
  - Notes: CPUs 4-19 only; do not rebuild `_core` in the big2-workers worktree while a Task 36 job runs; no sound; local only.
  - Scope widened 2026-11-27 (orchestrator, relaying Brian: "not just add expert, it probably should only be matched against expert"): `expert` replaces the r3 and ISMCTS-1000 evals and judges the hour-8 stop rule; check that `expert` is the strongest bot.
  - Built 2026-11-27 (slot 1): `experiments/selfplay_luck.py eval` now plays only `expert` = `ismcts:iterations=10000` (the app's settings, no wall-clock cap); `-Threads` and `-Final` in `selfplay_luck_job.ps1`. Measured cost: 25 / 16 CPU-s a deal (1+3 / ABAB), so ±0.5 in 3 CPU-h cannot be met; orchestrator's call: 1,000 deals per lineup (about 11 CPU-h, about ±0.66 per lineup, about ±0.3 on the gain over r3), the last checkpoint again at 2,000. Jobs: 356 (r3 and hour 8, stop rule, 16 threads), 357 (hour 4), then the Task 36 chain 358-366 above. The strongest-bot check and the with-search line are in `docs/study-results/selfplay-luck.md` (`expert` is the strongest measured; it ties ISMCTS-2,000; network PUCT costs 2× a move, not measured).
  - Then (a later run, once 357 has ended): copy `runs/selfplay_luck/table_expert.md` (r3, h4, h8) into `docs/study-results/selfplay-luck.md` and tick; the later hours arrive with Task 36.
  - Done 2026-11-27 (slot 1): table (r3, h4, h8) in `docs/study-results/selfplay-luck.md`. vs `expert`: r3 -1.93 / -1.99, hour 4 +3.21 / +2.59, hour 8 +3.47 / +3.05 (1+3 / ABAB, ranges about ±0.6-0.8); gain over r3 at hour 8 +3.29 [+2.87, +3.72], stop rule passed. Later hours: jobs 359, 361, 363, 365, 366 (Task 36 copies them).

- [ ] **Task 36c: each self-play checkpoint against its older self** (sonnet) `[parallel-ok]` (waiting: queue jobs 367-371)
  - Why: Brian, 2026-11-27, after hour 8 beat `expert` by +3.29: "perfect, keep training and observe the progress. It probably should match against older version of itself?".
  - What: now that the network is above `expert`, progress shows best head to head: each checkpoint (hour 12, 16, 20, 24) plays the previous 4-hour checkpoint and hour 8 (the first one above `expert`), networks only, no search, same fixed deals and both lineups (1+3, ABAB) as the expert matches, 1,000 deals (more if cheap: these need no ISMCTS). Also hour 8 vs hour 4 now. Queue them as cpu jobs after each training job, beside the expert matches (do not change jobs 357-366 or the training).
  - Done when: the queue holds the jobs (ids here); each result goes into the table in `docs/study-results/selfplay-luck.md` (new columns: vs previous, vs hour 8, with 95 % ranges) as the jobs end; the final inbox question uses both the expert and the head-to-head figures. Pushed.
  - Notes: CPUs 4-19 only; do not rebuild `_core` in the big2-workers worktree while a Task 36 job runs; no sound; local only.
  - Built 2026-11-27 (slot 1): `experiments/selfplay_luck.py h2h --hour N` (stage `h2h` in `selfplay_luck_job.ps1`): hour N against hour N-4, and for hours 16-24 also against hour 8; networks only (prior top move), 2,000 deals per lineup (first 1,000 are the expert matches' deals), seed 7, 95 % paired ranges; writes `runs/selfplay_luck/table_h2h.md`. Jobs (cpu): 367 = hour 8 vs hour 4 (runs now), 368 = hour 12 (after 358), 369 = hour 16 (after 360), 370 = hour 20 (after 362), 371 = hour 24 (after 364). Check: `jobq.ps1 status`, `runs/selfplay_luck/table_h2h.md`.
  - Then (a later run, as jobs end): copy `table_h2h.md` into `docs/study-results/selfplay-luck.md`; the final inbox question of Task 36 uses expert and head-to-head figures; tick.

- [x] **Task 36d: the same short self-play from r3 without luck removal** (sonnet) `[parallel-ok]` (checkpoint)
  - Why: Brian asked whether self-play works now "mainly due to luck removal?"; the orchestrator offered a check of the same 10-minute smoke run with luck removal off. Brian, 2026-11-27: "yes, run it, from r3 without luck removal to see if it gains the same or more".
  - What: repeat Task 36's chosen smoke setting (`runs/selfplay_luck/choice.json`: pull 0.02, 10 minutes, same seeds, same frozen-opponent mix, same network r3) with the luck removal switched off: no duplicate rotations and no leave-one-out baseline, and no oracle critic (plain PPO advantage from the game score, the way the earlier PPO runs did it; say exactly what was turned off). If cheap, also a second arm with the oracle critic but no duplicate, to split the two. Measure each end network exactly like the smoke run (vs r3, 2,000 duplicate deals per lineup, paired, 95 % ranges) and against `expert` with the 1,000 deals of Task 36b, beside the luck-removed 10-minute network measured the same way.
  - Done when: a table (arm, games played, KL at end, vs r3 1+3/ABAB, vs expert 1+3/ABAB, ranges) and a two-line verdict (does luck removal account for the gain, part of it, or none) in `docs/study-results/selfplay-luck.md` under "Task 36d"; pushed; result reported in the end message. Queue the GPU part as short jobs between Task 36's training jobs or on the cpu lane; never stop or reorder jobs 357-371.
  - Notes: CPUs 4-19 only; do not rebuild `_core` in the big2-workers worktree while a Task 36 job runs; no sound; local only.
  - Built 2026-11-27 (slot 1): `SelfPlayConfig.duplicate=False` (no rotations or copies: the same 2,048 games per iteration, each its own fresh deal; baseline = batch-mean return, mixed with the oracle critic by `oracle_weight`). Arms from r3, pull 0.02, seed 0, 10 minutes: `plain` (no duplicate, no critic: plain advantage = score minus batch mean, divided by its sd) and `oracle` (no duplicate, critic only). The luck-removed arm `dup` is Task 36's own smoke network (`runs/selfplay_luck_smoke/pull_0.02`: duplicate baseline, oracle_weight 0). Jobs: 372 (gpu, `nodup-train`, after 358; the orchestrator had job 360 also wait on 372, so both arms run right after hour-12 training and delay hour 16 by about 25 minutes), 373 (cpu, `nodup-eval`: each arm vs r3 at 2,000 deals and vs `expert` at 1,000 deals, both lineups; about 3 x 11 CPU-h). Check: `jobq.ps1 status`, `runs/selfplay_luck_nodup/table.md`.
  - Done 2026-11-27 (slot 1): table and verdict in `docs/study-results/selfplay-luck.md`; jobs 372 and 373 finished.

## Self-review

- Every acceptance criterion A1–A24 maps to a task (table above). Every task has a "Done when"
  with a command and an expected result.
- Shared files: `crates/big2/src/lib.rs` (Tasks 4, 5, 7, 13: add `pub mod` lines; run these
  helpers in sequence or in separate worktrees and merge), `python/big2/agents/__init__.py`
  (Tasks 5, 8, 14: spec parsing; same rule), `crates/pybind/src/lib.rs` (Tasks 4, 5, 7, 14, 18,
  20), `experiments/build_phase*.py` (one file per phase; tasks in the same phase add
  subcommands: sequence them or use worktrees), `docs/study-results/build-phase*.md` (same).
  `[parallel-ok]` marks tasks that do not touch those files at the same time as another marked
  task in the same phase.
- Shared resources: the GPU (one job at a time; check `nvidia-smi`), the CPU (two `stock_advisor`
  jobs run now; use 14 workers), the `big2` conda env (never `pip install -e .` while a job
  runs), `runs/` directories named per job. No ports, no databases.
- Nothing edits `crates/app-api/`, `web/`, `models/big2-distilled-duty.pt` or
  `web/package-lock.json`.

## Open questions for Brian (defaults apply; nothing blocks on them)

1. **Charlesworth's checkpoint (B2, Task 21).** His repo has no licence file. May we run his
   released weights locally as a benchmark opponent, never shipping or linking them? *Default:
   yes, local measurement only, recorded in `bot-stream\docs\decisions.md`.*
2. **Shared crates now or later (B9, proposal §8).** Create `hk-bot-core` now and build
   `shedding-core`, `eval-harness`, `teach-core` there, or build them as modules inside big2's
   crates with the boundaries named and extract them when Dou Dizhu needs them? *Default: inside
   big2 now (`plan.rs`, `harness.py`, `crates/teach`); extract when Dou Dizhu's engine lands.*
3. **Grade thresholds (Task 27).** The app's current lines are 1 / 3 / 9 points (⅓, 1, 3 cards).
   The proposal's are 0.5 / 1.5 / 4 points, then σ-scaled once σ is measured (≈ 0.5 / 2 / 5 at
   σ = 20). Which ships? *Default: the σ-scaled lines with the "unclear" grade, once Task 6
   measures σ; the app keeps its lines until the app stream switches.*
4. **Audits (B11).** Tasks 26 and 28 produce `hints-50.json` and `drills-50.json` for you to rate
   "right / irrelevant / wrong". When do you want them: after each phase, or once at the end?
   *Default: produced per phase, rated whenever you have time; not blocking.*
5. **Blind level ranking (Task 29).** The proposal's phase 4 asks a strong player to play ten
   deals per level and rank the levels blind. Will you do it, or do the measured gaps stand in?
   *Default: the measured gaps stand in; the blind test is optional.*
6. **Four-card plays as an engine flag (Task 21).** The B2 benchmark needs Charlesworth's rule.
   Keep it as a hidden `RuleSet` field (default off, not listed in `meta()`), or keep it out of
   the engine and play him under our rules only? *Default: hidden flag, off, benchmark only.*
7. **Three 10-hour probes (Task 22, 30 GPU-hours).** Run all three (ISMCTS-1000, DMC-10h, the
   final network), or only the final network plus one reference? *Default: all three, one at a
   time, each announced; it is what makes "hard to exploit" a comparison.*
8. **Loss weights in the v2 network (Task 12).** Distribution, all-seat and prior heads at
   1 : 1 : 1, or weight the actor's distribution higher? *Default: 1 : 1 : 1; Task 15's matches
   decide if a retune is needed.*

## Phase 6 - The expert solver's budget, an Elo ladder, graded mistakes and puzzles (the template for the other three games) (sonnet unless marked)

Brian, 2026-11-27 (all of it, the Why of every task below quotes the line it serves): "few things to do
for big2 [...] this will serve as a template for the other 3 games as well".

Facts this phase builds on (checked 2026-11-27):
- The self-play network (Task 36, `runs/selfplay_luck/hour_NN.pt` in this worktree, about 0.85 M
  parameters, r3's architecture with prior and cards-left heads) beats the app's `expert`
  (ISMCTS-10,000, no network, `crates/app-api/src/solver.rs`) by about +3.3 points a deal with no
  search (`docs/study-results/selfplay-luck.md`). Training ends with job 364 (hour 24, about 19:20
  on 2026-11-27); evals 365, 366, 371 follow. "The network" below is `hour_24.pt` unless Task 36's
  inbox answer picks another checkpoint; swapping it later is a file change, not a re-plan.
- Brian's playground test on 2026-11-27: the network's top move can be wrong where play-outs with
  the network find a better one (a 3-to-2 straight against a flush). So "expert solver" means
  network + search, and the study compares search settings on top of the network.
- App budgets: a bot move about 500 ms on a phone, a background `grade` 600 ms
  (hk-games `docs/architecture.md` 2a). Grade lines today: great 0.2, good 0.69, inaccuracy 2.75,
  mistake 6.86 points (hk-games D102; 0.025σ / 0.1σ / 0.25σ, σ 27.45). App levels today:
  random, easy (StructuredGreedy ε), medium (Q-net at a temperature), hard (ISMCTS-500),
  expert (ISMCTS-10,000) (`crates/app-api/src/api.rs`, `solver.rs`); hidden ratings 600 / 900 /
  1200 / 1500 / 1800 (hk-games `shared/ratings.json`, guesses, D63).
- The app's Elo is per deal, pairwise by final score against every other seat
  (hk-games `server/hkgames/elo.py` `pairwise_results`). The ladder fit uses the same rule, so a
  bot's Elo means what a player's Elo means.

Rules for this phase:
- Every task carries `(beside earlier phases)` (the project has `phaseOrder`); Task 17b's wait does not hold them.
- Match format as the plan's acceptance section: duplicate deals, seats rotated, last-card duty on,
  tiered scoring, no four-card plays (B22); paired differences with 95 % bootstrap ranges over deals.
- CPU work goes through the queue, cpu lane: `powershell -NoProfile -File
  C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <job.ps1> -ScriptArgs "..."`;
  timing runs in the `quiet` lane. Job scripts read `$env:HKQ_THREADS`, save progress at least
  every 30 minutes, resume when added again. CPUs 4-19 only; no sound; local only.
- Do not rebuild `_core` in this worktree while a Task 36 job (364-371) runs; build in your own
  worktree and point jobs at it, as Task 36 did.
- Shared files in this phase: `crates/search/src/lib.rs` (Task 38 only), `crates/app-api/` (Task 45
  only), `python/big2/teach/grading/` (created by Task 40; Tasks 42, 43, 44 add files beside it,
  never edit each other's), `docs/study-results/` (one file per task, named in the task).
- The game-agnostic part (Elo fit, loss distributions, line derivation, accuracy mapping, puzzle
  selection) lives in `python/big2/teach/grading/` and imports nothing from `big2`: it reads one
  file shape, `decisions.jsonl` (below), so the other three games only write that shape.

`decisions.jsonl` (one line per decision, written by Task 42, read by 43 and 44): `game`, `deal_id`,
`seat`, `band` (rung name), `rung_elo`, `position` (engine JSON), `legal_count`, `moves`
(`[{action, ev, se, prior}]`, every evaluated move, best first), `played`, `loss` (EV(best) -
EV(played), points), `loss_se`, `forced` (one legal move), `flat` (every evaluated move within the
noise band), `reference` (settings string), `seed`.

- [x] **Task 37: the self-play network in Rust** (sonnet) `[parallel-ok]` (beside earlier phases)
  - Why: Brian, 2026-11-27: "find out the optimal search depth that should be done for the expert solver" (the search runs in Rust and WebAssembly, so the network must too).
  - What: export `runs/selfplay_luck/hour_24.pt` with `big2 rl export` to `.pknn` v2 (prior and cards-left heads). If the self-play trainer's checkpoint keys differ from r3's, add the mapping in the exporter, not a copy of it.
  - Files: `python/big2/rl/export.py` (only if needed), `python/tests/test_export.py` (a case for a self-play checkpoint), `runs/p6_net/selfplay-h24.pknn` (gitignored), `docs/study-results/expert-search-budget.md` §0 (agreement and speed lines).
  - Done when: `python -m pytest python/tests/test_export.py -q` passes; on 1,000 positions from fixed seeds the Rust prior's top move equals PyTorch's on at least 99 % and the value outputs differ by at most 0.01 RMS; one forward pass timed native and in Node WebAssembly (median and 95th percentile ms) and written in §0.
  - Notes: ~0.5 CPU-h. The file is not shipped by this task.

- [x] **Task 38: search on top of the network (play-outs and PUCT), one settings struct** (opus) `[parallel-ok]` (beside earlier phases)
  - Why: Brian, 2026-11-27: "find out the optimal search depth that should be done for the expert solver. weigh the time taken to make decisions, as well as the diminishing return for increasing the depth".
  - What: in `crates/search`, a network play-out search: at the root take the top `k` legal moves by the network's prior (plus any forced-in move, e.g. the move being graded), sample `n` deals of the hidden hands consistent with what the seat knows (the existing determinizer, rules mask on), and for each deal play every candidate followed by `d` plies of every seat playing the network's top move; then score the position with the cards-left value head (`d = full` plays to the end and uses the real score). The same deals for every candidate (paired), so the result carries each candidate's mean, its standard error and the paired difference to the best. Settings struct `NetSearch { k, n, depth, value_mix, threads, seed, time_limit }`; `threads = 1` never touches rayon (plan rule). PUCT with this network (`puct.rs`, existing) stays as the second family, reachable with the same `.pknn`. Python binding through `big2._core` for matches; a CLI `big2 search --net <pknn> --k --n --depth` for one position.
  - Files: `crates/search/src/netsearch.rs` (new), `crates/search/src/lib.rs` (one `pub mod` line), `crates/pybind/src/lib.rs` (binding), `python/big2/agents/` (an agent spec `netsearch:k=..,n=..,depth=..`), tests beside them.
  - Done when: `cargo test -p big2-search netsearch` passes, including: with `k = 1` it plays the network's top move; with a fixed seed two runs give the same result; `threads = 1` gives the same result as before any rayon path; paired standard errors shrink about as `1/sqrt(n)` on a fixed position. If the orchestrator has put Brian's 2026-11-27 playground position (3-to-2 straight against a flush) in this task as position JSON, it is a test: the network's top move is the flush play and `k >= 2, n >= 64, depth = full` picks the move Brian's play-outs found. Otherwise say so in one line in the hand-back.
  - Notes: ~1 CPU-h of tests. No app change here. Orchestrator's call (2026-11-27): `value_mix` = weight of the value head at the depth cutoff, the rest the real score of playing on with the network (mirrors PUCT's `leaf_network_weight`), default 1, ignored at `depth = full`; Task 39 should sweep `value_mix < 1` too (the value head undervalued some moves by about 2 points in Brian's deal 23, `docs/solver.md` §6.6).
  - Done (slot 1, 2026-11-27): `crates/search/src/netsearch.rs`, `big2._core.net_search`, agent spec `netsearch:net=…,k=..,n=..,depth=..|full`, `big2 search`. `cargo test -p big2-search netsearch` 8 passed (k = 1 top move; seed reproducible; `threads = 1` matches the recorded result and 2, 3, 8 threads give the same numbers; 16× deals shrink the standard and paired errors 3-5.3×; depth 0 = value head; `value_mix = 0` = full play-out); Python 426 passed, 6 skipped. Brian's playground position (3-to-2 straight against a flush) was not given as position JSON here, so that test is not in.

- [ ] **Task 39: the depth and budget study: where the curve bends** (fable: picks the knee; that verdict sets the app's expert and every grade) `[parallel-ok]` (checkpoint) (beside earlier phases) needs: Task 37, Task 38
  - Why: Brian, 2026-11-27: "find out the optimal search depth that should be done for the expert solver. weigh the time taken to make decisions, as well as the diminishing return for increasing the depth".
  - What: every setting plays the raw network (top prior move, no search) on the same fixed deals, ABAB lineup (two seats each) and 1+3, duplicate seats; points a deal paired with 95 % range; and the time a move. Grid (screen): play-outs `k` in {2, 3, 5, 8}, `n` in {16, 32, 64, 128, 256}, `depth` in {0 (value head only), 4, 8, 16 plies, full}, not the full product: start at `k 3, n 64, depth full` and move one knob at a time (about 15 settings); PUCT with the network at {200, 500, 1,000, 2,000} simulations. Screen at 300 deals; the 5 best per unit of time go to 2,000 deals. Time: native one-thread ms per move (median, 95th percentile) on 300 positions from real games, and Node WebAssembly for the finalists (quiet lane), converted to phone time with the factor the app's own phone timings give (hk-games `docs/architecture.md`, the app-api README); if none is recorded, use 3× Node and say it is assumed. Also each finalist against the app's `expert` (1,000 deals, both lineups) so the gain is on the benchmark Brian trusts.
  - Knee rule (orchestrator's default, the task may argue for another in its report): the cheapest setting whose gain over the raw network is within 0.2 points a deal of the best measured gain (or whose 95 % range overlaps the best's), among settings whose phone 95th percentile is at most 500 ms for a bot move. Also report the cheapest setting within 600 ms for `grade`, and the gain per doubling of time so the bend is visible.
  - Files: `experiments/expert_search_budget.py` (driver, subcommands `screen`, `confirm`, `timing`), `experiments/expert_search_budget_job.ps1`, `docs/study-results/expert-search-budget.md` (method, the table, a curve of gain against ms per move as a PNG beside it, the knee and why), `runs/p6_budget/` (gitignored).
  - Done when: the table in the results file lists every setting with deals, gain over the raw network with 95 % range, ms per move native (median, p95) and, for finalists, Node and phone-estimate; the finalists' rows also show points against `expert`; the knee is named with the rule's numbers; an inbox question to Brian: "expert = <setting> (gain, ms); OK?" with the curve. The queue held the jobs (ids written here).
  - Notes: ~65 CPU-h (screen ~20, confirm ~45), cpu lane behind evals 365/366/371. A raw network move costs about a millisecond, so the cost is the searcher's seat.

- [ ] **Task 40: ladder rungs below the expert and their Elo** (fable: the Elo fit from four-player deals and the rung calibration) `[parallel-ok]` (beside earlier phases) needs: Task 37
  - Why: Brian, 2026-11-27: "re-adjust our options for other bots and lower difficulties so they span a reasonable distribution in the elo-space".
  - What: candidate rungs: `random`; today's `easy`, `medium`, `hard` exactly as the app runs them (so the old ladder maps onto the new scale); the network's prior sampled at temperature τ (softmax over legal moves) for several τ; the raw network (top move). Step 1, span: play `random`, `easy`, `network τ=1`, raw network and today's `expert` against each other (ABAB, 500 duplicate deals per pair) and fit; report the whole span in Elo. Step 2, rungs: choose τ values by bisection so the rungs sit at equal Elo steps over the span, aiming at 150-250 Elo apart (fewer, wider rungs if the span is narrow; see open question 1 in the notes). Step 3, fit: every rung plays the rung one and two above it (ABAB, duplicate deals, seats rotated); each deal gives the pairwise results of the app's rule (higher final score = 1, equal = 0.5, against every seat of the other rung); maximum-likelihood Bradley-Terry with ties counted as half, on the Elo scale (400 log10), anchored so the rung that becomes the app's `medium` is 1200 (D98; until Task 41 picks it, today's `medium` = 1200); 95 % range per rung by bootstrap over deals. Size each pairing to a 95 % range of at most ±25 Elo on each adjacent gap (measure 200 deals first, then size).
  - Files: `python/big2/teach/grading/elo.py` (game-agnostic: reads pairwise records, fits, bootstraps), `python/big2/teach/grading/__init__.py`, `python/tests/test_grading_elo.py` (a synthetic ladder with known Elo is recovered within its range), `experiments/elo_ladder.py`, `experiments/elo_ladder_job.ps1`, `docs/study-results/elo-ladder.md` (span, rungs, settings, fitted Elo with ranges, deals per pairing, thread-hours), `runs/p6_ladder/ladder.json`.
  - Done when: the test passes; `ladder.json` holds every rung's setting, Elo and 95 % range, each adjacent gap's range at most ±25 Elo (or the width reached and why); the rungs order strictly; the old app levels' Elo on the new scale are in the results file.
  - Notes: ~15 CPU-h (network-only rungs cost milliseconds a move; `hard` and today's `expert` dominate). The expert rung comes in Task 41.

- [ ] **Task 41: the expert rung, the final ladder and the app's levels** (sonnet) (checkpoint) (beside earlier phases) needs: Task 39, Task 40
  - Why: Brian, 2026-11-27: "re-adjust our options for other bots and lower difficulties so they span a reasonable distribution in the elo-space".
  - What: add the knee setting from Task 39 (network + search) as the top rung: it plays the two rungs below it with the same method and the fit is redone over all rungs. Then propose the app's five levels (random, easy, medium, hard, expert) as five rungs at near-equal Elo steps, `expert` = the knee setting, `medium` anchored at 1200; list the rungs not used by the app (they stay as mining bands).
  - Files: `experiments/elo_ladder.py` (subcommand `top`), `docs/study-results/elo-ladder.md` (final table; the five proposed app levels with setting, Elo, range, ms a move on the phone estimate), `runs/p6_ladder/ladder.json` (final).
  - Done when: the final table holds every rung including the expert rung, each adjacent gap with its range; an inbox question to Brian: "app levels = <five rungs> at <Elo>; OK?" The queue held the jobs (ids here).
  - Notes: ~10 CPU-h. Hand-over (text for the hk-games plan, Task E4): the final `ladder.json` path and the five levels.

- [ ] **Task 42a: mistake-mining pilot (two bands, 100 deals each)** (sonnet) `[parallel-ok]` (beside earlier phases) needs: Task 38, Task 40
  - Why: Brian, 2026-11-27: "have bots at various elos play against each others to find out the mistakes and their EV distributions" and "show me a few tens of puzzles for my review".
  - What: the Task 42 pipeline at small size, with a provisional reference that is already the network plus search (`netsearch k=8, n=256, depth=full`, offline, no time limit, standing in for Task 39's deeper setting) and a provisional in-game setting (`k=3, n=64, depth=full`) scored beside it, so the disagreement share is measured from the start and four bands from Task 40's rungs (lowest non-random, two middle, raw network): 100 same-rung deals per band, every non-forced decision labelled into `decisions.jsonl`; then the 60 strongest puzzle candidates (rules in Task 44, provisional lines D102) spread over the four bands, with theme tags from the existing tagger (`big2-teach` theme code used by plan Task 5). Measure the cost per labelled decision to size Task 42.
  - Files: `experiments/mine_mistakes.py` (shared with Task 42), `experiments/mine_mistakes_job.ps1`, `runs/p6_mistakes_pilot/decisions.jsonl`, `runs/p6_mistakes_pilot/candidates.json` (Drill shape plus `band`, `rung_elo`, the bot's move and its loss), `docs/study-results/mistakes.md` §0 (pilot counts, cost per decision, share forced and flat).
  - Done when: `candidates.json` has at least 40 puzzles, at least 8 per band, each with its theme, the bot's wrong move and every answer's loss; `decisions.jsonl` validates against the shape above (a pytest in `python/tests/test_mine_mistakes.py`); §0 has the cost per decision. Hand-over (text for hk-games plan Task E1): the path of `candidates.json`.
  - Notes: ~3 CPU-h.

- [ ] **Task 42: mistakes and their EV distributions per Elo band** (opus) (beside earlier phases) needs: Task 39, Task 41, Task 42a
  - Why: Brian, 2026-11-27: "have bots at various elos play against each others to find out the mistakes and their EV distributions, to re-adjust what moves should qualify as good or inaccurate or bad or etc.".
  - What: for every rung of the final ladder except `random`, same-rung tables (four seats of that rung, so the positions are the ones that band's games produce), 300 deals per band (more if Task 42a's cost allows within the budget below). Every decision of every seat is labelled against the strongest expert, which is the network plus search (Brian, 2026-11-27: "you have not specified but I assume that strongest expert includes search (with our decided optimal depth)"). The reference is the deeper of two settings: the knee setting of Task 39 (the in-game budget) is the floor, and offline the reference is the largest setting Task 39 measured (most candidates, most deals, full depth; time does not matter offline), unless that setting measured no stronger than the knee, in which case it is the knee with 4× its deals. Every decision is also scored at the knee setting, and the results file reports how often the two disagree on the best move, overall and per band, and the mean loss of the knee's pick under the deeper reference. EV and standard error of each legal move with prior at least 1 % plus the top 8 and the move played; `forced` if one legal move; `flat` if every evaluated move's paired loss is inside the noise band (95 % range of the best's lead covers 0 for all) or below 0.2 points. Per band: the loss distribution over non-forced, non-flat decisions (mean, median, 75/90/97/99th percentiles, share of losses above each D102 line), the share forced and flat. Repeatability: 5 % of decisions labelled again with a new seed; report the share whose word (D102 lines) is unchanged. The thresholds (Task 43) and the puzzles (Task 44) use the deeper reference's losses only.
  - Files: `experiments/mine_mistakes.py`, `experiments/mine_mistakes_job.ps1`, `python/big2/teach/grading/losses.py` (game-agnostic distributions from `decisions.jsonl`), `python/tests/test_grading_losses.py`, `runs/p6_mistakes/decisions-<band>.jsonl`, `docs/study-results/mistakes.md` (per-band table and a histogram PNG per band).
  - Done when: `decisions-<band>.jsonl` for every band, each at least 300 deals, every line carrying both the deeper reference's moves and the knee's pick (`knee_best`, `knee_loss`); the per-band table and histograms in the results file; the repeatability share and the knee-vs-deeper disagreement share (overall and per band) stated; tests pass; queue ids written here.
  - Notes: ~40 CPU-h (the deeper reference at roughly 2-4× the knee's cost a decision, about 48 decisions a deal, about a third forced and skipped by the search); if Task 42a's cost says the budget is exceeded, cut deals per band to 200 before cutting the reference.

- [ ] **Task 43: grade lines, the trivial flag and a performance measure from the distributions** (fable: derives the lines and the accuracy and Elo mappings; an unchecked number here changes every grade players see) (checkpoint) (beside earlier phases) needs: Task 42
  - Why: Brian, 2026-11-27: "re-adjust what moves should qualify as good or inaccurate or bad or etc." and "we should have a better representation of the performance rather than using how often the best/good is played, as a lot of the moves are trivial (e.g. you can only pass in certain spot)".
  - What, defaults the task may improve with reasons:
    1. Trivial: a decision is `forced` (one legal move, e.g. only pass) or `flat` (Task 42's rule). Trivial decisions get no grade word in the summary and count in no score.
    2. Lines (points lost, tiered, from the deeper reference's losses, never the knee's): great = the reference's noise (95 % half-width of a paired loss at the app budget, median over decisions), at least 0.2; good = the loss the top rung exceeds on 10 % of its non-trivial decisions; inaccuracy = the loss the top rung exceeds on 3 %; mistake = the loss the app's `medium` rung exceeds on 3 %; above that blunder. Lines must rise, each at least 1.5× the one below; compare with D102 (0.2 / 0.69 / 2.75 / 6.86) and with 0.25σ for "decides the deal". Table: each band's share of each word under D102 and under the new lines.
    3. Performance: average points lost per non-trivial decision (APL, like chess's average centipawn loss); a 0-100 accuracy per game = the mean over non-trivial decisions of `100 * exp(-loss / s)`, `s` set so a loss at the inaccuracy line scores 50; and an Elo estimate from APL, fitted across the rungs (Elo against log APL, with the per-game spread so the app knows how many games an estimate needs to reach ±100). Show each rung's mean accuracy and how well accuracy and APL order the rungs compared with today's best/good share (Spearman over rungs and over single games).
  - Files: `python/big2/teach/grading/lines.py`, `python/big2/teach/grading/accuracy.py` (game-agnostic), `python/tests/test_grading_lines.py`, `docs/study-results/grading-template.md` (method, tables, proposal), `runs/p6_grading/proposal.json` (`lines`, `trivial` rule, `accuracy_s`, `elo_from_apl` coefficients, games needed for ±100).
  - Done when: tests pass (lines are monotone; a synthetic set with known percentiles gives them back); the results file has the proposed lines next to D102 with per-band word shares, the per-rung accuracy and APL, and the ordering comparison; an inbox question to Brian with the proposal (lines, trivial rule, accuracy formula, whether to show the Elo estimate). Hand-over (text for hk-games plan Tasks E2, E3): `proposal.json` path and Brian's answer.
  - Notes: ~2 CPU-h (a small held-out set of games per rung for the per-game spread).

- [ ] **Task 44: puzzles from the mistakes, by Elo band and theme** (sonnet) `[parallel-ok]` (beside earlier phases) needs: Task 42
  - Why: Brian, 2026-11-27: "the mistakes should be used to enrich the puzzle reservoir, as well as being separated in elos given how different bots at different elos perform".
  - What: from `decisions-<band>.jsonl`: keep a decision when it is not trivial, the band's bot lost at least the inaccuracy line (D102 until Brian answers Task 43, then his lines), the best move is clear (second-best loses at least the inaccuracy line and the best's lead has a 95 % range above 0), and a deeper solve (2 new seeds, 4× deals) agrees on the best. Tag: `band` (where the mistake happened), theme by the existing tagger (one key per puzzle, `skill.b2_mixed` when none fits), and a puzzle Elo = the Elo at which a rung picks a good-or-better answer half the time (logistic fit over the rungs: exact probabilities for the temperature rungs from the softmax, 16 samples for search rungs), so puzzles match players on the play-Elo scale (the app maps it to its puzzle rating, hk-games Task E5). Dedupe on a position key (own hand, cards on the table and who played them, cards already played, cards left per seat, seat to move) within the new set and against `drills/big2-puzzles.json`. Also re-solve the existing 2,152 pool puzzles with the reference: keep, re-answer (best changed), or drop (no longer clear), with counts.
  - Files: `python/big2/teach/grading/puzzles.py` (selection and dedupe, game-agnostic), `python/big2/teach/puzzle_elo.py` (big2 rung probabilities), `python/tests/test_grading_puzzles.py`, `experiments/mistake_puzzles.py`, `experiments/mistake_puzzles_job.ps1`, `drills/big2-puzzles-v2.json` (Drill shape plus `band`, `elo`, `origin: "bot_mistake" | "pool"`, the bot's move), `docs/study-results/mistakes.md` §3 (counts per band and theme, kept/re-answered/dropped of the old pool).
  - Done when: tests pass; `drills/big2-puzzles-v2.json` validates with the plan Task 5 shape test (`pytest python/tests/test_drills.py`) plus the new fields; at least 2,000 puzzles in all, at least 150 per band, every theme at least 30 (or the shortfall per theme stated); no two share a position key; counts in §3. Hand-over (text for hk-games plan Task E5): the file path and counts.
  - Notes: ~10 CPU-h (deeper solve of candidates ~7, old pool ~3).

- [ ] **Task 45: the app's engine: network expert, one reference for hint, review and grade, the trivial flag** (opus) (beside earlier phases) needs: Task 39, Task 41, Task 43
  - Why: Brian, 2026-11-27: "find out the optimal search depth that should be done for the expert solver" and "a lot of the moves are trivial (e.g. you can only pass in certain spot)". As mahjong did (hk-games D139: hint, review and `expert` agree).
  - What: in `crates/app-api` (the app stream's crate; as Task 35 did, with a bot-notes entry): the network `.pknn` embedded beside the Q-net (f16 if the format allows, say the size); `expert` = the knee setting within `budget_ms`; the other levels = the rungs Brian approved in Task 41; `hint`, `review`, `grade` rank moves with the same network search (budget-capped), keep the current reasons (Task 35) and add per decision `trivial: "forced" | "flat" | null` and the paired `loss_se`. Existing keys keep their meaning and shape; new fields are added beside them. `bot_levels` names unchanged.
  - Files: `crates/app-api/src/solver.rs`, `crates/app-api/src/api.rs`, `crates/app-api/README.md` (new fields, budgets, the levels' settings), their tests, `models/big2-selfplay.pknn` (committed if under the repo's size rule, else say where), `docs/bot-notes-grading-entry.md` (the entry for hk-games bot-notes; the orchestrator pastes it).
  - Done when: `cargo test --workspace --release` passes; Python tests pass; the wasm web build succeeds and in Node WebAssembly `bot_action` at `expert` takes at most the phone budget's Node equivalent (Task 39's factor) at the 95th percentile over 300 positions, `grade` at most 600 ms on the same scale; a test shows a one-legal-move decision returns `trivial: "forced"` and a grade of `best`; the notes entry exists. Hand-over (text for hk-games plan Task E2): the commit to pin.
  - Notes: ~1 CPU-h plus builds.

- [ ] **Task 46: end-of-phase review** (opus) (beside earlier phases) needs: Task 44, Task 45
  - Why: write-plan: the pre-release-reviewer at the end of every run.
  - What: pre-release-reviewer on this phase's commit range plus the paths it touches; fix plain bugs; judgment calls to `docs/open-items.md` (by report).
  - Done when: the review's findings are fixed or recorded; full tests and lint pass (the plan's table).
