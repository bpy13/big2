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

- [ ] **Task 15: Phase 2 matches** **[long: ~4 h CPU, headless]** (after 14)
  (**results in, bars failed; fallback running.** Job finished 2026-10-29 19:47; tables and
  verdicts in `build-phase2.md` §1–§4. A9 fails: PUCT-2000 vs ISMCTS-1000 at 1,000 deals −0.12
  [−0.78, +0.56] in `1+3`, −1.78 [−2.42, −1.13] in `ABAB`; at 0.5 s per move −1.81 and −2.44.
  A10 fails: distribution head vs Task 7 qnet −1.34 (`ABAB`, tiered), −0.21 (`ABAB`, win_loss).
  `c_puct` 2.5 best, all negative. Fallback (proposal §10 row 2): calibration check and
  `leaf = λ·network + (1−λ)·rollout` at 300 deals, `build-phase2.md` §6.)
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

- [ ] **Task 16: Ladder v1, WebAssembly timing, bot-notes for phase 2** (after 15)
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

- [ ] **Task 17: Expert-iteration rounds 1–3** **[long: per round ~6 h CPU labelling + 2–4 GPU-h + 2 h CPU eval; headless]**
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

- [ ] **Task 18: Hard-constraint belief from the rules** `[parallel-ok]`
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

- [ ] **Task 19: Learned belief per round, kept only if it pays** `(fable)` **[GPU: ~2 h per round; CPU 1 h]**
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

- [ ] **Task 20: Information-set exact endgame with forced proofs** `(fable)` `[parallel-ok]`
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

- [ ] **Task 21: B2 — Charlesworth's network as a local opponent** **[~1 h CPU]** (needs open question 1)
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

- [ ] **Task 22: B3b — the best-response probe** **[GPU: 3 × 10 h, three separate headless jobs]** (after 17)
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

- [ ] **Task 23: Cross-play of two seeds** **[GPU: 2–4 h; CPU 2 h]** (after 17)
  - Files: `experiments/expert_iteration.py round 1 --seed 1 --tag seedB` (reuse); `build-phase3.md` §7.
  - Done when: §7 records PUCT-2000(seed 0) vs PUCT-2000(seed 1), 1,000 deals, both lineups; A18:
    the interval covers 0. If it does not, record it as a risk (proposal §10 row 4).

- [ ] **Task 24: Final 10,000-deal matches** **[long: ~9 h CPU each, headless]** (after 17, 21)
  - Files: `experiments/build_phase3.py final`; `build-phase3.md` §8; `runs/final_matches/run.ps1`.
  - Done when: §8 holds PUCT-2000(final) vs ISMCTS-1000 and vs Charlesworth (if run), 10,000
    deals each, both lineups, mean ± 95 % and win rate (A19). Check: `progress.json`. Resume: the
    script resumes from the last saved deal index in `progress.json`.

- [ ] **Task 25: Phase 3 docs and bot-notes** (after 17–24)
  - Files: `docs/study-results/build-phase3.md` summary table; `docs/solver.md` §6 (PUCT, rules
    mask, information-set endgame); `docs/rl.md` (v2 net, expert iteration loop, belief net);
    `docs/architecture.md` (new modules); `README.md` findings table; hk-games bot-notes entry.
  - Done when: every number in the summary table has a source `runs/<job>/results.json` named
    beside it; bot-notes lists the shipped model file name and size, the new public items
    (`RulesMask`, `NetSampler`, `infoset_endgame`), what `quality` would carry (`{"samples": n}` or
    `{"exact": true}`) once they accept B20, and the B2/B3 results in three lines.

## Phase 4 – Teaching depth (opus; Task 27 fable)

The teaching code lives in a new crate we own, `crates/teach` (package `big2-teach`), so the app
stream can call it from `crates/app-api` without us touching their crate.

- [ ] **Task 26: `big2-teach`: reason keys from the move table** `[parallel-ok]`
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

- [ ] **Task 27: Paired grader, σ thresholds with "unclear", luck ledger** `(fable)` `[parallel-ok]`
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

- [ ] **Task 28: Drill miner, seed ratings, mistake clustering** **[long: ~3 h CPU]** (after 26, 27)
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

- [ ] **Task 29: Error model and ladder v2** **[long: ~45 min CPU]** (after 26)
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

- [ ] **Task 30: Convention-gap check (scaffold)** `[parallel-ok]`
  - Files: create `python/big2/teach/convention_gap.py` (reads a game log in the hk-games review
    shape: list of `{position, played, best, loss, rating}`; groups disagreements by reason-key set
    and by player rating band; prints the sets whose disagreement rate does not fall with rating,
    tagged "convention"); test on a synthetic log; CLI `big2 teach convention-gap <log>`.
  - Done when: `pytest python/tests/test_convention_gap.py` passes (a synthetic log with one
    rating-independent disagreement and one rating-dependent one is split correctly).
  - Notes: real logs exist only after release with consent (B13); the script waits for them.

- [ ] **Task 31: Phase 4 docs, handover, bot-notes** (after 26–30)
  - Files: `docs/study-results/build-phase4.md` summary; `crates/teach/README.md` (keys, grades,
    thresholds, ledger, drills); `docs/architecture.md`, `README.md`, `docs/ui.md` if the web UI
    changed (it should not); hk-games bot-notes entry.
  - Done when: bot-notes lists: `big2-teach` and how to call `reasons_for`, `grade_paired`,
    `luck_ledger`; the full key table; the two threshold tables and which one Brian chose (or
    "pending, open question 3"); `drills/big2.json` as the handover (≥ 300 drills, shape checked)
    and where it lives; the ladder v2 constructors to switch the five levels to; the three audit
    files for Brian. Full tests and lint pass (the table at the top, minus the app crate's fmt).

---

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
