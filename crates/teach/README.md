# big2-teach

The teaching layer over the Big Two engine: reason keys from the move table, the paired grader
with σ lines and an `unclear` grade, and the luck ledger. Owned by the bot stream; the app
stream calls it from `crates/app-api` (add `big2-teach.workspace = true` to their
`Cargo.toml`). Everything runs on the calling thread (no rayon), so it works in WebAssembly.
All random draws use `ChaCha8Rng`, so a seed gives the same result on the phone and on a PC.

Results: `docs/study-results/build-phase4.md`.

## Calls

```rust
use big2_teach::{evaluate_paired, grade_from_table, grade_paired_table, luck_ledger, reasons, PairedConfig};

// One move table: every legal move on the same 300 sampled deals.
let cfg = PairedConfig { deals: 300, seed, max_deals: 1200, ..PairedConfig::default() };
let table = evaluate_paired(&state, &cfg);
let hint = table.best().action;
let why: Vec<Reason> = reasons(&state, &table, hint);          // [{key, args}], serde-serialisable

// Grading a played move (doubles the deals while `unclear`, up to max_deals):
let (graded, table) = grade_paired_table(&state, played, &cfg);  // loss, interval, grade, range, seed
let why = reasons(&state, table.as_ref().unwrap(), played);

// A finished deal: start position, every action, the seat to explain.
let ledger = luck_ledger(&start, &actions, seat, &cfg, &big2_big2::bots::Greedy { epsilon: 0.1 })?;
```

`reasons_for(state, table, candidates, chosen)` takes the plan's candidates
(`big2_big2::plan::candidates`) when the caller already has them. `MoveTable::from_analysis`
builds a table from any solver's `Analysis` (ISMCTS, PUCT); keys that need every seat's values
(`reason.helps_seat`) or paired deals then stay silent or use plain intervals.

## Reason keys

Keys and arguments only (hk-games D42). These are additions; the app's existing keys
(`reason.keeps_control`, `reason.cards_left`, …) stay as they are. *Chosen* is the move being
explained, *other* the best other row of the table, *unseen* every card neither played nor in
the actor's hand. Cards are card ids (`"9s"`); a combination is
`{"cards": ["9d", "9s"], "combo": "combo.pair"}`; seats are absolute seat indices; points are
rounded to 0.01, shares to 0.001.

| Key | Arguments | Fires when |
|---|---|---|
| `reason.plan.keeps` | `combo` | chosen keeps a fewest-plays plan, and a lower-rated play breaks a group of the best plan that chosen leaves whole; `combo` is that group (for the best-rated such play) |
| `reason.plan.breaks` | `combo`, `cost` | chosen breaks a group of the best plan; `cost` = extra plays to go out it causes (≥ 1) |
| `reason.control.pass_to_keep` | `card` | chosen is a pass rated above every play; `card` = the highest card of the best-rated play |
| `reason.value.score_vs_win` | `score_delta`, `win_delta` | another move has a higher win rate but a lower mean score than chosen (`win_delta` < 0) |
| `reason.tempo.plays_to_out` | `mine`, `theirs` | chosen leaves the fewest plays to go out of all legal plays, and some play leaves more; `theirs` = expected plays per opponent over the actor's sampled deals, seat order after the actor |
| `reason.control.safe_single` | `card` | chosen is a single, cards stay in hand, and no unseen card beats it |
| `reason.control.highest_out` | `card` | the actor holds the highest card not yet played, chosen keeps it, and some legal play spends it |
| `reason.value.gap` | `points` | two or more legal moves; `points` = chosen − other (positive: margin over the next best; negative: loss to the best) |
| `reason.counterfactual.outcomes` | `best`, `played`, `deals` | chosen is not the best row; `best`, `played` = `{"mean", "win"}` on the same `deals` (reviews) |
| `reason.helps_seat` | `seat`, `delta` | chosen gives an opponent ≥ 1 point more than other does (the opponent who gains most); in a paired table the gain's 95 % interval must exclude 0 |
| `reason.counterfactual.forced` | `result`, `winner`, `plays` | the exact endgame proved the best row's winner in every deal of the hidden cards and chosen is the best row; `result` is `"win"` or `"lose"` |

Each rule is unit-tested twice (`cargo test -p big2-teach reasons`): a position where it must
fire and one where it must not.

## Grades

`loss` = mean over the paired deals of (best − played), `interval` = its 95 % half-width. The
best row's own interval is that of its margin over the runner-up. With `lo`, `hi` the interval's
ends:

- `best`: `lo ≤ 0` and `hi` under the `good` line (cannot be told apart from the best move);
- `good`, `inaccuracy`, `mistake`, `blunder`: both ends in the same band;
- `unclear`: the ends fall in different bands; `range` names the two grades.

Two tables of lines (points, `tiered` scoring). Which one ships is open question 3 of the plan
(pending Brian; default: the σ lines, the app keeps its own until the app stream switches).

| Lines | good below | inaccuracy below | mistake below | blunder from | Constructor |
|---|---|---|---|---|---|
| σ-scaled, σ = 27.45 (Task 6) | 0.69 | 2.75 | 6.86 | 6.86 | `Thresholds::default()`, `Thresholds::from_sigma(σ)` |
| The app's current lines (⅓, 1, 3 cards) | 1 | 3 | 9 | 9 | `Thresholds::APP` |

σ is the standard deviation of one seat's score in one deal of ISMCTS self-play under
`tiered` scoring; other scoring rules need their own σ.

Repeatability (`build-phase4.md` §1): 90.2 % of decisions get the same grade word with a new
seed at 300 → 1,200 deals, 97.6 % a compatible one; the plan's bar is 95 % (open item).

## Luck ledger

`score = deal_strength + opponents_errors − own_errors + residual`, exact by construction.

- `deal_strength`: the seat's expected score at the start from its own view, every seat
  played by the rollout policy (the average over deals is 0).
- `opponents_errors`: at every opponent decision, the seat's mean score after the move played
  minus after that opponent's best move (the opponent's own table). Positive helped the seat.
- `own_errors`: the sum of the seat's losses.
- `residual`: the rest (how the hidden cards lay, play beyond what the tables expected, noise).

`Ledger::steps` lists each decision's contribution. `Ledger::from_tables` builds the ledger
from tables a review has already computed.

## Ladder v2

`big2_teach::ladder::LadderBot` (implements `Policy<State>`; `choose_explained` also returns the
mistake that fired, whose key a review can show):

| Level | Constructor | Mechanism |
|---|---|---|
| `random` | `LadderBot::random()` | a random legal move |
| `easy` | `LadderBot::easy()` | `StructuredGreedy` ε 0.1 + `ErrorModel` at 0.5 / 0.5 / 0.5 |
| `medium` | `LadderBot::medium(net)` | the distilled Q-network (`models/big2-qnet-duty.pknn`) sampled at temperature 1.0 + `ErrorModel` at 0.15 each |
| `hard` | `LadderBot::hard()` | ISMCTS, 2,000 simulations |
| `expert` | `LadderBot::expert()` | ISMCTS, 10,000 simulations + the exact information-set endgame |

The error model (`big2_big2::bots::ErrorModel`) has three named mistakes, each a key:
`mistake.hold_two_too_long` (keeps a 2 the policy would play), `mistake.break_pair_for_single`
(beats a single with a card from a pair), `mistake.miss_safe_single` (does not play a single
nobody can beat). The rates are first values; Task 29's 500-deal round tunes them
(`build-phase4.md` §3).

## Command-line runner

`cargo run --release -p big2-teach -- audit|regrade|ladder|mine --out PATH` (see
`src/main.rs`); used by `experiments/build_phase4.py` and `big2 teach mine-drills`.
