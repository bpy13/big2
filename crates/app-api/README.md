# big2-app-api

The app-facing adapter over the Big Two engine. It implements the calls of hk-games
`docs/engine-interface.md` (contract v0) as JSON strings in and out, and builds to
WebAssembly with `wasm-bindgen`. It holds no game logic of its own: rules, move
generation, bots and solvers come from `crates/big2` and `crates/search`.

Every response carries `"api": 0`. Every error is
`{"api": 0, "error": {"code": "error.*", "detail": "<developer text>"}}`.
The engine returns keys and numbers only, never display text: cards are ids such as `"3d"`,
combinations are `combo.*`, actions `action.*`, reasons `reason.*`, events `event.*`.

## Optional contract fields not emitted yet

`quality` (on `Hint` and `ReviewItem`), `interval` (on `ReviewItem` and `Grade`) and the `unclear`
grade are optional fields of the contract (`docs/engine-interface.md` in hk-games). This crate
does not emit them yet, so the app treats a missing `quality` or `interval` as "no information"
and never sees `unclear`. `API_VERSION` (in `shapes.rs`, now `0`) is the `"api"` number in every response.

## Build and test

```powershell
cargo test -p big2-app-api                       # the JSON calls end to end
cargo test -p big2-app-api --release -- --ignored --nocapture timing   # the timings below
cargo clippy -p big2-app-api --all-targets -- -D warnings
wasm-pack build crates/app-api --target web --release                   # -> crates/app-api/pkg (gitignored)
wasm-pack build crates/app-api --target nodejs --release --out-dir <dir>  # for a Node smoke run
```

The WebAssembly class is `Big2Api` (`new Big2Api()`); its methods are the calls below with
the same names. Strings in, strings out; `seat`, `budget_ms` and `iterations` are plain
numbers. Native Rust uses `big2_app_api::Api` with the same methods.

## Calls

| Call | Input | Output |
|---|---|---|
| `meta()` | — | `GameMeta` |
| `new_game(config)` | `GameConfig` | `{game_id}` |
| `view(game_id, seat)` | — | `SeatView` |
| `legal_actions(game_id)` | — | `{actions: [Action], seat}` for the seat to act (empty once the deal is over) |
| `apply(game_id, action)` | `Action` | `{events: [Event], view: SeatView}` (the view is for the seat that acted) |
| `bot_action(game_id, level, budget_ms)` | level from `bot_levels` | `Action` plus `budget_used_ms` (not applied: call `apply` with it) |
| `hint(game_id, seat, budget_ms)` | seat must be to move | `Hint` |
| `hint_with_iterations(game_id, seat, budget_ms, iterations)` | as `hint` | `Hint` with an explicit simulation count (measurements, drills) |
| `review(game_id, seat, budget_ms)` | — | `{items: [ReviewItem], seat, budget_used_ms}` |
| `grade(position, action, budget_ms)` | `Position`, `Action` | `Grade` |
| `snapshot(game_id)` | — | `Position` |
| `restore(position)` | `Position` | `{game_id}` |
| `drop_game(game_id)` | — | `{}` |

Two shapes differ from the contract's table because every response must be an object with
`api`: `legal_actions` wraps the list in `{actions}` and `review` wraps it in `{items}`.

`budget_ms` is a ceiling in milliseconds; `0` means no limit. ISMCTS checks the clock every
64 simulations; the review passes it to PIMC as `PimcConfig.time_limit`, which the engine checks between sampled deals. Every search runs
with one thread (the engine's `threads = 1` path), so nothing here needs a thread pool.

### Randomness

Everything is seeded. `new_game` deals from `seed`. Each decision draws its own generator from
`(seed, number of actions so far, call)`, so a game restored from a snapshot continues
exactly as the original would: the random bot picks the same move, the solver runs the same
search.

## Shapes

### GameMeta

```json
{ "api": 0, "game": "big2", "seats": [4, 4],
  "variants": { "straights": ["with_low_twos", "no_twos"], "flush_order": ["rank_first", "suit_first"],
                "pass_rule": ["may_reenter", "locked_out"], "scoring": ["tiered", "original", "win_loss"],
                "last_card_duty": ["on", "off"] },
  "bot_levels": ["random", "easy", "medium", "hard", "expert"],
  "has_hint": true, "has_review": true, "has_grade": true }
```

Each variant lists its options default first. See `docs/rules/big2.md` for what they mean.

### GameConfig

```json
{ "seats": 4, "variants": { "straights": "no_twos", "last_card_duty": "off" }, "seed": 12345, "human_seats": [0] }
```

`seats` must be 4 (`error.bad_config` otherwise). Variants left out keep the default; an
unknown key or value is `error.unknown_variant`. `last_card_duty` also accepts `true`/`false`.
`human_seats` is stored and echoed; the engine does not act on it.

### Action

```json
{ "type": "action.play", "cards": ["3d", "3s"], "combo": "combo.pair" }
{ "type": "action.pass" }
```

Cards are listed weakest first. On input `combo` is ignored (the engine classifies the
cards); `type` may be left out, and an empty card list then means pass.

Combo keys: `combo.single`, `combo.pair`, `combo.triple`, `combo.straight`, `combo.flush`,
`combo.full_house`, `combo.four_of_a_kind`, `combo.straight_flush`.

### SeatView

Only what `seat` may see. Other hands appear only once the deal is over.

```json
{ "api": 0, "game_id": "g1", "seat": 0,
  "rules": { "straights": "with_low_twos", "flush_order": "rank_first", "pass_rule": "may_reenter",
             "scoring": "tiered", "last_card_duty": "on" },
  "seed": 12345, "human_seats": [0],
  "hand": ["3d", "5c", "..."], "hand_sizes": [13, 13, 13, 13],
  "current_player": 0, "your_turn": true, "is_opening": true,
  "current_play": null,
  "trick": [], "history": [], "trick_starts": [0],
  "legal": [ Action, "..." ],
  "locked_out": [false, false, false, false],
  "over": false, "winner": null, "scores": null, "penalties": null,
  "final_hands": null, "initial_hands": null }
```

- `current_play`: `{player, cards, combo}` of the play to beat, or `null` when leading.
- `history`: `[{player, action}]`, every action so far. `trick` is the part of it since the
  current trick began; `trick_starts` holds the history index where each trick began.
- `legal` is filled only when it is this seat's turn.
- When `over`: `winner`, `scores` (pairwise settlement or win/loss, per the scoring rule),
  `penalties` (penalty points per seat), `final_hands` and, for a game that began with a
  full deal, `initial_hands`.

### Event

`apply` returns the events in order:

```json
{ "type": "event.play", "player": 2, "action": Action, "cards_left": 11 }
{ "type": "event.pass", "player": 3 }
{ "type": "event.trick_won", "player": 2 }
{ "type": "event.deal_over", "winner": 1, "scores": [-13, 39, -13, -13], "penalties": [5, 0, 4, 4] }
```

`event.trick_won` follows the pass that ends a trick; `player` leads the next one.
`event.deal_over` follows the play that empties a hand.

### Position

The whole hidden state. The start plus `history` rebuilds the game; the other fields describe
the resulting position and are ignored by `restore`.

```json
{ "api": 0, "game": "big2",
  "rules": { "...": "..." }, "seed": 12345, "human_seats": [0],
  "initial_hands": [["3d", "..."], ["..."], ["..."], ["..."]],
  "history": [ { "player": 0, "action": Action }, "..." ],
  "hands": [["..."], ["..."], ["..."], ["..."]],
  "current_player": 1, "current_play": { "player": 0, "cards": ["3d", "3s"], "combo": "combo.pair" },
  "trick_starts": [0], "winner": null }
```

A hand-written drill position starts mid-game instead of from a deal: give `start` and no
`initial_hands`.

```json
{ "game": "big2", "seed": 0,
  "start": { "hands": [["3d", "3c"], ["4d", "5d", "6d"], ["7d", "8d"], ["9d", "Td"]],
             "current_player": 0, "current_play": null },
  "history": [] }
```

Hands may hold 0 to 13 cards; cards in no hand count as already played; such a position is
never an opening position. A bad start or a history that does not replay is
`error.bad_position`.

### Hint

Top 3 moves, best first (ISMCTS, up to 20,000 simulations within the budget).

```json
{ "api": 0, "seat": 0, "simulations": 20000, "budget_used_ms": 258,
  "moves": [
    { "action": Action, "score": 2.27, "win_rate": 0.293,
      "numbers": { "cards_left_after": 11, "visits": 19538, "share": 0.977, "score_ci95": 0.15, "win_ci95": 0.006 },
      "reasons": [ { "key": "reason.cards_left", "args": { "cards_left": 11 } },
                   { "key": "reason.sheds_weakest", "args": { "card": "3d" } },
                   { "key": "reason.score_margin", "args": { "points": 4.89 } } ] } ] }
```

`score` is the expected final score (points under the scoring rule); `win_rate` the chance of
going out first; `share` the fraction of simulations spent on the move; the `ci95` figures
are 95% confidence half-widths.

### ReviewItem

One per decision the seat has made so far (the review may run before the deal ends).

```json
{ "turn": 3, "player": 0, "played": Action, "best": Action,
  "loss": 1.2, "grade": "inaccuracy",
  "reasons": [ "..." ], "best_reasons": [ "..." ],
  "numbers": { "played_score": -6.6, "best_score": -5.4, "played_win_rate": 0.05, "best_win_rate": 0.06,
               "deals": 300, "simulations": 900 } }
```

`turn` indexes `history`. `reasons` explain the move played (position facts plus
`reason.score_loss` when it lost points); `best_reasons` explain the best move. The review
uses PIMC (up to 300 sampled deals per decision, every legal move played in each), so the
played move gets as many simulations as the best one. A forced move (one legal action) gets
`grade: "best"`, `loss: 0`, `reasons: [reason.forced]`, empty `numbers` and no search. The
budget is shared equally among the decisions still to review.

### Grade

```json
{ "api": 0, "seat": 0, "grade": "good", "loss": 0.4, "best": Action,
  "reasons": [ "..." ], "best_reasons": [ "..." ],
  "numbers": { "played_score": 1.9, "best_score": 2.3, "played_win_rate": 0.27, "best_win_rate": 0.29, "simulations": 20000 },
  "budget_used_ms": 310 }
```

`grade` runs the hint search (ISMCTS, 20,000 simulations within the budget) on the position
and compares `action` with the solver's first choice. An illegal `action` is
`error.illegal_action`. The position is not registered as a game.

## Grade thresholds

`loss` is the best move's expected score minus the played move's, in points. One card left at
the end costs 3 points under pairwise settlement (`tiered`, `original`: a player's score is
`total − 4 × own penalty`), and 1 point is the unit under `win_loss`.

| Grade | Condition |
|---|---|
| `best` | the solver's first choice, or no loss |
| `good` | loss under ⅓ card (1 point tiered/original, ⅓ point win_loss) |
| `inaccuracy` | loss under 1 card (3 points, or 1 point) |
| `mistake` | loss under 3 cards (9 points, or 3 points) |
| `blunder` | 3 cards or more |

The solver's numbers are estimates with sampling error (`score_ci95`); a `good` at the margin
may be noise.

## Bot levels

| Level | What it plays |
|---|---|
| `random` | a uniformly random legal move |
| `easy` | `StructuredGreedy` (greedy play that keeps the hand's best decomposition into combinations), with 10% random moves (`epsilon = 0.1`) |
| `medium` | `QNetPolicy`, temperature 1.0: the trained network `models/big2-qnet-duty.pknn` (1.16 MB), embedded with `include_bytes!` and parsed once on the first `medium` move; no search |
| `hard` | ISMCTS, 2,000 simulations |
| `expert` | ISMCTS, 10,000 simulations |

`hard` and `expert` use one thread, the greedy rollout policy with 10% noise, and stop early at
`budget_ms`. `easy` and `medium` ignore `budget_ms` (they answer in well under a millisecond). All levels draw their
randomness from the decision seed, so a snapshot restored later continues identically.

## Reason keys

Every reason is a fact the code checks on the position; none is guessed.

| Key | Args | Means |
|---|---|---|
| `reason.forced` | — | the only legal move |
| `reason.pass_holds` | `playable` | passing although `playable` plays were legal: the cards stay for later |
| `reason.goes_out` | — | the play empties the hand |
| `reason.cards_left` | `cards_left` | cards left after the play |
| `reason.keeps_control` | — | no combination among the unseen cards beats this play, so the player leads again |
| `reason.lowest_beating` | — | the weakest combination that still beats the current play (others were legal) |
| `reason.sheds_weakest` | `card` | leading with the hand's weakest card, the hardest to shed later |
| `reason.saves_two` | `twos_kept` | keeps every 2 while another legal play would spend one |
| `reason.saves_high_card` | `card` | keeps the hand's highest card (not a 2) while another legal play would spend it |
| `reason.opponent_nearly_out` | `player`, `cards_left` | an opponent has 2 cards or fewer |
| `reason.score_margin` | `points` | the best move's expected score lead over the runner-up |
| `reason.score_loss` | `points` | how many points this move loses against the best |

## Error codes

| Code | When |
|---|---|
| `error.bad_json` | the input string is not valid JSON of the expected shape |
| `error.bad_config` | `seats` is not 4 |
| `error.bad_seat` | a seat outside 0–3 |
| `error.unknown_game` | no game with that `game_id` |
| `error.unknown_variant` | an unknown rule key or value |
| `error.unknown_level` | a bot level not in `bot_levels` |
| `error.bad_card` | a malformed or repeated card id |
| `error.bad_action` | an unknown `type`, or `action.play` without cards |
| `error.illegal_action` | a legal shape the rules reject (not in hand, does not beat the table, pass when leading, 3♦ missing, last-card duty) |
| `error.game_over` | acting, hinting or asking a bot after the deal ended |
| `error.not_your_turn` | `hint` for a seat that is not to move |
| `error.bad_position` | a `Position` that cannot be rebuilt |

## Measured timings

Release builds, one thread, opening position of seed 100, Windows 11 desktop
(2026-10-28). WebAssembly: `wasm-pack --release` with `wasm-opt -O`, Node 22.

| Call | Native | WebAssembly (Node) |
|---|---|---|
| `hint`, 5,000 simulations | 67 ms | 72 ms |
| `hint`, 20,000 simulations | 318 ms | 258 ms |
| `bot_action` expert (10,000) | 152 ms | 113 ms |
| `review`, 300 deals per decision | 16 decisions: 141 ms | 11 decisions: 25 ms |

The `.wasm` of the web target is 1,572,286 bytes (1.5 MB) before gzip (2026-10-30). It was 350,977 bytes (343 KB) before the 1.16 MB network was embedded for `medium`.

Per-move time of the new levels in WebAssembly (Node 22, one 25-move and one 22-move game, seed 11, 2026-10-30): `easy` mean 0.10 ms, first move 2.1 ms; `medium` mean 0.54 ms, first move 7.9 ms (the first move parses the network).

Rayon in the engine's search crate compiles for `wasm32-unknown-unknown`; with `threads = 1`
the single-thread path runs on the calling thread and no worker is spawned (checked by the
Node smoke run of `hint`, `bot_action` and `review`).
