## 2026-11-17 · bot stream · big2 phase 4: the teaching crate, the drill file, the ladder constructors
- **`big2-teach`** (`crates/teach`, runs on the calling thread, no rayon, seeded `ChaCha8Rng`; add
  `big2-teach.workspace = true` to `crates/app-api/Cargo.toml`). Calls, all in `crates/teach/README.md`:
  `evaluate_paired(&state, &PairedConfig { deals: 300, max_deals: 1200, seed, .. })` gives a `MoveTable`;
  `reasons(&state, &table, chosen)` (or `reasons_for` with the plan's candidates) gives `[{key, args}]`;
  `grade_paired_table(&state, played, &cfg)` gives `(Graded { loss, interval, grade, range, seed }, table)`;
  `luck_ledger(&start, &actions, seat, &cfg, &policy)` gives `Ledger { deal_strength, opponents_errors,
  own_errors, residual }`, whose terms sum to the seat's score (error 3.6e-15 over 200 seat-deals).
- **New reason keys** (additions; no existing key changed): `reason.plan.keeps {combo}`,
  `reason.plan.breaks {combo, cost}`, `reason.control.pass_to_keep {card}`,
  `reason.value.score_vs_win {score_delta, win_delta}`, `reason.tempo.plays_to_out {mine, theirs}`,
  `reason.control.safe_single {card}`, `reason.control.highest_out {card}`, `reason.value.gap {points}`,
  `reason.counterfactual.outcomes {best, played, deals}` (reviews only),
  `reason.helps_seat {seat, delta}`, `reason.counterfactual.forced {result, winner, plays}`.
  Meaning and firing rule of each: the key table in `crates/teach/README.md`; each rule has a firing
  and a not-firing unit test.
- **Grade lines (points, `tiered`).** σ-scaled (σ = 27.45): good < 0.69, inaccuracy < 2.75, mistake
  < 6.86, blunder from 6.86 (`Thresholds::default()`). The app's lines: 1 / 3 / 9 (`Thresholds::APP`).
  Which one ships is **pending: open question 3 for Brian** (default: σ lines with the `unclear`
  grade; the app keeps its lines until you switch). Repeatability with a new seed: 90.2 % same word
  at 300 → 1,200 deals, 97.6 % compatible; A21 counts compatible (decision 0003).
- **Drills:** `drills/big2.json` in the big2 repo: 553 drills in the engine-interface `Drill` shape
  (`id`, `game`, `skill`, `prompt`, `position`, `seat`, `rating`, `answers`, `solver`, `source`),
  ratings 600–1,400 (decision 0004). New skill candidates from the mistake clusters:
  `skill.control`, `skill.keep_the_plan`, `skill.score_vs_win`, `skill.kingmaking`. A second file,
  `drills/big2-puzzles.json` (2,142 rows with `brilliant_answers`), is the puzzle pool.
- **Ladder v2 constructors** (`big2_teach::ladder::LadderBot`): `random()`, `easy()`
  (StructuredGreedy ε 0.1 + mistake rates 0.5), `medium(net)` (`models/big2-qnet-duty.pknn`, T 1.0 +
  rates 0.15), `hard()`, `expert()`. As decided (inbox default, 2026-11-10): `hard` =
  ISMCTS-500, `expert` = ISMCTS-10,000 + exact endgame (both in the crate and in the app preset). Order
  is strict; gaps +6.56, +3.15, +2.72, +1.02 (the last
  short of the 1.5 bar, recorded).
- **Audit files for you** (not in git; `runs/` of the standalone checkout): `runs/p4_audit/hints-50.json`
  (50 hints, rate right / irrelevant / wrong) and `runs/p4_audit/drills-50.json` (50 drill tags). The
  plan names three; the third has no source (there is no third audit task).
- Details: `docs/study-results/build-phase4.md` in big2. No action for the app stream.
