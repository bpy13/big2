# Build run phase 4: teaching depth

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, phase 4 (Tasks 26–31). The teaching code is
the crate `crates/teach` (package `big2-teach`); its key table, grade lines and ledger are in
`crates/teach/README.md`. Experiments: `experiments/build_phase4.py`, which runs the native
`big2-teach` program (no rebuild of the Python extension needed).

Status 2026-10-30: Task 26 done; Task 27's code and run done, its repeatability bar not met
(§1, open item); Task 30 done (scaffold). Tasks 28 and 29: code and full runs done (§2, §3).
2026-10-31: Task 29's `hard` probes done (§3): `hard` = ISMCTS-500 gives a strict order, but
`expert` leads it by +1.02, short of A23's 1.5; the budgets are Brian's call.

## 0. Reason keys and the audit file (Task 26)

`python experiments/build_phase4.py audit` → `runs/p4_audit/hints-50.json` (3 s). 50 positions
from greedy self-play (ε 0.2), one per deal at a random decision with a choice. On each: the
paired move table (every legal move on the same 300 sampled deals, greedy rollouts ε 0.1,
exact endgame for forced results), its best move as the hint, the hint's reason keys, the top
five rows. Brian rates each entry right / irrelevant / wrong in its `rating` field (B11, open
question 4 of the plan).

How often each key fired on the 50 hints:

| Key | Hints | Key | Hints |
|---|---|---|---|
| `reason.value.gap` | 50 | `reason.helps_seat` | 12 |
| `reason.tempo.plays_to_out` | 20 | `reason.value.score_vs_win` | 12 |
| `reason.plan.keeps` | 14 | `reason.control.pass_to_keep` | 7 |
| `reason.control.highest_out` | 10 | `reason.control.safe_single` | 7 |
| `reason.plan.breaks` | 6 | `reason.counterfactual.forced` | 1 |

`reason.counterfactual.outcomes` never fires on a hint by design: it compares a played move
with the best one, so it belongs to reviews. Every hint carries at least one key
(`reason.value.gap`).

## 1. Grade repeatability and the luck ledger (Task 27)

`python experiments/build_phase4.py regrade` → `runs/p4_regrade/results.json` and `table.md`.
50 self-play deals, ISMCTS-300 in every seat. Every decision with a choice (1,386; 972 forced
decisions are not graded) graded twice with different seeds; every 5th graded decision of a deal
also by a 10× grader (3,000 deals); the luck ledger of all four seats of every deal. Lines:
σ-scaled at σ = 27.45 (Task 6): 0.69 / 2.75 / 6.86 points. Agreement counts:

- **strict**: the same grade word in both runs;
- **compatible**: the same grade, or an `unclear` range that covers the other run's grade
  (what a reader sees: "between inaccuracy and mistake" next to "mistake" is not a
  contradiction);
- **merged**: strict, with `best` and `good` counted as one.

Main run (the default: 300 deals, doubled while `unclear` up to 1,200):

| Measure | Value | Bar | Verdict |
|---|---|---|---|
| Same grade, two seeds (1,386 decisions) | 90.2 % | ≥ 95 % (A21) | not met |
| Compatible | 97.6 % | – | – |
| Same grade with `best` and `good` merged | 91.5 % | – | – |
| Share of `unclear` grades | 13.4 % | – | – |
| Phone grader = 10× grader (296 decisions) | 87.2 % | ≥ 90 % | not met |
| Phone vs 10×, compatible | 94.9 % | – | – |
| Ledger: largest sum error over 200 seat-deals | 3.6e-15 | ≤ 1e-9 (A21) | met |

Grades (seed A): best 1,076, good 23, inaccuracy 61, mistake 23, blunder 17, unclear 186.
Time: 52 s on 8 threads.

Budget probe (same deals and seeds; `runs/p4_regrade_probe/`):

| Grader deals | Strict | Compatible | Merged | Unclear | vs 10×: strict | vs 10×: compatible | Time (8 threads) |
|---|---|---|---|---|---|---|---|
| 300, fixed, first rule (below) | 87.1 % | 91.1 % | 90.3 % | 22.5 % | 82.8 % | 93.9 % | 24 s |
| 1,200, fixed, first rule | 89.8 % | 95.5 % | 93.0 % | 14.2 % | 84.8 % (vs 4,800) | 94.3 % | 54 s |
| 300, fixed | 88.2 % | 98.1 % | 89.1 % | 25.7 % | 80.7 % | 97.0 % | 52 s* |
| 300 → 1,200 while unclear | 90.2 % | 97.6 % | 91.5 % | 13.4 % | 87.2 % | 94.9 % | 52 s |
| 300 → 2,400 while unclear | 92.2 % | 97.2 % | 93.9 % | 9.5 % | 89.2 % | 94.9 % | 54 s |
| 300 → 4,800 while unclear | 93.8 % | 96.6 % | 96.2 % | 6.9 % | 89.9 % | 94.9 % | 78 s |

\* the CPU was shared with the round-1 job; times are only roughly comparable.

**First rule vs final rule.** The first rule called the solver's top move `best` with no
interval, and any other move `good` below 0.69 points. Most strict disagreements were then
`best`↔`good` on near-ties (losses of 0.01–0.08 points) and `best`↔`unclear`. The final rule
(`grade_span` in `grade.rs`): `best` = the interval reaches 0 and stays under the `good` line,
so a near-tie is `best` for both moves; the top move's interval is that of its margin over the
runner-up; `good` needs a real loss under the line. Doubling the deals while `unclear` follows
teaching-and-deployment §2.3.

**Verdict A21.** The ledger part is met (exact to 4e-15). The repeatability bar (≥ 95 % strict)
is not met at any budget a phone can afford: 90.2 % at 300 → 1,200 deals, 93.8 % at 16× the
app's budget. The remaining disagreements are grades on opposite sides of a line in two runs,
each with an interval that misses the line. Compatible agreement is 96.6–98.1 % everywhere.
Question for Brian in `docs/open-items.md` (default: count "compatible" for A21 and ship
300 → 1,200).

**Ledger sizes** (mean absolute value per seat-deal, points): score 13.76, deal strength 6.69,
opponents' errors 5.30, own errors 3.84, residual 15.69. The residual is larger than the score:
with greedy rollouts as the model, most of a deal's result is how the hidden cards lay and how
play differed from the rollouts. The terms are exact by construction; what they explain is
modest. A stronger evaluator (the round-1 network or ISMCTS) inside the tables would shrink the
residual at a higher cost.

## 2. Drills and mistake clusters (Task 28)

**Full run (2026-10-30),** `python experiments/build_phase4.py mine --threads 8`: 700 self-play
deals in 185 s (8 threads; the plan's estimate was ~20 min). Sources: `runs/p4_mine/summary.json`,
`runs/p4_mine/clusters.txt`, `runs/p4_mine/raw.jsonl`; outputs `drills/big2.json` (502 drills,
untracked so far) and `runs/p4_audit/drills-50.json` (50 drills for Brian's tag audit).

| Measure | Value | Bar (A22) | Verdict |
|---|---|---|---|
| Drills | 502 | ≥ 300 | met |
| Every drill has the §6 fields (`id`, `game`, `skill`, `prompt`, `position`, `seat`, `rating`, `answers`, `source`) | 502 of 502 | all | met |
| Rating spread | 645 – 1,193 (548 points) | ≥ 600 points | **not met** |
| Tag audit file | 50 drills | 50 | met |

Ratings by band: 600s 5, 700s 15, 800s 47, 900s 158, 1000s 226, 1100s 51 (5th percentile 828,
median 1,011, 95th 1,119). Value gaps (rounded): 2 points 127, 3 points 146, 4 points 78,
5 points 40, 6+ points 111. The formula `1000 + 400·share − 100·log2(gap)` gives a narrow spread
because every candidate already has `easy` and `medium` wrong and `expert` right (open item).

Drills per skill: `control` 182, `keep_the_plan` 173, `finish_in_two` 41, `saving_twos` 32,
`score_vs_win` 26, `when_to_pass` 20, `kingmaking` 14, `general` 14.

**Mistake clusters** (5,745 graded mistakes, grouped by the best move's reason keys; clusters of
at least 10; the 12 largest of 55 shown, full list in `clusters.txt`):

| Mistakes | Mean loss | Skill | In the app | Best move's keys |
|---:|---:|---|---|---|
| 594 | 1.98 | `keep_the_plan` | new | `plan.keeps`, `tempo.plays_to_out` |
| 466 | 2.03 | `keep_the_plan` | new | `plan.keeps`, `tempo.plays_to_out`, `value.score_vs_win` |
| 417 | 1.95 | `keep_the_plan` | new | `plan.breaks`, `value.score_vs_win` |
| 343 | 1.39 | `when_to_pass` | yes | `control.pass_to_keep` |
| 310 | 1.84 | `general` | new | (none) |
| 286 | 2.66 | `keep_the_plan` | new | `plan.breaks` |
| 268 | 1.43 | `score_vs_win` | new | `value.score_vs_win` |
| 195 | 2.66 | `keep_the_plan` | new | `control.highest_out`, `plan.keeps`, `tempo.plays_to_out` |
| 156 | 5.34 | `control` | new | `control.safe_single` |
| 154 | 5.72 | `finish_in_two` | yes | `tempo.plays_to_out` |
| 148 | 1.42 | `score_vs_win` | new | `helps_seat`, `value.score_vs_win` |
| 148 | 1.25 | `when_to_pass` | yes | `control.pass_to_keep`, `helps_seat` |

Against the app's skill list (`hk-games\app\src\drills\big2.json`: `endgame_counting`,
`finish_in_two`, `saving_twos`, `when_to_pass`): **new-skill candidates** `skill.control`,
`skill.keep_the_plan`, `skill.score_vs_win`, `skill.kingmaking` (and `skill.general` for
mistakes with no key). **App skills with no cluster:** `endgame_counting` (no key marks
counting yet) and `saving_twos` (32 drills carry it, from their keys, but no mistake cluster
of 10 or more).

**Verdict Task 28:** not ticked. Shape, count and clusters are done; the rating spread misses
A22 by 52 points. The default fix (rate by the gap alone within each share band) changes the
formula and waits for Brian (`docs/open-items.md`).

Tiny run (6 deals, `runs/p4_mine_tiny3/`): 2 drills (ratings 991 and 1085; skills
`keep_the_plan`, `saving_twos`), 57 graded mistakes. Largest clusters by the best move's keys:
`plan.keeps + tempo.plays_to_out + value.score_vs_win` (7), `plan.breaks` (5),
`plan.keeps + tempo.plays_to_out` (5). New-skill candidates so far: `skill.keep_the_plan`,
`skill.score_vs_win`. App skills with no cluster yet: `endgame_counting`, `finish_in_two`,
`saving_twos` (too few deals to say anything).

## 3. Ladder v2 (Task 29)

**Full run (2026-10-30),** `python experiments/build_phase4.py ladder --threads 8`: 500
duplicate deals per adjacent pair, A B A B (upper level in seats 0 and 2), each deal 4× with the
hands rotated; 777 s. Source: `runs/p4_ladder/results.json`, `table.md`. Levels: `random`;
`easy` = StructuredGreedy ε 0.1 + mistake rates 0.5; `medium` = Q-net
(`models/big2-qnet-duty.pknn`) T 1.0 + rates 0.15; `hard` = ISMCTS-2,000; `expert` =
ISMCTS-10,000 + exact endgame.

| Upper vs lower | Mean (points per game) | 95 % | Upper wins | Gap ≥ 1.5 |
|---|---:|---:|---:|---|
| easy vs random | +6.56 | ± 0.50 | 85.0 % | yes |
| medium vs easy | +3.15 | ± 0.56 | 63.6 % | yes |
| hard vs medium | +3.08 | ± 0.46 | 58.9 % | yes |
| expert vs hard | **−0.03** | ± 0.38 | 49.9 % | **no** |

Mistakes fired on 23.3 % of `easy`'s decisions with a choice and 5.5 % of `medium`'s.

**Verdict A23: not met.** The lower four levels order strictly with gaps of 3 points or more.
`expert` (ISMCTS-10,000 + endgame) does not beat `hard` (ISMCTS-2,000) at all. The plan's tuning
knobs (temperature, mistake rates) only touch `easy` and `medium`, so they cannot open this gap.
Which `hard` and `expert` ship is Brian's call (`docs/open-items.md`). Task 29 is not ticked.

**Probes with a weaker `hard` (2026-10-30, round-2 job).** Same run, same `expert`
(ISMCTS-10,000 + endgame), `hard` = ISMCTS-500 or ISMCTS-1,000; 6 threads, 710 s and 756 s.
Source: `runs/p4_ladder_h500/`, `runs/p4_ladder_h1000/` (`table.md`, `results.json`).

| `hard` | hard vs medium | expert vs hard | Strict order | Every gap ≥ 1.5 |
|---|---:|---:|---|---|
| ISMCTS-2,000 (full run above) | +3.08 ± 0.46 | −0.03 ± 0.38 | no | no |
| ISMCTS-1,000 | +2.76 ± 0.46 | +0.31 ± 0.40 | no (interval covers 0) | no |
| ISMCTS-500 | +2.72 ± 0.49 | **+1.02 ± 0.40** | yes | no |

The lower pairs do not change (easy vs random +6.56, medium vs easy +3.15). With `hard` =
ISMCTS-500, all five levels order strictly and `hard` stays 2.7 points above `medium`. `expert`
is then 1.02 points ahead: clearly ahead, but short of 1.5. ISMCTS gains little past a few
thousand iterations, so no `hard` budget probed reaches the 1.5 gap below `expert`. A weaker
`hard` (for example ISMCTS-250) or a stronger `expert` (a PUCT round that passes A9) are the
remaining levers; neither was run.

Tiny run (8 deals, `hard` ISMCTS-100 and `expert` ISMCTS-300 instead of 2,000 and 10,000;
`runs/p4_ladder_tiny/`): easy vs random +3.97 ± 3.35, medium vs easy +2.56 ± 3.10, hard vs
medium +1.41 ± 3.26, expert vs hard +0.34 ± 3.12. Mistakes fired on 25.4 % of `easy`'s
decisions with a choice and 6.2 % of `medium`'s. Nothing is decided at 8 deals.
