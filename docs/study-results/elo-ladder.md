# Elo ladder of the bots below the expert (Phase 6, Task 40)

Brian, 2026-11-27: "re-adjust our options for other bots and lower difficulties so they span a
reasonable distribution in the elo-space".

Status (2026-11-27): the method, the fit and the driver are built and tested; the full run is
queue job 398 (cpu lane, `jobq.ps1 log 398`; output `runs/p6_ladder/ladder.json` and
`ladder.md` in the big2-3 worktree). §3 and §4 get the numbers when it ends. A 1/25-size
pilot ran end to end (§5).

## 1. Method

**Candidate rungs.** `random`; today's app levels exactly as `crates/app-api` runs them
(`easy` = structured greedy with 10 % random moves, `medium` = Q-network at temperature 1,
`hard` = ISMCTS-500, `expert` = ISMCTS-10,000, the Python specs in `experiments/elo_ladder.py`
`OLD_LEVELS`); the self-play network's prior (Task 37's `runs/p6_net/selfplay-h24.pknn`)
sampled at a temperature τ (`netprior:net=…,temperature=τ`, new agent kind in
`python/big2/agents/netprior.py`: one Rust forward pass per decision, the prior over every legal
move raised to 1/τ); and the raw network (τ = 0, its top move).

**Matches.** Every pairing is an ABAB duplicate match: both rungs hold two seats, each deal is
played in its four rotations, last-card duty on, tiered scoring, no four-card plays. Each
finished game is one record `{deal, players, scores, seconds}` in
`runs/p6_ladder/pairs/<a>__<b>/block_*.json` (blocks of 50 deals, so the job resumes).

**Pairwise results and the fit** (`python/big2/teach/grading/elo.py`, game-agnostic: it reads
only those records). Per game, every seat of one rung against every seat of the other, by the
app's rule (hk-games `server/hkgames/elo.py`): higher final score = 1, equal = 0.5. Ratings are
the maximum-likelihood Bradley-Terry strengths (Hunter's MM iteration, draws as half a win each
way, a 0.5-game drawn prior per played pair so a 100 % result stays finite) on the Elo scale
(400 log10), anchored so today's `medium` is 1200 (hk-games D98, until Task 41 picks the
app's `medium`). 95 % ranges by percentile bootstrap over deals (a deal's four rotations
resample together), 1,000 resamples; a gap's range comes from the same resamples as the two
ratings. Test: `python/tests/test_grading_elo.py` recovers a synthetic five-rung ladder (Gumbel
scores, so the pairwise outcomes are exactly Bradley-Terry) within its ranges; a two-player
75 % result gives the closed-form 190.8 Elo.

**Steps** (`experiments/elo_ladder.py`, stages `span`, `rungs`, `fit`, `report`):
1. *Span*: `random`, `easy`, `medium`, `net-t1`, `net` against each other and all but `medium`
   against `expert`, 500 deals a pairing; fit; the span is Elo(`net`) − Elo(`random`).
2. *Rungs*: targets at equal steps over the span, 150-250 Elo apart (fewer, wider steps when
   the span is narrow); for each target a bisection on log τ (0.02-50): each probe plays 200
   deals against the two nearest known cheap rungs, the whole set is refitted, and the probe
   within ±25 Elo of the target (or the closest after 8 probes) becomes the rung.
3. *Fit*: every rung and old level plays the one and two above it in the fitted order (200
   deals first); then each adjacent gap's pairing grows by the 1/√n rule until its 95 %
   half-width is at most 25 Elo, capped at 3,000 deals (500 where `hard` or `expert` plays:
   0.26 s a move). The ladder's own adjacent rungs (old levels may sit between them) get
   direct pairings sized the same way.
4. *Report*: `ladder.json` (every rung's setting, Elo, range, deals; adjacent gaps with ranges;
   old levels on the new scale; pairings with thread-hours) and `ladder.md`.

## 2. Settings and cost

| item | value |
|---|---|
| network | `runs/p6_net/selfplay-h24.pknn` (Task 37) |
| anchor | `medium` (`qnet:temperature=1.0`) = 1200 |
| span deals per pairing | 500 (9 pairings among the five span rungs and `medium`, 4 with `expert`) |
| probe deals | 200 per probe and opponent |
| fit deals | 200 first, grown to a ±25 Elo gap or the cap |
| caps | 3,000 deals (network rungs), 500 (`hard`, `expert`) |
| seed | 11 (pairing seeds from the rung names; agents from deal seed and rotation) |

One non-forced move on this machine, one thread: `random` and `easy` under 0.01 ms,
`netprior` (any τ) 0.33-0.37 ms, `medium` 0.37 ms, `hard` 9 ms, `expert` 261 ms (median).
So the cost is the expert's seat: four span pairings of 500 deals ≈ 4 × 500 × 4 games × 2
seats × ~10 decisions × 0.26 s ≈ 12 thread-hours; everything else under an hour. Estimated
total 13-15 thread-hours, 2-3 hours wall on 8 workers.

## 3. Span and rungs

Pending job 398.

## 4. The ladder

Pending job 398. The tables (`ladder.md`) land here: rungs with Elo and ranges, adjacent
gaps with ranges and deals, old levels on the new scale, thread-hours.

## 5. Pilot (1/25 size, cheap searches)

`python experiments/elo_ladder.py all --quick` (20 deals a span pairing, `hard` = ISMCTS-100,
`expert` = ISMCTS-300, 8 workers, 3 minutes) ran every stage and resumed from its caches. Its
numbers are noise-level, but the shape is worth noting before the full run: the span from
`random` to the raw network came out near 300 Elo, which under the per-deal pairwise rule means
the raw network out-scores a random seat on about 84 % of deals. Four-player Big Two with card
luck compresses the per-deal Elo scale; if the full run confirms a span of a few hundred Elo,
the ladder holds few rungs (one or two temperatures between `random` and the network), as the
task's "fewer, wider rungs" clause allows, and the app's hidden ratings 600-1800 (D63) stretch
far wider than the bots' measured strengths.

Pilot ladder (not a result): `random` 961, `net-t4.34` 1102, `easy` 1166, `medium` 1200,
`hard`(100) 1206, `expert`(300) 1236, `net-t1` 1250, `net` 1255; strict order; every ladder
gap within ±25 after sizing.
