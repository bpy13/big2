# Elo ladder of the bots below the expert (Phase 6, Task 40)

Brian, 2026-11-27: "re-adjust our options for other bots and lower difficulties so they span a
reasonable distribution in the elo-space".

Status (2026-11-27, done): queue job 408 (cpu lane, `jobq.ps1 log 408`, exit 0; a re-queue of job 398,
cancelled by Task 47 at 400 of the 500 random-vs-expert deals while Brian picked the expert
network, which decision 0007 kept at hour 24) ran every stage in 30 minutes wall, 4.15
thread-hours. Results in §3 and §4; `runs/p6_ladder/ladder.json` and `ladder.md` are committed
beside this file (the job wrote them in the big2-3 worktree). The pilot is in §5.

**Verdict.** The per-deal scale is narrow: `random` to the raw network is 291 Elo, today's
`easy` to today's `expert` only 76. The ladder below the expert holds three cheap rungs
(`random` 955, network prior at τ = 4.34 1096, τ = 1 1230) and the raw network (1248); the
old `hard`, `expert` and the network's top move sit within 26 Elo of each other. Every adjacent
gap's range is within ±25 Elo and the order is strict. Task 41 picks the app's levels from §4.

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

**Span** (stage 1: 14 pairings, 500 duplicate deals each, `medium` anchored at 1200):

| rung | setting | Elo | 95 % range |
|---|---|---|---|
| `random` | `random` | 958 | [952, 963] |
| `easy` | `greedy:structured=1,epsilon=0.1` | 1160 | [1155, 1165] |
| `medium` | `qnet:temperature=1.0` | 1200 | anchor |
| `net-t1` | `netprior:net=<net>,temperature=1` | 1233 | [1229, 1238] |
| `expert` | `ismcts:iterations=10000` | 1239 | [1233, 1244] |
| `net` | `netprior:net=<net>,temperature=0` | 1249 | [1244, 1253] |

Span `random` → `net`: **291 Elo**, i.e. the raw network out-scores a random seat on about 84 %
of deals under the per-deal rule. The pilot (§5) had predicted this.

**Rungs** (stage 2). 291 Elo at 150-250 a step leaves one interior target, 1103 (the task's
"fewer, wider rungs" clause for a narrow span). Bisection on log τ, each probe 200 deals against
its two nearest cheap rungs, the whole set refitted after each probe:

| probe | τ | Elo after refit | opponents | verdict |
|---|---|---|---|---|
| 1 | 7.07 | 1053 | `random`, `easy` | too weak |
| 2 | 2.66 | 1162 | `net-t7.07`, `easy` | too strong |
| 3 | 4.34 | 1094 | `net-t7.07`, `easy` | within ±25 of 1103: rung |

The ladder is therefore `random`, `net-t4.34`, `net-t1`, `net`, with the four old levels fitted
beside it. A fourth cheap rung near 1160 (τ ≈ 2.7, probe 2) is available if Task 41 wants
more steps than three below the top; it was not needed to meet the 150-250 Elo spacing.

## 4. The ladder

Stage 3 (fit): each rung and old level played the one and two above it (200 deals first), then
each adjacent gap's pairing grew until its 95 % half-width was at most 25 Elo. Every gap met it
at its first size (200 or the span's 500 deals); no pairing needed growing. Anchor `medium` = 1200.

**Rungs and old levels on one scale** (deals = all deals the rung played):

| rung | kind | setting | Elo | 95 % range | deals |
|---|---|---|---|---|---|
| `random` | ladder | `random` | 955 | [949, 960] | 2900 |
| `net-t4.34` | ladder | `netprior:net=<net>,temperature=4.34` | 1096 | [1088, 1102] | 1000 |
| `easy` | old-level | `greedy:structured=1,epsilon=0.1` | 1161 | [1156, 1165] | 3300 |
| `medium` | old-level | `qnet:temperature=1.0` | 1200 | anchor | 2400 |
| `hard` | old-level | `ismcts:iterations=500` | 1222 | [1216, 1228] | 800 |
| `net-t1` | ladder | `netprior:net=<net>,temperature=1` | 1230 | [1225, 1234] | 2900 |
| `expert` | old-level | `ismcts:iterations=10000` | 1236 | [1231, 1242] | 2200 |
| `net` | ladder | `netprior:net=<net>,temperature=0` | 1248 | [1243, 1252] | 2500 |

`<net>` = `runs/p6_net/selfplay-h24.pknn`. Strict order: yes (every adjacent gap's range
excludes 0).

**Adjacent gaps, all eight rungs** (each from a direct pairing):

| lower | upper | gap | 95 % range | deals | within ±25 |
|---|---|---|---|---|---|
| `random` | `net-t4.34` | +140 | [133, 148] | 200 | yes |
| `net-t4.34` | `easy` | +65 | [58, 72] | 200 | yes |
| `easy` | `medium` | +39 | [35, 44] | 500 | yes |
| `medium` | `hard` | +22 | [16, 28] | 200 | yes |
| `hard` | `net-t1` | +8 | [2, 15] | 200 | yes |
| `net-t1` | `expert` | +6 | [2, 11] | 500 | yes |
| `expert` | `net` | +11 | [6, 16] | 500 | yes |

**The ladder's own gaps** (old levels left out):

| lower | upper | gap | 95 % range | deals | within ±25 |
|---|---|---|---|---|---|
| `random` | `net-t4.34` | +140 | [133, 148] | 200 | yes |
| `net-t4.34` | `net-t1` | +134 | [127, 142] | 200 | yes |
| `net-t1` | `net` | +18 | [13, 22] | 500 | yes |

**Old app levels on the new scale:** `easy` 1161 [1156, 1165], `medium` 1200 (anchor),
`hard` 1222 [1216, 1228], `expert` 1236 [1231, 1242]; `random` 955 [949, 960]. The whole old
ladder `easy` → `expert` is 76 Elo (the `expert` out-scores an `easy` seat on about 61 % of
deals); `medium` → `expert` is 36.

**Cost.** 27 pairings, 9,600 deals, 38,400 games, 4.15 thread-hours (estimate was 13-15: an
`expert` seat takes about 3,100 thread-seconds per 500-deal pairing, a third of the guess, since
the exact endgame and forced moves leave it fewer searched decisions a game). The four span
pairings with `expert` and `hard` vs `expert` are 3.9 of the 4.15 hours; every network, `easy`,
`medium` and `random` pairing is seconds. Per-pairing thread-hours are in `ladder.md`.

**Done-when check.** Test `python/tests/test_grading_elo.py` passes (6 tests). `ladder.json`
holds every rung's setting, Elo, 95 % range and deals; every adjacent gap's half-width is at most
7.4 Elo (bar 25); the rungs order strictly; the old levels' Elo are above.

**What the numbers say for Task 41.**
- The per-deal pairwise rule compresses strength: the raw network beats `expert` by about +3.6
  points a deal (selfplay-luck.md) yet sits only 11 Elo above it (52 % of deals), because a seat
  that loses by one card and a seat that loses by ten both score 0 against the winner, and card
  luck decides many deals outright. Elo on this scale measures how often a bot out-scores a seat,
  not by how much.
- Three of today's four levels (`medium`, `hard`, `expert`) and both network rungs above τ = 1
  lie within 48 Elo; a player cannot feel those steps. The useful ladder is `random` (955),
  τ = 4.34 (1096), τ ≈ 2.7 (~1160, probe 2, optional), τ = 1 (1230), then the expert rung from
  Task 39's search setting on top (Task 41).
- The app's hidden ratings run 600-1800 (D63) while the bots span under 300; Task 41 (or the
  app) has to choose whether bot Elo is reported on this per-deal scale or stretched onto the
  players' scale.

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
