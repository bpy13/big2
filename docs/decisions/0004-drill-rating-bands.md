# 0004 — Drill ratings by bands (A22 spread)

**Status:** accepted by default (2026-11-10, orchestrator; Brian may change it)

## Context

The seed rating `1000 + 400·share − 100·log2(gap)` spread only 548 points over 502 drills
(645–1,193); A22 needs at least 600 (`build-phase4.md` §2).

## Decision

Rank drills by the share of levels wrong, then within a share band by value gap (smaller gap is
harder), and stretch the rank evenly over 600–1,400. The re-run gives 553 drills spanning
600–1,400 (800 points). This is the inbox default for "Change the drill rating formula?".

## Consequences

* Task 28 is ticked; Task 31 no longer waits on it.
* The old formula stays one constant away: `RATING_FORMULA = "seed"` in `python/big2/teach/drills.py`.
* Ratings are ranks, so they depend on the drill set; they are not absolute Elo-like values.
