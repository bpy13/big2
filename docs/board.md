# Board

## Workers

## Results
- Task 19 (slot 2, 2026-11-10): code done and pushed, **not ticked**: queue job 46 (gpu lane, `hkq.ps1 log 46`) generates round-1 belief data, trains, and plays the 300-deal match; then §3 of build-phase3.md gets the numbers and the task is ticked.
- Task 29 (slot 3, 2026-11-10): ticked. `hard` = ISMCTS-500, `expert` = ISMCTS-10,000 + exact endgame; strict order, top gap +1.02 (A23 bar 1.5, short, recorded). Ladder test passes.
- Task 28 (slot 3, 2026-11-10): ticked. Banded drill ratings (decision 0004): 553 drills, ratings 600–1,400 (A22 met); `drills/big2.json` committed; old formula behind `RATING_FORMULA = "seed"`.
