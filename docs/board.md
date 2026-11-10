# Board

## Workers

## Results
- Task 33 (slot 1, 2026-11-10): code done and pushed, **not ticked**: queue job 49 (gpu lane, `hkq.ps1 log 49`; runs from the big2-workers worktree, needs its untracked `models/big2-distilled-duty.pt`) trains the from-scratch probe 10 h, then plays 1,000 deals; then §6.1 of build-phase3.md gets the numbers and the task is ticked. Smoke only confirmed launch (no full iteration); `pytest` not run here (no `big2._core` built in this worktree).
- Task 32 (slot 2, 2026-11-10): code done and pushed, **not ticked**: queue job 47 (cpu lane, `hkq.ps1 log 47`; `runs/rollout_br/progress.json`) plays 40 deals x 2 lineups of a rollout attacker (rollouts at ISMCTS-100, real opponents full `expert`); then §7a of build-phase3.md gets the numbers and the task is ticked.
- Task 19 (slot 2, 2026-11-10): code done and pushed, **not ticked**: queue job 46 (gpu lane, `hkq.ps1 log 46`) generates round-1 belief data, trains, and plays the 300-deal match; then §3 of build-phase3.md gets the numbers and the task is ticked.
- Task 29 (slot 3, 2026-11-10): ticked. `hard` = ISMCTS-500, `expert` = ISMCTS-10,000 + exact endgame; strict order, top gap +1.02 (A23 bar 1.5, short, recorded). Ladder test passes.
- Task 28 (slot 3, 2026-11-10): ticked. Banded drill ratings (decision 0004): 553 drills, ratings 600–1,400 (A22 met); `drills/big2.json` committed; old formula behind `RATING_FORMULA = "seed"`.
