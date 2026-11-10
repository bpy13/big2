# Milestones

This repo is the Big2 game of hk-games (submodule `games/big2`). Workers build only toward
the current milestone.

## Current: bots built to the plan
Started early on Brian's word (2026-11-10, hk-games: 3. default, i.e. start the game bot plans now,
alongside hk-games milestone 1). It serves what hk-games milestone 2 (Android launch) asks for "bots at an agreed strength".
Done when:
- Every task in `docs/plans/2026-10-29-big2-bot-build.md` is ticked.

Rules from hk-games: GPU or long CPU jobs go through the job queue
(`C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\tools\queue\hkq.ps1 add`), never started directly.
Never touch the standalone checkout `repo\
big2
` (Brian's files and running queue jobs live there);
workers use this clone and its worktrees only.
