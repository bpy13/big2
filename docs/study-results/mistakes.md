# Mistakes and their EV distributions per Elo band (plan Tasks 42a, 42)

## 0. Pilot (Task 42a)

Driver `experiments/mine_mistakes.py`, queue script `experiments/mine_mistakes_job.ps1`, tests
`python/tests/test_mine_mistakes.py`. Same-rung deals (all four seats one rung of
`runs/p6_ladder/ladder.json`), bands `net-t4.34` (1096), `easy` (1161), `net-t1` (1230), `net`
(1248); 100 deals per band. Every non-forced decision is searched by the provisional reference
(`netsearch k=8, n=256, depth=full`, widened to every move with prior >= 1 % plus the move
played) and the provisional knee (`k=3, n=64, depth=full`); the knee's pick is scored by the
reference's paired loss. Lines are D102 (0.2 / 0.69 / 2.75 / 6.86) read as upper limits of
great / good / inaccuracy / mistake. A puzzle candidate is a non-flat decision whose played
move lost at least 2.75, whose second-best move also loses at least 2.75 with its lead's 95 %
range above 0; `reasons` are empty (the theme comes from the structural rules of
`skill_for`), and no deeper confirmation solve yet (Task 44 does that).

**Cost, measured on a 1-deal-per-band smoke run (4 workers, one search thread each):**
reference 4.5-5.2 s and knee 0.5-0.9 s per non-forced decision; 42-57 % of decisions forced
(skipped). That is about 140 thread-s a deal, so 400 deals is about 15 thread-h, not the ~3
planned. Task 42 at 300 deals per band for ~6 bands with the deeper reference would be
several times that; its budget needs revisiting (cut deals per band or `n` first).

**Result (job 416, 100 deals per band, 40 min wall on 8 threads; `runs/p6_mistakes_pilot/summary.json`):**

| band (rung Elo) | decisions | forced | flat | played-move loss mean / p50 / p90 / p97 | share above 0.2 / 0.69 / 2.75 / 6.86 | knee picks a different best move | knee mean loss | ref s / knee s per decision |
|---|---|---|---|---|---|---|---|---|
| net-t4.34 (1096) | 5,729 | 46.5 % | 11.6 % | 2.22 / 0.26 / 6.35 / 13.48 | 51.2 / 43.6 / 24.5 / 8.7 % | 26.4 % | 0.218 | 1.65 / 0.23 |
| easy (1161) | 4,249 | 45.1 % | 6.6 % | 2.12 / 0.00 / 5.95 / 15.55 | 40.0 / 33.0 / 19.5 / 8.6 % | 25.3 % | 0.216 | 1.68 / 0.22 |
| net-t1 (1230) | 4,325 | 47.3 % | 9.4 % | 0.64 / 0.00 / 2.07 / 4.50 | 30.0 / 22.2 / 7.2 / 1.2 % | 26.6 % | 0.258 | 1.77 / 0.25 |
| net (1248) | 4,401 | 46.1 % | 9.8 % | 0.45 / 0.00 / 1.32 / 3.51 | 22.9 / 15.6 / 4.5 / 0.7 % | 24.1 % | 0.220 | 1.68 / 0.24 |

(Shares and losses are over non-forced decisions; seconds are per non-forced decision per
search thread.) Cost: the measured 140 thread-s a deal held up at about 1.7 s reference + 0.24 s
knee per labelled decision, so the 400 deals took about 11 thread-h on 8 threads, not the ~3
planned: Task 42 at 300 deals per band for ~6 bands with a deeper reference needs its budget cut
(200 deals per band first, as its notes say). Reading: the knee (k=3, n=64) names a different
best move than the reference on about a quarter of the non-forced decisions but costs only
0.22-0.26 points on average; about 46 % of decisions are forced and 7-12 % flat. The bands order
as expected on played-move loss (the 1096 and 1161 bands lose about 2.1-2.2 points a decision,
the 1230 and 1248 bands 0.45-0.64).

**Puzzle candidates** (`runs/p6_mistakes_pilot/candidates.json`, 60, 15 per band; every one has
theme, the bot's move, its loss and every answer's loss): themes `skill.b2_mixed` 20,
`skill.b2_stop_runner` 11, `skill.b2_singles_order` 11, `skill.saving_twos` 10,
`skill.b2_break_five` 5, `skill.when_to_pass` 3. `runs/` is git-ignored: `candidates.json` and
`summary.json` are committed with `git add -f`; `decisions.jsonl` (13.8 MB) stays in the
`big2-2` worktree's `runs/p6_mistakes_pilot/` and in this one, uncommitted.
