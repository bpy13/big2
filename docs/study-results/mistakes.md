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

Pilot counts and the final numbers: job 416 (`jobq.ps1 log 416`), then
`python experiments/mine_mistakes.py summary` writes `runs/p6_mistakes_pilot/summary.json`;
copy its table here.
