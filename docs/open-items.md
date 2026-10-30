# Open items (big2 bot stream)

Pending work and questions for Brian. Newest first. The plan is
`docs/plans/2026-10-29-big2-bot-build.md`; results are in `docs/study-results/`.

## 2026-10-31 · Round 2 beats ISMCTS-1000 but misses A9's bar; round 3 runs; ladder probes

**Where things stand.**
- Task 17 round 2 (`build-phase3.md` §1): r2 PUCT-2000 λ = 0.5 vs ISMCTS-1000 at 1,000 deals:
  +0.76 [+0.13, +1.37] (1+3), +0.86 [+0.29, +1.50] (ABAB). The intervals sit above 0 for the
  first time, but A9 needs ≥ +1.0. Trend r0 → r1 → r2 (ABAB): −1.78 → +0.16 → +0.86.
  Calibration slope 1.05.
- Round 3, the last approved round, runs now (`runs\b25_round3\run.ps1`; plan Task 17 status
  has the check and resume commands). The teacher stays ISMCTS-5000: no r2-vs-ISMCTS-5000
  match exists, so Brian's rule keeps it. Round 3 also plays r3 against ISMCTS-5000 at 1,000
  deals (`runs\round3\vs_teacher\`), so the teacher rule can be checked directly.
- Task 29 probes (`build-phase4.md` §3): `hard` = ISMCTS-500 gives a strict order; `expert`
  (ISMCTS-10,000 + endgame) leads it by +1.02 ± 0.40, short of 1.5. With `hard` =
  ISMCTS-1,000 the lead is +0.31 ± 0.40, so the order is not strict.

**Questions for Brian (defaults apply; nothing blocks on them).** These replace questions 1, 2
and 4 of the 2026-10-30 (evening) item below.
1. *What should `hard` and `expert` use in the app?* The app uses ISMCTS-2,000 and
   ISMCTS-10,000 today; `expert` does not beat `hard` there (−0.03). No probed `hard` budget
   reaches A23's 1.5-point gap below `expert`. *Default:* `hard` = ISMCTS-500, `expert` =
   ISMCTS-10,000 + exact endgame, and A23's top gap counts as met when the interval is above 0.
   Example: `expert` beats `hard` by +1.02 ± 0.40; `hard` beats `medium` by +2.72 ± 0.49.
   The scope-keeping option: probe ISMCTS-250 as `hard` first (about 15 minutes of CPU).
2. *If round 3 reaches A9's +1.0 bar, does PUCT replace ISMCTS as `expert`?* PUCT needs the
   network on the phone and costs about 18× ISMCTS-1000's time per move. *Default:* yes, if it
   also passes A12 (r3 beats r2) and the phone timing allows it (Task 16); otherwise ISMCTS
   ships. Example: r3 at +1.1 [+0.5, +1.7] in both lineups, beating r2: PUCT-2000 r3 ships.
3. *Stop expert iteration after round 3 if A9 still fails?* *Default:* yes; ISMCTS ships as
   `expert` and the GPU moves on. Example: r3 at +0.9 [+0.3, +1.5] ends the loop, even though
   it beats ISMCTS-1000. The other option is a round 4 (about 6–8 hours, mostly CPU), most
   useful if r3 also beats ISMCTS-5000 and can become the teacher.

## 2026-10-30 (evening) · Round 1 fails A9; round 2 queued; Phase 4 runs done

**Where things stand.**
- Task 17 round 1 (`build-phase3.md` §1): r1 PUCT-2000 λ = 0.5 vs ISMCTS-1000 at 1,000 deals:
  +0.46 [−0.18, +1.13] (1+3), +0.16 [−0.47, +0.81] (ABAB). A9 fails, but it is up from round
  0 (−0.12 / −1.78). The value calibration slope is now 1.03 (round 0: 0.86).
- Round 2 runs now as one detached job (`runs\b25_round2\run.ps1`; plan Task 17 status has
  the check and resume commands). Teacher stays ISMCTS-5000 (Brian's rule: "use teacher
  unless distilled model wins significantly"; r1 does not beat it).
- Task 28 (`build-phase4.md` §2): 502 drills, shape checked; rating spread 548 points, A22
  needs 600 (question 3).
- Task 29 (`build-phase4.md` §3): easy, medium, hard order with gaps of 3+ points; `expert`
  (ISMCTS-10,000 + endgame) vs `hard` (ISMCTS-2,000) is −0.03 ± 0.38 (question 4).

**Questions for Brian (defaults apply; nothing blocks on them):**
1. *Which search do the `hard` and `expert` levels ship?* PUCT needs the network and costs
   about 17× ISMCTS's time per move. *Default:* both stay ISMCTS unless a PUCT round passes
   A9. Example: if round 2 gives +0.8 [+0.1, +1.5], that is not ≥ +1.0, so ISMCTS ships.
2. *Stop expert iteration after round 3 if A9 still fails?* Each round costs about 6–7 hours
   of CPU and 1 GPU-hour. *Default:* yes; ship ISMCTS-5000 as `expert` and move the GPU to
   the next phase. Example: round 3 at +0.7 [−0.0, +1.4] ends the loop.
3. *Change the drill rating formula to reach the 600-point spread (A22)?* Today
   `1000 + 400·share − 100·log2(gap)` gives 645–1,193. *Default:* rate by the value gap
   within each band of "share of levels wrong", stretched to 600–1,400; the drills
   themselves stay the same. Example: the hardest drill would rate about 1,400 instead of
   1,193, the easiest about 600 instead of 645.
4. *How should `expert` get ahead of `hard`?* ISMCTS-10,000 does not beat ISMCTS-2,000
   (−0.03 ± 0.38 at 500 deals), and the app uses those two budgets today. Probes with
   `hard` = ISMCTS-500 and 1,000 run in the round-2 job (`runs\p4_ladder_h500`,
   `runs\p4_ladder_h1000`). *Default:* keep the app's budgets until the probes report, then
   propose the largest `hard` budget that is ≥ 1.5 below `expert` and ≥ 1.5 above `medium`.
   Example: if ISMCTS-500 gives +1.6 to `expert` and +2.0 over `medium`, `hard` becomes
   ISMCTS-500.

## 2026-10-30 · Phase 4 full runs (Tasks 28, 29): done, see the item above

Both ran in `runs\p4_queue\run.ps1` after the round-1 job (mine 185 s, ladder 777 s).

Two choices were made without Brian (non-blocking, easy to change):
- The ladder's `hard` and `expert` are ISMCTS-2,000 and ISMCTS-10,000 + exact endgame, not
  the plan's PUCT-300 / PUCT-2,000, following the 2026-10-30 item below ("bot levels stay on
  ISMCTS"). Drill positions come from ISMCTS self-play for the same reason.
- The ladder lives in `crates/teach/src/ladder.rs`, not `crates/big2/src/bots/ladder.rs`:
  `big2-big2` cannot depend on `big2-search`.

Risk for A22: the seed rating `1000 + 400·share − 100·log2(gap)` may not spread over 600
points, because a candidate already has easy and medium wrong and expert right (share about
0.4–0.8). If the full run shows a narrow spread, the default fix is to rate by the gap alone
within each share band; that is a formula change for Brian to accept.

## 2026-10-30 · Grade repeatability misses the 95 % bar (Task 27)

The paired grader gives the same grade word on a re-grade with a new seed on 90.2 % of 1,386
decisions (300 deals, doubled while `unclear` up to 1,200; `build-phase4.md` §1). The plan's
bar A21 is 95 %. Even 4,800 deals (16× the app's review budget) reach only 93.8 %. The rest
are grades on opposite sides of a line, each run sure of its side. Counting an `unclear`
range that covers the other run's grade as agreement gives 97.6 %. The phone grader also
agrees with a 10× grader on 87.2 % (bar 90 %; compatible 94.9 %).

**Question for Brian (default applies):** Which measure should A21 use?
1. *Default:* "compatible" agreement (≥ 95 %: met, 97.6 %), and ship 300 deals doubled up to
   1,200 while `unclear`. Example: run A says "mistake", run B says "between inaccuracy and
   mistake"; that counts as agreeing.
2. Strict agreement, with a larger budget or a stronger evaluator in the tables (more time per
   reviewed decision on the phone).

Also non-blocking: the luck ledger's residual (mean 15.7 points per seat-deal) is larger than
the score itself (13.8). The terms sum exactly, but with greedy rollouts they explain little of
a deal. A stronger evaluator would shrink it.

## 2026-10-30 · Exact endgame is slow beyond 10 cards in play (Task 20)

The information-set endgame solves each deal with plain max^n, which prunes nothing (about
1 M positions per second natively under load). Positions with more than about 10 cards in all
hands often need over 5 M positions (`build-phase3.md` §4), so the default `max_cards` is 10
and the hook only fires in the last few tricks. Options, none started: (1) "paranoid" search
(the observer against the other three as one side), which allows alpha-beta pruning but
changes the model; (2) solve a random sample of deals once more than ~2,000 exist; (3) better
move ordering plus a shared table across deals keyed on the observer's view. Non-blocking:
the queued A15 run measures what the current solver does.

## 2026-10-30 · Next step after Task 15's fallback (PUCT ties ISMCTS)

**Where things stand.** PUCT-2000 with the round-0 network fails A9 (`build-phase2.md` §1–§2).
The fallback of proposal §10 row 2 also fails the bar (§6): the prior is well calibrated, the
value is over-spread and optimistic, and a leaf value `0.5·network + 0.5·rollout` brings PUCT
to a tie with ISMCTS-1000 (+0.30 and +0.61 at 300 deals, intervals covering 0). It costs about
19× the time per move of ISMCTS-1000.

**Proposal §10 row 2 says:** if PUCT still ties after one more round, ship ISMCTS as `expert`
and move the GPU to phase 5. So the next step is one more round, not more tuning of round 0:

1. **Bot levels stay on ISMCTS.** `hard` and `expert` keep today's ISMCTS (the app stream's
   ISMCTS 2,000 / 10,000). The proposal's `hard` → PUCT-300, `expert` → PUCT-2000 mapping is
   **not** sent to the app stream. Task 16's bot-notes entry says so.
2. **Task 17, round 1 of expert iteration**, with three changes the results point to:
   - Label with the stronger search: ISMCTS (greedy rollout) at a larger budget than round 0's
     1,000 iterations, or PUCT λ = 0.5 if a 1,000-deal check shows it beats ISMCTS-1000. Round
     0's labels only copy ISMCTS-1000, so a network trained on them cannot beat it.
   - Train the value on every candidate's search score directly (as Task 7's mean-score
     network did, which beat the distribution head in §3), keeping the distributions as an
     auxiliary head; then check that the slope in §6.1 is near 1.
   - Measure the new network inside PUCT with λ = 0.5 and λ = 1, 300 deals each, then 1,000.
3. **Stop rule.** If round 1's PUCT still does not beat ISMCTS-1000 by +1.0 at 1,000 deals,
   ship ISMCTS-5000 as `expert` (or the app's ISMCTS-10,000 if the phone timing allows), and
   move the GPU to phase 5. The network still serves the teaching layer (`plays_to_out`,
   `lead_value`) and cheap hints.
4. **Cheaper simulations** (only if PUCT is kept): PUCT spends about 90 µs per simulation
   against ISMCTS's 14 µs per iteration. A cache of network outputs per position key, or
   fewer evaluations at opponent nodes, would narrow the equal-time gap of §2.

**Not done yet:** Task 16's WebAssembly timing (A11) waits until the CPU is free. A 1,000-deal
confirmation of λ = 0.5 against ISMCTS-1000 is optional (~2 h at 6 workers under load).

**Question for Brian (default applies):** Accept "one more round, then ISMCTS ships as
`expert` if PUCT still ties"? *Default: yes, as proposal §10 row 2 says.*
