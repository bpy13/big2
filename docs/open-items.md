# Open items (big2 bot stream)

Pending work and questions for Brian. Newest first. The plan is
`docs/plans/2026-10-29-big2-bot-build.md`; results are in `docs/study-results/`.

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
