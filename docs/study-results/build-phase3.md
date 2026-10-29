# Build run phase 3: iterate, belief, exact endgames, external bars

Plan: `docs/plans/2026-10-29-big2-bot-build.md`, phase 3 (Tasks 17–25). Match format as in
`build-phase2.md`: duplicate deals (each deal 4×, seats rotated), last-card duty on, tiered
scoring, no four-card plays, lineups `1+3` and `ABAB`, 95 % bootstrap intervals over deals.

Status 2026-10-30: code for Tasks 17 (round-1 preparation), 18 and 20 is in; their long runs
are queued in one detached job (`runs\b25_round1\run.ps1`, see the plan's Task 17 status).
Sections fill in from that job's `table.md` files.

## 1. Expert iteration (Task 17)

`experiments/expert_iteration.py round 1`. What changed from round 0, and why
(`docs/open-items.md`, 2026-10-30):

- **Teacher: ISMCTS-5000** (greedy rollout) instead of ISMCTS-1000. ISMCTS-5000 beat
  ISMCTS-1000 by +1.09 [+0.35, +1.87] paired (500 deals, `build-phase1.md` ladder v0).
  PUCT λ = 0.5 only tied ISMCTS-1000 (`build-phase2.md` §6.2), costs about 4× ISMCTS-5000's
  time per move, and would feed round 0's own value errors back into the labels.
- **Value on every candidate's search score.** Round 0 weighted the score loss 0.5 against
  three cross-entropies; at a validation score loss of 0.0083 it was about 0.4 % of the
  total, so the distributions learnt mainly from the played move's single outcome. Round 1:
  score 100, prior 1, actor and seats distributions 0.25 each; weights start from round 0's.
  Check: the calibration slope of `build-phase2.md` §6.1 near 1 (round 0: 0.86).
- **Matches:** PUCT-2000 (`c_puct` 2.5) with the new network at λ = 1 and λ = 0.5 against
  ISMCTS-1000, 300 deals each; the better λ at 1,000 deals (A9); round 1 vs round 0 at
  1,000 deals only if A9 passes (A12).

Results: `runs/round1/table.md` (pending).

## 2. Hard-constraint belief from the rules (Task 18)

`RulesMask` (`crates/search/src/belief.rs`): uniform deals, minus those under which an
action already taken would have been illegal. In Big Two only the last-card duty ties an
action to the hidden cards (a single played while the next player held one card was the
player's highest), so the check is one backward pass over the history
(`State::history_consistent`, tested equal to a full replay). After 1,000 rejections in a
row it returns the last draw and counts a fallback. `analyze(belief="rules")` for ISMCTS
and PUCT; agent spec option `belief=rules`.

Smoke run (2 self-play games, ISMCTS-100): 13 % of decisions rejected at least one draw;
those rejected 38 % of draws on average (largest 57 %); no fallbacks. The 300-deal match
(`experiments/build_phase3.py rules-mask`) is queued.

## 3. Learned belief (Task 19)

Not started (after Task 17's round 1).

## 4. Information-set exact endgame (Task 20)

`crates/search/src/infoset_endgame.rs`: when at most `threshold` (12) cards are hidden and
at most `max_cards` (10) are in all hands together, every deal of the hidden cards is listed
(`Game::for_each_deal`), solved by max^n (`Endgame::solve_actions_within`, one node budget for
all deals), and each action averaged, weighted by the sampler (0 for deals the rules mask
excludes). `Forced { winner, plays }` when every deal gives the same winner after the best
move. Hooked into ISMCTS and PUCT at the root (`infoset_endgame` setting); when the budget
runs out, the normal search runs (`exact: false`).

**Cost follows the cards in play, not the hidden cards.** A probe on positions from greedy
self-play (≤ 12 hidden, 5 M-node cap, one thread, CPU under outside load):

| Cards in all hands | Positions | Solved | Mean s (solved) | Max s (solved) |
|---:|---:|---:|---:|---:|
| 9 | 1 | 1 | 0.006 | 0.006 |
| 10 | 2 | 2 | 0.317 | 0.633 |
| 11 | 4 | 4 | 0.744 | 1.917 |
| 12 | 3 | 2 | 0.157 | 0.168 |
| 13 | 5 | 4 | 0.997 | 2.623 |
| 14 | 4 | 1 | 0.426 | 0.426 |
| 15–16 | 9 | 2 | 0.56 | 0.639 |
| 17–20 | 16 | 0 | – | – |

Plain max^n prunes nothing, at about 1 M positions per second here. The default `max_cards`
is therefore 10; the queued run (`build_phase3.py endgame`) measures agreement with
ISMCTS-100,000 (A15) and times by cards in play 8–14, then plays PUCT-2000 with and without it.
A faster per-deal solver is in `docs/open-items.md`.

## 5.–8. Tasks 21–24

Not started.
