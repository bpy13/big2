## 2026-11-17 · bot stream · big2: causes and merits in `reasons` (plan Task 35, D115)
- **What changed:** `review`, `grade` and `hint` (its moves other than the best) now add
  causes to the played move's `reasons`, and `review`/`grade` add merits to `best_reasons`. A
  cause is there only when its fact is true of the played move and false of the best move; a
  merit only when it is true of the best move and false of the played one, so no sentence
  states something both moves share. Causes come strongest first: the first one can be the
  headline instead of the points sentence. No existing key changed meaning or shape; the
  shapes of `ReviewItem` and `Grade` are the same (the arrays only gain entries).
- **New cause keys** (args; English sentence; seats are absolute 0-based like every `player`):
  - `reason.lets_opponent_out` (`player`, `cards_left`): "Player {player} has {cards_left} cards and could go out on this."
  - `reason.breaks_group` (`group` = `{cards, combo}`, `extra_plays`): "This breaks up your {group}: {extra_plays} more play(s) to go out."
  - `reason.more_plays_to_out` (`extra_plays`, rare): "This leaves you {extra_plays} more play(s) to go out."
  - `reason.can_be_beaten` (`beaters`): "Cards still out can beat this, so you may lose the lead."
  - `reason.misses_sure_out` (`plays`): "The best move goes out in {plays} plays nobody can stop; this one gives that up."
  - `reason.spends_two` (`cards`): "This spends your {cards}."
  - `reason.spends_high_card` (`card`): "This spends your highest card, {card}."
  - `reason.leaves_loose_singles` (`singles`, `best_singles`): "This leaves {singles} loose singles, against {best_singles} after the best move."
  - `reason.keeps_low_single` (`card`): "This leaves {card} on its own, a single that is hard to get rid of."
  - `reason.pass_returns_lead` (`player`, `cards_left`): "Passing lets player {player} keep the lead with {cards_left} cards left."
- **New merit keys** (in `best_reasons`):
  - `reason.sure_out` (`plays`): "This goes out in {plays} plays nobody can stop."
  - `reason.fewer_plays_to_out` (`plays`, `played_plays`): "This needs {plays} plays to go out, against {played_plays}."
- **Examples (the coach-voice examples):** 2 (Kd instead of pass, to beat Qs): `breaks_group`
  pair Kd Kh, +1 play. 4 (3d 3s instead of Kd Ks): `lets_opponent_out` player 3 with 2 cards,
  then `can_be_beaten`, `misses_sure_out` 2; best gets `sure_out` 2. 8 (7d instead of 8d 8c):
  `can_be_beaten`. All eight examples that showed only the points now have a cause.
- **Coverage:** on 200 graded decisions that lost points (stored drills), 180 (90 %) carry a
  cause; before, none did. Slowest call 3.9 ms, inside `grade`'s 600 ms.
- **Action for you:** dictionary entries (English, Traditional Chinese) for the 12 keys, and
  the headline rule: when the played move has a cause, show the first cause as the headline
  and the points under it. Full table: `crates/app-api/README.md` "Causes and merits" in big2;
  numbers: `docs/study-results/build-phase4.md` §4.
