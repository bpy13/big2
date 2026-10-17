# Big Two — rules specification

This document is the single source of truth for the Big Two rules implemented in
`crates/big2`. Every rule below has at least one test in `crates/big2/tests/` or in
the Python reference oracle (`python/tests/reference/big2_reference.py`). If the code
and this document disagree, the code is wrong.

Rules that differ between tables are collected in a `RuleSet`. The **default** value
of each variant is marked *(default)*.

## 1. Cards

A standard 52-card deck, no jokers.

**Rank order** (low → high): `3 4 5 6 7 8 9 T J Q K A 2`

**Suit order** (low → high): `♦ diamonds < ♣ clubs < ♥ hearts < ♠ spades`

A card's strength is compared by rank first, then suit. So `3♦` is the weakest
card in the deck and `2♠` the strongest; `K♠ < A♦` because rank decides first.

Text notation used throughout the code, CLI and tests: rank character followed by
suit letter, e.g. `3d`, `Tc` (ten of clubs), `Ah`, `2s`.

## 2. Deal and opening

1. Four players, 13 cards each. Seats are numbered 0–3 and play proceeds
   0 → 1 → 2 → 3 → 0 (counter-clockwise at a physical table; the direction does not
   matter for the rules).
2. The player holding `3♦` leads the first trick.
3. The first play of the deal **must include `3♦`** (as a single, or as part of a
   pair, triple or five-card combination).

## 3. Combinations

A play is one of the following. Any other set of cards is illegal (for example two
pairs, four of a kind on its own, or any 4-card set).

| Cards | Combination | Notes |
|---|---|---|
| 1 | Single | any card |
| 2 | Pair | two cards of the same rank |
| 3 | Triple | three cards of the same rank |
| 5 | Straight | five consecutive ranks, mixed suits (see §3.1) |
| 5 | Flush | five cards of the same suit, not a straight |
| 5 | Full house | a triple plus a pair |
| 5 | Four of a kind | four cards of one rank plus any fifth card ("kicker") |
| 5 | Straight flush | a straight whose cards share one suit |

### 3.1 Which straights exist

Ten rank sequences count as straights, listed weakest to strongest:

| # | Sequence | "Top card" (used for suit tie-breaks) |
|---|---|---|
| 1 | 3-4-5-6-7 | 7 |
| 2 | 4-5-6-7-8 | 8 |
| 3 | 5-6-7-8-9 | 9 |
| 4 | 6-7-8-9-T | T |
| 5 | 7-8-9-T-J | J |
| 6 | 8-9-T-J-Q | Q |
| 7 | 9-T-J-Q-K | K |
| 8 | T-J-Q-K-A | A |
| 9 | A-2-3-4-5 | 2 |
| 10 | 2-3-4-5-6 | 2 |

`J-Q-K-A-2`, `Q-K-A-2-3` and `K-A-2-3-4` are **not** straights (no wrap-around
through the 2 except the two sequences above).

> Variant `straights`:
> * `with_low_twos` *(default)* — all ten sequences above.
> * `no_twos` — only sequences 1–8 (no straight may contain a 2).

## 4. Beating the current play

A play can only follow a play **with the same number of cards**. A single beats
only a single, a pair only a pair, and so on. There are no "bombs": four of a kind
cannot be played on a single or a pair.

### 4.1 Single, pair, triple

* **Single:** higher card wins (rank, then suit).
* **Pair:** higher rank wins; for equal ranks, the pair holding the higher suit
  wins. Example: `9♠ 9♦` beats `9♥ 9♣` because ♠ > ♥.
* **Triple:** higher rank wins. (Two triples of the same rank cannot exist in one deck.)

### 4.2 Five-card plays

First compare the **category**:

```
straight  <  flush  <  full house  <  four of a kind  <  straight flush
```

Any play of a higher category beats any play of a lower one. Within one category:

* **Straight:** compare the sequence number from §3.1. If equal, compare the suit
  of the top card. Example: `T♦ J♣ Q♥ K♦ A♠` beats `T♠ J♠ Q♠ K♥ A♥` (top cards
  `A♠` vs `A♥`). `2-3-4-5-6` beats `A-2-3-4-5`, and both beat `T-J-Q-K-A`.
* **Flush:** compare the five ranks from highest to lowest, like poker; the first
  difference decides. If all five ranks are equal, the higher suit wins.
  Example: `3♦ 5♦ 7♦ 9♦ 2♦` beats `4♠ 6♠ 8♠ T♠ A♠` because `2 > A`.
* **Full house:** compare the rank of the triple. The pair does not matter.
* **Four of a kind:** compare the rank of the four. The kicker does not matter.
* **Straight flush:** like straights — sequence number, then suit.

> Variant `flush_order`:
> * `rank_first` *(default)* — as above.
> * `suit_first` — compare suit first, then ranks high to low.

## 5. Turn flow

1. The **leader** of a trick plays any legal combination. The leader cannot pass.
2. Play passes to the next seat. Each player either plays a combination that beats
   the **current play** (the most recent play in this trick) or passes.
3. When three players in a row pass, the trick is over. The player who made the
   current play becomes the new leader and starts a fresh trick with any combination.
4. A player who passed may play again later in the same trick if the turn comes
   back to them before three consecutive passes.

> Variant `pass_rule`:
> * `may_reenter` *(default)* — rule 4 as written.
> * `locked_out` — once a player passes, they skip every remaining turn of that
>   trick. The trick ends when every other player is locked out.

> Variant `last_card_duty` (default: **on**): when the **next** player holds exactly
> one card, any single the current player plays must be their highest card. Pairs,
> triples, five-card plays and passing are unaffected.
>
> The default changed from off to on on 2026-10-18. Results measured before then
> (Phase 1 and Phase 3 in `docs/study-results/`) and the networks trained before
> then use duty **off**.

## 6. End of the deal

The deal ends the moment one player plays their last card. That player is the
winner. Everyone else is left holding their remaining cards.

## 7. Scoring

### 7.1 Penalty

Each player's **penalty** is `cards left × multiplier`. The winner's penalty is 0.

| Cards left | `tiered` *(default)* | `original` |
|---|---|---|
| 1–7 | ×1 | ×1 |
| 8–9 | ×2 | ×1 |
| 10–12 | ×3 | ×2 |
| 13 | ×4 | ×3 |

### 7.2 Pairwise settlement

Every pair of players settles the difference in their penalties: the player with
the higher penalty pays the difference to the other. A player's **score** for the
deal is the sum of their four-way settlements. Scores always add up to zero.

Equivalent formula: with `n = 4` players and penalties `p_i`,
`score_i = (sum of all p) − n · p_i`.

**Worked example** (`tiered`):

| Player | Cards left | Penalty | Score |
|---|---|---|---|
| A (winner) | 0 | 0 | 0+3+16+52 − 4·0 = **+71** |
| B | 3 | 3 | 71 − 12 = **+59** |
| C | 8 | 16 | 71 − 64 = **+7** |
| D | 13 | 52 | 71 − 208 = **−137** |

B and C finish positive because they beat D by more than they lost to A. Strong
play therefore aims to **minimize cards left** when not winning, not only to go out
first.

> Variant `scoring`:
> * `tiered` *(default)*, `original` — as above.
> * `win_loss` — winner scores +3, everyone else −1.

## 8. Glossary

| Term | Meaning |
|---|---|
| Deal | One hand of Big Two, from the deal until someone goes out. |
| Trick | A sequence of plays from a lead until three consecutive passes. |
| Leader | The player who starts a trick. |
| Current play | The last combination played in the current trick. |
| Information set | Everything one player knows: their own hand, the full public history and each opponent's card count. |
