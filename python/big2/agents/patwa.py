"""Approximations of the two rule-based opponents in Patwa (2026).

Aalok Patwa, *Self-play reinforcement learning under imperfect information in Big 2*,
arXiv:2605.28863 (survey: ``docs/research/survey.md`` §2.4). The paper publishes no
code. These agents are re-implemented from its one-line descriptions, so they are
**approximations**: the weights below are ours, not his.

* ``PatwaGreedy`` — "lowest non-pass action by (type, rank key), ignoring hand
  structure". It passes only when it has no other legal action.
* ``PatwaSmart`` — "a deterministic scorer that prefers immediate wins, shedding
  cards, keeping 2s and avoiding orphan singles, with break penalties for pairs,
  triples and five-card structures". Each legal action gets a hand-written score;
  the highest wins, ties broken by the lowest (type, rank key). Passing scores
  ``-pass_cost``, so the agent passes only when every play costs a lot, in practice
  spending a 2 early. (A pass that cost nothing made the bot pass whole deals away
  rather than split a pair; it then lost to the greedy bot.)

Both see only their own hand and the legal actions (fair play, see
:mod:`big2.agents`).
"""

from __future__ import annotations

from dataclasses import dataclass

from big2._core import Big2State, classify_cards

#: Type order for the "lowest action" tiebreak: singles first, five-card hands last.
KIND_ORDER = {
    "single": 0,
    "pair": 1,
    "triple": 2,
    "straight": 3,
    "flush": 4,
    "full_house": 5,
    "four_of_a_kind": 6,
    "straight_flush": 7,
}

#: Bits of the four 2s (rank 12, card index ``rank * 4 + suit``).
TWOS_MASK = 0xF << 48


def action_key(cards: int, rules: dict[str, object]) -> tuple[int, int]:
    """``(type order, rank key)`` of a non-pass action; lower is weaker."""
    c = classify_cards(
        cards, straights=str(rules["straights"]), flush_order=str(rules["flush_order"])
    )
    assert c is not None, "a legal action is always a combination"
    return KIND_ORDER[c[0]], c[1]


def rank_counts(hand: int) -> list[int]:
    """Number of cards of each rank (index 0 = 3, 12 = 2) in a hand mask."""
    return [((hand >> (4 * r)) & 0xF).bit_count() for r in range(13)]


def five_card_combos(hand: int, rules: dict[str, object]) -> list[int]:
    """Every five-card combination (straight, flush, full house, ...) inside ``hand``.

    Enumerated by the engine: a position where the hand's owner leads lists every
    combination the hand holds. The three other players get one spare card each
    (the engine needs three non-empty hands) and the last-card duty is turned off,
    since it would restrict the lead to singles.
    """
    if hand.bit_count() < 5:
        return []
    spare = [i for i in range(52) if not (hand >> i) & 1][:3]
    lead = Big2State.from_position(
        [hand, 1 << spare[0], 1 << spare[1], 1 << spare[2]],
        0,
        None,
        straights=str(rules["straights"]),
        flush_order=str(rules["flush_order"]),
        last_card_duty=False,
    )
    return [a for a in lead.legal_actions() if a.bit_count() == 5]


@dataclass
class PatwaGreedy:
    """Plays the lowest non-pass legal action by (type, rank key)."""

    name: str = "patwa-greedy"

    def act(self, state: Big2State) -> int:
        plays = [a for a in state.legal_actions() if a]
        if not plays:
            return 0
        rules = state.rules()
        return min(plays, key=lambda a: action_key(a, rules))


@dataclass
class PatwaSmart:
    """Deterministic hand-written scorer (see the module docstring).

    Score of a play (higher is better), with ``h`` the hand before and ``r`` after:

    * emptying the hand: wins at once, always chosen;
    * ``+shed`` per card played;
    * ``-two`` per 2 played while the hand still has more than ``endgame`` cards;
    * ``-orphan`` per orphan single added (an orphan is a non-2 card that is in no
      pair, triple or remaining five-card hand); playing an orphan earns it back;
    * ``-pair`` / ``-triple`` per pair or triple broken (some, not all, of a rank
      group played);
    * ``-five`` if the play is not a five-card hand and leaves no five-card hand
      where ``h`` had one;
    * ``-low`` times the play's top card scaled to [0, 1], so it sheds low cards first.

    It passes (when allowed) if the best play scores below ``-pass_cost``; with the
    defaults that means a lone 2 while it holds more than ``endgame`` cards.

    The defaults come from a small grid on 2026-10-29 (400 duplicate deals each,
    A B A B against ``PatwaGreedy``, ``sum_of_others``): this setting scored
    +1.6 ± 0.25 per game and won 35 % of games (25 % is par), so it is stronger than
    the greedy bot, as Patwa's numbers imply his smart bot is.
    """

    shed: float = 1.0
    two: float = 7.0
    orphan: float = 0.5
    pair: float = 2.0
    triple: float = 2.5
    five: float = 3.0
    low: float = 3.0
    endgame: int = 6
    pass_cost: float = 8.0
    name: str = "patwa-smart"

    def act(self, state: Big2State) -> int:
        legal = state.legal_actions()
        plays = [a for a in legal if a]
        if not plays:
            return 0
        p = state.current_player
        assert p is not None
        hand = state.hand(p)
        rules = state.rules()
        fives = five_card_combos(hand, rules)
        orphans_before = self._orphans(hand, fives)
        can_pass = 0 in legal

        def score(a: int) -> tuple[float, tuple[int, int]]:
            key = action_key(a, rules)
            return self.score_play(hand, a, fives, orphans_before), (-key[0], -key[1])

        best = max(plays, key=score)
        if can_pass and score(best)[0] < -self.pass_cost:
            return 0
        return best

    def score_play(self, hand: int, play: int, fives: list[int], orphans_before: int) -> float:
        """The hand-written score of ``play`` from ``hand`` (see the class docstring)."""
        if play == hand:
            return 1e9
        rest = hand & ~play
        s = self.shed * play.bit_count()
        if hand.bit_count() > self.endgame:
            s -= self.two * (play & TWOS_MASK).bit_count()
        rest_fives = [f for f in fives if not f & play]
        s -= self.orphan * (self._orphans(rest, rest_fives) - orphans_before)
        before, after = rank_counts(hand), rank_counts(rest)
        for b, a in zip(before, after, strict=True):
            if 0 < a < b:  # a group partly played
                if b == 2:
                    s -= self.pair
                elif b >= 3:
                    s -= self.triple
        if play.bit_count() != 5 and fives and not rest_fives:
            s -= self.five
        s -= self.low * (play.bit_length() - 1) / 51.0  # the play's top card, 3d = 0 … 2s = 51
        return s

    @staticmethod
    def _orphans(hand: int, fives: list[int]) -> int:
        """Non-2 cards that are in no pair, triple or five-card hand of ``hand``."""
        in_five = 0
        for f in fives:
            in_five |= f
        counts = rank_counts(hand)
        n = 0
        for r in range(12):  # rank 12 (the 2s) never counts as an orphan
            if counts[r] == 1 and not (hand >> (4 * r)) & 0xF & (in_five >> (4 * r)):
                n += 1
        return n


__all__ = ["PatwaGreedy", "PatwaSmart", "action_key", "five_card_combos", "rank_counts"]
