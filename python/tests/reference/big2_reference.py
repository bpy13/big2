"""Independent, deliberately simple Big Two rules implementation.

This is a *test oracle*: it is written straight from ``docs/rules/big2.md`` without
looking at the Rust engine's tricks (bit masks, precomputed keys). The differential
test plays random games with the Rust engine and checks that, at every step, this
module agrees on the set of legal actions, and at the end on the scores.

Cards are ``(rank, suit)`` tuples with rank 0..12 meaning 3,4,…,K,A,2 and suit
0..3 meaning ♦,♣,♥,♠. Speed is not a goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

Card = tuple[int, int]  # (rank, suit)

RANKS = "3456789TJQKA2"
SUITS = "dchs"


def card_from_index(i: int) -> Card:
    """The engine's card index is rank*4 + suit (same rank and suit numbering)."""
    return (i // 4, i % 4)


def cards_from_mask(mask: int) -> list[Card]:
    return [card_from_index(i) for i in range(52) if mask >> i & 1]


def mask_from_cards(cards: list[Card] | tuple[Card, ...]) -> int:
    return sum(1 << (r * 4 + s) for r, s in cards)


# ---------------------------------------------------------------------------
# Combinations (rules §3, §4)
# ---------------------------------------------------------------------------

# Rank sequences of the ten straights, weakest first (rules §3.1).
_STRAIGHTS: list[tuple[int, ...]] = [tuple(range(k, k + 5)) for k in range(8)] + [
    (11, 12, 0, 1, 2),  # A 2 3 4 5
    (12, 0, 1, 2, 3),  # 2 3 4 5 6
]
_TOP_RANK = [k + 4 for k in range(8)] + [12, 12]

CATEGORY = {
    "straight": 1,
    "flush": 2,
    "full_house": 3,
    "four_of_a_kind": 4,
    "straight_flush": 5,
}


@dataclass(frozen=True)
class Combo:
    cards: tuple[Card, ...]
    kind: str
    strength: tuple[int, ...]  # compared lexicographically within the same size

    @property
    def size(self) -> int:
        return len(self.cards)


def classify(cards: tuple[Card, ...], straights: str, flush_order: str) -> Combo | None:
    cards = tuple(sorted(cards))
    n = len(cards)
    ranks = [r for r, _ in cards]
    suits = [s for _, s in cards]
    if n == 1:
        return Combo(cards, "single", (ranks[0], suits[0]))
    if n in (2, 3):
        if len(set(ranks)) != 1:
            return None
        if n == 2:
            return Combo(cards, "pair", (ranks[0], max(suits)))
        return Combo(cards, "triple", (ranks[0],))
    if n != 5:
        return None

    allowed = _STRAIGHTS if straights == "with_low_twos" else _STRAIGHTS[:8]
    is_flush = len(set(suits)) == 1
    seq = next((i for i, s in enumerate(allowed) if sorted(s) == sorted(ranks)), None)
    if seq is not None:
        top_suit = max(s for r, s in cards if r == _TOP_RANK[seq])
        if is_flush:
            return Combo(cards, "straight_flush", (CATEGORY["straight_flush"], seq, suits[0]))
        return Combo(cards, "straight", (CATEGORY["straight"], seq, top_suit))
    if is_flush:
        desc = tuple(sorted(ranks, reverse=True))
        if flush_order == "rank_first":
            return Combo(cards, "flush", (CATEGORY["flush"], *desc, suits[0]))
        return Combo(cards, "flush", (CATEGORY["flush"], suits[0], *desc))
    counts = sorted((ranks.count(r), r) for r in set(ranks))
    if [c for c, _ in counts] == [2, 3]:
        return Combo(cards, "full_house", (CATEGORY["full_house"], counts[1][1]))
    if [c for c, _ in counts] == [1, 4]:
        return Combo(cards, "four_of_a_kind", (CATEGORY["four_of_a_kind"], counts[1][1]))
    return None


def beats(a: Combo, b: Combo) -> bool:
    return a.size == b.size and a.strength > b.strength


# ---------------------------------------------------------------------------
# Game flow (rules §2, §5, §6, §7)
# ---------------------------------------------------------------------------


def penalty(n: int, scoring: str) -> int:
    if n == 0 or scoring == "win_loss":
        return 0
    if scoring == "tiered":
        mult = 1 if n <= 7 else 2 if n <= 9 else 3 if n <= 12 else 4
    else:  # original
        mult = 1 if n <= 9 else 2 if n <= 12 else 3
    return n * mult


@dataclass
class RefGame:
    hands: list[set[Card]]
    straights: str = "with_low_twos"
    flush_order: str = "rank_first"
    pass_rule: str = "may_reenter"
    scoring: str = "tiered"
    last_card_duty: bool = False
    current: int = 0
    current_play: Combo | None = None
    holder: int | None = None
    passed: set[int] = field(default_factory=set)
    consecutive_passes: int = 0
    opening: bool = True
    winner: int | None = None

    def __post_init__(self) -> None:
        self.current = next(p for p, h in enumerate(self.hands) if (0, 0) in h)

    def legal_actions(self) -> set[int]:
        """Legal actions as card masks; 0 is pass."""
        if self.winner is not None:
            return set()
        hand = sorted(self.hands[self.current])
        sizes = [self.current_play.size] if self.current_play else [1, 2, 3, 5]
        out: set[int] = set()
        next_has_one = len(self.hands[(self.current + 1) % 4]) == 1
        highest = max(hand)
        for k in sizes:
            for cs in combinations(hand, k):
                c = classify(cs, self.straights, self.flush_order)
                if c is None:
                    continue
                if self.opening and (0, 0) not in cs:
                    continue
                if self.current_play and not beats(c, self.current_play):
                    continue
                if self.last_card_duty and next_has_one and k == 1 and cs[0] != highest:
                    continue
                out.add(mask_from_cards(cs))
        if self.current_play is not None:
            out.add(0)
        return out

    def apply(self, mask: int) -> None:
        assert mask in self.legal_actions(), "illegal action for the reference engine"
        p = self.current
        if mask == 0:
            assert self.holder is not None
            if self.pass_rule == "may_reenter":
                self.consecutive_passes += 1
                if self.consecutive_passes == 3:
                    self._end_trick()
                else:
                    self.current = (p + 1) % 4
            else:
                self.passed.add(p)
                nxt = (p + 1) % 4
                while nxt != self.holder and nxt in self.passed:
                    nxt = (nxt + 1) % 4
                if nxt == self.holder:
                    self._end_trick()
                else:
                    self.current = nxt
            return
        cards = tuple(cards_from_mask(mask))
        combo = classify(cards, self.straights, self.flush_order)
        assert combo is not None
        self.hands[p] -= set(cards)
        self.current_play, self.holder = combo, p
        self.consecutive_passes = 0
        self.opening = False
        if not self.hands[p]:
            self.winner = p
            return
        nxt = (p + 1) % 4
        while self.pass_rule == "locked_out" and nxt in self.passed:
            nxt = (nxt + 1) % 4
        self.current = nxt

    def _end_trick(self) -> None:
        assert self.holder is not None
        self.current = self.holder
        self.current_play, self.holder = None, None
        self.consecutive_passes = 0
        self.passed = set()

    def scores(self) -> list[float]:
        assert self.winner is not None
        if self.scoring == "win_loss":
            return [3.0 if p == self.winner else -1.0 for p in range(4)]
        pen = [penalty(len(h), self.scoring) for h in self.hands]
        # Pairwise settlement, written out literally.
        return [float(sum(pen[j] - pen[i] for j in range(4) if j != i)) for i in range(4)]
