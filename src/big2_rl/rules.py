"""Pure Big Two card, combination, and scoring rules."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

RANK_NAMES = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
SUIT_NAMES = ("♦", "♣", "♥", "♠")
CARD_COUNT = 52
FIRST_CARD = 0
PASS_ACTION = 1 << 13
ACTION_COUNT = PASS_ACTION + 1

SCORE_TIERED = "custom_tiers"
SCORE_ORIGINAL = "original_tiers"
SCORING_LABELS = {
    SCORE_TIERED: "1–7 ×1 · 8–9 ×2 · 10–12 ×3 · 13 ×4",
    SCORE_ORIGINAL: "1–9 ×1 · 10–12 ×2 · 13 ×3",
}

SINGLE = 1
PAIR = 2
TRIPLE = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


@dataclass(frozen=True)
class Play:
    cards: tuple[int, ...]
    category: int
    tie_break: tuple[int, ...]

    @property
    def comparison_key(self) -> tuple[int, ...]:
        return (self.category,) + self.tie_break


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_name(card: int) -> str:
    return f"{RANK_NAMES[card_rank(card)]}{SUIT_NAMES[card_suit(card)]}"


def encode_card_mask(cards: Iterable[int], original_hand: tuple[int, ...]) -> int:
    positions = {card: index for index, card in enumerate(original_hand)}
    mask = 0
    for card in cards:
        mask |= 1 << positions[card]
    return mask


def decode_card_mask(mask: int, original_hand: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        card for index, card in enumerate(original_hand) if mask & (1 << index)
    )


def classify_play(cards: Iterable[int]) -> Play | None:
    hand = tuple(sorted(cards))
    size = len(hand)
    ranks = [card_rank(card) for card in hand]
    suits = [card_suit(card) for card in hand]

    if size == 1:
        return Play(hand, SINGLE, (ranks[0], suits[0]))
    if size == 2 and ranks[0] == ranks[1]:
        return Play(hand, PAIR, (ranks[0], max(suits)))
    if size == 3 and len(set(ranks)) == 1:
        return Play(hand, TRIPLE, (ranks[0], max(suits)))
    if size != 5:
        return None

    rank_counts = sorted(ranks.count(rank) for rank in set(ranks))
    is_flush = len(set(suits)) == 1
    is_straight = len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4

    if is_straight and is_flush:
        return Play(hand, STRAIGHT_FLUSH, (max(ranks), suits[0]))
    if rank_counts == [1, 4]:
        quad_rank = next(rank for rank in set(ranks) if ranks.count(rank) == 4)
        kicker = next(rank for rank in ranks if rank != quad_rank)
        return Play(hand, FOUR_OF_A_KIND, (quad_rank, kicker))
    if rank_counts == [2, 3]:
        triple_rank = next(rank for rank in set(ranks) if ranks.count(rank) == 3)
        pair_rank = next(rank for rank in set(ranks) if ranks.count(rank) == 2)
        return Play(hand, FULL_HOUSE, (triple_rank, pair_rank))
    if is_flush:
        return Play(
            hand,
            FLUSH,
            (suits[0],) + tuple(sorted(ranks, reverse=True)),
        )
    if is_straight:
        highest_card_suit = max(
            card_suit(card) for card in hand if card_rank(card) == max(ranks)
        )
        return Play(hand, STRAIGHT, (max(ranks), highest_card_suit))
    return None


def beats(candidate: Play, current: Play) -> bool:
    if len(candidate.cards) != len(current.cards):
        return False
    return candidate.comparison_key > current.comparison_key


def legal_plays(
    hand: Iterable[int], current_play: Play | None, must_include: int | None = None
) -> tuple[Play, ...]:
    cards = tuple(sorted(hand))
    plays = []
    for size in (1, 2, 3, 5):
        for subset in combinations(cards, size):
            if must_include is not None and must_include not in subset:
                continue
            play = classify_play(subset)
            if play is not None and (current_play is None or beats(play, current_play)):
                plays.append(play)
    return tuple(sorted(plays, key=lambda play: (play.comparison_key, play.cards)))


def penalty_points(cards_left: int, scoring_mode: str) -> int:
    if not 0 <= cards_left <= 13:
        raise ValueError("cards left must be between 0 and 13")
    if scoring_mode == SCORE_TIERED:
        multiplier = (
            1
            if cards_left <= 7
            else 2
            if cards_left <= 9
            else 3
            if cards_left <= 12
            else 4
        )
    elif scoring_mode == SCORE_ORIGINAL:
        multiplier = 1 if cards_left <= 9 else 2 if cards_left <= 12 else 3
    else:
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    return cards_left * multiplier


def pairwise_settlement(penalties: tuple[int, ...]) -> tuple[int, ...]:
    """Return zero-sum transfers when each pair settles their penalty difference."""
    return tuple(
        sum(
            other_penalty - penalty
            for other_index, other_penalty in enumerate(penalties)
            if other_index != player_index
        )
        for player_index, penalty in enumerate(penalties)
    )
