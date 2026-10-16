"""Big Two helpers on top of the Rust engine (``poker._core``).

The engine represents a set of cards as an ``int`` bit mask and an action as the
mask of cards played (``0`` = pass). This module adds what humans and the CLI
need: parsing and pretty-printing, rule presets, and rebuilding a game from one
player's point of view.

Card text notation: rank ``3 4 5 6 7 8 9 T J Q K A 2`` + suit ``d c h s``
(e.g. ``"3d Tc Ah 2s"``). Ranks run 3 (low) to 2 (high); suits ♦ < ♣ < ♥ < ♠.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from poker._core import Big2State, classify_cards, format_cards, parse_cards

__all__ = [
    "Big2State",
    "Rules",
    "PASS",
    "cards",
    "fmt",
    "pretty",
    "card_list",
    "kind_of",
    "parse_action",
    "state_from_observation",
]

PASS = 0
"""The pass action."""

_SUIT_SYMBOL = {"d": "♦", "c": "♣", "h": "♥", "s": "♠"}


@dataclass(frozen=True)
class Rules:
    """Rule variants (see ``docs/rules/big2.md``). Defaults are the project defaults."""

    straights: str = "with_low_twos"
    flush_order: str = "rank_first"
    pass_rule: str = "may_reenter"
    scoring: str = "tiered"
    last_card_duty: bool = False

    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Big2State`` constructors."""
        return asdict(self)

    def new_deal(self, seed: int | None = None) -> Big2State:
        return Big2State.new_deal(seed, **self.kwargs())


def cards(text: str) -> int:
    """Parses ``"3d 4c Ah"`` into a card mask. ``""`` and ``"pass"`` give ``PASS``."""
    t = text.strip().lower()
    if t in ("", "pass", "p"):
        return PASS
    return parse_cards(text)


def fmt(mask: int) -> str:
    """Formats a card mask as ``"3d 4c Ah"``; ``PASS`` as ``"pass"``."""
    return format_cards(mask) if mask else "pass"


def pretty(mask: int) -> str:
    """Like :func:`fmt` but with suit symbols: ``"3♦ 4♣ A♥"``."""
    if not mask:
        return "pass"
    return " ".join(
        c[:-1].replace("T", "10") + _SUIT_SYMBOL[c[-1]] for c in format_cards(mask).split()
    )


def card_list(mask: int) -> list[str]:
    """The cards in a mask as a list of codes, weakest first."""
    return format_cards(mask).split() if mask else []


def kind_of(mask: int, rules: Rules | None = None) -> str | None:
    """``"single"``, ``"pair"``, …, ``"straight_flush"``, or ``None`` if not a combination."""
    r = rules or Rules()
    c = classify_cards(mask, straights=r.straights, flush_order=r.flush_order)
    return c[0] if c else None


def parse_action(text: str) -> int:
    """Parses one action: card list or ``"pass"``."""
    return cards(text)


def state_from_observation(
    seat: int,
    hand: int,
    history: list[tuple[int, int]],
    rules: Rules | None = None,
    seed: int | None = None,
) -> Big2State:
    """Rebuilds a game from what ``seat`` knows.

    Args:
        seat: the observing player.
        hand: the observer's *current* hand.
        history: every action since the deal, as ``(player, cards)``; ``cards == 0``
            is a pass. The first entry must be the opening play (with 3♦), unless
            the history is empty, in which case the observer must hold 3♦.
        rules: rule variants.
        seed: seed for filling in hidden cards.

    Returns:
        A state consistent with the observation. Opponents' unseen cards are dealt
        at random, so only use it for things that do not depend on them, e.g. as
        the root of a search, which re-samples hidden cards anyway.

    Raises:
        ValueError: if the observation is inconsistent.
    """
    r = rules or Rules()
    played_by = [0, 0, 0, 0]
    for p, c in history:
        if not 0 <= p < 4:
            raise ValueError(f"no player {p}")
        if c & (sum(played_by) | hand):
            raise ValueError(f"card(s) {fmt(c & (sum(played_by) | hand))} appear twice")
        played_by[p] |= c
    originals = list(played_by)
    originals[seat] |= hand
    if originals[seat].bit_count() != 13:
        raise ValueError(
            f"seat {seat}: hand + played cards = {originals[seat].bit_count()} cards, expected 13"
        )
    if not history and not hand & 1:
        raise ValueError("with no history, the observer must hold 3d (they open)")

    unknown = [i for i in range(52) if not (sum(originals) >> i) & 1]
    random.Random(seed).shuffle(unknown)
    for p in range(4):
        if p == seat:
            continue
        need = 13 - originals[p].bit_count()
        if need < 0:
            raise ValueError(f"player {p} played more than 13 cards")
        for i in unknown[:need]:
            originals[p] |= 1 << i
        unknown = unknown[need:]

    # 3d must be with the opener; if hidden, it was dealt randomly: swap it in.
    if history:
        opener = history[0][0]
        if not originals[opener] & 1:
            holder = next(p for p in range(4) if originals[p] & 1)
            if holder == seat:
                raise ValueError("the observer holds 3d, so the first play must be theirs")
            # Swap 3d with any unplayed hidden card of the opener.
            swap = (originals[opener] & ~played_by[opener]) & -(
                originals[opener] & ~played_by[opener]
            )
            originals[opener] = (originals[opener] & ~swap) | 1
            originals[holder] = (originals[holder] & ~1) | swap

    state = Big2State.from_hands(originals, **r.kwargs())
    for p, c in history:
        if state.current_player != p:
            raise ValueError(
                f"history says player {p} acted, but it was player {state.current_player}'s turn"
            )
        state.apply(c)
    return state
