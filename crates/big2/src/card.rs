//! Big Two card indices.
//!
//! Every card gets an index `0..52` so that **a higher index is a stronger single
//! card**: `index = big_two_rank * 4 + suit`, where `big_two_rank` runs
//! `3 = 0, 4 = 1, …, A = 11, 2 = 12` and suit runs `♦ = 0, ♣ = 1, ♥ = 2, ♠ = 3`.
//! So `3♦ = 0`, `3♣ = 1`, …, `2♠ = 51`. A hand is a `u64` with bit `index` set for
//! each card held.

use poker_cards::{mask, parse_cards, Card, ParseCardError, Rank, Suit};

/// Number of cards in the deck.
pub const NUM_CARDS: u8 = 52;
/// Number of ranks.
pub const NUM_RANKS: u8 = 13;
/// Index of 3♦, the card that opens the game.
pub const THREE_OF_DIAMONDS: u8 = 0;
/// Mask with all 52 cards.
pub const FULL_DECK: u64 = (1u64 << NUM_CARDS) - 1;

/// All 13 cards of one suit (`♦ = 0 … ♠ = 3`).
#[inline]
pub fn suit_mask(suit: u8) -> u64 {
    0x0001_1111_1111_1111u64 << suit
}

/// Big Two rank (`3 = 0 … 2 = 12`) of a card index.
#[inline]
pub fn rank_of(index: u8) -> u8 {
    index >> 2
}

/// Suit (`♦ = 0 … ♠ = 3`) of a card index.
#[inline]
pub fn suit_of(index: u8) -> u8 {
    index & 3
}

/// The 4-bit set of suits held at Big Two rank `rank` in `hand`.
#[inline]
pub fn suits_at(hand: u64, rank: u8) -> u8 {
    ((hand >> (rank * 4)) & 0xF) as u8
}

/// Converts a card to its Big Two index.
pub fn index_of(card: Card) -> u8 {
    let big_two_rank = (card.rank.index() + 12) % 13; // Two(0) → 12, Three(1) → 0
    big_two_rank * 4 + card.suit.index()
}

/// Converts a Big Two index back to a card.
///
/// # Panics
/// Panics if `index >= 52`.
pub fn card_at(index: u8) -> Card {
    assert!(index < NUM_CARDS, "card index {index} out of range");
    let rank = Rank::from_index((rank_of(index) + 1) % 13);
    Card::new(rank, Suit::from_index(suit_of(index)))
}

/// Builds a mask from cards. Duplicates collapse into one bit.
pub fn mask_of(cards: &[Card]) -> u64 {
    cards.iter().fold(0, |m, &c| m | (1u64 << index_of(c)))
}

/// The cards in a mask, weakest first.
pub fn cards_in(m: u64) -> Vec<Card> {
    mask::bits(m).map(card_at).collect()
}

/// Parses a card list like `"3d 3c 5h"` into a mask.
///
/// Returns an error if a card is malformed or listed twice.
pub fn parse_mask(s: &str) -> Result<u64, ParseCardError> {
    let mut m = 0u64;
    for c in parse_cards(s)? {
        let bit = 1u64 << index_of(c);
        if m & bit != 0 {
            return Err(ParseCardError(format!("{c} listed twice")));
        }
        m |= bit;
    }
    Ok(m)
}

/// Formats a mask as `"3d 3c 5h"`, weakest card first.
pub fn format_mask(m: u64) -> String {
    cards_in(m)
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_order_matches_single_card_strength() {
        assert_eq!(parse_mask("3d").unwrap(), 1 << 0);
        assert_eq!(parse_mask("3c").unwrap(), 1 << 1);
        assert_eq!(parse_mask("4d").unwrap(), 1 << 4);
        assert_eq!(parse_mask("As").unwrap(), 1 << 47);
        assert_eq!(parse_mask("2d").unwrap(), 1 << 48);
        assert_eq!(parse_mask("2s").unwrap(), 1 << 51);
    }

    #[test]
    fn round_trip_all_cards() {
        for i in 0..NUM_CARDS {
            assert_eq!(index_of(card_at(i)), i);
        }
        assert_eq!(format_mask(parse_mask("2s 3d Tc").unwrap()), "3d Tc 2s");
    }

    #[test]
    fn duplicate_is_an_error() {
        assert!(parse_mask("3d 3d").is_err());
    }
}
