//! Game-independent playing-card vocabulary.
//!
//! This crate only knows what a card *is*: a [`Rank`], a [`Suit`] and their text
//! notation (`"Ah"`, `"Tc"`, `"3d"`). How cards are *ordered* differs between games
//! (in Big Two the 2 is the highest rank, in Hold'em the Ace is), so each game crate
//! defines its own ordering and its own index for packing cards into a `u64` bit mask.
//!
//! The [`mask`] module has small helpers for such masks.

use std::fmt;
use std::str::FromStr;

/// A card rank, in the conventional order Two → Ace.
///
/// The discriminant is the conventional poker rank index (`Two = 0`, `Ace = 12`).
/// Games that order ranks differently (Big Two puts Two on top) remap this index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Rank {
    Two = 0,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    /// All ranks, Two first.
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// The rank with conventional index `i` (`0 = Two … 12 = Ace`).
    ///
    /// # Panics
    /// Panics if `i > 12`.
    pub fn from_index(i: u8) -> Rank {
        Rank::ALL[i as usize]
    }

    /// The conventional index (`Two = 0 … Ace = 12`).
    pub fn index(self) -> u8 {
        self as u8
    }

    /// One-character notation: `2 3 … 9 T J Q K A`.
    pub fn to_char(self) -> char {
        b"23456789TJQKA"[self as usize] as char
    }

    /// Parses one rank character. Accepts upper or lower case; `"10"` is handled
    /// by [`Card::from_str`].
    pub fn from_char(c: char) -> Option<Rank> {
        let i = "23456789TJQKA".find(c.to_ascii_uppercase())?;
        Some(Rank::from_index(i as u8))
    }
}

/// A card suit. The discriminant follows the Big Two suit order
/// (♦ < ♣ < ♥ < ♠), which is also the usual bridge order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Suit {
    Diamonds = 0,
    Clubs,
    Hearts,
    Spades,
}

impl Suit {
    /// All suits, Diamonds first.
    pub const ALL: [Suit; 4] = [Suit::Diamonds, Suit::Clubs, Suit::Hearts, Suit::Spades];

    /// The suit with index `i` (`0 = ♦, 1 = ♣, 2 = ♥, 3 = ♠`).
    ///
    /// # Panics
    /// Panics if `i > 3`.
    pub fn from_index(i: u8) -> Suit {
        Suit::ALL[i as usize]
    }

    /// The index (`♦ = 0 … ♠ = 3`).
    pub fn index(self) -> u8 {
        self as u8
    }

    /// ASCII notation: `d c h s`.
    pub fn to_char(self) -> char {
        b"dchs"[self as usize] as char
    }

    /// Unicode symbol: `♦ ♣ ♥ ♠`.
    pub fn symbol(self) -> char {
        ['♦', '♣', '♥', '♠'][self as usize]
    }

    /// Parses `d c h s` (any case) or the Unicode symbols.
    pub fn from_char(c: char) -> Option<Suit> {
        match c {
            'd' | 'D' | '♦' => Some(Suit::Diamonds),
            'c' | 'C' | '♣' => Some(Suit::Clubs),
            'h' | 'H' | '♥' => Some(Suit::Hearts),
            's' | 'S' | '♠' => Some(Suit::Spades),
            _ => None,
        }
    }
}

/// A playing card.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Card {
        Card { rank, suit }
    }
}

impl fmt::Display for Card {
    /// Formats as ASCII notation, e.g. `Ah`, `Tc`, `3d`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank.to_char(), self.suit.to_char())
    }
}

/// Error returned when a card or card list cannot be parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseCardError(pub String);

impl fmt::Display for ParseCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot parse card {:?} (expected e.g. \"Ah\", \"Tc\", \"10♠\")",
            self.0
        )
    }
}

impl std::error::Error for ParseCardError {}

impl FromStr for Card {
    type Err = ParseCardError;

    /// Parses `"Ah"`, `"tc"`, `"10s"`, `"3♦"`.
    fn from_str(s: &str) -> Result<Card, ParseCardError> {
        let err = || ParseCardError(s.to_string());
        let t = s.trim();
        let (rank_part, suit_char) = {
            let suit_char = t.chars().last().ok_or_else(err)?;
            (&t[..t.len() - suit_char.len_utf8()], suit_char)
        };
        let rank = match rank_part {
            "10" => Rank::Ten,
            r if r.chars().count() == 1 => {
                Rank::from_char(r.chars().next().unwrap()).ok_or_else(err)?
            }
            _ => return Err(err()),
        };
        let suit = Suit::from_char(suit_char).ok_or_else(err)?;
        Ok(Card::new(rank, suit))
    }
}

/// Parses a whitespace- or comma-separated list of cards, e.g. `"3d 4c, 5h"`.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, ParseCardError> {
    s.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(str::parse)
        .collect()
}

/// Helpers for card sets stored as `u64` bit masks (bit `i` set ⇔ card with
/// game-specific index `i` is in the set).
pub mod mask {
    /// Iterates over the set bit positions of `m`, lowest first.
    pub fn bits(mut m: u64) -> impl Iterator<Item = u8> {
        std::iter::from_fn(move || {
            if m == 0 {
                None
            } else {
                let i = m.trailing_zeros() as u8;
                m &= m - 1;
                Some(i)
            }
        })
    }

    /// Number of cards in the set.
    pub fn count(m: u64) -> u32 {
        m.count_ones()
    }

    /// Index of the highest card in a non-empty set.
    pub fn highest(m: u64) -> Option<u8> {
        (m != 0).then(|| 63 - m.leading_zeros() as u8)
    }

    /// Index of the lowest card in a non-empty set.
    pub fn lowest(m: u64) -> Option<u8> {
        (m != 0).then(|| m.trailing_zeros() as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display_round_trip() {
        for rank in Rank::ALL {
            for suit in Suit::ALL {
                let c = Card::new(rank, suit);
                assert_eq!(c.to_string().parse::<Card>().unwrap(), c);
            }
        }
    }

    #[test]
    fn parse_variants() {
        assert_eq!(
            "10s".parse::<Card>().unwrap(),
            Card::new(Rank::Ten, Suit::Spades)
        );
        assert_eq!(
            "3♦".parse::<Card>().unwrap(),
            Card::new(Rank::Three, Suit::Diamonds)
        );
        assert_eq!(
            "ah".parse::<Card>().unwrap(),
            Card::new(Rank::Ace, Suit::Hearts)
        );
        assert!("1s".parse::<Card>().is_err());
        assert!("Ax".parse::<Card>().is_err());
        assert!("".parse::<Card>().is_err());
        assert_eq!(parse_cards("3d, 4c  5h").unwrap().len(), 3);
    }

    #[test]
    fn mask_helpers() {
        let m = 0b1011_0000u64;
        assert_eq!(mask::bits(m).collect::<Vec<_>>(), vec![4, 5, 7]);
        assert_eq!(mask::count(m), 3);
        assert_eq!(mask::highest(m), Some(7));
        assert_eq!(mask::lowest(m), Some(4));
        assert_eq!(mask::highest(0), None);
    }
}
