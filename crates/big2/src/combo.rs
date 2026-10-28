//! Combinations (rules §3) and how they compare (rules §4).
//!
//! A [`Combo`] stores its cards and a precomputed comparison `key`. Two combos of
//! the same size compare by key alone, so "does this beat the current play?" is
//! one integer comparison. The key layout is:
//!
//! ```text
//!   bits 24..   kind (Single = 1 … StraightFlush = 8)
//!   bits 0..24  strength within the kind (see `classify`)
//! ```
//!
//! Putting the kind in the high bits makes every flush beat every straight, every
//! full house beat every flush, and so on.

use crate::card::{rank_of, suit_mask, suits_at, NUM_RANKS};
use crate::rules::{FlushOrder, RuleSet, Straights};
use big2_cards::mask;
use std::fmt;

/// The kind of a combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Single = 1,
    Pair = 2,
    Triple = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

impl Kind {
    /// The snake_case name used in Python and logs.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Single => "single",
            Kind::Pair => "pair",
            Kind::Triple => "triple",
            Kind::Straight => "straight",
            Kind::Flush => "flush",
            Kind::FullHouse => "full_house",
            Kind::FourOfAKind => "four_of_a_kind",
            Kind::StraightFlush => "straight_flush",
        }
    }
}

/// A legal combination of cards with its comparison key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Combo {
    /// The cards, as a Big Two card mask.
    pub cards: u64,
    /// What kind of combination this is.
    pub kind: Kind,
    /// Comparison key; only meaningful against combos with the same number of cards.
    pub key: u32,
}

impl Combo {
    /// Number of cards (1, 2, 3 or 5).
    #[inline]
    pub fn size(&self) -> u32 {
        self.cards.count_ones()
    }

    /// `true` if `self` may be played on top of `current` (same size, higher key).
    #[inline]
    pub fn beats(&self, current: &Combo) -> bool {
        self.size() == current.size() && self.key > current.key
    }
}

impl fmt::Debug for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({})",
            self.kind.name(),
            crate::card::format_mask(self.cards)
        )
    }
}

impl fmt::Display for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&crate::card::format_mask(self.cards))
    }
}

/// Number of straight sequences when 2s are allowed in straights.
pub const NUM_STRAIGHTS: usize = 10;

/// Rank sets (13-bit, bit = Big Two rank) of each straight, weakest first
/// (rules §3.1): `3-7, 4-8, …, T-A, A-2-3-4-5, 2-3-4-5-6`.
pub const STRAIGHT_RANKS: [u16; NUM_STRAIGHTS] = {
    let mut out = [0u16; NUM_STRAIGHTS];
    let mut k = 0;
    while k < 8 {
        out[k] = 0b11111 << k;
        k += 1;
    }
    // Big Two ranks: 3=0 4=1 5=2 6=3 … A=11 2=12
    out[8] = (1 << 11) | (1 << 12) | 0b111; // A 2 3 4 5
    out[9] = (1 << 12) | 0b1111; // 2 3 4 5 6
    out
};

/// The rank whose card decides suit ties for each straight (its "top card").
pub const STRAIGHT_TOP_RANK: [u8; NUM_STRAIGHTS] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 12];

/// How many sequences of [`STRAIGHT_RANKS`] are allowed under `rules`.
#[inline]
pub fn num_straights(rules: &RuleSet) -> usize {
    match rules.straights {
        Straights::WithLowTwos => NUM_STRAIGHTS,
        Straights::NoTwos => 8,
    }
}

#[inline]
fn key(kind: Kind, detail: u32) -> u32 {
    debug_assert!(detail < (1 << 24));
    ((kind as u32) << 24) | detail
}

/// The 13-bit set of ranks present in `cards`.
#[inline]
fn rank_set(cards: u64) -> u16 {
    let mut s = 0u16;
    for r in 0..NUM_RANKS {
        if suits_at(cards, r) != 0 {
            s |= 1 << r;
        }
    }
    s
}

/// Classifies a set of cards. Returns `None` if the cards do not form a legal
/// combination under `rules`.
///
/// Within-kind strength (the low 24 bits of the key):
///
/// | Kind | Strength |
/// |---|---|
/// | Single | card index (rank·4 + suit) |
/// | Pair | rank·4 + highest suit |
/// | Triple | rank |
/// | Straight | sequence·4 + suit of top card |
/// | Flush, rank first | five ranks high→low (4 bits each), then suit (2 bits) |
/// | Flush, suit first | suit, then five ranks high→low |
/// | Full house | rank of the triple |
/// | Four of a kind | rank of the four |
/// | Straight flush | sequence·4 + suit |
pub fn classify(cards: u64, rules: &RuleSet) -> Option<Combo> {
    let n = cards.count_ones();
    let top = mask::highest(cards)?;
    let make = |kind, detail| {
        Some(Combo {
            cards,
            kind,
            key: key(kind, detail),
        })
    };
    match n {
        1 => make(Kind::Single, top as u32),
        2 | 3 => {
            let low = mask::lowest(cards)?;
            if rank_of(low) != rank_of(top) {
                return None;
            }
            if n == 2 {
                make(Kind::Pair, top as u32) // top = rank*4 + highest suit
            } else {
                make(Kind::Triple, rank_of(top) as u32)
            }
        }
        5 => classify_five(cards, rules),
        _ => None,
    }
}

fn classify_five(cards: u64, rules: &RuleSet) -> Option<Combo> {
    let make = |kind, detail| {
        Some(Combo {
            cards,
            kind,
            key: key(kind, detail),
        })
    };
    let ranks = rank_set(cards);
    let suit_masks = [0, 1, 2, 3].map(|s| cards & suit_mask(s));
    let flush_suit = suit_masks
        .iter()
        .position(|&m| m == cards)
        .map(|s| s as u32);

    if ranks.count_ones() == 5 {
        let straight = STRAIGHT_RANKS[..num_straights(rules)]
            .iter()
            .position(|&s| s == ranks);
        if let Some(seq) = straight {
            return match flush_suit {
                Some(suit) => make(Kind::StraightFlush, seq as u32 * 4 + suit),
                None => {
                    let top_suits = suits_at(cards, STRAIGHT_TOP_RANK[seq]);
                    let top_suit = 7 - top_suits.leading_zeros(); // highest set bit of a u8
                    make(Kind::Straight, seq as u32 * 4 + top_suit)
                }
            };
        }
        if let Some(suit) = flush_suit {
            // Ranks high → low, 4 bits each.
            let mut ranks_desc = 0u32;
            for r in (0..NUM_RANKS).rev() {
                if ranks & (1 << r) != 0 {
                    ranks_desc = (ranks_desc << 4) | r as u32;
                }
            }
            let detail = match rules.flush_order {
                FlushOrder::RankFirst => (ranks_desc << 2) | suit,
                FlushOrder::SuitFirst => (suit << 20) | ranks_desc,
            };
            return make(Kind::Flush, detail);
        }
        return None;
    }

    // Two distinct ranks: full house (3+2) or four of a kind (4+1).
    if ranks.count_ones() == 2 {
        let lo = ranks.trailing_zeros() as u8;
        let hi = 15 - ranks.leading_zeros() as u8;
        let (n_lo, n_hi) = (
            suits_at(cards, lo).count_ones(),
            suits_at(cards, hi).count_ones(),
        );
        return match (n_lo, n_hi) {
            (3, 2) => make(Kind::FullHouse, lo as u32),
            (2, 3) => make(Kind::FullHouse, hi as u32),
            (4, 1) => make(Kind::FourOfAKind, lo as u32),
            (1, 4) => make(Kind::FourOfAKind, hi as u32),
            _ => None,
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_mask;

    fn c(s: &str) -> Option<Combo> {
        classify(parse_mask(s).unwrap(), &RuleSet::default())
    }
    fn kind(s: &str) -> Option<Kind> {
        c(s).map(|x| x.kind)
    }
    fn beats(a: &str, b: &str) -> bool {
        c(a).unwrap().beats(&c(b).unwrap())
    }

    #[test]
    fn kinds() {
        assert_eq!(kind("3d"), Some(Kind::Single));
        assert_eq!(kind("9s 9d"), Some(Kind::Pair));
        assert_eq!(kind("9s 8d"), None);
        assert_eq!(kind("Ks Kd Kh"), Some(Kind::Triple));
        assert_eq!(kind("3d 4c 5h 6s 7d"), Some(Kind::Straight));
        assert_eq!(kind("3d 5d 7d 9d 2d"), Some(Kind::Flush));
        assert_eq!(kind("3d 3c 3h 9s 9d"), Some(Kind::FullHouse));
        assert_eq!(kind("3d 3c 3h 3s 9d"), Some(Kind::FourOfAKind));
        assert_eq!(kind("3h 4h 5h 6h 7h"), Some(Kind::StraightFlush));
        assert_eq!(kind("3d 3c 9h 9s"), None); // two pairs: 4 cards
        assert_eq!(kind("3d 3c 9h 9s Kd"), None); // two pairs + kicker
        assert_eq!(kind(""), None);
    }

    #[test]
    fn straights_with_low_twos() {
        assert_eq!(kind("Ad 2c 3h 4s 5d"), Some(Kind::Straight));
        assert_eq!(kind("2c 3h 4s 5d 6d"), Some(Kind::Straight));
        assert_eq!(kind("Jd Qc Kh As 2d"), None, "no wrap through the 2");
        assert_eq!(kind("Qd Kc Ah 2s 3d"), None);
        assert_eq!(kind("Kd Ac 2h 3s 4d"), None);
        // order: T-A < A-5 < 2-6
        assert!(beats("Ad 2c 3h 4s 5d", "Td Jc Qh Ks As"));
        assert!(beats("2d 3c 4h 5s 6d", "Ad 2s 3h 4s 5s"));
    }

    #[test]
    fn straights_without_twos() {
        let rules = RuleSet {
            straights: Straights::NoTwos,
            ..Default::default()
        };
        let m = parse_mask("Ad 2c 3h 4s 5d").unwrap();
        assert_eq!(classify(m, &rules), None);
        let sf = parse_mask("Ad 2d 3d 4d 5d").unwrap();
        assert_eq!(classify(sf, &rules).unwrap().kind, Kind::Flush);
    }

    #[test]
    fn straight_ties_break_on_top_card_suit() {
        // rules doc §4.2 example
        assert!(beats("Td Jc Qh Kd As", "Ts Js Qs Kh Ah"));
        // for A-2-3-4-5 the 2 is the top card
        assert!(beats("Ad 2s 3d 4d 5c", "As 2h 3s 4s 5s"));
    }

    #[test]
    fn pairs_compare_by_rank_then_highest_suit() {
        assert!(beats("9s 9d", "9h 9c"));
        assert!(beats("Td Tc", "9s 9h"));
        assert!(!beats("9h 9c", "9s 9d"));
    }

    #[test]
    fn five_card_category_order() {
        let straight = "9d Tc Jh Qs Kd";
        let flush = "3d 5d 7d 9d Jd";
        let full_house = "4d 4c 4h 3s 3d";
        let quads = "5d 5c 5h 5s 3c";
        let straight_flush = "3c 4c 5c 6c 7c";
        let order = [straight, flush, full_house, quads, straight_flush];
        for (i, a) in order.iter().enumerate() {
            for b in &order[..i] {
                assert!(beats(a, b), "{a} should beat {b}");
                assert!(!beats(b, a));
            }
        }
    }

    #[test]
    fn flush_rank_first_vs_suit_first() {
        // rules doc §4.2 example: the 2-high diamond flush beats the A-high spade flush
        assert!(beats("3d 5d 7d 9d 2d", "4s 6s 8s Ts As"));
        let suit_first = RuleSet {
            flush_order: FlushOrder::SuitFirst,
            ..Default::default()
        };
        let a = classify(parse_mask("3d 5d 7d 9d 2d").unwrap(), &suit_first).unwrap();
        let b = classify(parse_mask("4s 6s 8s Ts As").unwrap(), &suit_first).unwrap();
        assert!(b.beats(&a));
        // rank first: later ranks matter when the top ranks tie
        assert!(beats("4d 6d 8d Td Ad", "3s 6s 8s Ts As"));
        // full rank tie → suit
        assert!(beats("3s 5s 7s 9s Js", "3h 5h 7h 9h Jh"));
    }

    #[test]
    fn full_house_and_quads_ignore_the_extra_cards() {
        assert!(beats("4d 4c 4h 3s 3d", "3c 3h 3s 2s 2d"));
        assert!(beats("6d 6c 6h 6s 3c", "5d 5c 5h 5s 2s"));
    }

    #[test]
    fn sizes_never_mix() {
        assert!(!c("2s").unwrap().beats(&c("3d 3c").unwrap()));
        assert!(
            !c("5d 5c 5h 5s 3c").unwrap().beats(&c("3d").unwrap()),
            "no bombs"
        );
    }
}
