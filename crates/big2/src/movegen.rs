//! Legal-move generation.
//!
//! Combinations are built directly from the hand's per-rank suit sets instead of
//! testing every subset of the hand (a 13-card hand has 1,287 five-card subsets,
//! of which usually only a few dozen are legal). Every generated set still goes
//! through [`classify`], so classification has a single source of truth.

use crate::card::{suit_mask, suits_at, NUM_RANKS};
use crate::combo::{classify, num_straights, Combo, Kind, STRAIGHT_RANKS};
use crate::rules::RuleSet;
use poker_cards::mask;

/// Calls `f` for each subset of the 4-bit suit set `suits` with exactly `k` suits.
#[inline]
fn for_each_suit_subset(suits: u8, k: u32, mut f: impl FnMut(u8)) {
    let mut sub = suits;
    while sub != 0 {
        if sub.count_ones() == k {
            f(sub);
        }
        sub = (sub - 1) & suits;
    }
}

/// Places a 4-bit suit set at Big Two rank `rank` in a card mask.
#[inline]
fn at_rank(suits: u8, rank: u8) -> u64 {
    (suits as u64) << (rank * 4)
}

/// Appends every legal combination in `hand` with `size` cards (1, 2, 3 or 5).
pub fn combos_of_size(hand: u64, size: u32, rules: &RuleSet, out: &mut Vec<Combo>) {
    let mut push = |cards: u64| {
        if let Some(c) = classify(cards, rules) {
            out.push(c);
        }
    };
    match size {
        1 => mask::bits(hand).for_each(|i| push(1u64 << i)),
        2 | 3 => {
            for r in 0..NUM_RANKS {
                for_each_suit_subset(suits_at(hand, r), size, |s| push(at_rank(s, r)));
            }
        }
        5 => five_card_combos(hand, rules, &mut push),
        _ => {}
    }
}

fn five_card_combos(hand: u64, rules: &RuleSet, push: &mut impl FnMut(u64)) {
    let suits: [u8; 13] = std::array::from_fn(|r| suits_at(hand, r as u8));

    // Straights and straight flushes: one card from each rank of the sequence.
    for &ranks in &STRAIGHT_RANKS[..num_straights(rules)] {
        let rs: Vec<u8> = (0..NUM_RANKS).filter(|r| ranks & (1 << r) != 0).collect();
        if rs.iter().any(|&r| suits[r as usize] == 0) {
            continue;
        }
        for a in mask::bits(suits[rs[0] as usize] as u64) {
            for b in mask::bits(suits[rs[1] as usize] as u64) {
                for c in mask::bits(suits[rs[2] as usize] as u64) {
                    for d in mask::bits(suits[rs[3] as usize] as u64) {
                        for e in mask::bits(suits[rs[4] as usize] as u64) {
                            push(
                                at_rank(1 << a, rs[0])
                                    | at_rank(1 << b, rs[1])
                                    | at_rank(1 << c, rs[2])
                                    | at_rank(1 << d, rs[3])
                                    | at_rank(1 << e, rs[4]),
                            );
                        }
                    }
                }
            }
        }
    }

    // Flushes: any five cards of one suit that are not a straight flush
    // (straight flushes were produced above).
    for s in 0..4u8 {
        let in_suit: Vec<u8> = mask::bits(hand & suit_mask(s)).collect();
        let n = in_suit.len();
        if n < 5 {
            continue;
        }
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    for l in k + 1..n {
                        for m in l + 1..n {
                            let cards = [i, j, k, l, m]
                                .iter()
                                .fold(0u64, |acc, &x| acc | 1u64 << in_suit[x]);
                            if let Some(c) = classify(cards, rules) {
                                if c.kind == Kind::Flush {
                                    push(cards);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Full houses: a triple plus a pair of another rank.
    for t in 0..NUM_RANKS {
        if suits[t as usize].count_ones() < 3 {
            continue;
        }
        for_each_suit_subset(suits[t as usize], 3, |ts| {
            for p in 0..NUM_RANKS {
                if p != t {
                    for_each_suit_subset(suits[p as usize], 2, |ps| {
                        push(at_rank(ts, t) | at_rank(ps, p))
                    });
                }
            }
        });
    }

    // Four of a kind: all four cards of a rank plus any other card.
    for q in 0..NUM_RANKS {
        if suits[q as usize] == 0xF {
            let quad = at_rank(0xF, q);
            for kicker in mask::bits(hand & !quad) {
                push(quad | 1u64 << kicker);
            }
        }
    }
}

/// All legal combinations a player holding `hand` may play.
///
/// * `to_beat`: the current play, or `None` when leading a trick.
/// * `must_include`: cards every play must contain (3♦ on the opening play), or 0.
///
/// The pass action is not included; see [`crate::State::legal_actions`].
pub fn legal_combos(
    hand: u64,
    to_beat: Option<&Combo>,
    must_include: u64,
    rules: &RuleSet,
    out: &mut Vec<Combo>,
) {
    out.clear();
    match to_beat {
        Some(current) => {
            combos_of_size(hand, current.size(), rules, out);
            out.retain(|c| c.key > current.key);
        }
        None => {
            for size in [1, 2, 3, 5] {
                combos_of_size(hand, size, rules, out);
            }
        }
    }
    if must_include != 0 {
        out.retain(|c| c.cards & must_include == must_include);
    }
}

/// The highest single card in `hand` (used by the last-card duty).
pub fn highest_card(hand: u64) -> Option<u8> {
    mask::highest(hand)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_mask;

    /// Reference: test every subset of the hand.
    fn brute_force(hand: u64, size: u32, rules: &RuleSet) -> Vec<u64> {
        let cards: Vec<u8> = mask::bits(hand).collect();
        let n = cards.len();
        let mut out = Vec::new();
        for sub in 0u32..(1 << n) {
            if sub.count_ones() != size {
                continue;
            }
            let m = (0..n)
                .filter(|i| sub & (1 << i) != 0)
                .fold(0u64, |acc, i| acc | 1 << cards[i]);
            if classify(m, rules).is_some() {
                out.push(m);
            }
        }
        out.sort();
        out
    }

    fn generated(hand: u64, size: u32, rules: &RuleSet) -> Vec<u64> {
        let mut v = Vec::new();
        combos_of_size(hand, size, rules, &mut v);
        let mut m: Vec<u64> = v.iter().map(|c| c.cards).collect();
        m.sort();
        m
    }

    #[test]
    fn matches_brute_force_on_random_hands() {
        use rand::{seq::SliceRandom, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let mut deck: Vec<u8> = (0..52).collect();
        for rules in [
            RuleSet::default(),
            RuleSet {
                straights: crate::Straights::NoTwos,
                ..Default::default()
            },
        ] {
            for _ in 0..300 {
                deck.shuffle(&mut rng);
                let hand = deck[..13].iter().fold(0u64, |m, &i| m | 1 << i);
                for size in [1, 2, 3, 5] {
                    assert_eq!(
                        generated(hand, size, &rules),
                        brute_force(hand, size, &rules),
                        "hand {hand:x} size {size}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_duplicates() {
        let hand = parse_mask("3d 4d 5d 6d 7d 8d 9d Td 3c 3h 3s 4c 4h").unwrap();
        let mut v = Vec::new();
        legal_combos(hand, None, 0, &RuleSet::default(), &mut v);
        let mut cards: Vec<u64> = v.iter().map(|c| c.cards).collect();
        let n = cards.len();
        cards.sort();
        cards.dedup();
        assert_eq!(cards.len(), n);
    }

    #[test]
    fn opening_must_include_three_of_diamonds() {
        let hand = parse_mask("3d 3c 4h 5s 6d 7c 9h Ts Jd Qc Kh As 2d").unwrap();
        let mut v = Vec::new();
        legal_combos(hand, None, 1, &RuleSet::default(), &mut v);
        assert!(!v.is_empty());
        assert!(v.iter().all(|c| c.cards & 1 == 1));
        // single 3d, pair 3d3c, and straights 3-7 containing 3d
        assert!(v.iter().any(|c| c.kind == Kind::Single));
        assert!(v.iter().any(|c| c.kind == Kind::Pair));
        assert!(v.iter().any(|c| c.kind == Kind::Straight));
    }

    #[test]
    fn following_filters_by_size_and_strength() {
        let hand = parse_mask("3d 9c 9h Ks 2d").unwrap();
        let current = classify(parse_mask("9d 9s").unwrap(), &RuleSet::default()).unwrap();
        let mut v = Vec::new();
        legal_combos(hand, Some(&current), 0, &RuleSet::default(), &mut v);
        assert!(v.is_empty(), "9c9h does not beat 9d9s: {v:?}");
        let current = classify(parse_mask("9d 9c").unwrap(), &RuleSet::default()).unwrap();
        legal_combos(
            parse_mask("9h 9s Kd").unwrap(),
            Some(&current),
            0,
            &RuleSet::default(),
            &mut v,
        );
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn helpers() {
        assert_eq!(highest_card(parse_mask("3d 2s 9h").unwrap()), Some(51));
    }
}
