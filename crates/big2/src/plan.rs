//! Hand planning: the fewest plays that shed a hand, every way to reach that
//! number, and how each legal action relates to those plans.
//!
//! Both `tiered` and `original` scoring count the cards left (rules §7), so the
//! first fact about a hand is how many plays it takes to empty it
//! ([`min_plays`]). The ways of reaching that number are the hand's
//! [`Decomposition`]s: the groups a player tries to keep together.
//! [`candidates`] labels every legal action with the decomposition it keeps or
//! the group it breaks, so bots and reason keys can say "this play breaks the
//! pair of 9s" without guessing.
//!
//! This module is the seed of `shedding-core` (strategy proposal §8): it knows
//! nothing of [`State`] apart from [`candidates`].
//!
//! # How the planner works
//!
//! Cards of different ranks only meet in five-card plays. A hand without
//! five-card plays therefore splits by rank: one play for every rank present,
//! plus one more for a rank held in all four suits (triple + single or
//! pair + pair). The planner enumerates the five-card combinations the way
//! [`crate::movegen`] does (from per-rank suit sets), tries every set of
//! disjoint ones (at most two fit in 13 cards), and adds the rank split of what
//! is left. The brute-force check in the tests covers every subset of the hand.
//!
//! # Tiebreak between decompositions
//!
//! [`decompositions`] lists the plans with the fewest plays in this fixed order:
//!
//! 1. fewest singles first (a single is the easiest play to be stuck with);
//! 2. then the plan whose five-card groups are strongest first (compare the
//!    strongest five-card group, then the next, …);
//! 3. then the plans' full group lists, strongest first (size, key, cards), so
//!    the order is deterministic.

use crate::combo::{classify, Combo, Kind};
use crate::movegen::combos_of_size;
use crate::rules::RuleSet;
use crate::state::{Action, State};
use big2_games::Game;
use std::cmp::Ordering;

/// Bit 0 of every rank's 4-bit suit set (`♦` of every rank).
const RANK_LOW_BITS: u64 = 0x0001_1111_1111_1111;

/// Plays needed to shed `hand` with singles, pairs and triples only: one per
/// rank present, two for a rank held in all four suits.
#[inline]
fn rank_split_plays(hand: u64) -> u32 {
    let any = (hand | hand >> 1 | hand >> 2 | hand >> 3) & RANK_LOW_BITS;
    let all = (hand & hand >> 1 & hand >> 2 & hand >> 3) & RANK_LOW_BITS;
    any.count_ones() + all.count_ones()
}

/// The fewest plays that shed every card of `hand` under `rules`.
///
/// `0` for an empty hand. Takes a few microseconds for a 13-card hand.
pub fn min_plays(hand: u64, rules: &RuleSet) -> u32 {
    if hand == 0 {
        return 0;
    }
    let mut best = rank_split_plays(hand);
    if hand.count_ones() < 5 {
        return best;
    }
    let mut fives = Vec::with_capacity(32);
    combos_of_size(hand, 5, rules, &mut fives);
    search_min(hand, &fives, 0, 0, &mut best);
    best
}

/// Tries every set of disjoint five-card combos from `fives[from..]` on top of
/// `used` already chosen, lowering `best` when a set does better.
fn search_min(rest: u64, fives: &[Combo], from: usize, used: u32, best: &mut u32) {
    for (i, c) in fives.iter().enumerate().skip(from) {
        if c.cards & rest != c.cards {
            continue;
        }
        let r = rest & !c.cards;
        let plays = used + 1 + rank_split_plays(r);
        if plays < *best {
            *best = plays;
        }
        // Another five-card group only helps if it can still beat `best`.
        if r.count_ones() >= 5 && used + 2 < *best {
            search_min(r, fives, i + 1, used + 1, best);
        }
    }
}

/// One way to shed a hand in the fewest plays.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Decomposition {
    /// The groups, strongest first: five-card groups, then triples, pairs and
    /// singles, each ordered by key (then cards) descending.
    pub combos: Vec<Combo>,
}

/// Sort key for groups inside a decomposition and for the final tiebreak.
#[inline]
fn group_key(c: &Combo) -> (u32, u32, u64) {
    (c.size(), c.key, c.cards)
}

impl Decomposition {
    fn new(mut combos: Vec<Combo>) -> Self {
        combos.sort_by_key(|c| std::cmp::Reverse(group_key(c)));
        Decomposition { combos }
    }

    /// Number of plays.
    pub fn plays(&self) -> usize {
        self.combos.len()
    }

    /// Number of singles among the groups.
    pub fn singles(&self) -> usize {
        self.combos
            .iter()
            .filter(|c| c.kind == Kind::Single)
            .count()
    }

    /// Keys of the five-card groups, strongest first.
    fn five_card_keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.combos.iter().filter(|c| c.size() == 5).map(|c| c.key)
    }

    /// `true` if `cards` is exactly one of the groups.
    pub fn contains(&self, cards: u64) -> bool {
        self.combos.iter().any(|c| c.cards == cards)
    }

    /// The group that `cards` takes the most cards from (ties: the larger, then
    /// stronger group), or `None` when `cards` touches no group.
    pub fn group_broken_by(&self, cards: u64) -> Option<&Combo> {
        self.combos
            .iter()
            .filter(|c| c.cards & cards != 0)
            .max_by_key(|c| ((c.cards & cards).count_ones(), group_key(c)))
    }

    /// Tiebreak order (see the module docs).
    fn cmp_order(&self, other: &Self) -> Ordering {
        self.singles()
            .cmp(&other.singles())
            .then_with(|| other.five_card_keys().cmp(self.five_card_keys()))
            .then_with(|| {
                other
                    .combos
                    .iter()
                    .map(group_key)
                    .cmp(self.combos.iter().map(group_key))
            })
    }
}

/// Every way to shed `hand` in [`min_plays`] plays, best first (tiebreak in the
/// module docs), at most `max` of them.
///
/// Returns one empty decomposition for an empty hand.
pub fn decompositions(hand: u64, rules: &RuleSet, max: usize) -> Vec<Decomposition> {
    let target = min_plays(hand, rules);
    let mut fives = Vec::with_capacity(32);
    if hand.count_ones() >= 5 {
        combos_of_size(hand, 5, rules, &mut fives);
    }
    let mut out = Vec::new();
    let mut chosen = Vec::new();
    collect_five_card_sets(hand, &fives, 0, target, rules, &mut chosen, &mut out);
    out.sort_by(Decomposition::cmp_order);
    out.truncate(max);
    out
}

/// Recurses over sets of disjoint five-card groups; for each set that reaches
/// `target` plays, expands the rank split of the rest into decompositions.
fn collect_five_card_sets(
    rest: u64,
    fives: &[Combo],
    from: usize,
    target: u32,
    rules: &RuleSet,
    chosen: &mut Vec<Combo>,
    out: &mut Vec<Decomposition>,
) {
    let used = chosen.len() as u32;
    if used + rank_split_plays(rest) == target {
        expand_rank_split(rest, rules, chosen, out);
    }
    if rest.count_ones() < 5 || used + 1 > target {
        return;
    }
    for (i, c) in fives.iter().enumerate().skip(from) {
        if c.cards & rest != c.cards {
            continue;
        }
        chosen.push(*c);
        collect_five_card_sets(rest & !c.cards, fives, i + 1, target, rules, chosen, out);
        chosen.pop();
    }
}

/// Adds every rank split of `rest` (ranks with 1–3 cards have one group; a
/// rank with all four suits has seven) on top of `prefix`.
fn expand_rank_split(rest: u64, rules: &RuleSet, prefix: &[Combo], out: &mut Vec<Decomposition>) {
    let mut fixed: Vec<Combo> = prefix.to_vec();
    let mut quads: Vec<u64> = Vec::new();
    for r in 0..crate::card::NUM_RANKS {
        let m = rest & (0xFu64 << (r * 4));
        match m.count_ones() {
            0 => {}
            4 => quads.push(m),
            _ => fixed.push(classify(m, rules).expect("one rank is a legal group")),
        }
    }
    expand_quads(&quads, rules, &mut fixed, out);
}

fn expand_quads(
    quads: &[u64],
    rules: &RuleSet,
    acc: &mut Vec<Combo>,
    out: &mut Vec<Decomposition>,
) {
    let Some((&q, more)) = quads.split_first() else {
        out.push(Decomposition::new(acc.clone()));
        return;
    };
    let low = q.trailing_zeros();
    let bit = |s: u32| 1u64 << (low + s);
    // Triple + single: four ways.
    for s in 0..4 {
        acc.push(classify(q & !bit(s), rules).expect("triple"));
        acc.push(classify(bit(s), rules).expect("single"));
        expand_quads(more, rules, acc, out);
        acc.pop();
        acc.pop();
    }
    // Pair + pair: three ways (♦ with each other suit).
    for s in 1..4 {
        let a = bit(0) | bit(s);
        acc.push(classify(a, rules).expect("pair"));
        acc.push(classify(q & !a, rules).expect("pair"));
        expand_quads(more, rules, acc, out);
        acc.pop();
        acc.pop();
    }
}

/// A legal action labelled against the hand's decompositions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub action: Action,
    /// Index of the best (lowest) decomposition that has this play as a group.
    /// `None` for a pass or a play that is a group of no decomposition.
    pub keeps: Option<usize>,
    /// For a play that keeps no decomposition: the group of the best
    /// decomposition it takes the most cards from (there always is one, since
    /// every card belongs to a group). `None` for a pass or when it keeps.
    pub breaks: Option<Combo>,
}

/// Every legal action of the player to act, labelled against `decomps` (the
/// decompositions of that player's hand, as [`decompositions`] returns them).
///
/// Order: actions that keep a decomposition first, by the index they keep (then
/// the engine's legal-action order), then the plays that break a group, then
/// the pass. Nothing is pruned: the list is the engine's legal actions with
/// labels. Callers who want fewer moves drop the breakers.
pub fn candidates(state: &State, decomps: &[Decomposition]) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = state
        .legal_actions_vec()
        .into_iter()
        .map(|action| {
            let cards = action.cards();
            if cards == 0 {
                return Candidate {
                    action,
                    keeps: None,
                    breaks: None,
                };
            }
            let keeps = decomps.iter().position(|d| d.contains(cards));
            let breaks = match keeps {
                Some(_) => None,
                None => decomps
                    .first()
                    .and_then(|d| d.group_broken_by(cards).copied()),
            };
            Candidate {
                action,
                keeps,
                breaks,
            }
        })
        .collect();
    out.sort_by_key(|c| (c.keeps.is_none(), c.keeps.unwrap_or(0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_mask;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    /// Reference: minimum set cover over every legal subset of the hand, as a
    /// DP over subsets of the hand's cards (each subset's lowest card must be
    /// in the group that covers it).
    fn brute_force_min_plays(hand: u64, rules: &RuleSet) -> u32 {
        let cards: Vec<u8> = big2_cards::mask::bits(hand).collect();
        let n = cards.len();
        // Legal groups as compressed masks over the card positions.
        let mut groups: Vec<u32> = Vec::new();
        for sub in 1u32..(1 << n) {
            if !matches!(sub.count_ones(), 1 | 2 | 3 | 5) {
                continue;
            }
            let m = (0..n)
                .filter(|i| sub & (1 << i) != 0)
                .fold(0u64, |acc, i| acc | 1 << cards[i]);
            if classify(m, rules).is_some() {
                groups.push(sub);
            }
        }
        let mut dp = vec![u32::MAX; 1 << n];
        dp[0] = 0;
        for s in 1u32..(1 << n) {
            let low = 1 << s.trailing_zeros();
            let mut best = u32::MAX;
            for &g in &groups {
                if g & low != 0 && g & !s == 0 {
                    best = best.min(dp[(s & !g) as usize] + 1);
                }
            }
            dp[s as usize] = best;
        }
        dp[(1usize << n) - 1]
    }

    fn random_hand(rng: &mut rand::rngs::SmallRng, n: usize) -> u64 {
        let mut deck: Vec<u8> = (0..52).collect();
        deck.shuffle(rng);
        deck[..n].iter().fold(0u64, |m, &i| m | 1 << i)
    }

    fn h(s: &str) -> u64 {
        parse_mask(s).unwrap()
    }

    fn names(d: &Decomposition) -> Vec<String> {
        d.combos.iter().map(|c| format!("{c:?}")).collect()
    }

    #[test]
    fn min_plays_matches_brute_force_on_random_hands() {
        let rules = RuleSet::default();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(4);
        for i in 0..10_000 {
            let n = 1 + i % 13;
            let hand = random_hand(&mut rng, n);
            assert_eq!(
                min_plays(hand, &rules),
                brute_force_min_plays(hand, &rules),
                "hand {}",
                crate::card::format_mask(hand)
            );
        }
    }

    #[test]
    fn min_plays_edge_cases() {
        let rules = RuleSet::default();
        assert_eq!(min_plays(0, &rules), 0);
        assert_eq!(min_plays(h("3d"), &rules), 1);
        assert_eq!(min_plays(h("3d 3c 3h 3s"), &rules), 2);
        // Two five-card groups plus three singles.
        assert_eq!(
            min_plays(h("3d 4c 5h 6s 7d 9d Td Jd Qd Kd As 2s 8c"), &rules),
            5
        );
    }

    #[test]
    fn rules_section_3_examples_decompose_as_one_group() {
        let rules = RuleSet::default();
        for (cards, kind) in [
            ("3d 3c 3h 9s 9d", Kind::FullHouse),
            ("3d 4c 5h 6s 7d", Kind::Straight),
            ("3d 5d 7d 9d 2d", Kind::Flush),
            ("3d 3c 3h 3s 9d", Kind::FourOfAKind),
            ("3h 4h 5h 6h 7h", Kind::StraightFlush),
        ] {
            assert_eq!(min_plays(h(cards), &rules), 1, "{cards}");
            let d = decompositions(h(cards), &rules, 8);
            assert_eq!(d.len(), 1, "{cards}: {d:?}");
            assert_eq!(d[0].combos.len(), 1);
            assert_eq!(d[0].combos[0].kind, kind, "{cards}");
        }
    }

    #[test]
    fn full_house_plus_single_beats_triple_pair_single_but_not_two_triples() {
        let rules = RuleSet::default();
        // Full house + K (2 plays) rather than triple + pair + K (3 plays).
        let d = decompositions(h("3d 3c 3h 9s 9d Kd"), &rules, 8);
        assert_eq!(d.len(), 1);
        assert_eq!(
            names(&d[0]),
            vec!["full_house(3d 3c 3h 9d 9s)", "single(Kd)"]
        );
        // Two triples (0 singles) come before any full house + single.
        let d = decompositions(h("3d 3c 3h 9s 9d 9c"), &rules, 8);
        assert_eq!(d.len(), 7, "{d:?}");
        assert_eq!(names(&d[0]), vec!["triple(9d 9c 9s)", "triple(3d 3c 3h)"]);
        assert!(d[1..].iter().all(|x| x.singles() == 1));
        // Among full houses, the strongest kept five-card group comes first.
        assert_eq!(d[1].combos[0].kind, Kind::FullHouse);
        assert!(d[1].combos[0].key >= d[2].combos[0].key);
    }

    #[test]
    fn highest_kept_flush_first() {
        let rules = RuleSet::default();
        let d = decompositions(h("3d 4d 5d 6d 8d 9d"), &rules, 8);
        assert_eq!(d.len(), 6);
        assert_eq!(names(&d[0]), vec!["flush(4d 5d 6d 8d 9d)", "single(3d)"]);
        assert_eq!(names(&d[5]), vec!["flush(3d 4d 5d 6d 8d)", "single(9d)"]);
        assert_eq!(decompositions(h("3d 4d 5d 6d 8d 9d"), &rules, 2).len(), 2);
    }

    #[test]
    fn four_of_a_rank_splits_seven_ways_pairs_first() {
        let rules = RuleSet::default();
        let d = decompositions(h("3d 3c 3h 3s"), &rules, 8);
        assert_eq!(d.len(), 7);
        assert_eq!(names(&d[0]), vec!["pair(3h 3s)", "pair(3d 3c)"]);
        assert!(d[..3].iter().all(|x| x.singles() == 0));
        assert!(d[3..].iter().all(|x| x.singles() == 1));
        let mut distinct = d.clone();
        distinct.dedup();
        assert_eq!(distinct.len(), 7);
    }

    #[test]
    fn empty_hand_has_one_empty_decomposition() {
        let d = decompositions(0, &RuleSet::default(), 8);
        assert_eq!(d, vec![Decomposition { combos: vec![] }]);
    }

    #[test]
    fn every_decomposition_covers_the_hand_in_min_plays() {
        let rules = RuleSet::default();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
        for _ in 0..500 {
            let hand = random_hand(&mut rng, 13);
            let target = min_plays(hand, &rules) as usize;
            for d in decompositions(hand, &rules, usize::MAX) {
                assert_eq!(d.plays(), target);
                let mut seen = 0u64;
                for c in &d.combos {
                    assert_eq!(seen & c.cards, 0, "overlap in {d:?}");
                    assert_eq!(classify(c.cards, &rules), Some(*c));
                    seen |= c.cards;
                }
                assert_eq!(seen, hand, "{d:?} does not cover the hand");
            }
        }
    }

    #[test]
    fn decomposition_and_candidate_counts_on_random_deals() {
        let rules = RuleSet::default();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
        let n = 1_000;
        let (mut short_lists, mut short_uncapped, mut few_candidates) = (0, 0, 0);
        for _ in 0..n {
            let state = State::new_deal(rules, &mut rng);
            let hand = state.hand(state.current_player().unwrap());
            if decompositions(hand, &rules, 8).len() <= 20 {
                short_lists += 1;
            }
            let all = decompositions(hand, &rules, usize::MAX);
            if all.len() <= 20 {
                short_uncapped += 1;
            }
            let cands = candidates(&state, &all);
            assert_eq!(cands.len(), state.legal_actions_vec().len());
            if cands.len() <= 24 {
                few_candidates += 1;
            }
        }
        eprintln!(
            "decompositions(hand, 8) <= 20: {short_lists}/{n}; uncapped <= 20: {short_uncapped}/{n}; \
             opening candidates <= 24: {few_candidates}/{n}"
        );
        assert!(short_lists * 100 >= 95 * n);
        assert!(few_candidates * 100 >= 95 * n);
    }

    #[test]
    fn candidates_label_keeps_and_breaks() {
        let rules = RuleSet::default();
        // Seat 0 leads with 9s 9d 3c Kd: best plan is pair of 9s, K, 3.
        let hands = [
            h("3c 9d 9s Kd"),
            h("4d 4c 4h"),
            h("5d 5c 5h"),
            h("6d 6c 6h"),
        ];
        let state = State::from_position(rules, hands, 0, None).unwrap();
        let decomps = decompositions(hands[0], &rules, 8);
        assert_eq!(decomps.len(), 1);
        let cands = candidates(&state, &decomps);
        let find = |cards: &str| {
            cands
                .iter()
                .find(|c| c.action.cards() == h(cards))
                .unwrap_or_else(|| panic!("{cards} missing from {cands:?}"))
        };
        assert_eq!(find("9d 9s").keeps, Some(0));
        assert_eq!(find("Kd").keeps, Some(0));
        assert_eq!(find("3c").keeps, Some(0));
        let single_nine = find("9s");
        assert_eq!(single_nine.keeps, None);
        assert_eq!(single_nine.breaks.map(|c| c.cards), Some(h("9d 9s")));
        // Keepers come first; no pass when leading.
        assert!(cands.iter().take(3).all(|c| c.keeps == Some(0)));
        assert!(cands.iter().all(|c| c.action != Action::Pass));

        // Following a pair: the pass is last, labelled with nothing.
        let state = State::from_position(rules, hands, 0, Some((3, h("8d 8c")))).unwrap();
        let cands = candidates(&state, &decomps);
        let last = cands.last().unwrap();
        assert_eq!(last.action, Action::Pass);
        assert_eq!((last.keeps, last.breaks), (None, None));
    }
}
