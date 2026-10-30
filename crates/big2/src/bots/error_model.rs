//! [`ErrorModel`]: human-style named mistakes on top of any policy (strategy
//! proposal §4, ladder v2).
//!
//! Weaker levels should lose the way people lose, not by random noise: they
//! hold a 2 too long, break a pair to beat a single, miss a single nobody can
//! beat. The model wraps a policy; at each decision it checks, in a fixed order,
//! whether each mistake *applies* (the position allows it and the inner policy
//! did not already play that way). An applicable mistake fires with its rate;
//! the first that fires replaces the inner policy's move. The mistake that fired
//! is returned with the move ([`ErrorModel::choose_explained`]), so a review can
//! name it with its reason key ([`Mistake::key`]).
//!
//! Every replacement is taken from the legal actions, so the model never plays
//! an illegal move (last-card duty included: the engine has already filtered the
//! singles).

use crate::card::{rank_of, FULL_DECK};
use crate::combo::Kind;
use crate::state::{Action, State};
use big2_cards::mask;
use big2_games::{Game, Policy};
use rand::Rng;

/// Big Two rank index of a 2.
const RANK_TWO: u8 = 12;

/// The cards of rank 2 in `cards`.
fn twos(cards: u64) -> u64 {
    cards & (0xFu64 << (RANK_TWO * 4))
}

/// A named mistake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mistake {
    /// The inner policy plays a 2; the mistake keeps every 2 instead (a pass when
    /// following, else the weakest play without a 2).
    HoldTwoTooLong,
    /// Following a single: plays the weakest single that comes out of a pair
    /// (or larger set of one rank) while the inner policy did not.
    BreakPairForSingle,
    /// The inner policy plays a single that no unseen card beats; the mistake
    /// plays the weakest other action instead (a pass when following).
    MissSafeSingle,
}

impl Mistake {
    /// All mistakes, in the order they are checked.
    pub const ALL: [Mistake; 3] = [
        Mistake::HoldTwoTooLong,
        Mistake::BreakPairForSingle,
        Mistake::MissSafeSingle,
    ];

    /// The reason key a review shows for this mistake (hk-games D42: a key, no text).
    pub fn key(self) -> &'static str {
        match self {
            Mistake::HoldTwoTooLong => "mistake.hold_two_too_long",
            Mistake::BreakPairForSingle => "mistake.break_pair_for_single",
            Mistake::MissSafeSingle => "mistake.miss_safe_single",
        }
    }
}

/// A policy with named mistakes at given rates (each in `0..=1`: the share of
/// the decisions where the mistake applies on which it fires).
#[derive(Clone, Debug)]
pub struct ErrorModel<P> {
    pub inner: P,
    pub hold_two_too_long: f64,
    pub break_pair_for_single: f64,
    pub miss_safe_single: f64,
}

impl<P> ErrorModel<P> {
    /// The rate of `m`.
    pub fn rate(&self, m: Mistake) -> f64 {
        match m {
            Mistake::HoldTwoTooLong => self.hold_two_too_long,
            Mistake::BreakPairForSingle => self.break_pair_for_single,
            Mistake::MissSafeSingle => self.miss_safe_single,
        }
    }
}

/// The weakest play in `plays` by the [`super::Greedy`] order (following: lowest key;
/// leading: plays with the weakest card first, larger first, then lowest key).
fn weakest<'a>(state: &State, plays: impl Iterator<Item = &'a Action>) -> Option<Action> {
    let hand = state.hand(state.current_player()?);
    let low = mask::lowest(hand)?;
    let following = state.current_play().is_some();
    plays
        .filter_map(|a| match a {
            Action::Play(c) => Some(c),
            Action::Pass => None,
        })
        .min_by_key(|c| {
            let has_low = c.cards & (1u64 << low) != 0;
            if following {
                (false, std::cmp::Reverse(0), c.key)
            } else {
                (!has_low, std::cmp::Reverse(c.size()), c.key)
            }
        })
        .map(|c| Action::Play(*c))
}

/// The action a mistake plays instead of `chosen`, when it applies.
pub fn mistake_move(state: &State, legal: &[Action], chosen: Action, m: Mistake) -> Option<Action> {
    let me = state.current_player()?;
    let hand = state.hand(me);
    let following = state.current_play().is_some();
    match m {
        Mistake::HoldTwoTooLong => {
            if twos(chosen.cards()) == 0 {
                return None;
            }
            if following && legal.contains(&Action::Pass) {
                return Some(Action::Pass);
            }
            weakest(state, legal.iter().filter(|a| twos(a.cards()) == 0))
        }
        Mistake::BreakPairForSingle => {
            let (_, current) = state.current_play()?;
            if current.kind != Kind::Single {
                return None;
            }
            let from_set = |a: &Action| match a {
                Action::Play(c) if c.kind == Kind::Single => {
                    let r = rank_of(c.cards.trailing_zeros() as u8);
                    (hand >> (r * 4) & 0xF).count_ones() >= 2
                }
                _ => false,
            };
            if from_set(&chosen) {
                return None;
            }
            weakest(state, legal.iter().filter(|a| from_set(a)))
        }
        Mistake::MissSafeSingle => {
            let Action::Play(c) = chosen else {
                return None;
            };
            if c.kind != Kind::Single || hand == c.cards {
                return None;
            }
            let unseen = FULL_DECK & !state.played() & !hand;
            let card = c.cards.trailing_zeros() as u8;
            if mask::highest(unseen).is_some_and(|u| u > card) {
                return None;
            }
            if following && legal.contains(&Action::Pass) {
                return Some(Action::Pass);
            }
            weakest(state, legal.iter().filter(|a| **a != chosen))
        }
    }
}

impl<P: Policy<State>> ErrorModel<P> {
    /// The inner policy's move, or a mistake's replacement, with the mistake
    /// that fired.
    pub fn choose_explained<R: Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> (Action, Option<Mistake>) {
        let chosen = self.inner.choose(state, legal, rng);
        if legal.len() < 2 {
            return (chosen, None);
        }
        for m in Mistake::ALL {
            let rate = self.rate(m);
            if rate <= 0.0 {
                continue;
            }
            if let Some(a) = mistake_move(state, legal, chosen, m) {
                if a != chosen && rng.gen::<f64>() < rate {
                    debug_assert!(legal.contains(&a));
                    return (a, Some(m));
                }
            }
        }
        (chosen, None)
    }
}

impl<P: Policy<State>> Policy<State> for ErrorModel<P> {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        self.choose_explained(state, legal, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::StructuredGreedy;
    use crate::card::parse_mask;
    use crate::rules::RuleSet;
    use rand::SeedableRng;

    fn h(s: &str) -> u64 {
        parse_mask(s).unwrap()
    }

    fn state(hands: [&str; 4], current: usize, play: Option<(usize, &str)>) -> State {
        State::from_position(
            RuleSet::default(),
            hands.map(h),
            current,
            play.map(|(p, c)| (p, h(c))),
        )
        .unwrap()
    }

    /// An inner policy that always plays the action with exactly `cards`.
    struct Plays(u64);
    impl Policy<State> for Plays {
        fn choose<R: Rng + ?Sized>(&self, _: &State, legal: &[Action], _: &mut R) -> Action {
            *legal.iter().find(|a| a.cards() == self.0).unwrap()
        }
    }

    fn only(m: Mistake, rate: f64, inner: Plays) -> ErrorModel<Plays> {
        ErrorModel {
            inner,
            hold_two_too_long: if m == Mistake::HoldTwoTooLong {
                rate
            } else {
                0.0
            },
            break_pair_for_single: if m == Mistake::BreakPairForSingle {
                rate
            } else {
                0.0
            },
            miss_safe_single: if m == Mistake::MissSafeSingle {
                rate
            } else {
                0.0
            },
        }
    }

    /// Fires `m` on `n` draws and returns the share fired and the replacement seen.
    fn share(model: &ErrorModel<Plays>, s: &State, m: Mistake, n: usize) -> (f64, Option<Action>) {
        let legal = s.legal_actions_vec();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let (mut fired, mut seen) = (0usize, None);
        for _ in 0..n {
            let (a, why) = model.choose_explained(s, &legal, &mut rng);
            assert!(legal.contains(&a));
            if why == Some(m) {
                fired += 1;
                seen = Some(a);
            } else {
                assert_eq!(why, None);
            }
        }
        (fired as f64 / n as f64, seen)
    }

    #[test]
    fn error_model_hold_two_fires_at_its_rate() {
        // Following a single 9d: the inner policy plays the 2s; the mistake passes.
        let s = state(
            ["4c 4d Jc 2s", "5d 5c", "6d 6c", "7d 7c"],
            0,
            Some((3, "9d")),
        );
        let model = only(Mistake::HoldTwoTooLong, 0.3, Plays(h("2s")));
        let (p, a) = share(&model, &s, Mistake::HoldTwoTooLong, 100_000);
        assert!((p - 0.3).abs() < 0.01, "{p}");
        assert_eq!(a, Some(Action::Pass));
    }

    #[test]
    fn error_model_break_pair_fires_at_its_rate() {
        // Following a single 3c: the inner plays Jc; the mistake plays a 4 from the pair.
        let s = state(
            ["4c 4d Jc 2s", "5d 5c", "6d 6c", "7d 7c"],
            0,
            Some((3, "3c")),
        );
        let model = only(Mistake::BreakPairForSingle, 0.5, Plays(h("Jc")));
        let (p, a) = share(&model, &s, Mistake::BreakPairForSingle, 100_000);
        assert!((p - 0.5).abs() < 0.01, "{p}");
        assert_eq!(a.map(|a| a.cards()), Some(h("4d")));
    }

    #[test]
    fn error_model_miss_safe_single_fires_at_its_rate() {
        // Everything above the K is played except seat 0's 2s: the 2s is safe.
        let s = state(["3c Kd 2s", "4d 4c", "5d 5c", "Ad 6d"], 0, None);
        let model = only(Mistake::MissSafeSingle, 0.1, Plays(h("2s")));
        let (p, a) = share(&model, &s, Mistake::MissSafeSingle, 100_000);
        assert!((p - 0.1).abs() < 0.01, "{p}");
        assert_eq!(a.map(|a| a.cards()), Some(h("3c")));
    }

    #[test]
    fn error_model_mistakes_do_not_apply_where_they_cannot() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        // No 2 in the chosen play: hold-two does not apply.
        let s = state(
            ["4c 4d Jc 2s", "5d 5c", "6d 6c", "7d 7c"],
            0,
            Some((3, "9d")),
        );
        assert_eq!(
            mistake_move(
                &s,
                &s.legal_actions_vec(),
                Action::Pass,
                Mistake::HoldTwoTooLong
            ),
            None
        );
        // Following a pair, break-pair-for-single does not apply.
        let s = state(
            ["4c 4d 9c 9h", "5d 5c", "6d 6c", "7d 7c"],
            0,
            Some((3, "3c 3h")),
        );
        let legal = s.legal_actions_vec();
        let chosen = legal[0];
        assert_eq!(
            mistake_move(&s, &legal, chosen, Mistake::BreakPairForSingle),
            None
        );
        // A single that an unseen Ad beats is not safe.
        let s = state(["3c Kd 2s", "4d 4c", "5d 5c", "Ad 6d"], 0, None);
        let kd = *s
            .legal_actions_vec()
            .iter()
            .find(|a| a.cards() == h("Kd"))
            .unwrap();
        assert_eq!(
            mistake_move(&s, &s.legal_actions_vec(), kd, Mistake::MissSafeSingle),
            None
        );
        // Rate 0 never fires.
        let model = only(Mistake::HoldTwoTooLong, 0.0, Plays(h("2s")));
        let s = state(
            ["4c 4d Jc 2s", "5d 5c", "6d 6c", "7d 7c"],
            0,
            Some((3, "9d")),
        );
        let legal = s.legal_actions_vec();
        assert!((0..1000).all(|_| model.choose_explained(&s, &legal, &mut rng).1.is_none()));
    }

    #[test]
    fn error_model_never_plays_an_illegal_move() {
        let model = ErrorModel {
            inner: StructuredGreedy {
                epsilon: 0.2,
                ..StructuredGreedy::default()
            },
            hold_two_too_long: 0.7,
            break_pair_for_single: 0.7,
            miss_safe_single: 0.7,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let mut fired = [0usize; 3];
        for _ in 0..300 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                let (a, why) = model.choose_explained(&s, &legal, &mut rng);
                assert_eq!(s.check(&a), Ok(()), "{a:?} in {s:?}");
                if let Some(m) = why {
                    fired[Mistake::ALL.iter().position(|x| *x == m).unwrap()] += 1;
                }
                s.apply(a);
            }
        }
        assert!(
            fired.iter().all(|&f| f > 0),
            "every mistake fires in play: {fired:?}"
        );
    }

    #[test]
    fn error_model_keys_are_keys() {
        for m in Mistake::ALL {
            assert!(m.key().starts_with("mistake."));
        }
    }
}
