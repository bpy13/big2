//! Neural-network input features for Big Two (Phase 3).
//!
//! Everything is encoded from the point of view of the player to act, using only
//! what that player can see. Seats are *relative*: 0 = me, 1 = next player,
//! 2 = across, 3 = previous player. All features are 0/1, stored as `u8`.
//!
//! **State features** ([`STATE_LEN`] bytes):
//!
//! | Offset | Size | Meaning |
//! |---:|---:|---|
//! | 0 | 52 | my hand |
//! | 52 | 52 | unseen cards (held by someone else) |
//! | 104 | 4×52 | cards played so far by each relative seat |
//! | 312 | 3×14 | cards left for relative seats 1–3, one-hot 0–13 |
//! | 354 | 52 | the play to beat (all zero when leading) |
//! | 406 | 4 | who made it (relative seat), all zero when leading |
//! | 410 | 1 | I am leading |
//! | 411 | 1 | opening play (must include 3♦) |
//! | 412 | 3 | consecutive passes since the last play, one-hot 0–2 |
//! | 415 | 4 | relative seats locked out of this trick (`locked_out` rule) |
//! | 419 | 12×57 | last 12 actions, newest first: 52 card bits, 4 relative-seat bits, 1 pass bit |
//!
//! **Action features** ([`ACTION_LEN`] bytes): 52 card bits, then a one-hot kind
//! (pass, single, pair, triple, straight, flush, full house, four of a kind,
//! straight flush).
//!
//! **Full-information features** ([`FULL_LEN`] bytes, [`encode_full`]) for value
//! networks that judge a fully dealt position (used inside search, where the
//! hidden cards have been sampled). Seats are relative to the player to act:
//!
//! | Offset | Size | Meaning |
//! |---:|---:|---|
//! | 0 | 4×52 | each relative seat's hand |
//! | 208 | 52 | the play to beat (all zero when leading) |
//! | 260 | 4 | who made it (relative seat) |
//! | 264 | 1 | the player to act leads |
//! | 265 | 1 | opening play (must include 3♦) |
//! | 266 | 3 | consecutive passes since the last play, one-hot 0–2 |
//! | 269 | 4 | relative seats locked out of this trick |
//!
//! The network is trained for one rule set; rules are not part of the features.

use crate::combo::Kind;
use crate::state::{Action, State, NUM_PLAYERS};
use poker_cards::mask;
use poker_games::Game;

/// Actions of history included in the state features.
pub const HISTORY_LEN: usize = 12;
const HIST_ITEM: usize = 52 + 4 + 1;

pub const OFF_HAND: usize = 0;
pub const OFF_UNSEEN: usize = 52;
pub const OFF_PLAYED: usize = 104;
pub const OFF_COUNTS: usize = 312;
pub const OFF_CURRENT: usize = 354;
pub const OFF_HOLDER: usize = 406;
pub const OFF_LEADING: usize = 410;
pub const OFF_OPENING: usize = 411;
pub const OFF_PASSES: usize = 412;
pub const OFF_LOCKED: usize = 415;
pub const OFF_HISTORY: usize = 419;
/// Length of the state feature vector.
pub const STATE_LEN: usize = OFF_HISTORY + HISTORY_LEN * HIST_ITEM;
/// Length of the action feature vector.
pub const ACTION_LEN: usize = 52 + 9;

fn put_cards(out: &mut [u8], cards: u64) {
    for i in mask::bits(cards) {
        out[i as usize] = 1;
    }
}

/// Relative seat of `player` as seen from `me`.
#[inline]
fn rel(me: usize, player: usize) -> usize {
    (player + NUM_PLAYERS - me) % NUM_PLAYERS
}

/// Appends the state features for the player to act to `out`.
///
/// # Panics
/// Panics if the game is over.
pub fn encode_state(s: &State, out: &mut Vec<u8>) {
    let me = s.current_player().expect("game not over");
    let start = out.len();
    out.resize(start + STATE_LEN, 0);
    let f = &mut out[start..];

    let hand = s.hand(me);
    put_cards(&mut f[OFF_HAND..], hand);
    put_cards(
        &mut f[OFF_UNSEEN..],
        crate::card::FULL_DECK & !hand & !s.played(),
    );

    let mut played_by = [0u64; NUM_PLAYERS];
    for t in s.history() {
        played_by[t.player as usize] |= t.action.cards();
    }
    for (p, &cards) in played_by.iter().enumerate() {
        put_cards(&mut f[OFF_PLAYED + 52 * rel(me, p)..], cards);
    }
    let sizes = s.hand_sizes();
    for (p, &n) in sizes.iter().enumerate() {
        let r = rel(me, p);
        if r > 0 {
            f[OFF_COUNTS + 14 * (r - 1) + n as usize] = 1;
        }
    }
    match s.current_play() {
        Some((holder, combo)) => {
            put_cards(&mut f[OFF_CURRENT..], combo.cards);
            f[OFF_HOLDER + rel(me, holder)] = 1;
        }
        None => f[OFF_LEADING] = 1,
    }
    if s.is_opening() {
        f[OFF_OPENING] = 1;
    }
    // Consecutive passes at the end of the history (only meaningful mid-trick).
    if s.current_play().is_some() {
        let passes = s
            .history()
            .iter()
            .rev()
            .take_while(|t| t.action == Action::Pass)
            .count();
        f[OFF_PASSES + passes.min(2)] = 1;
    }
    for (p, &locked) in s.locked_out().iter().enumerate() {
        if locked {
            f[OFF_LOCKED + rel(me, p)] = 1;
        }
    }
    for (k, t) in s.history().iter().rev().take(HISTORY_LEN).enumerate() {
        let item = &mut f[OFF_HISTORY + k * HIST_ITEM..OFF_HISTORY + (k + 1) * HIST_ITEM];
        put_cards(item, t.action.cards());
        item[52 + rel(me, t.player as usize)] = 1;
        if t.action == Action::Pass {
            item[56] = 1;
        }
    }
}

pub const FULL_HANDS: usize = 0;
pub const FULL_CURRENT: usize = 208;
pub const FULL_HOLDER: usize = 260;
pub const FULL_LEADING: usize = 264;
pub const FULL_OPENING: usize = 265;
pub const FULL_PASSES: usize = 266;
pub const FULL_LOCKED: usize = 269;
/// Length of the full-information feature vector.
pub const FULL_LEN: usize = 273;
/// Name of this encoding, stored in exported networks' metadata.
pub const FULL_FEATURES_NAME: &str = "big2-full-v1";

/// Indices of the full-information features that are 1, for the player to act
/// (see the module docs). Every hand is visible: use only on sampled states.
///
/// # Panics
/// Panics if the game is over.
pub fn active_full(s: &State, out: &mut Vec<usize>) {
    let me = s.current_player().expect("game not over");
    out.clear();
    for p in 0..NUM_PLAYERS {
        let base = FULL_HANDS + 52 * rel(me, p);
        out.extend(mask::bits(s.hand(p)).map(|c| base + c as usize));
    }
    match s.current_play() {
        Some((holder, combo)) => {
            out.extend(mask::bits(combo.cards).map(|c| FULL_CURRENT + c as usize));
            out.push(FULL_HOLDER + rel(me, holder));
            let passes = s
                .history()
                .iter()
                .rev()
                .take_while(|t| t.action == Action::Pass)
                .count();
            out.push(FULL_PASSES + passes.min(2));
        }
        None => out.push(FULL_LEADING),
    }
    if s.is_opening() {
        out.push(FULL_OPENING);
    }
    for (p, &locked) in s.locked_out().iter().enumerate() {
        if locked {
            out.push(FULL_LOCKED + rel(me, p));
        }
    }
}

/// Appends the full-information features (0/1 bytes) for the player to act.
pub fn encode_full(s: &State, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + FULL_LEN, 0);
    let mut active = Vec::with_capacity(64);
    active_full(s, &mut active);
    for i in active {
        out[start + i] = 1;
    }
}

/// Index of an action's kind in the action features' one-hot (pass = 0).
pub fn kind_index(a: &Action) -> usize {
    match a {
        Action::Pass => 0,
        Action::Play(c) => match c.kind {
            Kind::Single => 1,
            Kind::Pair => 2,
            Kind::Triple => 3,
            Kind::Straight => 4,
            Kind::Flush => 5,
            Kind::FullHouse => 6,
            Kind::FourOfAKind => 7,
            Kind::StraightFlush => 8,
        },
    }
}

/// Appends the action features of `a` to `out`.
pub fn encode_action(a: &Action, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + ACTION_LEN, 0);
    put_cards(&mut out[start..], a.cards());
    out[start + 52 + kind_index(a)] = 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::RuleSet;
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn opening_state_features() {
        let mut rng = SmallRng::seed_from_u64(1);
        let s = State::new_deal(RuleSet::default(), &mut rng);
        let mut f = Vec::new();
        encode_state(&s, &mut f);
        assert_eq!(f.len(), STATE_LEN);
        assert_eq!(
            f[OFF_HAND..OFF_HAND + 52]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            13
        );
        assert_eq!(
            f[OFF_UNSEEN..OFF_UNSEEN + 52]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            39
        );
        assert_eq!(f[OFF_HAND], 1, "the opener holds 3d");
        assert_eq!(f[OFF_LEADING], 1);
        assert_eq!(f[OFF_OPENING], 1);
        for r in 0..3 {
            assert_eq!(
                f[OFF_COUNTS + 14 * r + 13],
                1,
                "every opponent has 13 cards"
            );
        }
        assert!(f[OFF_HISTORY..].iter().all(|&x| x == 0));
    }

    #[test]
    fn features_never_reveal_hidden_cards() {
        // Two states that differ only in hidden cards encode identically.
        let mut rng = SmallRng::seed_from_u64(4);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..17 {
            let legal = s.legal_actions_vec();
            s.apply(legal[0]);
        }
        let me = s.current_player().unwrap();
        let d = s.determinize(me, &mut rng);
        assert_ne!(d.hands(), s.hands());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        encode_state(&s, &mut a);
        encode_state(&d, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn history_is_relative_and_newest_first() {
        let mut rng = SmallRng::seed_from_u64(2);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let opener = s.current_player().unwrap();
        let first = s.legal_actions_vec()[0];
        s.apply(first);
        let mut f = Vec::new();
        encode_state(&s, &mut f);
        // The next player sees the opener as relative seat 3 (previous player).
        let item = &f[OFF_HISTORY..OFF_HISTORY + HIST_ITEM];
        assert_eq!(item[52 + 3], 1);
        assert_eq!(item[0], 1, "3d was played");
        assert_eq!(s.current_player(), Some((opener + 1) % 4));
        assert_eq!(f[OFF_HOLDER + 3], 1);
        assert_eq!(f[OFF_PASSES], 1, "no passes yet");
    }

    #[test]
    fn full_features_rotate_with_the_player_to_act() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut f = Vec::new();
        encode_full(&s, &mut f);
        assert_eq!(f.len(), FULL_LEN);
        assert_eq!(
            f.iter().map(|&x| x as u32).sum::<u32>(),
            52 + 2,
            "52 cards, lead, opening"
        );
        assert_eq!(f[FULL_HANDS], 1, "the opener (relative seat 0) holds 3d");
        let opener = s.current_player().unwrap();
        let first = s.legal_actions_vec()[0];
        s.apply(first);
        f.clear();
        encode_full(&s, &mut f);
        // For the next player, the opener is the previous seat (3) and made the play.
        assert_eq!(f[FULL_HOLDER + 3], 1);
        assert_eq!(f[FULL_CURRENT], 1, "3d is the play to beat");
        assert_eq!(f[FULL_PASSES], 1);
        let mine = s.hand(s.current_player().unwrap());
        for c in mask::bits(mine) {
            assert_eq!(f[FULL_HANDS + c as usize], 1);
        }
        assert_eq!(s.current_player(), Some((opener + 1) % 4));
    }

    #[test]
    fn action_features() {
        let mut out = Vec::new();
        encode_action(&Action::Pass, &mut out);
        assert_eq!(out.len(), ACTION_LEN);
        assert_eq!(out[52], 1);
        assert_eq!(out[..52].iter().sum::<u8>(), 0);
    }
}
