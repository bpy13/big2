//! Simple rule-based Big Two policies.
//!
//! They are deliberately simple: they serve as baseline opponents for measuring
//! the solver and the RL agents, and as fast rollout policies inside search.

use crate::state::{Action, State};
use big2_cards::mask;
use big2_games::{Game, Policy};
use rand::Rng;

/// "Get rid of the weakest cards first."
///
/// * **Leading:** play a combination containing the weakest card in hand,
///   preferring the largest such combination (5 > 3 > 2 > 1 cards), weakest first.
/// * **Following:** play the weakest combination that beats the current play;
///   pass only when nothing beats it.
/// * With probability `epsilon`, play a uniformly random legal action instead.
///   A small `epsilon` makes rollouts less predictable.
#[derive(Clone, Copy, Debug)]
pub struct Greedy {
    pub epsilon: f64,
}

impl Default for Greedy {
    fn default() -> Self {
        Greedy { epsilon: 0.0 }
    }
}

impl Policy<State> for Greedy {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        if self.epsilon > 0.0 && rng.gen::<f64>() < self.epsilon {
            return legal[rng.gen_range(0..legal.len())];
        }
        let plays = legal.iter().filter_map(|a| match a {
            Action::Play(c) => Some(c),
            Action::Pass => None,
        });
        if state.current_play().is_some() {
            return plays
                .min_by_key(|c| c.key)
                .map_or(Action::Pass, |c| Action::Play(*c));
        }
        let hand = state.hand(state.current_player().expect("game not over"));
        let weakest = 1u64 << mask::lowest(hand).expect("non-empty hand");
        plays
            .filter(|c| c.cards & weakest != 0)
            .min_by_key(|c| (std::cmp::Reverse(c.size()), c.key))
            .map(|c| Action::Play(*c))
            .unwrap_or(legal[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_mask;
    use crate::rules::RuleSet;
    use rand::SeedableRng;

    /// Gives player 0 `hand` and deals the rest of the deck to players 1–3.
    fn deal_with(hand: &str) -> State {
        let h0 = parse_mask(hand).unwrap();
        let mut rest = mask::bits(crate::card::FULL_DECK & !h0);
        let mut hands = [h0, 0, 0, 0];
        for h in hands.iter_mut().skip(1) {
            for _ in 0..13 {
                *h |= 1 << rest.next().unwrap();
            }
        }
        State::from_hands(RuleSet::default(), hands).unwrap()
    }

    #[test]
    fn leads_largest_combo_with_weakest_card() {
        let state = deal_with("3d 4c 5h 6s 7d 9c 9h Ts Jd Qc Kh As 2d");
        let legal = state.legal_actions_vec();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        match Greedy::default().choose(&state, &legal, &mut rng) {
            Action::Play(c) => assert_eq!(c.cards, parse_mask("3d 4c 5h 6s 7d").unwrap()),
            Action::Pass => panic!("leader cannot pass"),
        }
    }
}
