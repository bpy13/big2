//! [`StructuredGreedy`]: the greedy bot with a hand plan.

use super::Greedy;
use crate::combo::Combo;
use crate::plan::min_plays;
use crate::state::{Action, State};
use big2_cards::mask;
use big2_games::{Game, Policy};
use rand::Rng;

/// Default for [`StructuredGreedy::plan_depth`].
pub const DEFAULT_PLAN_DEPTH: usize = 8;

/// "Shed the weakest cards first, but keep the plan."
///
/// A play *keeps the plan* when it does not raise the number of plays the hand
/// needs: `min_plays(hand − play) + 1 == min_plays(hand)` (see
/// [`crate::plan::min_plays`]). Those are exactly the plays that are a group of
/// one of the hand's best decompositions ([`crate::plan::decompositions`]).
///
/// * **Following:** the weakest beating play that keeps the plan, so a pair,
///   triple or five-card group is not broken to beat a single while a lone
///   single can do it. If every beating play breaks a group, as [`Greedy`]
///   (the weakest beating play; it never passes while it can beat).
/// * **Leading:** among the plays that contain the weakest card, in [`Greedy`]'s
///   order (largest first, then weakest), the first that keeps the plan; if
///   none does, as [`Greedy`].
/// * With probability `epsilon`, a uniformly random legal action instead.
///
/// Why not pass instead of breaking: a variant that passed whenever every
/// single that beats a single broke a group lost 7.6 ± 0.5 points per game to
/// [`Greedy`] (2,000 duplicate deals, A B A B); this version wins 1.1 ± 0.2
/// (build-phase1.md §1).
///
/// `plan_depth` caps how many plays (in the order above) the planner checks per
/// decision; each check costs one `min_plays` call (a few µs on a full hand).
/// When none of the checked plays keeps the plan, the "if none" rule applies.
/// `plan_depth = 0` turns the planner off: the bot then plays exactly as
/// [`Greedy`] with the same `epsilon`.
#[derive(Clone, Copy, Debug)]
pub struct StructuredGreedy {
    pub epsilon: f64,
    pub plan_depth: usize,
}

impl Default for StructuredGreedy {
    fn default() -> Self {
        StructuredGreedy {
            epsilon: 0.0,
            plan_depth: DEFAULT_PLAN_DEPTH,
        }
    }
}

impl StructuredGreedy {
    /// The first of `plays` (already in preference order, at most `plan_depth`
    /// checked) that keeps the plan of `hand`.
    fn first_keeper<'a>(
        &self,
        hand: u64,
        state: &State,
        plays: impl Iterator<Item = &'a Combo>,
    ) -> Option<Combo> {
        let rules = state.rules();
        let target = min_plays(hand, rules).saturating_sub(1);
        plays
            .take(self.plan_depth)
            .find(|c| min_plays(hand & !c.cards, rules) <= target)
            .copied()
    }
}

impl Policy<State> for StructuredGreedy {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        if self.epsilon > 0.0 && rng.gen::<f64>() < self.epsilon {
            return legal[rng.gen_range(0..legal.len())];
        }
        let mut greedy = || Greedy { epsilon: 0.0 }.choose(state, legal, rng);
        if self.plan_depth == 0 {
            return greedy();
        }
        let hand = state.hand(state.current_player().expect("game not over"));
        let mut plays: Vec<&Combo> = legal
            .iter()
            .filter_map(|a| match a {
                Action::Play(c) => Some(c),
                Action::Pass => None,
            })
            .collect();
        if state.current_play().is_some() {
            plays.sort_by_key(|c| c.key);
        } else {
            let weakest = 1u64 << mask::lowest(hand).expect("non-empty hand");
            plays.retain(|c| c.cards & weakest != 0);
            plays.sort_by_key(|c| (std::cmp::Reverse(c.size()), c.key));
        }
        match self.first_keeper(hand, state, plays.iter().copied()) {
            Some(c) => Action::Play(c),
            None => greedy(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_mask;
    use crate::rules::RuleSet;
    use big2_games::play_out;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn h(s: &str) -> u64 {
        parse_mask(s).unwrap()
    }

    /// Seat 0 to act with `hand`, following `to_beat` played by seat 3.
    fn following(hand: &str, to_beat: &str, others: [&str; 3]) -> State {
        let hands = [h(hand), h(others[0]), h(others[1]), h(others[2])];
        State::from_position(RuleSet::default(), hands, 0, Some((3, h(to_beat)))).unwrap()
    }

    fn choose(bot: &StructuredGreedy, s: &State) -> Action {
        let legal = s.legal_actions_vec();
        bot.choose(s, &legal, &mut SmallRng::seed_from_u64(0))
    }

    #[test]
    fn structured_beats_a_low_single_with_the_lone_single_not_the_pair() {
        let s = following("3c 9d 9s Kd", "5d", ["4d 4c 4h", "6d 6c 6h", "7d 7c 7h"]);
        let legal = s.legal_actions_vec();
        let mut rng = SmallRng::seed_from_u64(0);
        // Greedy breaks the pair of 9s; the planner keeps it and plays the king.
        assert_eq!(
            Greedy::default().choose(&s, &legal, &mut rng).cards(),
            h("9d")
        );
        assert_eq!(choose(&StructuredGreedy::default(), &s).cards(), h("Kd"));
    }

    #[test]
    fn structured_breaks_like_greedy_when_every_beating_play_breaks() {
        // Only the 9s beat the 8, and both are in the pair: play as Greedy.
        let s = following("3c 9d 9s", "8d", ["4d 4c 4h", "6d 6c 6h", "7d 7c 7h"]);
        assert_eq!(choose(&StructuredGreedy::default(), &s).cards(), h("9d"));
    }

    #[test]
    fn structured_obeys_the_last_card_duty() {
        // The next seat holds one card: a single played must be the highest one.
        let s = following("3c 9d 9s Kd", "5d", ["4d", "6d 6c 6h", "7d 7c 7h"]);
        let legal = s.legal_actions_vec();
        let a = choose(&StructuredGreedy::default(), &s);
        assert!(legal.contains(&a), "{a:?} not in {legal:?}");
        assert_eq!(a.cards(), h("Kd"));
    }

    #[test]
    fn structured_keeps_a_triple_when_following_a_pair() {
        // 7s pair beats the 6s; the 9s triple would also beat it but breaks.
        let s = following(
            "3c 7d 7s 9d 9c 9h",
            "6d 6c",
            ["4d 4c 4h", "5d 5c 5h", "8d 8c 8h"],
        );
        assert_eq!(choose(&StructuredGreedy::default(), &s).cards(), h("7d 7s"));
        // Without the 7s, the pair can only come from the triple: play as Greedy.
        let s = following("3c 9d 9c 9h", "6d 6c", ["4d 4c 4h", "5d 5c 5h", "8d 8c 8h"]);
        let legal = s.legal_actions_vec();
        let g = Greedy::default().choose(&s, &legal, &mut SmallRng::seed_from_u64(0));
        assert_eq!(choose(&StructuredGreedy::default(), &s), g);
    }

    #[test]
    fn structured_leads_a_group_that_keeps_the_plan() {
        // On every lead of 300 greedy self-play deals: if one of the first
        // `plan_depth` plays holding the weakest card keeps the plan, the lead does.
        let rules = RuleSet::default();
        let mut rng = SmallRng::seed_from_u64(5);
        let bot = StructuredGreedy::default();
        let mut leads_checked = 0;
        for _ in 0..300 {
            let mut s = State::new_deal(rules, &mut rng);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                if s.current_play().is_none() {
                    let hand = s.hand(s.current_player().unwrap());
                    let keeps = |cards: u64| {
                        min_plays(hand & !cards, &rules) + 1 == min_plays(hand, &rules)
                    };
                    let weakest = 1u64 << mask::lowest(hand).unwrap();
                    let mut with_weakest: Vec<Combo> = legal
                        .iter()
                        .filter_map(|a| match a {
                            Action::Play(c) if c.cards & weakest != 0 => Some(*c),
                            _ => None,
                        })
                        .collect();
                    with_weakest.sort_by_key(|c| (std::cmp::Reverse(c.size()), c.key));
                    let a = bot.choose(&s, &legal, &mut rng);
                    assert!(legal.contains(&a));
                    if with_weakest
                        .iter()
                        .take(bot.plan_depth)
                        .any(|c| keeps(c.cards))
                    {
                        assert!(keeps(a.cards()), "{a:?} breaks the plan of {s:?}");
                    }
                    leads_checked += 1;
                }
                let g = Greedy::default().choose(&s, &legal, &mut rng);
                s.apply(g);
            }
        }
        assert!(leads_checked > 1_000);
    }

    #[test]
    fn plan_depth_zero_is_greedy() {
        let rules = RuleSet::default();
        let mut rng = SmallRng::seed_from_u64(3);
        let off = StructuredGreedy {
            epsilon: 0.0,
            plan_depth: 0,
        };
        for _ in 0..50 {
            let mut s = State::new_deal(rules, &mut rng);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                let g = Greedy::default().choose(&s, &legal, &mut rng);
                assert_eq!(off.choose(&s, &legal, &mut rng), g);
                s.apply(g);
            }
        }
    }

    /// A B A B on duplicate deals (each deal twice, the pairs of seats swapped):
    /// the planner's bot must beat Greedy. Measured +1.07 ± 0.23 per seat-game
    /// on these 2,000 deals.
    #[test]
    fn structured_beats_greedy_head_to_head() {
        let rules = RuleSet::default();
        let mut rng = SmallRng::seed_from_u64(77);
        let sg = StructuredGreedy::default();
        let g = Greedy::default();
        let (mut sum, mut sq, n) = (0.0, 0.0, 2000);
        for _ in 0..n {
            let deal = State::new_deal(rules, &mut rng);
            let mut d = 0.0;
            for rot in 0..2 {
                let mut s = deal.clone();
                while let Some(p) = s.current_player() {
                    let legal = s.legal_actions_vec();
                    let a = if (p + rot) % 2 == 0 {
                        sg.choose(&s, &legal, &mut rng)
                    } else {
                        g.choose(&s, &legal, &mut rng)
                    };
                    s.apply(a);
                }
                let o = s.outcome().unwrap();
                for p in 0..4 {
                    if (p + rot) % 2 == 0 {
                        d += o.scores[p] / 4.0;
                    }
                }
            }
            sum += d;
            sq += d * d;
        }
        let m = sum / n as f64;
        let ci = 1.96 * ((sq / n as f64 - m * m) / n as f64).sqrt();
        eprintln!("structured vs greedy per seat-game {m:+.2} ± {ci:.2}");
        assert!(m - ci > 0.0, "{m:+.2} ± {ci:.2}");
    }

    #[test]
    fn structured_rollouts_finish_with_legal_moves() {
        let rules = RuleSet::default();
        let mut rng = SmallRng::seed_from_u64(8);
        let bot = StructuredGreedy {
            epsilon: 0.1,
            ..Default::default()
        };
        for _ in 0..500 {
            let mut s = State::new_deal(rules, &mut rng);
            // `State::apply` panics on an illegal action.
            let o = play_out(&mut s, &bot, &mut rng);
            assert!(o.winner.is_some());
        }
    }
}
