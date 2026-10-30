//! Ladder v2: the five bot levels as named constructors (strategy proposal §4,
//! plan Task 29).
//!
//! | Level | Mechanism |
//! |---|---|
//! | `random` | a uniformly random legal move |
//! | `easy` | `StructuredGreedy` (ε 0.1) plus the error model at [`EASY_RATES`] |
//! | `medium` | the distilled Q-network sampled at [`MEDIUM_TEMPERATURE`] plus a thin error model ([`MEDIUM_RATES`]) |
//! | `hard` | ISMCTS, [`HARD_ITERATIONS`] simulations |
//! | `expert` | ISMCTS, [`EXPERT_ITERATIONS`] simulations, plus the exact information-set endgame |
//!
//! `hard` and `expert` are ISMCTS, not the proposal's PUCT-300 / PUCT-2000: PUCT
//! has not beaten ISMCTS yet (`docs/open-items.md`, 2026-10-30: "bot levels stay on
//! ISMCTS"), and these are the app's current budgets (2,000 and 10,000).
//!
//! The ladder lives here, not in `crates/big2/src/bots/ladder.rs` as the plan
//! said: `hard` and `expert` need `big2-search`, which `big2-big2` cannot depend
//! on (`big2-search` already uses `big2-big2` in its tests). The error model
//! itself is in `big2_big2::bots::error_model`.
//!
//! The rates and the temperature are hand-set first values; Task 29's
//! round-robin (`build-phase4.md` §3) tunes them until adjacent levels are at
//! least 1.5 points apart, and the final values are recorded here.

use big2_big2::bots::{ErrorModel, Mistake, StructuredGreedy};
use big2_big2::qnet::{QNet, QNetPolicy};
use big2_big2::{Action, State};
use big2_games::{Game, Policy};
use big2_search::{ismcts, InfosetConfig, IsmctsConfig, Objective};
use rand::Rng;
use std::sync::Arc;

/// `easy`'s mistake rates: hold a 2 too long, break a pair for a single, miss a
/// safe single (each the share of the decisions where it applies).
pub const EASY_RATES: [f64; 3] = [0.5, 0.5, 0.5];
/// `medium`'s mistake rates.
pub const MEDIUM_RATES: [f64; 3] = [0.15, 0.15, 0.15];
/// `medium`'s sampling temperature in points (Task 9's ladder v0 used 1.0).
pub const MEDIUM_TEMPERATURE: f64 = 1.0;
/// `easy`'s random-move share.
pub const EASY_EPSILON: f64 = 0.1;
/// ISMCTS simulations of `hard` and `expert`.
pub const HARD_ITERATIONS: u64 = 2_000;
pub const EXPERT_ITERATIONS: u64 = 10_000;

/// ISMCTS as a policy (one thread, greedy rollouts with ε 0.1, as the app). A
/// forced move is played without searching; the search seed comes from `rng`.
#[derive(Clone, Debug)]
pub struct IsmctsBot {
    pub iterations: u64,
    pub infoset_endgame: Option<InfosetConfig>,
}

impl Policy<State> for IsmctsBot {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = state.current_player().expect("game not over");
        let cfg = IsmctsConfig {
            iterations: self.iterations,
            threads: 1,
            seed: rng.gen(),
            objective: Objective::Score,
            infoset_endgame: self.infoset_endgame.clone(),
            ..IsmctsConfig::default()
        };
        let rollout = big2_big2::bots::Greedy { epsilon: 0.1 };
        ismcts(state, me, &cfg, &rollout).best()
    }
}

/// One level of the ladder.
#[derive(Clone)]
pub enum LadderBot {
    Random,
    Easy(ErrorModel<StructuredGreedy>),
    Medium(ErrorModel<QNetPolicy>),
    Search(IsmctsBot),
}

fn rates<P>(inner: P, r: [f64; 3]) -> ErrorModel<P> {
    ErrorModel {
        inner,
        hold_two_too_long: r[0],
        break_pair_for_single: r[1],
        miss_safe_single: r[2],
    }
}

/// The five level names, weakest first.
pub const LEVELS: [&str; 5] = ["random", "easy", "medium", "hard", "expert"];

impl LadderBot {
    pub fn random() -> LadderBot {
        LadderBot::Random
    }

    pub fn easy() -> LadderBot {
        LadderBot::Easy(rates(
            StructuredGreedy {
                epsilon: EASY_EPSILON,
                ..StructuredGreedy::default()
            },
            EASY_RATES,
        ))
    }

    /// `net` is the distilled Q-network (`models/big2-qnet-duty.pknn`).
    pub fn medium(net: Arc<QNet>) -> LadderBot {
        LadderBot::Medium(rates(
            QNetPolicy {
                net,
                temperature: MEDIUM_TEMPERATURE,
            },
            MEDIUM_RATES,
        ))
    }

    pub fn hard() -> LadderBot {
        LadderBot::Search(IsmctsBot {
            iterations: HARD_ITERATIONS,
            infoset_endgame: None,
        })
    }

    pub fn expert() -> LadderBot {
        LadderBot::Search(IsmctsBot {
            iterations: EXPERT_ITERATIONS,
            infoset_endgame: Some(InfosetConfig::default()),
        })
    }

    /// The level by name (`LEVELS`); `medium` needs `net`.
    pub fn by_name(name: &str, net: Option<Arc<QNet>>) -> Result<LadderBot, String> {
        Ok(match name {
            "random" => LadderBot::random(),
            "easy" => LadderBot::easy(),
            "medium" => LadderBot::medium(net.ok_or("medium needs the Q-network file")?),
            "hard" => LadderBot::hard(),
            "expert" => LadderBot::expert(),
            other => return Err(format!("unknown level {other}")),
        })
    }

    /// The move and the mistake that produced it, if any (review shows its key).
    pub fn choose_explained<R: Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> (Action, Option<Mistake>) {
        match self {
            LadderBot::Random => (legal[rng.gen_range(0..legal.len())], None),
            LadderBot::Easy(m) => m.choose_explained(state, legal, rng),
            LadderBot::Medium(m) => m.choose_explained(state, legal, rng),
            LadderBot::Search(s) => (s.choose(state, legal, rng), None),
        }
    }
}

impl Policy<State> for LadderBot {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        self.choose_explained(state, legal, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use big2_big2::qnet::synthetic_graph;
    use big2_big2::RuleSet;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn ladder_levels_play_legal_moves_to_the_end() {
        let net = Arc::new(QNet::new(synthetic_graph(16, 8, 8, 16, 3)).unwrap());
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let bots: Vec<LadderBot> = LEVELS
            .iter()
            .map(|n| match *n {
                // Small searches keep the test fast; the constructors are checked below.
                "hard" | "expert" => LadderBot::Search(IsmctsBot {
                    iterations: 30,
                    infoset_endgame: Some(InfosetConfig::default()),
                }),
                _ => LadderBot::by_name(n, Some(net.clone())).unwrap(),
            })
            .collect();
        for _ in 0..3 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while !s.is_terminal() {
                let p = s.current_player().unwrap();
                let legal = s.legal_actions_vec();
                let a = bots[p % bots.len()].choose(&s, &legal, &mut rng);
                assert!(legal.contains(&a));
                s.apply(a);
            }
        }
    }

    #[test]
    fn ladder_constructors_match_the_table() {
        let LadderBot::Search(h) = LadderBot::hard() else {
            panic!()
        };
        let LadderBot::Search(e) = LadderBot::expert() else {
            panic!()
        };
        assert_eq!((h.iterations, e.iterations), (2_000, 10_000));
        assert!(h.infoset_endgame.is_none() && e.infoset_endgame.is_some());
        let LadderBot::Easy(m) = LadderBot::easy() else {
            panic!()
        };
        assert_eq!(m.hold_two_too_long, EASY_RATES[0]);
        assert!(LadderBot::by_name("medium", None).is_err());
        assert!(LadderBot::by_name("grandmaster", None).is_err());
    }
}
