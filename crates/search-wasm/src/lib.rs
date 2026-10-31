//! Timing harness for the search, the same code in Node WebAssembly and natively.
//!
//! One entry point, [`Harness::run`]: it takes a step as a line of text and returns a
//! JSON array with one row per position. `experiments/wasm_timing.mjs` sends the same
//! steps to the WebAssembly build and to the native example `puct_timing`, so the two
//! columns of a table come from identical code and identical positions. The rows are
//! built by hand: this crate wants as little in its `.wasm` as the timing allows.
//!
//! Steps (`seed` picks the deals; the search seed of position `i` is `i`):
//!
//! * `opening puct <simulations> <lambda> <count> <seed>`: PUCT on the first decision of
//!   `count` deals.
//! * `opening ismcts <iterations> <count> <seed>`: the same with ISMCTS.
//! * `midgame ismcts <iterations> <count> <seed>`: ISMCTS on `anytime`'s positions.
//! * `anytime <simulations> <lambda> <first_ms> <count> <seed>`: PUCT on `count` positions
//!   from greedy self-play, recording whether the move changes after `first_ms`.
//! * `endgame <threshold> <max_cards> <max_nodes> <position>`: the information-set exact
//!   endgame on one position (a line of `parse_position`).

use big2_big2::bots::Greedy;
use big2_big2::pvnet::PvNet;
use big2_big2::{RuleSet, State};
use big2_games::{Game, Policy};
use big2_search::{
    infoset_endgame, ismcts, puct_with_rollout, InfosetConfig, IsmctsConfig, Objective, PuctConfig,
    UniformSampler,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use web_time::Instant;

/// PUCT's exploration constant in every match of `build-phase3.md`.
const C_PUCT: f64 = 2.5;
/// The rollout's random-move share in every match (pybind's default).
const ROLLOUT_EPSILON: f64 = 0.1;

/// The first decision of `count` deals.
pub fn opening_positions(count: usize, seed: u64) -> Vec<State> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| State::new_deal(RuleSet::default(), &mut rng))
        .collect()
}

/// One random decision with a choice from each of `count` greedy self-play games.
/// Forced moves are left out: no search runs for them.
pub fn midgame_positions(count: usize, seed: u64) -> Vec<State> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bot = Greedy { epsilon: 0.1 };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut choices = Vec::new();
        while s.current_player().is_some() {
            let legal = s.legal_actions_vec();
            if legal.len() > 1 {
                choices.push(s.clone());
            }
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        if !choices.is_empty() {
            let i = rng.gen_range(0..choices.len() as u32) as usize;
            out.push(choices.swap_remove(i));
        }
    }
    out
}

/// A position as text: the four starting hands, then every action so far, all as
/// decimal card masks (`experiments/wasm_positions.py` writes them).
pub fn parse_position(words: &[&str]) -> Result<State, String> {
    let n: Vec<u64> = words
        .iter()
        .map(|w| w.parse::<u64>().map_err(|e| format!("{w:?}: {e}")))
        .collect::<Result<_, _>>()?;
    if n.len() < 4 {
        return Err(format!("{} numbers, need at least 4 hands", n.len()));
    }
    let mut s = State::from_hands(RuleSet::default(), [n[0], n[1], n[2], n[3]])?;
    for &cards in &n[4..] {
        let a = s
            .action_for_cards(cards)
            .map_err(|e| format!("{cards}: {e:?}"))?;
        s.apply(a);
    }
    Ok(s)
}

fn number<T: std::str::FromStr>(words: &[&str], i: usize) -> Result<T, String> {
    words
        .get(i)
        .and_then(|w| w.parse().ok())
        .ok_or_else(|| format!("argument {i} missing or not a number"))
}

/// The network the search runs with; the steps run on the calling thread.
pub struct Harness {
    net: PvNet,
}

impl Harness {
    /// `pknn`: a policy-value network file (`big2 rl export`).
    pub fn new(pknn: &[u8]) -> Result<Harness, String> {
        Ok(Harness {
            net: PvNet::from_bytes(pknn)?,
        })
    }

    /// Runs one step (module docs); the JSON text of its rows.
    pub fn run(&self, step: &str) -> Result<String, String> {
        let words: Vec<&str> = step.split_whitespace().collect();
        let rows: Vec<String> = match words.as_slice() {
            ["opening", "puct", ..] => {
                let (sims, lambda) = (number(&words, 2)?, number(&words, 3)?);
                let (count, seed) = (number(&words, 4)?, number(&words, 5)?);
                opening_positions(count, seed)
                    .iter()
                    .enumerate()
                    .map(|(i, s)| self.puct_row(s, sims, lambda, None, i as u64))
                    .collect()
            }
            ["opening", "ismcts", ..] => {
                let (iterations, count) = (number(&words, 2)?, number(&words, 3)?);
                let seed = number(&words, 4)?;
                opening_positions(count, seed)
                    .iter()
                    .enumerate()
                    .map(|(i, s)| ismcts_row(s, iterations, i as u64))
                    .collect()
            }
            ["midgame", "ismcts", ..] => {
                let (iterations, count) = (number(&words, 2)?, number(&words, 3)?);
                let seed = number(&words, 4)?;
                midgame_positions(count, seed)
                    .iter()
                    .enumerate()
                    .map(|(i, s)| ismcts_row(s, iterations, i as u64))
                    .collect()
            }
            ["anytime", ..] => {
                let (sims, lambda) = (number(&words, 1)?, number(&words, 2)?);
                let first_ms: u64 = number(&words, 3)?;
                let (count, seed) = (number(&words, 4)?, number(&words, 5)?);
                midgame_positions(count, seed)
                    .iter()
                    .enumerate()
                    .map(|(i, s)| self.puct_row(s, sims, lambda, Some(first_ms), i as u64))
                    .collect()
            }
            ["endgame", ..] if words.len() >= 8 => {
                let cfg = InfosetConfig {
                    threshold: number(&words, 1)?,
                    max_cards: number(&words, 2)?,
                    max_nodes: number(&words, 3)?,
                };
                vec![endgame_row(&parse_position(&words[4..])?, &cfg)]
            }
            _ => return Err(format!("unknown step {step:?}")),
        };
        Ok(format!("[{}]", rows.join(",")))
    }

    fn puct_row(
        &self,
        state: &State,
        simulations: u64,
        lambda: f64,
        first_ms: Option<u64>,
        seed: u64,
    ) -> String {
        let seat = state.current_player().expect("a decision");
        let cfg = PuctConfig {
            simulations,
            c_puct: C_PUCT,
            seed,
            leaf_network_weight: lambda,
            anytime_first_ms: first_ms,
            ..PuctConfig::default()
        };
        let rollout = Greedy {
            epsilon: ROLLOUT_EPSILON,
        };
        let start = Instant::now();
        let r = puct_with_rollout(state, seat, &cfg, &self.net, &UniformSampler, &rollout);
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let best = r.analysis.best();
        // No first answer: the whole search ended before `first_ms`, so nothing can change.
        let changed = r.first_answer.is_some_and(|a| a != best);
        format!(
            r#"{{"ms":{ms:.3},"legal":{},"simulations":{},"reached_first_ms":{},"first_simulations":{},"changed":{changed}}}"#,
            state.legal_actions_vec().len(),
            r.analysis.simulations,
            r.first_answer.is_some(),
            r.first_answer_simulations,
        )
    }
}

fn ismcts_row(state: &State, iterations: u64, seed: u64) -> String {
    let seat = state.current_player().expect("a decision");
    let cfg = IsmctsConfig {
        iterations,
        seed,
        ..IsmctsConfig::default()
    };
    let rollout = Greedy {
        epsilon: ROLLOUT_EPSILON,
    };
    let start = Instant::now();
    let r = ismcts(state, seat, &cfg, &rollout);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    format!(
        r#"{{"ms":{ms:.3},"legal":{},"simulations":{}}}"#,
        state.legal_actions_vec().len(),
        r.simulations
    )
}

fn endgame_row(state: &State, cfg: &InfosetConfig) -> String {
    let seat = state.current_player().expect("a decision");
    let start = Instant::now();
    let r = infoset_endgame(state, seat, cfg, Objective::Score, &UniformSampler);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let (applies, exact) = match &r {
        Some(r) => (true, r.exact),
        None => (false, false),
    };
    format!(r#"{{"ms":{ms:.3},"applies":{applies},"exact":{exact}}}"#)
}

#[cfg(target_arch = "wasm32")]
mod bindings {
    use wasm_bindgen::prelude::*;

    /// [`crate::Harness`] for Node.
    #[wasm_bindgen]
    pub struct Harness(crate::Harness);

    #[wasm_bindgen]
    impl Harness {
        #[wasm_bindgen(constructor)]
        pub fn new(pknn: &[u8]) -> Result<Harness, JsError> {
            crate::Harness::new(pknn)
                .map(Harness)
                .map_err(|e| JsError::new(&e))
        }

        pub fn run(&self, step: &str) -> Result<String, JsError> {
            self.0.run(step).map_err(|e| JsError::new(&e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_repeatable_decisions_with_a_choice() {
        let a = midgame_positions(5, 3);
        let b = midgame_positions(5, 3);
        assert_eq!(a.len(), 5);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.hands(), y.hands());
            assert!(x.legal_actions_vec().len() > 1);
        }
    }

    #[test]
    fn a_position_line_replays_to_the_same_state() {
        let s = opening_positions(1, 9).remove(0);
        let h = s.hands();
        let a = s.legal_actions_vec()[0];
        let line = format!("{} {} {} {} {}", h[0], h[1], h[2], h[3], a.cards());
        let words: Vec<&str> = line.split_whitespace().collect();
        let got = parse_position(&words).unwrap();
        let mut want = s.clone();
        want.apply(a);
        assert_eq!(got.hands(), want.hands());
        assert_eq!(got.current_player(), want.current_player());
    }

    #[test]
    fn a_bad_step_is_an_error_not_a_panic() {
        // parse_position runs before any network is needed.
        assert!(parse_position(&["1", "2"]).is_err());
        assert!(number::<u64>(&["x"], 0).is_err());
    }
}
