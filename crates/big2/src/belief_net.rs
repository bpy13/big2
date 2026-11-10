//! The learned belief about the hidden cards (plan Task 19), read from a `.pknn` file
//! written by `big2 rl belief-train --export` (`big2.rl.belief`).
//!
//! One network (a correction to the hand-size baseline: seat `s` holds a hidden card with
//! probability `cards_s / hidden cards`) from the observer's state features ([`encode_state`], what the Q-network
//! sees) to `52 × 3` logits: for each card, which of the three opponents (relative
//! seats 1, 2, 3) holds it. Only the cards the observer cannot see matter. The network
//! knows nothing of how a deal is drawn: [`BeliefNet::deal_log_weight`] scores a whole
//! candidate deal as the sum of its cards' log-probabilities, which is the importance
//! weight of a uniform draw towards the net's product-of-cards belief.

use std::collections::BTreeMap;

use big2_games::Game;
use big2_nn::{Dense, Graph, Mlp};

use crate::card::FULL_DECK;
use crate::features::{encode_state, STATE_LEN};
use crate::state::{State, NUM_PLAYERS};

/// `kind` metadata of a belief file.
pub const BELIEF_KIND: &str = "big2-belief";
/// `features` metadata: the state encoding of [`encode_state`].
pub const BELIEF_FEATURES: &str = "big2-state-v1";

/// Per card, the log-probability that relative seat 1, 2, 3 holds it.
pub type CardLogProbs = [[f32; 3]; 52];

/// The belief network.
#[derive(Clone, Debug)]
pub struct BeliefNet {
    mlp: Mlp,
    /// The file's metadata.
    pub meta: BTreeMap<String, String>,
}

impl BeliefNet {
    /// Builds the network from a graph with one tower `belief`, checking the metadata
    /// and the sizes against this crate's features.
    pub fn new(g: Graph) -> Result<BeliefNet, String> {
        let kind = g.meta.get("kind").map(String::as_str);
        if kind != Some(BELIEF_KIND) {
            return Err(format!("not a belief file (kind {kind:?})"));
        }
        let features = g.meta.get("features").map(String::as_str);
        if features != Some(BELIEF_FEATURES) {
            return Err(format!(
                "network expects features {features:?}, not {BELIEF_FEATURES:?}"
            ));
        }
        let mlp = g
            .tower("belief")
            .cloned()
            .ok_or_else(|| "missing tower \"belief\"".to_string())?;
        if mlp.n_in() != STATE_LEN || mlp.n_out() != 52 * 3 {
            return Err(format!(
                "belief tower is {}→{}, expected {STATE_LEN}→156",
                mlp.n_in(),
                mlp.n_out()
            ));
        }
        Ok(BeliefNet { mlp, meta: g.meta })
    }

    /// Reads a `.pknn` version 2 file (see [`big2_nn`]).
    pub fn from_bytes(bytes: &[u8]) -> Result<BeliefNet, String> {
        BeliefNet::new(Graph::from_bytes(bytes)?)
    }

    /// The belief of the player to act in `state`, from what they can see.
    ///
    /// # Panics
    /// Panics if the game is over.
    pub fn card_log_probs(&self, state: &State) -> CardLogProbs {
        let mut features = Vec::with_capacity(STATE_LEN);
        encode_state(state, &mut features);
        let active: Vec<usize> = features
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, _)| i)
            .collect();
        let logits = self.mlp.forward_sparse(&active);
        // The baseline the net corrects: seat `s` holds a hidden card with probability
        // `cards_s / hidden cards` (what a uniform deal gives).
        let me = state.current_player().expect("game not over");
        let sizes = state.hand_sizes();
        let n: [f32; 3] = std::array::from_fn(|r| sizes[(me + r + 1) % NUM_PLAYERS] as f32);
        let total = n.iter().sum::<f32>().max(1e-9);
        let base: [f32; 3] = std::array::from_fn(|r| (n[r].max(1e-9) / total).ln());
        let mut out = [[0.0f32; 3]; 52];
        for (c, row) in out.iter_mut().enumerate() {
            let l: [f32; 3] = std::array::from_fn(|r| logits[c * 3 + r] + base[r]);
            let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z = l.iter().map(|x| (x - m).exp()).sum::<f32>().ln() + m;
            for (r, x) in row.iter_mut().zip(l.iter()) {
                *r = x - z;
            }
        }
        out
    }

    /// `Σ` over the cards `observer` cannot see of the log-probability the belief gave
    /// to the seat that holds each one in `deal`.
    pub fn deal_log_weight(lp: &CardLogProbs, observer: usize, deal: &State) -> f64 {
        let own = deal.hand(observer);
        let mut unseen = FULL_DECK & !own & !deal.played();
        let mut total = 0.0f64;
        while unseen != 0 {
            let c = unseen.trailing_zeros() as usize;
            unseen &= unseen - 1;
            for rel in 1..NUM_PLAYERS {
                if deal.hand((observer + rel) % NUM_PLAYERS) >> c & 1 == 1 {
                    total += f64::from(lp[c][rel - 1]);
                    break;
                }
            }
        }
        total
    }
}

/// A belief graph with pseudo-random weights, for tests (a real network comes from
/// `big2 rl belief-train`).
#[doc(hidden)]
pub fn synthetic_graph(hidden: usize, seed: u64) -> Graph {
    let mut x = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f32 / (1u64 << 52) as f32 - 1.0
    };
    let mut layer = |n_in: usize, n_out: usize| {
        let s = 1.0 / (n_in as f32).sqrt();
        Dense {
            n_in,
            n_out,
            weight: (0..n_in * n_out).map(|_| next() * s).collect(),
            bias: (0..n_out).map(|_| next() * s).collect(),
        }
    };
    let layers = vec![layer(STATE_LEN, hidden), layer(hidden, 52 * 3)];
    Graph {
        meta: BTreeMap::from([
            ("kind".to_string(), BELIEF_KIND.to_string()),
            ("features".to_string(), BELIEF_FEATURES.to_string()),
        ]),
        towers: vec![(
            "belief".to_string(),
            Mlp::new(layers, BTreeMap::new()).unwrap(),
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::RuleSet;
    use big2_nn::Precision;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn mid_game(seed: u64, moves: usize) -> State {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..moves {
            let legal = s.legal_actions_vec();
            s.apply(legal[rng.gen_range(0..legal.len())]);
        }
        s
    }

    #[test]
    fn file_round_trip_and_normalised_rows() {
        let bytes = synthetic_graph(16, 1).to_bytes(Precision::F32);
        let net = BeliefNet::from_bytes(&bytes).unwrap();
        let lp = net.card_log_probs(&mid_game(5, 12));
        for row in lp {
            let p: f32 = row.iter().map(|x| x.exp()).sum();
            assert!((p - 1.0).abs() < 1e-5, "row sums to {p}");
        }
    }

    #[test]
    fn rejects_other_files() {
        let mut g = synthetic_graph(8, 1);
        g.meta.insert("kind".into(), "big2-qnet".into());
        assert!(BeliefNet::new(g).is_err());
        let mut g = synthetic_graph(8, 1);
        g.meta.insert("features".into(), "other".into());
        assert!(BeliefNet::new(g).is_err());
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn deal_weight_adds_the_holders_log_probs() {
        let net = BeliefNet::new(synthetic_graph(16, 2)).unwrap();
        let s = mid_game(9, 10);
        let me = s.current_player().unwrap();
        let lp = net.card_log_probs(&s);
        let mut rng = SmallRng::seed_from_u64(3);
        let d = s.determinize(me, &mut rng);
        let w = BeliefNet::deal_log_weight(&lp, me, &d);
        // By hand, card by card.
        let mut expect = 0.0f64;
        for c in 0..52 {
            for rel in 1..4 {
                if d.hand((me + rel) % 4) >> c & 1 == 1 {
                    expect += f64::from(lp[c][rel - 1]);
                }
            }
        }
        assert!((w - expect).abs() < 1e-9);
        assert!(w < 0.0);
    }
}
