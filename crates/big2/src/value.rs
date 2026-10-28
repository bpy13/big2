//! A learned value network as a search leaf evaluator.
//!
//! The network reads the full-information features of a (sampled) position
//! ([`crate::features::encode_full`], relative to the player to act) and outputs
//! eight numbers: each relative seat's expected score divided by `score_scale`,
//! then each seat's win logit. It is trained in Python (`big2 rl value-train`)
//! on positions from self-play and exported to the `.pknn` format of
//! [`big2_nn`].

use crate::features::{active_full, FULL_FEATURES_NAME, FULL_LEN};
use crate::state::{State, NUM_PLAYERS};
use big2_games::{Estimate, Evaluator, Game};
use big2_nn::Mlp;

/// A value network for Big Two positions.
#[derive(Clone, Debug)]
pub struct ValueNet {
    mlp: Mlp,
    score_scale: f32,
}

impl ValueNet {
    /// Checks that `mlp` takes the full-information features and has 8 outputs.
    pub fn new(mlp: Mlp) -> Result<ValueNet, String> {
        let features = mlp.meta.get("features").map(String::as_str);
        if features != Some(FULL_FEATURES_NAME) {
            return Err(format!(
                "network expects features {features:?}, not {FULL_FEATURES_NAME:?}"
            ));
        }
        if mlp.n_in() != FULL_LEN || mlp.n_out() != 2 * NUM_PLAYERS {
            return Err(format!(
                "network shape {}→{} does not match {FULL_LEN}→{}",
                mlp.n_in(),
                mlp.n_out(),
                2 * NUM_PLAYERS
            ));
        }
        let score_scale = match mlp.meta.get("score_scale") {
            Some(v) => v.parse().map_err(|_| format!("bad score_scale {v:?}"))?,
            None => 1.0,
        };
        Ok(ValueNet { mlp, score_scale })
    }

    /// Reads a `.pknn` file.
    pub fn from_bytes(bytes: &[u8]) -> Result<ValueNet, String> {
        ValueNet::new(Mlp::from_bytes(bytes)?)
    }
}

impl Evaluator<State> for ValueNet {
    /// Scores are made zero-sum (as real scores are) by subtracting their mean;
    /// win probabilities are a softmax over the four seats.
    fn evaluate(&self, s: &State) -> Estimate {
        let me = s.current_player().expect("evaluate a game in progress");
        let mut active = Vec::with_capacity(64);
        active_full(s, &mut active);
        let out = self.mlp.forward_sparse(&active);
        let (score_rel, logit_rel) = out.split_at(NUM_PLAYERS);
        let mean = score_rel.iter().sum::<f32>() / NUM_PLAYERS as f32;
        let max = logit_rel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logit_rel.iter().map(|l| (l - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        let mut scores = vec![0.0; NUM_PLAYERS];
        let mut wins = vec![0.0; NUM_PLAYERS];
        for r in 0..NUM_PLAYERS {
            let seat = (me + r) % NUM_PLAYERS;
            scores[seat] = ((score_rel[r] - mean) * self.score_scale) as f64;
            wins[seat] = (exp[r] / total) as f64;
        }
        Estimate { scores, wins }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::RuleSet;
    use big2_nn::Dense;
    use rand::{rngs::SmallRng, SeedableRng};
    use std::collections::BTreeMap;

    #[test]
    fn outputs_map_from_relative_to_absolute_seats() {
        // One layer, zero weights, biases favour relative seat 1 (the next player).
        let layer = Dense {
            n_in: FULL_LEN,
            n_out: 8,
            weight: vec![0.0; FULL_LEN * 8],
            bias: vec![-1.0, 3.0, -1.0, -1.0, 0.0, 5.0, 0.0, 0.0],
        };
        let meta = BTreeMap::from([
            ("features".to_string(), FULL_FEATURES_NAME.to_string()),
            ("score_scale".to_string(), "10".to_string()),
        ]);
        let net = ValueNet::new(Mlp::new(vec![layer], meta).unwrap()).unwrap();
        let s = State::new_deal(RuleSet::default(), &mut SmallRng::seed_from_u64(0));
        let me = s.current_player().unwrap();
        let est = net.evaluate(&s);
        let next = (me + 1) % 4;
        assert!((est.scores[next] - 30.0).abs() < 1e-4, "{est:?}");
        assert!((est.scores.iter().sum::<f64>()).abs() < 1e-4);
        assert!(est.wins[next] > 0.9);
        assert!((est.wins.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
