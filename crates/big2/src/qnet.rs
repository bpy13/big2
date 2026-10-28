//! The distilled Q-network in Rust: scores every legal move of a decision.
//!
//! This is the PyTorch `QNet` of `python/big2/rl/model.py`, read from a `.pknn`
//! version 2 file written by `big2 rl export` (`big2.rl.export`). The file holds
//! three towers ([`big2_nn::Graph`]):
//!
//! ```text
//! state  (STATE_LEN)           ─ 512 ─ 256 ─┐ ReLU
//!                                           ├─ head: 256 ─ 128 ─ 2  (score, win logit)
//! action (ACTION_LEN + 52 hand) ─ 128 ──────┘ ReLU
//! ```
//!
//! The action tower's input is the move's features ([`encode_action`]) followed by
//! the 52 cards the player would have left after it. Both towers end in a ReLU (the
//! file's towers do not include it; [`QNet`] applies it). The state tower runs once
//! per decision; the first head layer is split so its state half also runs once,
//! and only the move half runs per legal move.
//!
//! The network sees only what the player to act can see ([`encode_state`]), so its
//! scores never depend on hidden cards.

use std::collections::BTreeMap;
use std::sync::Arc;

use big2_games::{Game, Policy};
use big2_nn::{Dense, Graph, Mlp};
use rand::Rng;

use crate::features::{encode_action, encode_state, ACTION_LEN, STATE_LEN};
use crate::state::{Action, State};

/// `kind` metadata of a Q-network file.
pub const QNET_KIND: &str = "big2-qnet";
/// `features` metadata: the state encoding of [`encode_state`] plus the action
/// encoding of [`encode_action`] with the hand left after the move.
pub const QNET_FEATURES: &str = "big2-state-v1";

/// The Q-network: per legal move, the acting player's predicted final score (in
/// points) and the logit of their probability of going out first.
#[derive(Clone, Debug)]
pub struct QNet {
    state: Mlp,
    action: Mlp,
    /// First head layer's weights for the state half, transposed (`state_out × hidden`).
    head_state_t: Vec<f32>,
    /// First head layer's weights for the move half, transposed (`action_out × hidden`).
    head_action_t: Vec<f32>,
    head_bias: Vec<f32>,
    /// Head layers after the first (ReLU is applied before them).
    head_rest: Option<Mlp>,
    score_scale: f32,
    /// The file's metadata.
    pub meta: BTreeMap<String, String>,
}

/// Transposes the columns `cols` of a row-major `n_out × n_in` matrix into
/// `cols.len() × n_out`.
fn transpose_cols(l: &Dense, cols: std::ops::Range<usize>) -> Vec<f32> {
    let mut t = Vec::with_capacity(cols.len() * l.n_out);
    for i in cols {
        t.extend((0..l.n_out).map(|o| l.weight[o * l.n_in + i]));
    }
    t
}

/// `y += Σ x[i] · t[i]` over the rows `t[i]` of a transposed matrix, skipping zeros
/// (half the inputs are zero after a ReLU).
fn add_rows(y: &mut [f32], x: &[f32], t: &[f32]) {
    let n = y.len();
    for (i, &v) in x.iter().enumerate() {
        if v != 0.0 {
            for (y, w) in y.iter_mut().zip(&t[i * n..(i + 1) * n]) {
                *y += v * w;
            }
        }
    }
}

fn relu(v: &mut [f32]) {
    for x in v {
        *x = x.max(0.0);
    }
}

impl QNet {
    /// Builds the network from a graph with towers `state`, `action` and `head`,
    /// checking the metadata and every size against this crate's features.
    pub fn new(g: Graph) -> Result<QNet, String> {
        let kind = g.meta.get("kind").map(String::as_str);
        if kind != Some(QNET_KIND) {
            return Err(format!("not a Q-network file (kind {kind:?})"));
        }
        let features = g.meta.get("features").map(String::as_str);
        if features != Some(QNET_FEATURES) {
            return Err(format!(
                "network expects features {features:?}, not {QNET_FEATURES:?}"
            ));
        }
        let score_scale = match g.meta.get("score_scale") {
            Some(v) => v.parse().map_err(|_| format!("bad score_scale {v:?}"))?,
            None => return Err("missing score_scale".into()),
        };
        let tower = |name: &str| {
            g.tower(name)
                .cloned()
                .ok_or_else(|| format!("missing tower {name:?}"))
        };
        let (state, action, head) = (tower("state")?, tower("action")?, tower("head")?);
        if state.n_in() != STATE_LEN {
            return Err(format!(
                "state tower takes {}, not {STATE_LEN}",
                state.n_in()
            ));
        }
        if action.n_in() != ACTION_LEN + 52 {
            return Err(format!(
                "action tower takes {}, not {}",
                action.n_in(),
                ACTION_LEN + 52
            ));
        }
        let (s_out, a_out) = (state.n_out(), action.n_out());
        if head.n_in() != s_out + a_out || head.n_out() != 2 {
            return Err(format!(
                "head is {}→{}, expected {}→2",
                head.n_in(),
                head.n_out(),
                s_out + a_out
            ));
        }
        let first = &head.layers[0];
        let head_rest = if head.layers.len() > 1 {
            Some(Mlp::new(head.layers[1..].to_vec(), BTreeMap::new())?)
        } else {
            None
        };
        Ok(QNet {
            head_state_t: transpose_cols(first, 0..s_out),
            head_action_t: transpose_cols(first, s_out..s_out + a_out),
            head_bias: first.bias.clone(),
            head_rest,
            state,
            action,
            score_scale,
            meta: g.meta,
        })
    }

    /// Reads a `.pknn` version 2 file (see [`big2_nn`]).
    pub fn from_bytes(bytes: &[u8]) -> Result<QNet, String> {
        QNet::new(Graph::from_bytes(bytes)?)
    }

    /// Scores each of `legal` (moves of the player to act in `state`), in the same
    /// order: `(predicted final score in points, win logit)`.
    ///
    /// # Panics
    /// Panics if the game is over.
    pub fn score_moves(&self, state: &State, legal: &[Action]) -> Vec<(f32, f32)> {
        let mut features = Vec::with_capacity(STATE_LEN);
        encode_state(state, &mut features);
        let active: Vec<usize> = (0..STATE_LEN).filter(|&i| features[i] == 1).collect();
        let mut h = self.state.forward_sparse(&active);
        relu(&mut h);
        // The state half of the first head layer, shared by every move.
        let mut base = self.head_bias.clone();
        add_rows(&mut base, &h, &self.head_state_t);

        let hand = state.hand(state.current_player().expect("game not over"));
        let mut act = Vec::with_capacity(ACTION_LEN);
        let mut active = Vec::with_capacity(32);
        legal
            .iter()
            .map(|a| {
                act.clear();
                encode_action(a, &mut act);
                active.clear();
                active.extend((0..ACTION_LEN).filter(|&i| act[i] == 1));
                let left = hand & !a.cards();
                active.extend(big2_cards::mask::bits(left).map(|c| ACTION_LEN + c as usize));
                let mut m = self.action.forward_sparse(&active);
                relu(&mut m);
                let mut x = base.clone();
                add_rows(&mut x, &m, &self.head_action_t);
                let out = match &self.head_rest {
                    Some(rest) => {
                        relu(&mut x);
                        rest.forward(&x)
                    }
                    None => x,
                };
                (out[0] * self.score_scale, out[1])
            })
            .collect()
    }
}

/// Plays with a [`QNet`]: the move with the best predicted score when
/// `temperature` is 0, otherwise a move drawn with probability proportional to
/// `exp(score / temperature)` (`temperature` in points).
#[derive(Clone, Debug)]
pub struct QNetPolicy {
    pub net: Arc<QNet>,
    pub temperature: f64,
}

impl Policy<State> for QNetPolicy {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let scores = self.net.score_moves(state, legal);
        let best = scores
            .iter()
            .map(|s| s.0 as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        if self.temperature <= 0.0 {
            let i = scores.iter().position(|s| s.0 as f64 == best).unwrap();
            return legal[i];
        }
        let weights: Vec<f64> = scores
            .iter()
            .map(|s| ((s.0 as f64 - best) / self.temperature).exp())
            .collect();
        let mut x = rng.gen::<f64>() * weights.iter().sum::<f64>();
        for (a, w) in legal.iter().zip(&weights) {
            if x < *w {
                return *a;
            }
            x -= w;
        }
        legal[legal.len() - 1]
    }
}

/// A Q-network graph with the given sizes and pseudo-random weights in ±`1/√n_in`,
/// for tests and benchmarks (a real network comes from `big2 rl export`).
#[doc(hidden)]
pub fn synthetic_graph(
    state_hidden: usize,
    state_out: usize,
    action_out: usize,
    head_hidden: usize,
    seed: u64,
) -> Graph {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
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
    let mlp = |layers: Vec<Dense>| Mlp::new(layers, BTreeMap::new()).unwrap();
    let state = mlp(vec![
        layer(STATE_LEN, state_hidden),
        layer(state_hidden, state_out),
    ]);
    let action = mlp(vec![layer(ACTION_LEN + 52, action_out)]);
    let head = mlp(vec![
        layer(state_out + action_out, head_hidden),
        layer(head_hidden, head_hidden / 2),
        layer(head_hidden / 2, 2),
    ]);
    Graph {
        meta: BTreeMap::from([
            ("kind".to_string(), QNET_KIND.to_string()),
            ("features".to_string(), QNET_FEATURES.to_string()),
            ("score_scale".to_string(), "50".to_string()),
        ]),
        towers: vec![
            ("state".to_string(), state),
            ("action".to_string(), action),
            ("head".to_string(), head),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::RuleSet;
    use big2_nn::Precision;
    use rand::{rngs::SmallRng, SeedableRng};

    fn tiny_file(precision: Precision) -> Vec<u8> {
        synthetic_graph(16, 8, 8, 16, 3).to_bytes(precision)
    }

    /// The same network computed the plain way: concatenate, then run each layer.
    fn reference(g: &Graph, state: &State, legal: &[Action]) -> Vec<(f32, f32)> {
        let mut s = Vec::new();
        encode_state(state, &mut s);
        let s: Vec<f32> = s.iter().map(|&v| v as f32).collect();
        let mut h = g.tower("state").unwrap().forward(&s);
        relu(&mut h);
        let hand = state.hand(state.current_player().unwrap());
        legal
            .iter()
            .map(|a| {
                let mut f = Vec::new();
                encode_action(a, &mut f);
                let left = hand & !a.cards();
                let mut x: Vec<f32> = f.iter().map(|&v| v as f32).collect();
                x.extend((0..52).map(|c| ((left >> c) & 1) as f32));
                let mut m = g.tower("action").unwrap().forward(&x);
                relu(&mut m);
                let out = g.tower("head").unwrap().forward(&[h.clone(), m].concat());
                (out[0] * 50.0, out[1])
            })
            .collect()
    }

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
    fn qnet_loads_a_synthetic_v2_file_and_matches_the_plain_computation() {
        let bytes = tiny_file(Precision::F32);
        let g = Graph::from_bytes(&bytes).unwrap();
        let net = QNet::from_bytes(&bytes).unwrap();
        for seed in 0..5 {
            let s = mid_game(seed, 9);
            let legal = s.legal_actions_vec();
            let got = net.score_moves(&s, &legal);
            let want = reference(&g, &s, &legal);
            assert_eq!(got.len(), legal.len());
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-4,
                    "{a:?} {b:?}"
                );
            }
        }
        // The int8 file loads too and stays close.
        let q = QNet::from_bytes(&tiny_file(Precision::Int8 { group: 32 })).unwrap();
        let s = mid_game(1, 9);
        let legal = s.legal_actions_vec();
        for (a, b) in q
            .score_moves(&s, &legal)
            .iter()
            .zip(net.score_moves(&s, &legal))
        {
            assert!((a.0 - b.0).abs() < 0.5, "{a:?} {b:?}");
        }
    }

    #[test]
    fn qnet_scores_never_depend_on_hidden_cards() {
        // As features.rs's invariance test: two states differing only in hidden cards.
        let net = QNet::from_bytes(&tiny_file(Precision::Int8 { group: 32 })).unwrap();
        let mut rng = SmallRng::seed_from_u64(4);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..17 {
            let legal = s.legal_actions_vec();
            s.apply(legal[0]);
        }
        let me = s.current_player().unwrap();
        let d = s.determinize(me, &mut rng);
        assert_ne!(d.hands(), s.hands());
        let legal = s.legal_actions_vec();
        assert_eq!(legal, d.legal_actions_vec());
        assert_eq!(net.score_moves(&s, &legal), net.score_moves(&d, &legal));
    }

    #[test]
    fn qnet_rejects_files_of_the_wrong_kind_or_shape() {
        let mut g = synthetic_graph(16, 8, 8, 16, 3);
        g.meta.insert("kind".into(), "big2-value".into());
        assert!(QNet::new(g).unwrap_err().contains("kind"));
        let mut g = synthetic_graph(16, 8, 8, 16, 3);
        g.towers.remove(1);
        assert!(QNet::new(g).unwrap_err().contains("action"));
        let mut g = synthetic_graph(16, 8, 8, 16, 3);
        g.meta.insert("features".into(), "big2-full-v1".into());
        assert!(QNet::new(g).is_err());
    }

    #[test]
    fn qnet_policy_plays_the_best_move_cold_and_varies_when_warm() {
        let net = Arc::new(QNet::from_bytes(&tiny_file(Precision::F32)).unwrap());
        let s = (0..)
            .map(|seed| mid_game(seed, 4))
            .find(|s| s.legal_actions_vec().len() > 2)
            .unwrap();
        let legal = s.legal_actions_vec();
        let scores = net.score_moves(&s, &legal);
        let best = (0..legal.len())
            .max_by(|&a, &b| scores[a].0.total_cmp(&scores[b].0))
            .unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        let cold = QNetPolicy {
            net: net.clone(),
            temperature: 0.0,
        };
        for _ in 0..10 {
            assert_eq!(cold.choose(&s, &legal, &mut rng), legal[best]);
        }
        let hot = QNetPolicy {
            net,
            temperature: 1e6,
        };
        let picks: std::collections::HashSet<_> =
            (0..200).map(|_| hot.choose(&s, &legal, &mut rng)).collect();
        assert!(picks.len() > 1 && picks.iter().all(|a| legal.contains(a)));
    }

    #[test]
    fn qnet_plays_whole_games() {
        let net = Arc::new(QNet::from_bytes(&tiny_file(Precision::Int8 { group: 32 })).unwrap());
        let policy = QNetPolicy {
            net,
            temperature: 1.0,
        };
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..3 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            let out = big2_games::play_out(&mut s, &policy, &mut rng);
            assert_eq!(out.scores.iter().sum::<f64>(), 0.0);
        }
    }
}
