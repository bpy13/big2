//! The policy-value network (network v2) in Rust: a prior over the candidate moves
//! and, per candidate, each seat's distribution of cards left at the end of the deal.
//!
//! This is the PyTorch `QNetV2` of `python/big2/rl/model.py`, read from a `.pknn`
//! version 2 file written by `big2 rl export` (`big2.rl.export.export_pvnet_pknn`).
//! The towers are those of [`crate::qnet`]; only the head's output differs:
//!
//! ```text
//! state  (STATE_LEN)            ─ 512 ─ 256 ─┐ ReLU
//!                                            ├─ head: 256 ─ 128 ─ 57
//! action (ACTION_LEN + 52 hand) ─ 128 ───────┘ ReLU
//! ```
//!
//! The 57 outputs per candidate are 4 × 14 distribution logits (seat 0 = the player
//! to act, then the next, across and previous seats; class `k` = finishing with `k`
//! cards) and one prior logit. The expected score of any [`Scoring`] follows from the
//! distributions ([`seat_scores`]), so one network serves `tiered`, `original` and
//! `win_loss`.
//!
//! The network sees only what the player to act can see ([`encode_state`]).

use std::collections::BTreeMap;

use big2_games::Game;
use big2_nn::{Dense, Graph, Mlp};
use rand::Rng;

use crate::features::{encode_action, encode_state, ACTION_LEN, STATE_LEN};
use crate::plan::min_plays;
use crate::rules::Scoring;
use crate::state::{Action, State, NUM_PLAYERS};

/// `kind` metadata of a policy-value network file.
pub const PVNET_KIND: &str = "big2-pvnet";
/// `features` metadata: the same encoding as [`crate::qnet::QNET_FEATURES`].
pub const PVNET_FEATURES: &str = "big2-state-v1";
/// Classes of a cards-left distribution: 0 to 13 cards.
pub const CLASSES: usize = 14;
/// Seats per distribution set, relative to the player to act.
pub const SEATS: usize = NUM_PLAYERS;
/// Head outputs per candidate: the distribution logits, then the prior logit.
pub const OUTPUTS: usize = SEATS * CLASSES + 1;

/// Per seat (relative to the player to act), the probability of each cards-left count.
pub type Dist = [[f32; CLASSES]; SEATS];

/// Penalty per cards-left count 0–13 under `scoring` (rules §7.1). `WinLoss` gets
/// 0 for the winner and 1 for everyone else: with it, the pairwise settlement of
/// [`seat_scores`] gives the winner +3 and the others −1, as the rules do.
pub fn payoff(scoring: Scoring) -> [f32; CLASSES] {
    std::array::from_fn(|n| match scoring {
        Scoring::WinLoss => (n > 0) as u32 as f32,
        s => s.penalty(n as u32) as f32,
    })
}

/// Each seat's expected final score from cards-left distributions: with `E[p_i]`
/// the expected penalty of seat `i`, `score_i = Σ_j E[p_j] − 4·E[p_i]` (the pairwise
/// settlement is linear in the penalties, so its expectation needs only these).
pub fn seat_scores(dist: &Dist, scoring: Scoring) -> [f32; SEATS] {
    let pay = payoff(scoring);
    let pen: [f32; SEATS] =
        std::array::from_fn(|s| dist[s].iter().zip(&pay).map(|(p, v)| p * v).sum());
    let total: f32 = pen.iter().sum();
    std::array::from_fn(|s| total - SEATS as f32 * pen[s])
}

/// What the network says about one decision.
#[derive(Clone, Debug)]
pub struct PvOutput {
    /// Probability of each candidate (softmax of the prior logits), in their order.
    pub prior: Vec<f32>,
    /// Per candidate, the cards-left distributions after playing it.
    pub cand_dist: Vec<Dist>,
    /// The position's distributions: the candidates' mixed by the prior.
    pub dist: Dist,
    /// The player to act's expected score from [`PvOutput::dist`] under the state's
    /// scoring, in points.
    pub score: f32,
}

impl PvOutput {
    /// Every seat's expected score of the position under `scoring` (seats relative
    /// to the player to act).
    pub fn seat_scores(&self, scoring: Scoring) -> [f32; SEATS] {
        seat_scores(&self.dist, scoring)
    }

    /// The player to act's expected score of the position under `scoring`.
    pub fn score_for(&self, scoring: Scoring) -> f32 {
        self.seat_scores(scoring)[0]
    }

    /// The player to act's expected score after candidate `i` under `scoring`.
    pub fn candidate_score(&self, i: usize, scoring: Scoring) -> f32 {
        seat_scores(&self.cand_dist[i], scoring)[0]
    }
}

/// The policy-value network.
#[derive(Clone, Debug)]
pub struct PvNet {
    state: Mlp,
    action: Mlp,
    /// First head layer's weights for the state half, transposed (`state_out × hidden`).
    head_state_t: Vec<f32>,
    /// First head layer's weights for the move half, transposed (`action_out × hidden`).
    head_action_t: Vec<f32>,
    head_bias: Vec<f32>,
    /// Head layers after the first (ReLU is applied before them).
    head_rest: Option<Mlp>,
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

/// `y += Σ x[i] · t[i]` over the rows `t[i]` of a transposed matrix, skipping zeros.
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

/// Softmax of `logits` in place.
fn softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in logits.iter_mut() {
        *x /= sum;
    }
}

impl PvNet {
    /// Builds the network from a graph with towers `state`, `action` and `head`,
    /// checking the metadata and every size against this crate's features.
    pub fn new(g: Graph) -> Result<PvNet, String> {
        let kind = g.meta.get("kind").map(String::as_str);
        if kind != Some(PVNET_KIND) {
            return Err(format!("not a policy-value network file (kind {kind:?})"));
        }
        let features = g.meta.get("features").map(String::as_str);
        if features != Some(PVNET_FEATURES) {
            return Err(format!(
                "network expects features {features:?}, not {PVNET_FEATURES:?}"
            ));
        }
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
        if head.n_in() != s_out + a_out || head.n_out() != OUTPUTS {
            return Err(format!(
                "head is {}→{}, expected {}→{OUTPUTS}",
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
        Ok(PvNet {
            head_state_t: transpose_cols(first, 0..s_out),
            head_action_t: transpose_cols(first, s_out..s_out + a_out),
            head_bias: first.bias.clone(),
            head_rest,
            state,
            action,
            meta: g.meta,
        })
    }

    /// Reads a `.pknn` version 2 file (see [`big2_nn`]).
    pub fn from_bytes(bytes: &[u8]) -> Result<PvNet, String> {
        PvNet::new(Graph::from_bytes(bytes)?)
    }

    /// The raw head outputs (`OUTPUTS` per candidate, in order): 56 distribution
    /// logits (seat-major), then the prior logit.
    pub fn raw_outputs(&self, state: &State, candidates: &[Action]) -> Vec<[f32; OUTPUTS]> {
        let mut features = Vec::with_capacity(STATE_LEN);
        encode_state(state, &mut features);
        let active: Vec<usize> = (0..STATE_LEN).filter(|&i| features[i] == 1).collect();
        let mut h = self.state.forward_sparse(&active);
        relu(&mut h);
        // The state half of the first head layer, shared by every candidate.
        let mut base = self.head_bias.clone();
        add_rows(&mut base, &h, &self.head_state_t);

        let hand = state.hand(state.current_player().expect("game not over"));
        let mut act = Vec::with_capacity(ACTION_LEN);
        let mut active = Vec::with_capacity(32);
        candidates
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
                std::array::from_fn(|k| out[k])
            })
            .collect()
    }

    /// Evaluates the decision of the player to act in `state` over `candidates`
    /// (their legal moves, in any order; usually `plan::candidates`' order).
    ///
    /// # Panics
    /// Panics if the game is over or `candidates` is empty.
    pub fn evaluate(&self, state: &State, candidates: &[Action]) -> PvOutput {
        assert!(!candidates.is_empty(), "no candidates");
        let raw = self.raw_outputs(state, candidates);
        let mut prior: Vec<f32> = raw.iter().map(|o| o[OUTPUTS - 1]).collect();
        softmax(&mut prior);
        let cand_dist: Vec<Dist> = raw
            .iter()
            .map(|o| {
                std::array::from_fn(|s| {
                    let mut d: [f32; CLASSES] = std::array::from_fn(|k| o[s * CLASSES + k]);
                    softmax(&mut d);
                    d
                })
            })
            .collect();
        let mut dist = [[0.0f32; CLASSES]; SEATS];
        for (p, d) in prior.iter().zip(&cand_dist) {
            for s in 0..SEATS {
                for k in 0..CLASSES {
                    dist[s][k] += p * d[s][k];
                }
            }
        }
        let score = seat_scores(&dist, state.rules().scoring)[0];
        PvOutput {
            prior,
            cand_dist,
            dist,
            score,
        }
    }

    /// The fewest plays the player to act needs to shed their hand (the planner's
    /// [`min_plays`]), passed through for the teaching layer.
    ///
    /// # Panics
    /// Panics if the game is over.
    pub fn plays_to_out(state: &State) -> u32 {
        let me = state.current_player().expect("game not over");
        min_plays(state.hand(me), state.rules())
    }

    /// What holding the lead is worth to the player to act, in points under the
    /// state's scoring: their expected score if they lead the next trick minus their
    /// expected score if the other side leads it, from two evaluations.
    ///
    /// "The other side" is the holder of the current play (the seat that leads next
    /// if everyone passes), or the next seat when the player to act is leading.
    /// Both positions are rebuilt from the current hands with a fresh trick
    /// ([`State::from_position`], which keeps no history, so the network sees less
    /// than in the real position). The second evaluation is from the other seat's
    /// view, so it runs on `samples` deals of the hidden cards drawn from the actor's
    /// view (never the true hands), averaged.
    ///
    /// # Panics
    /// Panics if the game is over or `samples` is 0.
    pub fn lead_value<R: Rng + ?Sized>(&self, state: &State, samples: usize, rng: &mut R) -> f32 {
        assert!(samples > 0, "samples must be at least 1");
        let me = state.current_player().expect("game not over");
        let other = match state.current_play() {
            Some((holder, _)) if holder != me => holder,
            _ => (me + 1) % NUM_PLAYERS,
        };
        let scoring = state.rules().scoring;
        let lead_by = |s: &State, seat: usize| {
            State::from_position(*s.rules(), s.hands(), seat, None)
                .expect("the hands of a valid state form a valid position")
        };
        let value = |s: &State| {
            let legal = s.legal_actions_vec();
            self.evaluate(s, &legal).seat_scores(scoring)
        };
        let mine = value(&lead_by(state, me))[0];
        let rel = (me + NUM_PLAYERS - other) % NUM_PLAYERS; // the actor seen from `other`
        let theirs: f32 = (0..samples)
            .map(|_| value(&lead_by(&state.determinize(me, rng), other))[rel])
            .sum::<f32>()
            / samples as f32;
        mine - theirs
    }
}

/// A policy-value graph with the given sizes and pseudo-random weights in
/// ±`1/√n_in`, for tests and benchmarks (a real network comes from `big2 rl export`).
#[doc(hidden)]
pub fn synthetic_graph(
    state_hidden: usize,
    state_out: usize,
    action_out: usize,
    head_hidden: usize,
    seed: u64,
) -> Graph {
    let mut g =
        crate::qnet::synthetic_graph(state_hidden, state_out, action_out, head_hidden, seed);
    // Same towers as the Q-network; replace the head's last layer by a 57-output one.
    let mut x = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f32 / (1u64 << 52) as f32 - 1.0
    };
    let (_, head) = g.towers.iter_mut().find(|(n, _)| n == "head").unwrap();
    let mut layers = head.layers.clone();
    let n_in = layers.last().unwrap().n_in;
    let s = 1.0 / (n_in as f32).sqrt();
    *layers.last_mut().unwrap() = Dense {
        n_in,
        n_out: OUTPUTS,
        weight: (0..n_in * OUTPUTS).map(|_| next() * s).collect(),
        bias: (0..OUTPUTS).map(|_| next() * s).collect(),
    };
    *head = Mlp::new(layers, BTreeMap::new()).unwrap();
    g.meta = BTreeMap::from([
        ("kind".to_string(), PVNET_KIND.to_string()),
        ("features".to_string(), PVNET_FEATURES.to_string()),
    ]);
    g
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

    fn mid_game(seed: u64, moves: usize) -> State {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..moves {
            let legal = s.legal_actions_vec();
            s.apply(legal[rng.gen_range(0..legal.len())]);
        }
        s
    }

    /// The same network computed the plain way: concatenate, then run each tower.
    fn reference(g: &Graph, state: &State, legal: &[Action]) -> Vec<Vec<f32>> {
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
                g.tower("head").unwrap().forward(&[h.clone(), m].concat())
            })
            .collect()
    }

    #[test]
    fn pvnet_payoff_and_settlement_match_the_rules() {
        let t = payoff(Scoring::Tiered);
        assert_eq!(
            t.map(|v| v as u32),
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 18, 30, 33, 36, 52]
        );
        // One-hot distributions of finished deals give the engine's own scores.
        let mut rng = SmallRng::seed_from_u64(11);
        for scoring in [Scoring::Tiered, Scoring::Original, Scoring::WinLoss] {
            let rules = RuleSet {
                scoring,
                ..RuleSet::default()
            };
            for _ in 0..20 {
                let mut s = State::new_deal(rules, &mut rng);
                let out = big2_games::play_out(&mut s, &big2_games::UniformRandom, &mut rng);
                let sizes = s.hand_sizes();
                let mut d = [[0.0f32; CLASSES]; SEATS];
                for (seat, &n) in sizes.iter().enumerate() {
                    d[seat][n as usize] = 1.0;
                }
                let got = seat_scores(&d, scoring);
                for (g, want) in got.iter().zip(&out.scores) {
                    assert_eq!(*g as f64, *want, "{scoring:?} {sizes:?}");
                }
            }
        }
    }

    #[test]
    fn pvnet_loads_a_synthetic_file_and_matches_the_plain_computation() {
        let bytes = tiny_file(Precision::F32);
        let g = Graph::from_bytes(&bytes).unwrap();
        let net = PvNet::from_bytes(&bytes).unwrap();
        for seed in 0..5 {
            let s = mid_game(seed, 9);
            let legal = s.legal_actions_vec();
            let raw = net.raw_outputs(&s, &legal);
            let want = reference(&g, &s, &legal);
            for (a, b) in raw.iter().zip(&want) {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() < 1e-4, "{x} {y}");
                }
            }
            let out = net.evaluate(&s, &legal);
            assert_eq!(out.prior.len(), legal.len());
            assert!((out.prior.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            for d in out.cand_dist.iter().chain([&out.dist]) {
                for row in d {
                    assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                }
            }
            // The position's score is the prior-weighted mean of the candidates'.
            let mixed: f32 = (0..legal.len())
                .map(|i| out.prior[i] * out.candidate_score(i, Scoring::Tiered))
                .sum();
            assert!((mixed - out.score).abs() < 1e-3, "{mixed} {}", out.score);
            assert_eq!(out.score, out.score_for(Scoring::Tiered));
            assert!(out.seat_scores(Scoring::Tiered).iter().sum::<f32>().abs() < 1e-3);
            assert!(out.score_for(Scoring::WinLoss).abs() <= 3.0 + 1e-5);
        }
        // The int8 file loads too and stays close.
        let q = PvNet::from_bytes(&tiny_file(Precision::Int8 { group: 32 })).unwrap();
        let s = mid_game(1, 9);
        let legal = s.legal_actions_vec();
        let (a, b) = (q.evaluate(&s, &legal), net.evaluate(&s, &legal));
        for (x, y) in a.prior.iter().zip(&b.prior) {
            assert!((x - y).abs() < 0.05, "{x} {y}");
        }
    }

    #[test]
    fn pvnet_never_depends_on_hidden_cards() {
        // As features.rs's invariance test: two states differing only in hidden cards.
        let net = PvNet::from_bytes(&tiny_file(Precision::Int8 { group: 32 })).unwrap();
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
        let (a, b) = (net.evaluate(&s, &legal), net.evaluate(&d, &legal));
        assert_eq!((a.prior, a.dist, a.score), (b.prior, b.dist, b.score));
        // lead_value draws its hidden cards from the actor's view: the same seed
        // gives the same number whatever the true hidden cards are.
        let lv = |st: &State| net.lead_value(st, 4, &mut SmallRng::seed_from_u64(7));
        assert_eq!(lv(&s), lv(&d));
        assert!(lv(&s).is_finite());
    }

    #[test]
    fn pvnet_plays_to_out_is_the_planners() {
        let s = mid_game(2, 0);
        let me = s.current_player().unwrap();
        assert_eq!(PvNet::plays_to_out(&s), min_plays(s.hand(me), s.rules()));
        assert!((1..=13).contains(&PvNet::plays_to_out(&s)));
    }

    #[test]
    fn pvnet_rejects_files_of_the_wrong_kind_or_shape() {
        let q = crate::qnet::synthetic_graph(16, 8, 8, 16, 3);
        assert!(PvNet::new(q).unwrap_err().contains("kind"));
        let mut g = synthetic_graph(16, 8, 8, 16, 3);
        g.meta.insert("kind".into(), "big2-pvnet".into());
        g.towers.remove(1);
        assert!(PvNet::new(g).unwrap_err().contains("action"));
        // A Q-network head (2 outputs) under the policy-value kind is refused.
        let mut q = crate::qnet::synthetic_graph(16, 8, 8, 16, 3);
        q.meta.insert("kind".into(), PVNET_KIND.into());
        assert!(PvNet::new(q).unwrap_err().contains("head"));
    }
}
