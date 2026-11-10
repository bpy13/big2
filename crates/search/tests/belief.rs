//! Tests for opponent-aware determinization and leaf evaluators, on Big Two.

use big2_big2::bots::Greedy;
use big2_big2::{RuleSet, State};
use big2_games::{Estimate, Evaluator, Game, Policy, Rewind};
use big2_search::belief::log_likelihood;
use big2_search::{
    ismcts, ismcts_with, Belief, BeliefConfig, IsmctsConfig, Leaf, PolicyModel, UniformSampler,
};
use rand::{rngs::SmallRng, SeedableRng};

/// A deal played `moves` actions deep by greedy bots.
fn greedy_game(seed: u64, moves: usize) -> State {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut legal = Vec::new();
    for _ in 0..moves {
        if s.is_terminal() {
            break;
        }
        s.legal_actions(&mut legal);
        let a = Greedy::default().choose(&s, &legal, &mut rng);
        s.apply(a);
    }
    s
}

#[test]
fn rewind_replays_to_the_same_state() {
    for seed in 0..20 {
        let s = greedy_game(seed, 25);
        let (mut start, actions) = s.rewind().expect("a dealt game can be rewound");
        assert_eq!(start.history().len(), 0);
        for a in actions {
            start.apply(a);
        }
        assert_eq!(start.hands(), s.hands());
        assert_eq!(start.history(), s.history());
        assert_eq!(start.current_player(), s.current_player());
    }
    let pos = State::from_position(RuleSet::default(), [1, 2, 4, 8], 0, None).unwrap();
    assert!(pos.rewind().is_none());
}

#[test]
fn the_true_deal_explains_greedy_play_well() {
    // Opponents really are greedy bots, and the model says so: the true deal
    // explains their actions better than a typical sampled deal. (Not always better
    // than every sample: a deal in which an opponent had fewer options makes their
    // action slightly more probable.)
    let model = PolicyModel {
        policy: Greedy::default(),
        epsilon: 0.2,
    };
    for seed in 0..10 {
        let s = greedy_game(100 + seed, 20);
        let Some(observer) = s.current_player() else {
            continue;
        };
        let truth = log_likelihood(&s, observer, &model);
        let mut rng = SmallRng::seed_from_u64(seed);
        let samples: Vec<f64> = (0..200)
            .map(|_| log_likelihood(&s.determinize(observer, &mut rng), observer, &model))
            .collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(truth > mean, "seed {seed}: truth {truth} vs mean {mean}");
        // Some samples contradict a greedy action outright (a factor ε/n).
        assert!(samples.iter().any(|&ll| ll < truth - 1.0), "seed {seed}");
    }
}

#[test]
fn belief_concentrates_on_consistent_deals() {
    let model = PolicyModel {
        policy: Greedy::default(),
        epsilon: 0.2,
    };
    let s = greedy_game(7, 24);
    let observer = s.current_player().unwrap();
    let cfg = BeliefConfig {
        candidates: 300,
        min_ess: 0.25,
    };
    let b = Belief::new(&s, observer, &model, &cfg, &mut SmallRng::seed_from_u64(1)).unwrap();
    assert!(b.ess >= 0.25 * 300.0 - 1.0, "ess {}", b.ess);
    assert!(b.tempering > 0.0 && b.tempering <= 1.0);
    // Deals drawn from the belief explain the history better than uniform ones.
    let mut rng = SmallRng::seed_from_u64(2);
    let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let weighted = mean(
        (0..300)
            .map(|_| log_likelihood(b.draw(&mut rng), observer, &model))
            .collect(),
    );
    let uniform = mean(
        (0..300)
            .map(|_| log_likelihood(&s.determinize(observer, &mut rng), observer, &model))
            .collect(),
    );
    assert!(weighted > uniform, "{weighted} vs {uniform}");
    // Drawn deals keep the observer's hand and the public state.
    let d = b.draw(&mut rng);
    assert_eq!(d.hand(observer), s.hand(observer));
    assert_eq!(d.hand_sizes(), s.hand_sizes());
    assert_eq!(d.history(), s.history());
}

#[test]
fn ismcts_with_uniform_sampler_matches_ismcts() {
    let s = greedy_game(3, 9);
    let p = s.current_player().unwrap();
    let cfg = IsmctsConfig {
        iterations: 400,
        seed: 5,
        ..Default::default()
    };
    let rollout = Greedy { epsilon: 0.1 };
    let a = ismcts(&s, p, &cfg, &rollout);
    let b = ismcts_with::<_, _, _, big2_search::NoEvaluator>(
        &s,
        p,
        &cfg,
        &rollout,
        &UniformSampler,
        None,
    );
    let key = |r: &big2_search::Analysis<_>| {
        r.moves
            .iter()
            .map(|m| (m.action, m.visits))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&a), key(&b));
}

/// Says every position is won by seat `0`.
struct SeatZeroWins;

impl Evaluator<State> for SeatZeroWins {
    fn evaluate(&self, _s: &State) -> Estimate {
        let p = 0;
        let scores = (0..4).map(|q| if q == p { 30.0 } else { -10.0 }).collect();
        let wins = (0..4).map(|q| (q == p) as u8 as f64).collect();
        Estimate { scores, wins }
    }
}

#[test]
fn leaf_evaluator_values_reach_the_root() {
    // Early in the deal no simulation reaches the end, so only the evaluator counts.
    let s = (0..)
        .map(|seed| greedy_game(seed, 0))
        .find(|s| s.current_player() != Some(0))
        .unwrap();
    let p = s.current_player().unwrap();
    let cfg = IsmctsConfig {
        iterations: 300,
        ..Default::default()
    };
    let leaf = Leaf {
        evaluator: &SeatZeroWins,
        rollout_weight: 0.0,
    };
    let r = ismcts_with(
        &s,
        p,
        &cfg,
        &Greedy::default(),
        &UniformSampler,
        Some(&leaf),
    );
    for m in r.moves.iter().filter(|m| m.visits > 0) {
        assert_eq!(m.win_rate, 0.0, "{m:?}");
        assert_eq!(m.mean_score, -10.0, "{m:?}");
    }
}

// --- Task 19: the learned-belief sampler -------------------------------------

use big2_big2::belief_net::{synthetic_graph, BeliefNet};
use big2_search::{DealPrior, NetSampler, Sampler};

/// A prior that likes deals where the next seat holds the highest unseen card.
struct LikesNextSeat;

impl DealPrior<State> for LikesNextSeat {
    fn log_weights(&self, root: &State, observer: usize, deals: &[State]) -> Vec<f64> {
        let unseen = big2_big2::card::FULL_DECK & !root.hand(observer) & !root.played();
        let top = 63 - unseen.leading_zeros();
        deals
            .iter()
            .map(|d| {
                if d.hand((observer + 1) % 4) >> top & 1 == 1 {
                    20.0
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// The network as a prior: what `crates/pybind` does.
struct NetPrior(BeliefNet);

impl DealPrior<State> for NetPrior {
    fn log_weights(&self, root: &State, observer: usize, deals: &[State]) -> Vec<f64> {
        let lp = self.0.card_log_probs(root);
        deals
            .iter()
            .map(|d| BeliefNet::deal_log_weight(&lp, observer, d))
            .collect()
    }
}

#[test]
fn net_sampler_draws_deals_the_observer_cannot_tell_apart_from_the_root() {
    let s = greedy_game(7, 14);
    let me = s.current_player().unwrap();
    let net = NetPrior(BeliefNet::new(synthetic_graph(16, 4)).unwrap());
    let mut rng = SmallRng::seed_from_u64(1);
    let ns = NetSampler::new(&s, me, &net, &BeliefConfig::default(), &mut rng).unwrap();
    assert!(ns.tempering() >= 0.0 && ns.tempering() <= 1.0);
    for _ in 0..50 {
        let d = ns.sample(&s, me, &mut rng);
        assert_eq!(d.hand(me), s.hand(me));
        assert_eq!(d.hand_sizes(), s.hand_sizes());
        assert_eq!(d.played(), s.played());
        assert!(d.history_consistent());
    }
}

#[test]
fn net_sampler_follows_a_strong_prior() {
    let s = greedy_game(8, 10);
    let me = s.current_player().unwrap();
    let cfg = BeliefConfig {
        candidates: 400,
        min_ess: 0.001,
    };
    let mut rng = SmallRng::seed_from_u64(2);
    let ns = NetSampler::new(&s, me, &LikesNextSeat, &cfg, &mut rng).unwrap();
    let unseen = big2_big2::card::FULL_DECK & !s.hand(me) & !s.played();
    let top = 63 - unseen.leading_zeros();
    let hits = (0..400)
        .filter(|_| ns.sample(&s, me, &mut rng).hand((me + 1) % 4) >> top & 1 == 1)
        .count();
    assert!(
        hits > 300,
        "{hits} of 400 draws give the next seat the top card"
    );
    // With the default tempering the same prior is flattened, not followed blindly.
    let ns = NetSampler::new(&s, me, &LikesNextSeat, &BeliefConfig::default(), &mut rng).unwrap();
    assert!(ns.ess() >= 0.25 * 256.0 - 1.0);
}
