//! Opponent-aware determinization: sample hidden cards in proportion to how well
//! they explain what the opponents have done.
//!
//! [`poker_games::Game::determinize`] deals the hidden cards uniformly: every
//! deal consistent with the hand sizes and the cards played is equally likely.
//! But the opponents' choices carry information. A player who passed on a low
//! single probably holds nothing that beats it; a player who broke up a pair
//! probably did not have a straight. Bayes' rule makes this precise:
//!
//! ```text
//! P(deal | history) ∝ P(deal) · Π  P(opponent's action | their view in that deal)
//! ```
//!
//! with the product over every opponent decision so far. [`Belief`] approximates
//! it by **importance resampling**:
//!
//! 1. Draw `candidates` uniform determinizations of the current position.
//! 2. For each, rewind to the start of the game ([`poker_games::Rewind`]) and
//!    replay the history, adding up the log-probability of each opponent action
//!    under an [`ActionModel`] (for example [`PolicyModel`], a softened bot).
//!    Deals in which an observed action would have been illegal get weight 0.
//! 3. The search then draws its determinizations from the candidates, with
//!    probability proportional to their weights.
//!
//! **Tempering.** Over dozens of decisions the likelihoods multiply into weights
//! that differ by many orders of magnitude, so one or two deals would take all
//! the weight and the search would become overconfident about the hidden cards
//! (and inherit every error of the opponent model). The log-likelihoods are
//! therefore scaled by the largest `τ ∈ [0, 1]` that keeps the *effective sample
//! size* `(Σw)² / Σw²` at least `min_ess` × `candidates`. `τ = 1` is exact Bayes
//! under the model; `τ = 0` is the uniform sampler.

use poker_games::{ActionModel, Game, Player, Policy, Rewind};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Draws the full states a search simulates: what Monte Carlo search uses in
/// place of the hidden information.
pub trait Sampler<G: Game>: Sync {
    /// A full state the observer cannot tell apart from `root`.
    fn sample(&self, root: &G, observer: Player, rng: &mut SmallRng) -> G;
}

/// Uniform determinization ([`Game::determinize`]), the default.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformSampler;

impl<G: Game> Sampler<G> for UniformSampler {
    fn sample(&self, root: &G, observer: Player, rng: &mut SmallRng) -> G {
        root.determinize(observer, rng)
    }
}

/// Settings for [`Belief::new`].
#[derive(Clone, Debug)]
pub struct BeliefConfig {
    /// Uniform determinizations drawn and weighted.
    pub candidates: usize,
    /// Smallest effective sample size allowed, as a fraction of `candidates`
    /// (see the module docs on tempering).
    pub min_ess: f64,
}

impl Default for BeliefConfig {
    fn default() -> Self {
        BeliefConfig {
            candidates: 256,
            min_ess: 0.25,
        }
    }
}

/// A deterministic policy turned into a probability model: it plays like
/// `policy` with probability `1 − epsilon`, else uniformly at random.
#[derive(Clone, Copy, Debug)]
pub struct PolicyModel<P> {
    pub policy: P,
    pub epsilon: f64,
}

impl<G: Game, P: Policy<G>> ActionModel<G> for PolicyModel<P> {
    fn log_prob(&self, state: &G, legal: &[G::Action], action: G::Action) -> f64 {
        // The policy must not depend on the rng (only epsilon-exploration would).
        let mut rng = SmallRng::seed_from_u64(0);
        let chosen = self.policy.choose(state, legal, &mut rng);
        let uniform = self.epsilon / legal.len() as f64;
        let p = if chosen == action {
            1.0 - self.epsilon + uniform
        } else {
            uniform
        };
        p.ln()
    }
}

/// Weighted determinizations of one position (see the module docs).
#[derive(Clone, Debug)]
pub struct Belief<G> {
    deals: Vec<G>,
    /// Cumulative weights, ending at 1.
    cumulative: Vec<f64>,
    /// The tempering exponent used.
    pub tempering: f64,
    /// Effective sample size of the weights.
    pub ess: f64,
}

impl<G: Rewind> Belief<G> {
    /// Weights `cfg.candidates` determinizations of `root` by `model`.
    ///
    /// Returns `None` if `root` cannot be rewound (a position without a known
    /// start); callers then fall back to uniform sampling.
    pub fn new<M: ActionModel<G>>(
        root: &G,
        observer: Player,
        model: &M,
        cfg: &BeliefConfig,
        rng: &mut SmallRng,
    ) -> Option<Belief<G>> {
        root.rewind()?;
        let n = cfg.candidates.max(1);
        let mut deals = Vec::with_capacity(n);
        let mut loglik = Vec::with_capacity(n);
        for _ in 0..n {
            let d = root.determinize(observer, rng);
            loglik.push(log_likelihood(&d, observer, model));
            deals.push(d);
        }
        let (weights, tempering) = temper(&loglik, cfg.min_ess);
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None; // no candidate explains the history (should not happen)
        }
        let ess = total * total / weights.iter().map(|w| w * w).sum::<f64>();
        let mut acc = 0.0;
        let cumulative = weights
            .iter()
            .map(|w| {
                acc += w / total;
                acc
            })
            .collect();
        Some(Belief {
            deals,
            cumulative,
            tempering,
            ess,
        })
    }
}

impl<G: Game> Belief<G> {
    /// Draws one of the weighted deals.
    pub fn draw(&self, rng: &mut SmallRng) -> &G {
        let u: f64 = rng.gen();
        let i = self.cumulative.partition_point(|&c| c < u);
        &self.deals[i.min(self.deals.len() - 1)]
    }
}

impl<G: Game> Sampler<G> for Belief<G> {
    fn sample(&self, _root: &G, _observer: Player, rng: &mut SmallRng) -> G {
        self.draw(rng).clone()
    }
}

/// Log-probability of the opponents' actions in `deal`'s history, under `model`.
/// `-inf` if one of them would have been illegal with these cards.
pub fn log_likelihood<G: Rewind, M: ActionModel<G>>(deal: &G, observer: Player, model: &M) -> f64 {
    let Some((mut s, actions)) = deal.rewind() else {
        return 0.0;
    };
    let mut legal = Vec::new();
    let mut ll = 0.0;
    for a in actions {
        let p = s
            .current_player()
            .expect("history ends before the game does");
        if p != observer {
            s.legal_actions(&mut legal);
            if !legal.contains(&a) {
                return f64::NEG_INFINITY;
            }
            if legal.len() > 1 {
                ll += model.log_prob(&s, &legal, a);
            }
        }
        s.apply(a);
    }
    ll
}

/// Weights `exp(τ · loglik)` with the largest `τ ∈ [0, 1]` keeping the effective
/// sample size ≥ `min_ess` × n. Impossible candidates (`-inf`) always get 0.
fn temper(loglik: &[f64], min_ess: f64) -> (Vec<f64>, f64) {
    let max = loglik.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = |tau: f64| -> Vec<f64> {
        loglik
            .iter()
            .map(|&l| {
                if l == f64::NEG_INFINITY {
                    0.0
                } else {
                    (tau * (l - max)).exp()
                }
            })
            .collect()
    };
    let ess = |w: &[f64]| {
        let s: f64 = w.iter().sum();
        let s2: f64 = w.iter().map(|x| x * x).sum();
        if s2 > 0.0 {
            s * s / s2
        } else {
            0.0
        }
    };
    let target = min_ess * loglik.len() as f64;
    let w1 = weights(1.0);
    if ess(&w1) >= target {
        return (w1, 1.0);
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if ess(&weights(mid)) >= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (weights(lo), lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tempering_keeps_the_effective_sample_size() {
        let ll: Vec<f64> = (0..100).map(|i| -(i as f64)).collect();
        let (w, tau) = temper(&ll, 0.25);
        let s: f64 = w.iter().sum();
        let ess = s * s / w.iter().map(|x| x * x).sum::<f64>();
        assert!(tau > 0.0 && tau < 1.0);
        assert!((ess - 25.0).abs() < 0.5, "ess {ess}");
        // Flat likelihoods need no tempering.
        let (w, tau) = temper(&[-3.0; 10], 0.25);
        assert_eq!(tau, 1.0);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn impossible_candidates_get_no_weight() {
        let (w, _) = temper(&[0.0, f64::NEG_INFINITY, -1.0], 0.0);
        assert_eq!(w[1], 0.0);
        assert!(w[0] > w[2]);
    }
}
