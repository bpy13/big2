//! Opponent-aware determinization: sample hidden cards in proportion to how well
//! they explain what the opponents have done.
//!
//! [`big2_games::Game::determinize`] deals the hidden cards uniformly: every
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
//! 2. For each, rewind to the start of the game ([`big2_games::Rewind`]) and
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

use big2_games::{ActionModel, Game, Player, Policy, Rewind};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};

/// Draws the full states a search simulates: what Monte Carlo search uses in
/// place of the hidden information.
pub trait Sampler<G: Game>: Sync {
    /// A full state the observer cannot tell apart from `root`.
    fn sample(&self, root: &G, observer: Player, rng: &mut SmallRng) -> G;

    /// The weight this sampler gives `deal` relative to a uniform draw, for the
    /// exact information-set endgame ([`crate::infoset_endgame`]), which lists every
    /// deal instead of drawing them. `1` (the default) is the uniform weight; a
    /// sampler that rejects deals returns `0` for them.
    fn weight(&self, _deal: &G) -> f64 {
        1.0
    }
}

/// Uniform determinization ([`Game::determinize`]), the default.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformSampler;

impl<G: Game> Sampler<G> for UniformSampler {
    fn sample(&self, root: &G, observer: Player, rng: &mut SmallRng) -> G {
        root.determinize(observer, rng)
    }
}

/// **Cheats**: every "sample" is the true state, hidden cards included. Only for
/// measuring how much perfect knowledge of the hidden cards would be worth (an
/// upper bound for any belief model); never for play against others.
#[derive(Clone, Copy, Debug, Default)]
pub struct OracleSampler;

impl<G: Game> Sampler<G> for OracleSampler {
    fn sample(&self, root: &G, _observer: Player, _rng: &mut SmallRng) -> G {
        root.clone()
    }
}

/// The hard constraints of the rules as a sampler: uniform determinizations, minus
/// every deal under which an action already taken would have been illegal
/// ([`Rewind::history_consistent`]). In Big Two that is the last-card duty: a
/// player who played a single while the next player held one card held nothing
/// higher, so a deal that gives them a higher card now is impossible.
///
/// It draws uniform deals and rejects the impossible ones (exact: what remains is
/// uniform over the possible deals). After `max_tries` rejections in a row it gives
/// up and returns the last draw, so a search never stalls; the counters say how
/// often that happened. No opponent model, no likelihoods: cheap enough for a
/// phone (one backward pass over the history per draw).
#[derive(Debug)]
pub struct RulesMask {
    /// Draws per sample before giving up (at least 1).
    pub max_tries: u32,
    draws: AtomicU64,
    rejections: AtomicU64,
    fallbacks: AtomicU64,
}

/// What a [`RulesMask`] has done so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RulesMaskStats {
    /// Deals drawn (accepted plus rejected).
    pub draws: u64,
    /// Draws rejected as impossible.
    pub rejections: u64,
    /// Samples that gave up after `max_tries` rejections and returned an impossible deal.
    pub fallbacks: u64,
}

impl RulesMaskStats {
    /// Rejected draws as a share of all draws (0 before any draw).
    pub fn rejection_share(&self) -> f64 {
        if self.draws == 0 {
            0.0
        } else {
            self.rejections as f64 / self.draws as f64
        }
    }
}

impl Default for RulesMask {
    fn default() -> Self {
        RulesMask::new(1_000)
    }
}

impl Clone for RulesMask {
    /// A fresh mask with the same settings (counters start at zero).
    fn clone(&self) -> Self {
        RulesMask::new(self.max_tries)
    }
}

impl RulesMask {
    pub fn new(max_tries: u32) -> Self {
        RulesMask {
            max_tries: max_tries.max(1),
            draws: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
        }
    }

    /// The counters so far (summed over every thread that sampled).
    pub fn stats(&self) -> RulesMaskStats {
        RulesMaskStats {
            draws: self.draws.load(Ordering::Relaxed),
            rejections: self.rejections.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }
}

impl<G: Rewind> Sampler<G> for RulesMask {
    fn sample(&self, root: &G, observer: Player, rng: &mut SmallRng) -> G {
        let mut tries = 0u32;
        loop {
            let d = root.determinize(observer, rng);
            tries += 1;
            if d.history_consistent() {
                self.draws.fetch_add(u64::from(tries), Ordering::Relaxed);
                self.rejections
                    .fetch_add(u64::from(tries - 1), Ordering::Relaxed);
                return d;
            }
            if tries >= self.max_tries {
                self.draws.fetch_add(u64::from(tries), Ordering::Relaxed);
                self.rejections
                    .fetch_add(u64::from(tries), Ordering::Relaxed);
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return d;
            }
        }
    }

    fn weight(&self, deal: &G) -> f64 {
        if deal.history_consistent() {
            1.0
        } else {
            0.0
        }
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

    use big2_big2::card::format_mask;
    use big2_big2::{Action, Kind, RuleSet, State};
    use big2_games::replay_consistent;

    /// A real game in which seat 1 played a single while seat 2 held one card (the
    /// last-card duty), seat 1 still holds cards and the deal goes on: returns the
    /// state, that single's card and the moves played. Found by seeded random play.
    fn duty_history() -> (State, u64) {
        for seed in 0..10_000u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            let mut legal = Vec::new();
            let mut constrained = None;
            while let Some(p) = s.current_player() {
                s.legal_actions(&mut legal);
                // Singles first, so duty situations come up often.
                let singles: Vec<Action> = legal
                    .iter()
                    .copied()
                    .filter(|a| matches!(a, Action::Play(c) if c.kind == Kind::Single))
                    .collect();
                let pool = if singles.is_empty() || rng.gen_bool(0.3) {
                    &legal
                } else {
                    &singles
                };
                let a = pool[rng.gen_range(0..pool.len())];
                let duty = s.hand_sizes()[(p + 1) % 4] == 1;
                s.apply(a);
                if p == 1 && duty && a.cards().count_ones() == 1 {
                    constrained = Some(a.cards());
                }
                if let Some(x) = constrained {
                    // Stop while seat 1 still has cards, the deal is running and some
                    // card higher than the single is hidden from seat 0.
                    let hidden = s.hand(1) | s.hand(2) | s.hand(3);
                    let higher = hidden & !((x << 1) - 1);
                    if s.current_player().is_some() && s.hand(1) != 0 && higher != 0 {
                        return (s, x);
                    }
                    break;
                }
            }
        }
        panic!("no duty history found");
    }

    /// The plan's test (Task 18): seat 1 played single `x` while seat 2 held one
    /// card, so `x` was seat 1's highest card. Every deal that gives seat 1 a card
    /// above `x` would make that play illegal, and the mask rejects it.
    #[test]
    fn rules_mask_rejects_deals_that_break_the_last_card_duty() {
        let (s, x) = duty_history();
        assert!(s.history_consistent(), "the true deal is always possible");
        let above_x = !((x << 1) - 1);
        let mut rng = SmallRng::seed_from_u64(3);
        let (mut rejected, mut accepted, mut with_higher) = (0, 0, 0);
        for _ in 0..2_000 {
            let d = s.determinize(0, &mut rng);
            let ok = d.history_consistent();
            assert_eq!(ok, replay_consistent(&d), "fast check vs replay");
            if d.hand(1) & above_x != 0 {
                with_higher += 1;
                assert!(
                    !ok,
                    "seat 1 holds {} above its duty single {}",
                    format_mask(d.hand(1) & above_x),
                    format_mask(x)
                );
            }
            if ok {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
        assert!(
            with_higher > 0 && rejected > 0 && accepted > 0,
            "{rejected} {accepted}"
        );

        // The sampler returns only possible deals and counts the rejections.
        let mask = RulesMask::default();
        for _ in 0..500 {
            let d = mask.sample(&s, 0, &mut rng);
            assert!(d.history_consistent());
            assert_eq!(d.hand(1) & above_x, 0);
            assert_eq!(d.hand(0), s.hand(0));
            assert_eq!(d.hand_sizes(), s.hand_sizes());
        }
        let st = mask.stats();
        assert_eq!(st.fallbacks, 0);
        assert!(st.draws >= 500 && st.rejections == st.draws - 500, "{st:?}");
        assert!(st.rejection_share() > 0.0 && st.rejection_share() < 1.0);
        assert_eq!(Sampler::<State>::weight(&mask, &s), 1.0);
    }

    /// The backward check gives the same answer as replaying the whole deal, on
    /// random determinizations of random games (with and without the duty rule).
    #[test]
    fn rules_mask_fast_check_matches_the_replay() {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut disagreements = 0;
        let mut rejected = 0;
        for game in 0..300 {
            let rules = RuleSet {
                last_card_duty: game % 5 != 0,
                ..RuleSet::default()
            };
            let mut s = State::new_deal(rules, &mut rng);
            let stop = rng.gen_range(10..60);
            for _ in 0..stop {
                if s.current_player().is_none() {
                    break;
                }
                let legal = s.legal_actions_vec();
                s.apply(legal[rng.gen_range(0..legal.len())]);
            }
            let Some(me) = s.current_player() else {
                continue;
            };
            for _ in 0..20 {
                let d = s.determinize(me, &mut rng);
                let fast = d.history_consistent();
                disagreements += usize::from(fast != replay_consistent(&d));
                rejected += usize::from(!fast);
            }
        }
        assert_eq!(disagreements, 0);
        assert!(rejected > 0, "no rejection seen: the test checks nothing");
    }

    /// Gives up after `max_tries` rejections and says so; a fixed seed gives the
    /// same deals.
    #[test]
    fn rules_mask_falls_back_and_is_reproducible() {
        let (s, _) = duty_history();
        let one = RulesMask::new(1);
        let mut rng = SmallRng::seed_from_u64(5);
        for _ in 0..300 {
            one.sample(&s, 0, &mut rng);
        }
        let st = one.stats();
        assert_eq!(st.draws, 300);
        assert_eq!(st.fallbacks, st.rejections);
        assert!(st.fallbacks > 0);
        let a: Vec<_> = {
            let mut r = SmallRng::seed_from_u64(9);
            (0..50)
                .map(|_| RulesMask::default().sample(&s, 0, &mut r).hands())
                .collect()
        };
        let b: Vec<_> = {
            let mut r = SmallRng::seed_from_u64(9);
            (0..50)
                .map(|_| RulesMask::default().sample(&s, 0, &mut r).hands())
                .collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn impossible_candidates_get_no_weight() {
        let (w, _) = temper(&[0.0, f64::NEG_INFINITY, -1.0], 0.0);
        assert_eq!(w[1], 0.0);
        assert!(w[0] > w[2]);
    }
}
