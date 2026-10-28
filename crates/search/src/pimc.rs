//! Flat Monte Carlo over determinizations ("perfect-information Monte Carlo").
//!
//! For each sampled deal, play every legal action once (or several times) and
//! finish the game with the rollout policy. An action's value is its average
//! result over all samples. There is no tree: every action gets the same number
//! of simulations, which makes the per-action numbers easy to interpret but
//! spends as much effort on bad actions as on good ones.
//!
//! This was the approach of early strong Bridge programs (GIB, 1999). Here it is
//! a baseline for [`crate::ismcts`].

use crate::analysis::{Analysis, MoveStats, Objective, Sums};
use crate::endgame::Endgame;
use crate::ismcts::{simulate, IsmctsConfig};
use big2_games::{Player, Policy, Transposable};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

/// Settings for [`pimc`].
#[derive(Clone, Debug)]
pub struct PimcConfig {
    /// Sampled deals.
    pub determinizations: u64,
    /// Rollouts per action per deal.
    pub rollouts_per_action: u32,
    pub threads: usize,
    pub seed: u64,
    /// Used to rank actions in the result.
    pub objective: Objective,
    /// As in [`IsmctsConfig::endgame_threshold`].
    pub endgame_threshold: usize,
    pub endgame_max_nodes: u64,
}

impl Default for PimcConfig {
    fn default() -> Self {
        PimcConfig {
            determinizations: 200,
            rollouts_per_action: 1,
            threads: 1,
            seed: 0,
            objective: Objective::Score,
            endgame_threshold: 0,
            endgame_max_nodes: 20_000,
        }
    }
}

/// Analyzes `state` for the player to move, who must be `observer`.
pub fn pimc<G, P>(state: &G, observer: Player, cfg: &PimcConfig, rollout: &P) -> Analysis<G::Action>
where
    G: Transposable,
    P: Policy<G>,
{
    assert_eq!(
        state.current_player(),
        Some(observer),
        "the observer must be the player to move"
    );
    let start = Instant::now();
    let legal = state.legal_actions_vec();
    let sim_cfg = IsmctsConfig {
        endgame_threshold: cfg.endgame_threshold,
        endgame_max_nodes: cfg.endgame_max_nodes,
        objective: cfg.objective,
        ..Default::default()
    };
    let threads = cfg.threads.max(1) as u64;
    let per_thread: Vec<Vec<Sums>> = (0..threads)
        .into_par_iter()
        .map(|t| {
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed.wrapping_add(t.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let mut endgame = Endgame::<G>::new(cfg.objective);
            let mut sums = vec![Sums::default(); legal.len()];
            let n = cfg.determinizations / threads + u64::from(t < cfg.determinizations % threads);
            for _ in 0..n {
                let d = state.determinize(observer, &mut rng);
                for (i, &a) in legal.iter().enumerate() {
                    for _ in 0..cfg.rollouts_per_action {
                        let mut s = d.clone();
                        s.apply(a);
                        let o = simulate(&mut s, &sim_cfg, rollout, &mut endgame, &mut rng);
                        sums[i].add(&o, observer);
                    }
                }
            }
            sums
        })
        .collect();

    let mut merged = vec![Sums::default(); legal.len()];
    for sums in &per_thread {
        for (m, s) in merged.iter_mut().zip(sums) {
            m.merge(s);
        }
    }
    let mut moves: Vec<MoveStats<G::Action>> = legal
        .iter()
        .zip(&merged)
        .map(|(&a, s)| MoveStats::from_sums(a, s))
        .collect();
    let key = |m: &MoveStats<G::Action>| match cfg.objective {
        Objective::Score => m.mean_score,
        Objective::Win => m.win_rate,
    };
    moves.sort_by(|a, b| key(b).total_cmp(&key(a)));
    Analysis {
        player: observer,
        simulations: merged.iter().map(|s| s.n).sum(),
        moves,
        elapsed: start.elapsed(),
    }
}
