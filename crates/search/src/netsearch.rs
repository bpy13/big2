//! Network play-out search: paired play-outs of the network's best few moves.
//!
//! A policy-value network ([`PriorValue`]) proposes and judges; the search only
//! checks its proposals against each other on the same sampled deals:
//!
//! 1. **Candidates** — the root's legal moves, ranked by the network's prior; the
//!    top `k` are kept, plus any move the caller forces in (for example the move
//!    being graded).
//! 2. **Deals** — `n` deals of the hidden cards consistent with what the searching
//!    seat knows, drawn by a [`Sampler`] (the rules mask in play).
//! 3. **Play-outs** — on every deal, every candidate is played and followed by
//!    `depth` plies in which each seat plays the network's top move (a seat with
//!    one legal move plays it without asking the network). The position reached is
//!    scored with the network's value (the cards-left head in Big Two):
//!    `value_mix · value + (1 − value_mix) · the real score of playing on to the end
//!    the same way`. With `depth = full` the play-out always runs to the end and
//!    the real score is used; a deal that ends sooner also gives its real score.
//! 4. **Paired statistics** — every candidate saw the same deals, so the result
//!    carries each candidate's mean and standard error and the paired difference
//!    to the best candidate with its own standard error (much smaller than the
//!    candidates' errors when the deals' luck is shared).
//!
//! Play-outs are deterministic given the deal (the network's top move, first one
//! on ties), so the randomness is in the deals only. Deal `j` is drawn from a
//! generator seeded by `(seed, j)`, which makes the result the same for any number
//! of threads when no time limit is set. With `threads <= 1` the search runs on
//! the calling thread and never touches rayon (required in WebAssembly).

use crate::analysis::{Analysis, MoveStats};
use crate::belief::Sampler;
use big2_games::{Game, Player, PriorValue};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Duration;
use web_time::Instant;

/// How far each play-out runs before the network's value scores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    /// This many plies after the candidate (0 = value the position right after it).
    Plies(u32),
    /// To the end of the deal; the real score.
    Full,
}

/// Settings for [`net_search`].
#[derive(Clone, Debug)]
pub struct NetSearch {
    /// Candidates: the network's top `k` moves by prior (at least 1).
    pub k: usize,
    /// Sampled deals of the hidden cards; every candidate is played on each.
    pub n: u64,
    /// Plies of network play after the candidate.
    pub depth: Depth,
    /// Weight of the network's value at the depth cutoff: `value_mix · value +
    /// (1 − value_mix) · real score from playing on to the end`. `1` (the default)
    /// is the value alone; ignored with [`Depth::Full`]. It mirrors PUCT's
    /// `leaf_network_weight` (orchestrator, 2026-11-27).
    pub value_mix: f64,
    /// Parallel workers; with `threads <= 1` the search never touches rayon.
    pub threads: usize,
    /// Base seed of the deals.
    pub seed: u64,
    /// Optional wall-clock limit, checked between deals (a deal in progress is
    /// finished, so every candidate keeps the same deals).
    pub time_limit: Option<Duration>,
}

impl Default for NetSearch {
    fn default() -> Self {
        NetSearch {
            k: 3,
            n: 64,
            depth: Depth::Full,
            value_mix: 1.0,
            threads: 1,
            seed: 0,
            time_limit: None,
        }
    }
}

/// One candidate's result, from the searching seat's point of view (points).
#[derive(Clone, Debug)]
pub struct NetCandidate<A> {
    pub action: A,
    /// The network's prior of this move at the root.
    pub prior: f64,
    /// Mean score over the deals.
    pub mean: f64,
    /// Standard error of `mean`.
    pub se: f64,
    /// Mean probability of going out first (real or the network's estimate).
    pub win_rate: f64,
    /// Paired difference to the best candidate, `mean(best − this)` (0 for the best).
    pub loss: f64,
    /// Standard error of `loss` over the deals (0 for the best).
    pub loss_se: f64,
}

/// What [`net_search`] returns.
#[derive(Clone, Debug)]
pub struct NetSearchResult<A> {
    pub player: Player,
    /// The candidates, best (highest mean) first.
    pub candidates: Vec<NetCandidate<A>>,
    /// Deals played (each with every candidate).
    pub deals: u64,
    /// Network evaluations made.
    pub net_calls: u64,
    pub elapsed: Duration,
}

impl<A: Copy> NetSearchResult<A> {
    /// The recommended move.
    pub fn best(&self) -> A {
        self.candidates[0].action
    }

    /// The result as the solvers' common [`Analysis`] (`visits` = deals,
    /// `score_ci95` = 1.96 standard errors).
    pub fn to_analysis(&self) -> Analysis<A> {
        let moves = self
            .candidates
            .iter()
            .map(|c| MoveStats {
                action: c.action,
                visits: self.deals,
                mean_score: c.mean,
                score_ci95: 1.96 * c.se,
                win_rate: c.win_rate,
                win_ci95: 1.96
                    * (c.win_rate * (1.0 - c.win_rate) / self.deals.max(1) as f64).sqrt(),
            })
            .collect();
        Analysis {
            player: self.player,
            moves,
            simulations: self.deals * self.candidates.len() as u64,
            elapsed: self.elapsed,
        }
    }
}

/// The network's top move among `legal` (first on ties), and its value estimate.
fn top_move<G: Game, N: PriorValue<G>>(
    net: &N,
    s: &G,
    legal: &[G::Action],
) -> (G::Action, big2_games::Estimate) {
    let (prior, est) = net.prior_value(s, legal);
    let mut best = 0;
    for (i, p) in prior.iter().enumerate() {
        if *p > prior[best] {
            best = i;
        }
    }
    (legal[best], est)
}

/// Plays `s` on with the network's top moves and scores it for `observer`:
/// `(score, win)`. `calls` counts network evaluations.
fn play_out<G: Game, N: PriorValue<G>>(
    mut s: G,
    observer: Player,
    depth: Depth,
    value_mix: f64,
    net: &N,
    legal: &mut Vec<G::Action>,
    calls: &mut u64,
) -> (f64, f64) {
    let mut plies = 0u32;
    // Set once the cutoff is passed with value_mix < 1: the value head's part.
    let mut head: Option<(f64, f64)> = None;
    loop {
        if let Some(o) = s.outcome() {
            let real = (
                o.scores[observer],
                (o.winner == Some(observer)) as u8 as f64,
            );
            return match head {
                Some((v, w)) => (
                    value_mix * v + (1.0 - value_mix) * real.0,
                    value_mix * w + (1.0 - value_mix) * real.1,
                ),
                None => real,
            };
        }
        s.legal_actions(legal);
        let cutoff = head.is_none() && matches!(depth, Depth::Plies(d) if plies >= d);
        if cutoff && value_mix > 0.0 {
            *calls += 1;
            let (a, est) = top_move(net, &s, legal);
            let v = (est.scores[observer], est.wins[observer]);
            if value_mix >= 1.0 {
                return v;
            }
            head = Some(v);
            s.apply(a);
        } else {
            if cutoff {
                // value_mix = 0: the real score alone.
                head = Some((0.0, 0.0));
            }
            let a = if legal.len() == 1 {
                legal[0]
            } else {
                *calls += 1;
                top_move(net, &s, legal).0
            };
            s.apply(a);
        }
        plies += 1;
    }
}

/// One deal's results: every candidate's `(score, win)`.
type Row = Vec<(f64, f64)>;

/// The generator of deal `j`.
fn deal_rng(seed: u64, j: u64) -> SmallRng {
    // SplitMix64 of (seed, j): independent streams per deal.
    let mut z = seed ^ j.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    SmallRng::seed_from_u64(z ^ (z >> 31))
}

/// Analyzes `state` for the player to move, who must be `observer`: the network's
/// top `cfg.k` moves plus every move in `extra` (each must be legal), played out on
/// `cfg.n` paired deals drawn by `sampler`.
///
/// # Panics
/// Panics if the game is over, `observer` is not the player to move, or a move in
/// `extra` is not legal.
pub fn net_search<G, N, S>(
    state: &G,
    observer: Player,
    cfg: &NetSearch,
    net: &N,
    sampler: &S,
    extra: &[G::Action],
) -> NetSearchResult<G::Action>
where
    G: Game,
    N: PriorValue<G>,
    S: Sampler<G>,
{
    assert_eq!(
        state.current_player(),
        Some(observer),
        "the observer must be the player to move"
    );
    let start = Instant::now();
    let deadline = cfg.time_limit.map(|t| start + t);
    let legal = state.legal_actions_vec();
    let (prior, _) = net.prior_value(state, &legal);
    let mut order: Vec<usize> = (0..legal.len()).collect();
    // Stable: equal priors keep the legal order.
    order.sort_by(|&a, &b| prior[b].total_cmp(&prior[a]));
    order.truncate(cfg.k.max(1));
    for a in extra {
        let i = legal
            .iter()
            .position(|l| l == a)
            .unwrap_or_else(|| panic!("forced-in move {a:?} is not legal"));
        if !order.contains(&i) {
            order.push(i);
        }
    }
    let cands: Vec<G::Action> = order.iter().map(|&i| legal[i]).collect();

    // One worker plays deals `lo..hi`: per deal, every candidate's (score, win).
    let worker = |lo: u64, hi: u64| -> (Vec<Row>, u64) {
        let mut buf = Vec::new();
        let mut calls = 0u64;
        let mut rows = Vec::with_capacity((hi - lo) as usize);
        for j in lo..hi {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            let mut rng = deal_rng(cfg.seed, j);
            let d = sampler.sample(state, observer, &mut rng);
            let row = cands
                .iter()
                .map(|&a| {
                    let mut s = d.clone();
                    s.apply(a);
                    play_out(
                        s,
                        observer,
                        cfg.depth,
                        cfg.value_mix,
                        net,
                        &mut buf,
                        &mut calls,
                    )
                })
                .collect();
            rows.push(row);
        }
        (rows, calls)
    };
    let threads = cfg.threads.max(1) as u64;
    let chunk = |t: u64| {
        let per = cfg.n / threads;
        let rem = cfg.n % threads;
        let lo = t * per + t.min(rem);
        (lo, lo + per + u64::from(t < rem))
    };
    let parts: Vec<(Vec<Row>, u64)> = if threads == 1 {
        vec![worker(0, cfg.n)]
    } else {
        (0..threads)
            .into_par_iter()
            .map(|t| {
                let (lo, hi) = chunk(t);
                worker(lo, hi)
            })
            .collect()
    };
    let mut net_calls = 1;
    let mut rows = Vec::new();
    for (r, c) in parts {
        rows.extend(r);
        net_calls += c;
    }

    let n = rows.len();
    let nf = n.max(1) as f64;
    // Mean and standard error of a column of values.
    let mean_se = |f: &dyn Fn(&Row) -> f64| -> (f64, f64) {
        let m = rows.iter().map(f).sum::<f64>() / nf;
        let var = if n > 1 {
            rows.iter().map(|r| (f(r) - m).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        (m, (var / nf).sqrt())
    };
    let stats: Vec<(f64, f64, f64)> = (0..cands.len())
        .map(|i| {
            let (m, se) = mean_se(&|r| r[i].0);
            let w = rows.iter().map(|r| r[i].1).sum::<f64>() / nf;
            (m, se, w)
        })
        .collect();
    // Best: highest mean; ties go to the earlier candidate (higher prior).
    let mut best = 0;
    for (i, s) in stats.iter().enumerate() {
        if s.0 > stats[best].0 {
            best = i;
        }
    }
    let mut out: Vec<NetCandidate<G::Action>> = cands
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let (loss, loss_se) = if i == best {
                (0.0, 0.0)
            } else {
                mean_se(&|r| r[best].0 - r[i].0)
            };
            NetCandidate {
                action: a,
                prior: prior[order[i]],
                mean: stats[i].0,
                se: stats[i].1,
                win_rate: stats[i].2,
                loss,
                loss_se,
            }
        })
        .collect();
    // Best first, then by loss (stable: ties keep the prior order).
    out.sort_by(|a, b| a.loss.total_cmp(&b.loss));
    NetSearchResult {
        player: observer,
        candidates: out,
        deals: n as u64,
        net_calls,
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{RulesMask, UniformSampler};
    use big2_big2::pvnet::{synthetic_graph, PvNet};
    use big2_big2::{RuleSet, State};
    use big2_nn::Precision;
    use rand::Rng;

    fn tiny_net() -> PvNet {
        PvNet::from_bytes(&synthetic_graph(16, 8, 8, 16, 3).to_bytes(Precision::F32)).unwrap()
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

    /// A position with several legal moves (a fresh lead after a few random moves).
    fn open_position(seed: u64) -> State {
        (0..)
            .map(|m| mid_game(seed, m))
            .find(|s| s.legal_actions_vec().len() >= 4)
            .unwrap()
    }

    /// FNV-1a over every candidate's action and the exact bits of its numbers.
    fn fingerprint(r: &NetSearchResult<big2_big2::Action>) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        eat(&r.deals.to_le_bytes());
        for c in &r.candidates {
            eat(format!("{:?}", c.action).as_bytes());
            for x in [c.prior, c.mean, c.se, c.win_rate, c.loss, c.loss_se] {
                eat(&x.to_bits().to_le_bytes());
            }
        }
        h
    }

    fn run(s: &State, cfg: &NetSearch) -> NetSearchResult<big2_big2::Action> {
        let me = s.current_player().unwrap();
        net_search(s, me, cfg, &tiny_net(), &UniformSampler, &[])
    }

    #[test]
    fn netsearch_with_k1_plays_the_networks_top_move() {
        let net = tiny_net();
        for seed in 0..6 {
            let s = open_position(seed);
            let legal = s.legal_actions_vec();
            let (prior, _) = net.prior_value(&s, &legal);
            let top = (0..legal.len())
                .max_by(|&a, &b| prior[a].total_cmp(&prior[b]).then(b.cmp(&a)))
                .unwrap();
            let cfg = NetSearch {
                k: 1,
                n: 8,
                ..Default::default()
            };
            let r = run(&s, &cfg);
            assert_eq!(r.candidates.len(), 1);
            assert_eq!(r.best(), legal[top], "seed {seed}");
        }
    }

    #[test]
    fn netsearch_keeps_the_top_k_by_prior_and_forced_in_moves() {
        let net = tiny_net();
        let s = open_position(3);
        let me = s.current_player().unwrap();
        let legal = s.legal_actions_vec();
        let (prior, _) = net.prior_value(&s, &legal);
        let mut by_prior: Vec<usize> = (0..legal.len()).collect();
        by_prior.sort_by(|&a, &b| prior[b].total_cmp(&prior[a]));
        let worst = legal[*by_prior.last().unwrap()];
        let cfg = NetSearch {
            k: 2,
            n: 8,
            ..Default::default()
        };
        let r = net_search(&s, me, &cfg, &net, &UniformSampler, &[worst]);
        assert_eq!(r.candidates.len(), 3);
        let got: Vec<_> = r.candidates.iter().map(|c| c.action).collect();
        for i in &by_prior[..2] {
            assert!(got.contains(&legal[*i]));
        }
        assert!(got.contains(&worst));
        // A forced-in move already among the top k is not duplicated.
        let r = net_search(&s, me, &cfg, &net, &UniformSampler, &[legal[by_prior[0]]]);
        assert_eq!(r.candidates.len(), 2);
    }

    #[test]
    fn netsearch_with_a_fixed_seed_is_reproducible() {
        let s = open_position(7);
        let cfg = NetSearch {
            k: 3,
            n: 32,
            seed: 11,
            ..Default::default()
        };
        assert_eq!(fingerprint(&run(&s, &cfg)), fingerprint(&run(&s, &cfg)));
        let other = NetSearch { seed: 12, ..cfg };
        assert_ne!(fingerprint(&run(&s, &cfg)), fingerprint(&run(&s, &other)));
    }

    /// `threads = 1` runs on the calling thread (no rayon) and gives exactly the
    /// recorded result; more threads give the same numbers (deals seeded by index).
    #[test]
    fn netsearch_single_thread_matches_the_recorded_result_and_threads_agree() {
        let s = open_position(7);
        let cfg = NetSearch {
            k: 3,
            n: 40,
            seed: 5,
            threads: 1,
            ..Default::default()
        };
        let one = run(&s, &cfg);
        assert_eq!(one.deals, 40);
        assert_eq!(fingerprint(&one), RECORDED);
        for threads in [2, 3, 8] {
            let many = run(
                &s,
                &NetSearch {
                    threads,
                    ..cfg.clone()
                },
            );
            assert_eq!(fingerprint(&many), fingerprint(&one), "threads {threads}");
        }
    }

    /// Recorded from the first single-thread run (no rayon path involved).
    const RECORDED: u64 = 0x0cf6_e155_ed07_6b34;

    #[test]
    fn netsearch_paired_standard_errors_shrink_as_one_over_sqrt_n() {
        let s = open_position(9);
        let cfg = |n| NetSearch {
            k: 3,
            n,
            seed: 21,
            threads: 4,
            ..Default::default()
        };
        let small = run(&s, &cfg(100));
        let large = run(&s, &cfg(1_600));
        // 16× the deals: errors about 4× smaller.
        let ratio = |f: &dyn Fn(&NetCandidate<big2_big2::Action>) -> f64| {
            let a: f64 = small.candidates.iter().map(f).sum();
            let b: f64 = large.candidates.iter().map(f).sum();
            a / b
        };
        let se = ratio(&|c| c.se);
        let paired = ratio(&|c| c.loss_se);
        assert!((3.0..5.3).contains(&se), "se ratio {se}");
        assert!((3.0..5.3).contains(&paired), "paired ratio {paired}");
        // Pairing helps: the paired errors are below the unpaired ones.
        for c in &large.candidates[1..] {
            let unpaired = (c.se.powi(2) + large.candidates[0].se.powi(2)).sqrt();
            assert!(c.loss_se < unpaired, "{} vs {unpaired}", c.loss_se);
        }
    }

    #[test]
    fn netsearch_depth_zero_is_the_value_after_each_candidate() {
        let net = tiny_net();
        let s = open_position(4);
        let me = s.current_player().unwrap();
        let cfg = NetSearch {
            k: 3,
            n: 16,
            depth: Depth::Plies(0),
            ..Default::default()
        };
        let r = run(&s, &cfg);
        for c in &r.candidates {
            let mut after = s.clone();
            after.apply(c.action);
            if after.is_terminal() {
                continue;
            }
            // The value from the next seat's view (their own hand) on the same deals.
            let want = (0..cfg.n)
                .map(|j| {
                    let mut d = UniformSampler.sample(&s, me, &mut deal_rng(cfg.seed, j));
                    d.apply(c.action);
                    let legal = d.legal_actions_vec();
                    net.prior_value(&d, &legal).1.scores[me]
                })
                .sum::<f64>()
                / cfg.n as f64;
            assert!((c.mean - want).abs() < 1e-9, "{} {want}", c.mean);
        }
    }

    #[test]
    fn netsearch_value_mix_zero_is_the_full_play_out() {
        let s = open_position(6);
        let full = run(
            &s,
            &NetSearch {
                n: 24,
                seed: 3,
                ..Default::default()
            },
        );
        let mixed = run(
            &s,
            &NetSearch {
                n: 24,
                seed: 3,
                depth: Depth::Plies(4),
                value_mix: 0.0,
                ..Default::default()
            },
        );
        assert_eq!(fingerprint(&full), fingerprint(&mixed));
        // A mix lies between its two ends, candidate by candidate.
        let head = run(
            &s,
            &NetSearch {
                n: 24,
                seed: 3,
                depth: Depth::Plies(4),
                ..Default::default()
            },
        );
        let half = run(
            &s,
            &NetSearch {
                n: 24,
                seed: 3,
                depth: Depth::Plies(4),
                value_mix: 0.5,
                ..Default::default()
            },
        );
        for c in &half.candidates {
            let mean = |r: &NetSearchResult<_>| {
                r.candidates
                    .iter()
                    .find(|x| x.action == c.action)
                    .unwrap()
                    .mean
            };
            let want = 0.5 * mean(&head) + 0.5 * mean(&full);
            assert!((c.mean - want).abs() < 1e-9, "{} {want}", c.mean);
        }
    }

    #[test]
    fn netsearch_runs_with_the_rules_mask_and_stops_at_the_time_limit() {
        let s = open_position(2);
        let me = s.current_player().unwrap();
        let mask = RulesMask::default();
        let cfg = NetSearch {
            n: 1_000_000,
            time_limit: Some(Duration::from_millis(50)),
            ..Default::default()
        };
        let start = Instant::now();
        let r = net_search(&s, me, &cfg, &tiny_net(), &mask, &[]);
        assert!(start.elapsed() < Duration::from_millis(500));
        assert!(r.deals > 0 && r.deals < 1_000_000);
        assert!(mask.stats().draws >= r.deals);
        let a = r.to_analysis();
        assert_eq!(a.best(), r.best());
        assert!(a.moves.iter().all(|m| m.visits == r.deals));
    }
}
