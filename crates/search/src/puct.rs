//! PUCT with a policy-value network inside: information-set search guided by the
//! network's prior, with the network's value at the leaves instead of rollouts.
//!
//! The tree is the one [`crate::ismcts`] builds: nodes are sequences of *public*
//! actions seen from the searching player's viewpoint, and every iteration starts
//! from a fresh sample of the hidden cards. What differs is how the tree grows:
//!
//! 1. **Determinize** — sample hidden cards consistent with what the observer knows.
//! 2. **Select** — at each node, among the actions legal in this sample, pick the
//!    one with the highest `Q + c_puct · P · √N / (1 + n)` (Rosin's PUCT as used by
//!    AlphaZero): `Q` is the mean utility for the player to move, `P` the network's
//!    prior renormalized over the legal actions, `N` the visits of those actions,
//!    `n` the action's own visits. An action never tried has `Q` equal to the
//!    network's value of the node for that player ("first-play urgency"), so the
//!    prior alone orders the first tries.
//! 3. **Expand and evaluate** — the first time an action is tried, a node is added
//!    and the network evaluates the position it leads to: its prior becomes the
//!    node's `P`, its per-player scores and win probabilities the leaf value. A
//!    finished game gives its outcome; a small position is solved exactly by
//!    [`crate::endgame`] when `endgame_threshold` allows. By default there are no
//!    rollouts. With `leaf_network_weight = λ < 1` ([`puct_with_rollout`]) the leaf
//!    value is `λ · network + (1 − λ) · rollout`: a rollout policy plays on from the
//!    leaf, to the end of the deal or for `rollout_moves` moves, after which the
//!    network values the position it stopped at. The prior still comes from the
//!    network, so `λ = 0` is "the network orders the tries, rollouts value them".
//! 4. **Backpropagate** — as in ISMCTS, every node on the path records the value for
//!    the player who chose its action (multi-player: each player maximizes their
//!    own result).
//!
//! The player to move at an opponent node holds different sampled cards on each
//! visit, so the set of legal actions there varies. The node's prior table covers
//! every action seen legal so far; when a sample brings a new one, the network is
//! asked again and the new actions are added. The observer's own nodes never need
//! that: their legal actions are fixed.
//!
//! **Threads.** With `threads <= 1` everything runs on the calling thread and rayon
//! is never touched (the phone has no thread pool). With more, each thread builds
//! its own tree and the root statistics are summed.
//!
//! **Anytime answer.** `anytime_first_ms` records the move the search would have
//! played had it stopped after that many milliseconds, so the app can measure how
//! often the full search changes its mind (plan Task 16). With several threads it is
//! tree 0's answer.

use crate::analysis::{Analysis, MoveStats, Objective, Sums};
use crate::belief::{Sampler, UniformSampler};
use crate::endgame::Endgame;
use crate::infoset_endgame::{infoset_endgame, InfosetConfig};
use big2_games::{Estimate, Game, Player, Policy, PriorValue, Transposable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Duration;
use web_time::Instant;

/// Settings for [`puct`].
#[derive(Clone, Debug)]
pub struct PuctConfig {
    /// Total simulations (one network evaluation each, plus re-evaluations at
    /// opponent nodes when a sample brings a new legal action) across all threads.
    pub simulations: u64,
    /// Optional wall-clock limit; whichever of `simulations` and this comes first.
    pub time_limit: Option<Duration>,
    /// Exploration constant of `Q + c_puct · P · √N / (1 + n)`. Utilities are in
    /// `[-1, 1]` (scores divided by the game's largest possible score).
    pub c_puct: f64,
    /// Independent trees searched in parallel; with `threads <= 1` the search
    /// runs on the calling thread and never touches rayon.
    pub threads: usize,
    /// Base random seed. With a fixed seed, thread count and no time limit, the
    /// search is reproducible.
    pub seed: u64,
    /// What the searching player maximizes.
    pub objective: Objective,
    /// Solve exactly once [`Game::endgame_size`] is at most this (0 = never).
    pub endgame_threshold: usize,
    /// Node budget per exact solve; when exceeded, the network's value is used.
    pub endgame_max_nodes: u64,
    /// Record the best move as of this many milliseconds into the search
    /// ([`PuctAnalysis::first_answer`]); `None` records nothing.
    pub anytime_first_ms: Option<u64>,
    /// Weight `λ` of the network in a new leaf's value, `λ · network + (1 − λ) ·
    /// rollout` (build plan Task 15's fallback). `1` (the default) plays no rollout
    /// and leaves the search exactly as without this setting; below 1 the search
    /// needs a rollout policy ([`puct_with_rollout`]).
    pub leaf_network_weight: f64,
    /// Moves the leaf rollout plays before it stops (`0` = to the end of the deal).
    /// A rollout that stops early is valued by the network at the position it
    /// reached (one more evaluation).
    pub rollout_moves: u32,
    /// Answer exactly at the root once few cards are hidden from the observer
    /// ([`crate::infoset_endgame`]: every deal solved, no network); when its node
    /// budget runs out, the normal search runs. `None` (the default) never does.
    pub infoset_endgame: Option<InfosetConfig>,
}

impl Default for PuctConfig {
    fn default() -> Self {
        PuctConfig {
            simulations: 2_000,
            time_limit: None,
            c_puct: 1.5,
            threads: 1,
            seed: 0,
            objective: Objective::Score,
            endgame_threshold: 0,
            endgame_max_nodes: 20_000,
            anytime_first_ms: None,
            leaf_network_weight: 1.0,
            rollout_moves: 0,
            infoset_endgame: None,
        }
    }
}

/// Placeholder rollout type for searches that play no rollouts ([`puct`],
/// [`puct_with`]); never called.
pub struct NoRollout;

impl<G: Game> Policy<G> for NoRollout {
    fn choose<R: Rng + ?Sized>(&self, _state: &G, _legal: &[G::Action], _rng: &mut R) -> G::Action {
        unreachable!("NoRollout is never called: leaf_network_weight < 1 needs puct_with_rollout")
    }
}

/// What [`puct`] returns: the usual [`Analysis`] plus the anytime answer.
#[derive(Clone, Debug)]
pub struct PuctAnalysis<A> {
    pub analysis: Analysis<A>,
    /// The most visited root move when `anytime_first_ms` had passed, if that was
    /// before the search ended (`None` when the setting is off or the whole search
    /// finished sooner).
    pub first_answer: Option<A>,
    /// Simulations done when `first_answer` was taken.
    pub first_answer_simulations: u64,
}

struct Node<A> {
    /// Action leading here from the parent (`None` at the root).
    action: Option<A>,
    /// Player who chose `action`.
    player: Player,
    children: Vec<usize>,
    visits: u64,
    /// Sum of `player`'s utility over visits.
    utility: f64,
    /// Score/win sums for reporting (only read for root children).
    sums: Sums,
    /// The network's prior for every action seen legal at this node, in the order
    /// first seen (not renormalized: selection does that over the legal subset).
    priors: Vec<(A, f64)>,
    /// The network's utility of this position for the player to move here: the
    /// `Q` of an action not yet tried.
    value_to_move: f64,
}

impl<A> Node<A> {
    fn new(action: Option<A>, player: Player) -> Self {
        Node {
            action,
            player,
            children: Vec::new(),
            visits: 0,
            utility: 0.0,
            sums: Sums::default(),
            priors: Vec::new(),
            value_to_move: 0.0,
        }
    }
}

/// Root-child statistics from one tree, how many simulations it ran, and its
/// anytime answer (move and simulations at the time).
type TreeResult<A> = (Vec<(A, Sums)>, u64, Option<(A, u64)>);

/// Analyzes `state` for the player to move, who must be `observer`, with uniform
/// determinization.
///
/// # Panics
/// Panics if the game is over or `observer` is not the player to move.
pub fn puct<G, N>(state: &G, observer: Player, cfg: &PuctConfig, net: &N) -> PuctAnalysis<G::Action>
where
    G: Transposable,
    N: PriorValue<G>,
{
    puct_with(state, observer, cfg, net, &UniformSampler)
}

/// [`puct`] with a custom determinization `sampler`.
///
/// # Panics
/// Also panics if `cfg.leaf_network_weight < 1` (that needs [`puct_with_rollout`]).
pub fn puct_with<G, N, S>(
    state: &G,
    observer: Player,
    cfg: &PuctConfig,
    net: &N,
    sampler: &S,
) -> PuctAnalysis<G::Action>
where
    G: Transposable,
    N: PriorValue<G>,
    S: Sampler<G>,
{
    assert!(
        cfg.leaf_network_weight >= 1.0,
        "leaf_network_weight < 1 mixes in rollouts: use puct_with_rollout"
    );
    run(state, observer, cfg, net, sampler, None::<&NoRollout>)
}

/// [`puct_with`] plus a `rollout` policy for the leaf value
/// `λ · network + (1 − λ) · rollout` (`λ = cfg.leaf_network_weight`). With `λ = 1`
/// the rollout is never played and the result is identical to [`puct_with`]'s.
pub fn puct_with_rollout<G, N, S, P>(
    state: &G,
    observer: Player,
    cfg: &PuctConfig,
    net: &N,
    sampler: &S,
    rollout: &P,
) -> PuctAnalysis<G::Action>
where
    G: Transposable,
    N: PriorValue<G>,
    S: Sampler<G>,
    P: Policy<G>,
{
    run(state, observer, cfg, net, sampler, Some(rollout))
}

fn run<G, N, S, P>(
    state: &G,
    observer: Player,
    cfg: &PuctConfig,
    net: &N,
    sampler: &S,
    rollout: Option<&P>,
) -> PuctAnalysis<G::Action>
where
    G: Transposable,
    N: PriorValue<G>,
    S: Sampler<G>,
    P: Policy<G>,
{
    assert!(
        (0.0..=1.0).contains(&cfg.leaf_network_weight),
        "leaf_network_weight must be in [0, 1], got {}",
        cfg.leaf_network_weight
    );
    assert_eq!(
        state.current_player(),
        Some(observer),
        "the observer must be the player to move"
    );
    let start = Instant::now();
    let deadline = cfg.time_limit.map(|t| start + t);
    let anytime = cfg
        .anytime_first_ms
        .map(|ms| start + Duration::from_millis(ms));
    let threads = cfg.threads.max(1);
    if let Some(ic) = &cfg.infoset_endgame {
        if let Some(r) = infoset_endgame(state, observer, ic, cfg.objective, sampler) {
            if r.exact {
                return PuctAnalysis {
                    analysis: r.analysis,
                    first_answer: None,
                    first_answer_simulations: 0,
                };
            }
        }
    }

    let one_tree = |t: usize| -> TreeResult<G::Action> {
        let sims = cfg.simulations / threads as u64
            + u64::from((t as u64) < cfg.simulations % threads as u64);
        let mut rng = SmallRng::seed_from_u64(
            cfg.seed
                .wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let ctx = Ctx {
            cfg,
            net,
            sampler,
            rollout,
        };
        search_tree(state, observer, &ctx, sims, deadline, anytime, &mut rng)
    };
    // One thread: run on the calling thread so rayon's global pool is never built
    // (it must not be in WebAssembly).
    let per_thread: Vec<TreeResult<G::Action>> = if threads == 1 {
        vec![one_tree(0)]
    } else {
        (0..threads).into_par_iter().map(one_tree).collect()
    };

    // Merge root statistics across threads; keep every legal action, even unvisited ones.
    let legal = state.legal_actions_vec();
    let mut merged: Vec<(G::Action, Sums)> = legal.iter().map(|&a| (a, Sums::default())).collect();
    let mut simulations = 0;
    for (stats, sims, _) in &per_thread {
        simulations += sims;
        for (a, s) in stats {
            if let Some(slot) = merged.iter_mut().find(|(b, _)| b == a) {
                slot.1.merge(s);
            }
        }
    }
    let mean_utility = |s: &Sums| match cfg.objective {
        Objective::Score => s.score / s.n.max(1) as f64,
        Objective::Win => s.wins / s.n.max(1) as f64,
    };
    // Most-visited first ("robust child"); ties by the objective's mean.
    merged.sort_by(|(_, x), (_, y)| {
        y.n.cmp(&x.n)
            .then(mean_utility(y).total_cmp(&mean_utility(x)))
    });
    let (first_answer, first_answer_simulations) = match per_thread[0].2 {
        Some((a, n)) => (Some(a), n),
        None => (None, 0),
    };

    PuctAnalysis {
        analysis: Analysis {
            player: observer,
            moves: merged
                .iter()
                .map(|(a, s)| MoveStats::from_sums(*a, s))
                .collect(),
            simulations,
            elapsed: start.elapsed(),
        },
        first_answer,
        first_answer_simulations,
    }
}

/// What one search thread works with.
struct Ctx<'a, N, S, P> {
    cfg: &'a PuctConfig,
    net: &'a N,
    sampler: &'a S,
    /// The leaf rollout policy; only played when `leaf_network_weight < 1`.
    rollout: Option<&'a P>,
}

/// Runs one tree for `simulations` (or until `deadline`) and returns root-child sums.
fn search_tree<G, N, S, P>(
    root: &G,
    observer: Player,
    ctx: &Ctx<'_, N, S, P>,
    simulations: u64,
    deadline: Option<Instant>,
    anytime: Option<Instant>,
    rng: &mut SmallRng,
) -> TreeResult<G::Action>
where
    G: Transposable,
    N: PriorValue<G>,
    S: Sampler<G>,
    P: Policy<G>,
{
    let cfg = ctx.cfg;
    let mut tree: Vec<Node<G::Action>> = vec![Node::new(None, observer)];
    let mut endgame = Endgame::<G>::new(cfg.objective);
    let mut legal = Vec::new();
    let mut path: Vec<usize> = Vec::new();
    let mut done = 0u64;
    let mut first_answer = None;
    // Checking the clock every simulation would cost more than a simulation on a
    // phone's coarse timer, so it is read every 16.
    const CLOCK_EVERY: u64 = 16;

    while done < simulations {
        if done.is_multiple_of(CLOCK_EVERY) {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            if first_answer.is_none() && anytime.is_some_and(|t| Instant::now() >= t) {
                first_answer = best_root_child(&tree, cfg.objective).map(|a| (a, done));
            }
        }
        let mut s = ctx.sampler.sample(root, observer, rng);
        let mut node = 0usize;
        path.clear();

        // Selection and expansion; `value` is the leaf's estimate.
        let value: Estimate = loop {
            let Some(to_move) = s.current_player() else {
                break Estimate::from(&s.outcome().expect("a finished game has an outcome"));
            };
            s.legal_actions(&mut legal);
            ensure_priors(&mut tree[node], &s, &legal, ctx, root, to_move);

            let best = select(&tree, node, &legal, cfg.c_puct);
            let existing = tree[node]
                .children
                .iter()
                .copied()
                .find(|&c| tree[c].action == Some(best));
            s.apply(best);
            match existing {
                Some(c) => {
                    path.push(c);
                    node = c;
                }
                None => {
                    let child = tree.len();
                    tree.push(Node::new(Some(best), to_move));
                    tree[node].children.push(child);
                    path.push(child);
                    break evaluate_leaf(&mut tree[child], &mut s, ctx, root, &mut endgame, rng);
                }
            }
        };

        // Backpropagation.
        for &n in &path {
            let node = &mut tree[n];
            node.visits += 1;
            node.utility += cfg.objective.utility_estimate(root, &value, node.player);
            node.sums
                .add_value(value.scores[node.player], value.wins[node.player]);
        }
        done += 1;
    }

    let stats = tree[0]
        .children
        .iter()
        .map(|&c| (tree[c].action.unwrap(), tree[c].sums))
        .collect();
    (stats, done, first_answer)
}

/// Asks the network for the prior of any legal action the node has not seen yet
/// (always on the first visit; later only at opponent nodes whose sampled hand
/// allows a new action). The node's value is set on the first call.
fn ensure_priors<G, N, S, P>(
    node: &mut Node<G::Action>,
    s: &G,
    legal: &[G::Action],
    ctx: &Ctx<'_, N, S, P>,
    root: &G,
    to_move: Player,
) where
    G: Game,
    N: PriorValue<G>,
{
    let covered = legal
        .iter()
        .all(|a| node.priors.iter().any(|(b, _)| b == a));
    if covered {
        return;
    }
    let first = node.priors.is_empty();
    let (prior, est) = ctx.net.prior_value(s, legal);
    debug_assert_eq!(prior.len(), legal.len(), "one prior per legal action");
    for (&a, &p) in legal.iter().zip(&prior) {
        if !node.priors.iter().any(|(b, _)| *b == a) {
            node.priors.push((a, p));
        }
    }
    if first {
        node.value_to_move = ctx.cfg.objective.utility_estimate(root, &est, to_move);
    }
}

/// The PUCT choice among `legal` at `node`: `Q + c · P · √N / (1 + n)`, with `P`
/// renormalized over the legal actions and `N` their visits.
fn select<A: Copy + Eq>(tree: &[Node<A>], node: usize, legal: &[A], c: f64) -> A {
    let n = &tree[node];
    let child_of = |a: A| {
        n.children
            .iter()
            .copied()
            .find(|&c| tree[c].action == Some(a))
    };
    let prior_of = |a: A| {
        n.priors
            .iter()
            .find(|(b, _)| *b == a)
            .map_or(0.0, |(_, p)| *p)
    };
    let mass: f64 = legal.iter().map(|&a| prior_of(a)).sum();
    let mass = if mass > 0.0 { mass } else { 1.0 };
    let total: u64 = legal
        .iter()
        .filter_map(|&a| child_of(a))
        .map(|c| tree[c].visits)
        .sum();
    let sqrt_total = ((total + 1) as f64).sqrt();
    let mut best = legal[0];
    let mut best_score = f64::NEG_INFINITY;
    for &a in legal {
        let (q, visits) = match child_of(a) {
            Some(c) if tree[c].visits > 0 => {
                (tree[c].utility / tree[c].visits as f64, tree[c].visits)
            }
            _ => (n.value_to_move, 0),
        };
        let score = q + c * (prior_of(a) / mass) * sqrt_total / (1.0 + visits as f64);
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

/// The value of a new node's position: the outcome if the game is over, the exact
/// result if the position is small enough and solvable within budget, else the
/// network's estimate (whose prior the node keeps for its own expansion), mixed
/// with a rollout from `s` when `leaf_network_weight < 1` (`s` is then played on;
/// the caller discards it).
fn evaluate_leaf<G, N, S, P>(
    node: &mut Node<G::Action>,
    s: &mut G,
    ctx: &Ctx<'_, N, S, P>,
    root: &G,
    endgame: &mut Endgame<G>,
    rng: &mut SmallRng,
) -> Estimate
where
    G: Transposable,
    N: PriorValue<G>,
    P: Policy<G>,
{
    let Some(to_move) = s.current_player() else {
        return Estimate::from(&s.outcome().expect("a finished game has an outcome"));
    };
    let cfg = ctx.cfg;
    if s.endgame_size() <= cfg.endgame_threshold {
        if let Some(o) = endgame.solve(s, cfg.endgame_max_nodes) {
            return Estimate::from(&o);
        }
    }
    let legal = s.legal_actions_vec();
    let (prior, est) = ctx.net.prior_value(s, &legal);
    debug_assert_eq!(prior.len(), legal.len(), "one prior per legal action");
    node.priors = legal.iter().copied().zip(prior).collect();
    node.value_to_move = cfg.objective.utility_estimate(root, &est, to_move);
    let lambda = cfg.leaf_network_weight;
    if lambda >= 1.0 {
        // Today's behaviour: no rollout, and the random generator is not touched.
        return est;
    }
    let rollout = ctx
        .rollout
        .expect("leaf_network_weight < 1 needs a rollout policy");
    let played = short_rollout(s, cfg.rollout_moves, rollout, ctx.net, rng);
    let mix = |net: &[f64], played: &[f64]| -> Vec<f64> {
        net.iter()
            .zip(played)
            .map(|(n, p)| lambda * n + (1.0 - lambda) * p)
            .collect()
    };
    Estimate {
        scores: mix(&est.scores, &played.scores),
        wins: mix(&est.wins, &played.wins),
    }
}

/// Plays `s` on with `rollout` for every seat: to the end of the deal when
/// `max_moves` is 0, else for at most `max_moves` moves, after which the network
/// values the position reached.
fn short_rollout<G, N, P>(
    s: &mut G,
    max_moves: u32,
    rollout: &P,
    net: &N,
    rng: &mut SmallRng,
) -> Estimate
where
    G: Game,
    N: PriorValue<G>,
    P: Policy<G>,
{
    let mut legal = Vec::new();
    let mut moves = 0u32;
    while s.current_player().is_some() {
        s.legal_actions(&mut legal);
        if max_moves > 0 && moves == max_moves {
            return net.prior_value(s, &legal).1;
        }
        let a = rollout.choose(s, &legal, rng);
        s.apply(a);
        moves += 1;
    }
    Estimate::from(&s.outcome().expect("a finished game has an outcome"))
}

/// The most visited root child, ties by the objective's mean; `None` before any visit.
fn best_root_child<A: Copy>(tree: &[Node<A>], objective: Objective) -> Option<A> {
    let mean = |n: &Node<A>| match objective {
        Objective::Score => n.sums.score / n.sums.n.max(1) as f64,
        Objective::Win => n.sums.wins / n.sums.n.max(1) as f64,
    };
    tree[0]
        .children
        .iter()
        .map(|&c| &tree[c])
        .filter(|n| n.visits > 0)
        .max_by(|x, y| x.visits.cmp(&y.visits).then(mean(x).total_cmp(&mean(y))))
        .and_then(|n| n.action)
}

/// A bot that plays [`puct`]'s most visited move. Each decision seeds its search
/// from the policy's random generator, so a game is reproducible from that
/// generator's seed while no two decisions share a seed.
#[derive(Clone, Debug)]
pub struct PuctPolicy<'a, N> {
    pub net: &'a N,
    pub cfg: PuctConfig,
}

impl<G, N> Policy<G> for PuctPolicy<'_, N>
where
    G: Transposable,
    N: PriorValue<G>,
{
    fn choose<R: Rng + ?Sized>(&self, state: &G, _legal: &[G::Action], rng: &mut R) -> G::Action {
        let me = state.current_player().expect("game not over");
        let cfg = PuctConfig {
            seed: rng.gen(),
            ..self.cfg.clone()
        };
        puct(state, me, &cfg, self.net).analysis.best()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ismcts::test_support::fingerprint;
    use big2_big2::bots::Greedy;
    use big2_big2::card::parse_mask;
    use big2_big2::pvnet::{synthetic_graph, PvNet};
    use big2_big2::{Action, RuleSet, State};
    use big2_nn::Precision;

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

    fn cfg(simulations: u64) -> PuctConfig {
        PuctConfig {
            simulations,
            ..Default::default()
        }
    }

    /// (a) One thread and a seed: two runs give byte-identical analyses.
    #[test]
    fn puct_single_thread_is_reproducible() {
        let net = tiny_net();
        let s = mid_game(7, 5);
        let me = s.current_player().unwrap();
        let c = PuctConfig {
            seed: 7,
            ..cfg(300)
        };
        let a = puct(&s, me, &c, &net);
        let b = puct(&s, me, &c, &net);
        assert_eq!(a.analysis.simulations, 300);
        assert_eq!(fingerprint(&a.analysis), fingerprint(&b.analysis));
        assert_eq!(a.analysis.moves.len(), s.legal_actions_vec().len());
        let visits: u64 = a.analysis.moves.iter().map(|m| m.visits).sum();
        assert_eq!(visits, 300);
        // A different seed samples different deals and (almost surely) differs.
        let d = puct(&s, me, &PuctConfig { seed: 8, ..c }, &net);
        assert_ne!(fingerprint(&a.analysis), fingerprint(&d.analysis));
    }

    /// (b) The search never depends on the true hidden cards: two states that
    /// differ only in cards the observer cannot see give the same analysis.
    #[test]
    fn puct_never_depends_on_hidden_cards() {
        let net = tiny_net();
        let mut rng = SmallRng::seed_from_u64(4);
        let s = mid_game(4, 17);
        let me = s.current_player().unwrap();
        let d = s.determinize(me, &mut rng);
        assert_ne!(d.hands(), s.hands());
        let c = PuctConfig {
            seed: 11,
            endgame_threshold: 20,
            ..cfg(200)
        };
        let (a, b) = (puct(&s, me, &c, &net), puct(&d, me, &c, &net));
        assert_eq!(fingerprint(&a.analysis), fingerprint(&b.analysis));
    }

    /// (c) One move ends the game at once: PUCT-100 plays it.
    #[test]
    fn puct_100_takes_an_immediate_win() {
        let net = tiny_net();
        let rules = RuleSet::default();
        // Player 0 leads holding a pair of 2s: the pair goes out; a single 2 does not.
        let hands = [
            parse_mask("2h 2s").unwrap(),
            parse_mask("3c 4c 5c 6c 7c").unwrap(),
            parse_mask("8d 9d Td Jd Qd Kd").unwrap(),
            parse_mask("3h 4h 5h 6h 7h 8h 9h").unwrap(),
        ];
        let s = State::from_position(rules, hands, 0, None).unwrap();
        let legal = s.legal_actions_vec();
        assert!(legal.len() >= 3, "{legal:?}");
        let r = puct(&s, 0, &cfg(100), &net);
        let pair = Action::Play(big2_big2::classify(parse_mask("2h 2s").unwrap(), &rules).unwrap());
        assert_eq!(r.analysis.best(), pair, "{:?}", r.analysis.moves);
        assert_eq!(r.analysis.moves[0].win_rate, 1.0);
    }

    /// (d) A 50 ms time limit returns well before 120 ms however many simulations
    /// were asked for, and records the anytime answer.
    #[test]
    fn puct_time_limit_stops_the_search() {
        let net = tiny_net();
        let s = mid_game(2, 0);
        let me = s.current_player().unwrap();
        let c = PuctConfig {
            simulations: 1_000_000,
            time_limit: Some(Duration::from_millis(50)),
            anytime_first_ms: Some(10),
            ..Default::default()
        };
        let t = Instant::now();
        let r = puct(&s, me, &c, &net);
        let took = t.elapsed();
        assert!(took < Duration::from_millis(120), "{took:?}");
        assert!(r.analysis.simulations > 0);
        assert!(r.analysis.simulations < 1_000_000);
        assert!(r.first_answer.is_some());
        assert!(r.first_answer_simulations > 0);
        assert!(r.first_answer_simulations <= r.analysis.simulations);
    }

    /// Two threads give the sum of two trees' simulations; no anytime setting, no answer.
    #[test]
    fn puct_threads_merge_root_statistics() {
        let net = tiny_net();
        let s = mid_game(9, 3);
        let me = s.current_player().unwrap();
        let r = puct(
            &s,
            me,
            &PuctConfig {
                threads: 2,
                ..cfg(101)
            },
            &net,
        );
        assert_eq!(r.analysis.simulations, 101);
        assert_eq!(r.analysis.moves.iter().map(|m| m.visits).sum::<u64>(), 101);
        assert!(r.first_answer.is_none());
    }

    /// Positions and settings of the golden fingerprints: (deal seed, random moves
    /// played, endgame threshold, threads) and the fingerprint of PUCT-400's analysis.
    /// The fingerprints were printed by the code before `leaf_network_weight`
    /// existed (2026-10-30), so equality proves λ = 1 changes nothing.
    const GOLDEN: [(u64, usize, usize, usize, u64); 4] = [
        (7, 5, 0, 1, 0x5d0c_27eb_3adc_0da6),
        (4, 17, 20, 1, 0x4ba0_4ddf_8f80_187c),
        (9, 3, 0, 2, 0xb3d6_27a9_f6a4_1b18),
        (21, 30, 12, 1, 0xa979_79fc_5958_fff5),
    ];

    fn golden_cfg(seed: u64, endgame_threshold: usize, threads: usize) -> PuctConfig {
        PuctConfig {
            seed: seed + 100,
            endgame_threshold,
            threads,
            ..cfg(400)
        }
    }

    /// The default `leaf_network_weight = 1` is bit-identical to the search before
    /// the setting existed, through `puct` and through `puct_with_rollout` (whose
    /// rollout is then never played).
    #[test]
    fn puct_lambda_one_is_bit_identical_to_before() {
        let net = tiny_net();
        let greedy = Greedy { epsilon: 0.1 };
        assert_eq!(PuctConfig::default().leaf_network_weight, 1.0);
        for (seed, moves, eg, threads, want) in GOLDEN {
            let s = mid_game(seed, moves);
            let me = s.current_player().unwrap();
            let c = golden_cfg(seed, eg, threads);
            let plain = puct(&s, me, &c, &net);
            assert_eq!(fingerprint(&plain.analysis), want, "puct, deal {seed}");
            let with = puct_with_rollout(&s, me, &c, &net, &UniformSampler, &greedy);
            assert_eq!(
                fingerprint(&with.analysis),
                want,
                "puct_with_rollout, deal {seed}"
            );
        }
    }

    /// λ = 0.5 with a rollout: reproducible with a seed, every simulation counted,
    /// and different from the network-only search.
    #[test]
    fn puct_rollout_leaf_is_reproducible_and_changes_the_search() {
        let net = tiny_net();
        let greedy = Greedy { epsilon: 0.1 };
        let (seed, moves, eg, threads, want) = GOLDEN[0];
        let s = mid_game(seed, moves);
        let me = s.current_player().unwrap();
        let c = PuctConfig {
            leaf_network_weight: 0.5,
            ..golden_cfg(seed, eg, threads)
        };
        let a = puct_with_rollout(&s, me, &c, &net, &UniformSampler, &greedy);
        let b = puct_with_rollout(&s, me, &c, &net, &UniformSampler, &greedy);
        assert_eq!(fingerprint(&a.analysis), fingerprint(&b.analysis));
        assert_ne!(fingerprint(&a.analysis), want);
        assert_eq!(a.analysis.moves.iter().map(|m| m.visits).sum::<u64>(), 400);
        // A rollout cut after 3 moves (valued by the network there) differs from a full one.
        let short = PuctConfig {
            leaf_network_weight: 0.0,
            rollout_moves: 3,
            ..c.clone()
        };
        let full = PuctConfig {
            rollout_moves: 0,
            ..short.clone()
        };
        let x = puct_with_rollout(&s, me, &short, &net, &UniformSampler, &greedy);
        let y = puct_with_rollout(&s, me, &full, &net, &UniformSampler, &greedy);
        assert_ne!(fingerprint(&x.analysis), fingerprint(&y.analysis));
        assert_eq!(x.analysis.simulations, 400);
    }

    /// With pure rollout values (λ = 0, full rollouts) every leaf value is a played-out
    /// deal: each root move's wins are a whole number (a network value would give
    /// fractions), and the immediate win is valued as one.
    #[test]
    fn puct_pure_rollout_values_are_played_out_deals() {
        let net = tiny_net();
        let rules = RuleSet::default();
        let hands = [
            parse_mask("2h 2s").unwrap(),
            parse_mask("3c 4c 5c 6c 7c").unwrap(),
            parse_mask("8d 9d Td Jd Qd Kd").unwrap(),
            parse_mask("3h 4h 5h 6h 7h 8h 9h").unwrap(),
        ];
        let s = State::from_position(rules, hands, 0, None).unwrap();
        let c = PuctConfig {
            leaf_network_weight: 0.0,
            ..cfg(200)
        };
        let r = puct_with_rollout(&s, 0, &c, &net, &UniformSampler, &Greedy { epsilon: 0.0 });
        let pair = Action::Play(big2_big2::classify(parse_mask("2h 2s").unwrap(), &rules).unwrap());
        // (Leading the single 2♠ also wins here once the others pass, so the most
        // visited move may be either; the pair's own value must be a certain win.)
        let m = r.analysis.moves.iter().find(|m| m.action == pair).unwrap();
        assert_eq!(m.win_rate, 1.0, "{:?}", r.analysis.moves);
        for m in &r.analysis.moves {
            let wins = m.win_rate * m.visits as f64;
            assert!((wins - wins.round()).abs() < 1e-9, "{m:?}");
        }
        // The same search with network values gives fractional wins somewhere.
        let net_only = puct(&s, 0, &cfg(200), &net);
        assert!(net_only.analysis.moves.iter().any(|m| {
            let wins = m.win_rate * m.visits as f64;
            (wins - wins.round()).abs() > 1e-6
        }));
    }

    /// `puct_with` has no rollout policy, so λ < 1 is refused.
    #[test]
    #[should_panic(expected = "puct_with_rollout")]
    fn puct_without_rollout_refuses_lambda_below_one() {
        let net = tiny_net();
        let s = mid_game(7, 5);
        let me = s.current_player().unwrap();
        let c = PuctConfig {
            leaf_network_weight: 0.5,
            ..cfg(10)
        };
        puct(&s, me, &c, &net);
    }

    /// The policy plays a legal move and finishes games against the greedy bot.
    #[test]
    fn puct_policy_plays_whole_games() {
        let net = tiny_net();
        let policy = PuctPolicy {
            net: &net,
            cfg: cfg(30),
        };
        let greedy = Greedy { epsilon: 0.0 };
        let mut rng = SmallRng::seed_from_u64(1);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut legal = Vec::new();
        while let Some(p) = s.current_player() {
            s.legal_actions(&mut legal);
            let a = if p == 0 {
                policy.choose(&s, &legal, &mut rng)
            } else {
                greedy.choose(&s, &legal, &mut rng)
            };
            assert!(legal.contains(&a));
            s.apply(a);
        }
        assert!(s.outcome().is_some());
    }

    /// Quick strength check against a real network, for the hand-back (not a bar):
    /// `BIG2_PVNET=<path.pknn> cargo test -p big2-search puct_quick -- --ignored --nocapture`.
    /// Seat 0 plays PUCT; seats 1–3 the greedy bot; mean score per deal and time per move.
    #[test]
    #[ignore]
    fn puct_quick_strength_vs_greedy() {
        let Ok(path) = std::env::var("BIG2_PVNET") else {
            eprintln!("BIG2_PVNET not set; skipping");
            return;
        };
        let net = PvNet::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        let deals: u64 = std::env::var("BIG2_DEALS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        let sims: u64 = std::env::var("BIG2_SIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        let policy = PuctPolicy {
            net: &net,
            cfg: cfg(sims),
        };
        let greedy = Greedy { epsilon: 0.0 };
        let mut rng = SmallRng::seed_from_u64(2026);
        let (mut total, mut sq, mut wins, mut moves, mut secs) = (0.0, 0.0, 0u64, 0u64, 0.0);
        let mut legal = Vec::new();
        for _ in 0..deals {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                s.legal_actions(&mut legal);
                let a = if p == 0 {
                    let t = Instant::now();
                    let a = policy.choose(&s, &legal, &mut rng);
                    secs += t.elapsed().as_secs_f64();
                    moves += 1;
                    a
                } else {
                    greedy.choose(&s, &legal, &mut rng)
                };
                s.apply(a);
            }
            let o = s.outcome().unwrap();
            total += o.scores[0];
            sq += o.scores[0] * o.scores[0];
            wins += (o.winner == Some(0)) as u64;
        }
        let n = deals as f64;
        let mean = total / n;
        let sd = (sq / n - mean * mean).max(0.0).sqrt();
        eprintln!(
            "PUCT-{sims} seat 0 vs 3 greedy, {deals} deals: mean {mean:+.2} ± {:.2} (95 %), \
             win rate {:.0} %, {:.1} ms per move",
            1.96 * sd / n.sqrt(),
            100.0 * wins as f64 / n,
            1000.0 * secs / moves as f64
        );
    }
}
