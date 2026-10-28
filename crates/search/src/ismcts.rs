//! Single-observer information-set Monte Carlo tree search (SO-ISMCTS).
//!
//! Reference: Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree
//! Search", IEEE TCIAIG 4(2), 2012.
//!
//! Plain MCTS needs to know the full state. ISMCTS instead keeps one tree whose
//! nodes are *sequences of public actions* seen from the searching player's
//! viewpoint, and on every iteration:
//!
//! 1. **Determinize** — sample hidden cards consistent with what the searching
//!    player knows.
//! 2. **Select** — walk down the tree, at each node considering only the children
//!    whose action is legal in this sample, picking by UCB1 (best average for the
//!    player to move plus an exploration bonus).
//! 3. **Expand** — add one child for an action not tried yet.
//! 4. **Simulate** — finish the game with a fast rollout policy, or solve it
//!    exactly with [`crate::endgame`] if few cards remain.
//! 5. **Backpropagate** — every node on the path records the result for the
//!    player who chose the action leading to it.
//!
//! Each node therefore estimates how good an action is *averaged over the hands
//! the opponents might hold*. Multi-player is handled by storing, at every node,
//! the utility of the player who acted there.
//!
//! The UCB1 exploration term uses the *availability* count (how often the action
//! was legal when its parent was visited) instead of the parent's visit count,
//! as in the paper.
//!
//! Two parts can be swapped through [`ismcts_with`]:
//!
//! * the **sampler** of step 1 — uniform by default, or opponent-aware
//!   ([`crate::belief::Belief`]);
//! * the **leaf evaluation** of step 4 — a rollout by default, or a [`Leaf`]
//!   evaluator (e.g. a learned value network), optionally mixed with a rollout.

use crate::analysis::{Analysis, MoveStats, Objective, Sums};
use crate::belief::{Sampler, UniformSampler};
use crate::endgame::Endgame;
use big2_games::{play_out, Estimate, Evaluator, Game, Outcome, Player, Policy, Transposable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Settings for [`ismcts`].
#[derive(Clone, Debug)]
pub struct IsmctsConfig {
    /// Total iterations (simulated games) across all threads.
    pub iterations: u64,
    /// Optional wall-clock limit; whichever of `iterations` and this comes first.
    pub time_limit: Option<Duration>,
    /// UCB1 exploration constant `c` in `mean + c·√(ln avail / visits)`.
    /// Utilities are scaled to `[-1, 1]` by the game's maximum score, so typical
    /// Big Two scores are small numbers; see `docs/solver.md` for tuning.
    pub exploration: f64,
    /// Independent trees searched in parallel ("root parallelization"); their
    /// root statistics are summed.
    pub threads: usize,
    /// Base random seed. With a fixed seed, thread count and no time limit, the
    /// search is reproducible.
    pub seed: u64,
    /// What the searching player maximizes.
    pub objective: Objective,
    /// Solve exactly once [`Game::endgame_size`] is at most this (0 = never).
    pub endgame_threshold: usize,
    /// Node budget per exact solve; when exceeded, fall back to a rollout.
    pub endgame_max_nodes: u64,
}

impl Default for IsmctsConfig {
    fn default() -> Self {
        IsmctsConfig {
            iterations: 10_000,
            time_limit: None,
            exploration: 0.25,
            threads: 1,
            seed: 0,
            objective: Objective::Score,
            endgame_threshold: 0,
            endgame_max_nodes: 20_000,
        }
    }
}

/// Evaluates the position at the end of each simulation instead of (or as well
/// as) a rollout: `value = rollout_weight · rollout + (1 − rollout_weight) · evaluator`.
/// With `rollout_weight = 0` no rollout is played.
pub struct Leaf<'a, E> {
    pub evaluator: &'a E,
    pub rollout_weight: f64,
}

/// Placeholder evaluator type for searches without one (see [`ismcts`]).
pub struct NoEvaluator;

impl<G: Game> Evaluator<G> for NoEvaluator {
    fn evaluate(&self, _state: &G) -> Estimate {
        unreachable!("NoEvaluator is never called")
    }
}

/// Root-child statistics from one tree, and the number of iterations it ran.
type TreeResult<A> = (Vec<(A, Sums)>, u64);

struct Node<A> {
    /// Action leading here from the parent (`None` at the root).
    action: Option<A>,
    /// Player who chose `action`.
    player: Player,
    children: Vec<usize>,
    visits: u64,
    /// Times this node's action was legal when the parent was visited.
    avail: u64,
    /// Sum of `player`'s utility over visits.
    utility: f64,
    /// Score/win sums for reporting (only read for root children).
    sums: Sums,
}

impl<A> Node<A> {
    fn new(action: Option<A>, player: Player) -> Self {
        Node {
            action,
            player,
            children: Vec::new(),
            visits: 0,
            avail: 0,
            utility: 0.0,
            sums: Sums::default(),
        }
    }
}

/// Analyzes `state` for the player to move, who must be `observer`.
///
/// `rollout` plays the simulated games; a fast, reasonable policy (not uniform
/// random) gives much better estimates per simulation.
///
/// # Panics
/// Panics if the game is over or `observer` is not the player to move.
pub fn ismcts<G, P>(
    state: &G,
    observer: Player,
    cfg: &IsmctsConfig,
    rollout: &P,
) -> Analysis<G::Action>
where
    G: Transposable,
    P: Policy<G>,
{
    ismcts_with::<G, P, UniformSampler, NoEvaluator>(
        state,
        observer,
        cfg,
        rollout,
        &UniformSampler,
        None,
    )
}

/// [`ismcts`] with a custom determinization `sampler` and, optionally, a `leaf`
/// evaluator in place of (or mixed with) rollouts.
pub fn ismcts_with<G, P, S, E>(
    state: &G,
    observer: Player,
    cfg: &IsmctsConfig,
    rollout: &P,
    sampler: &S,
    leaf: Option<&Leaf<'_, E>>,
) -> Analysis<G::Action>
where
    G: Transposable,
    P: Policy<G>,
    S: Sampler<G>,
    E: Evaluator<G>,
{
    assert_eq!(
        state.current_player(),
        Some(observer),
        "the observer must be the player to move"
    );
    let start = Instant::now();
    let deadline = cfg.time_limit.map(|t| start + t);
    let threads = cfg.threads.max(1);

    let per_thread: Vec<TreeResult<G::Action>> = (0..threads)
        .into_par_iter()
        .map(|t| {
            let iters = cfg.iterations / threads as u64
                + u64::from((t as u64) < cfg.iterations % threads as u64);
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed
                    .wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let ctx = Ctx {
                cfg,
                rollout,
                sampler,
                leaf,
            };
            search_tree(state, observer, &ctx, iters, deadline, &mut rng)
        })
        .collect();

    // Merge root statistics across threads; keep every legal action, even unvisited ones.
    let legal = state.legal_actions_vec();
    let mut merged: Vec<(G::Action, Sums)> = legal.iter().map(|&a| (a, Sums::default())).collect();
    let mut simulations = 0;
    for (stats, sims) in &per_thread {
        simulations += sims;
        for (a, s) in stats {
            if let Some(slot) = merged.iter_mut().find(|(b, _)| b == a) {
                slot.1.merge(s);
            }
        }
    }
    // Most-visited first ("robust child"); ties by the objective's mean.
    let mean_utility = |s: &Sums| match cfg.objective {
        Objective::Score => s.score / s.n.max(1) as f64,
        Objective::Win => s.wins / s.n.max(1) as f64,
    };
    merged.sort_by(|(_, x), (_, y)| {
        y.n.cmp(&x.n)
            .then(mean_utility(y).total_cmp(&mean_utility(x)))
    });

    Analysis {
        player: observer,
        moves: merged
            .iter()
            .map(|(a, s)| MoveStats::from_sums(*a, s))
            .collect(),
        simulations,
        elapsed: start.elapsed(),
    }
}

/// What one search thread works with.
struct Ctx<'a, 'l, P, S, E> {
    cfg: &'a IsmctsConfig,
    rollout: &'a P,
    sampler: &'a S,
    leaf: Option<&'a Leaf<'l, E>>,
}

/// Runs one tree for `iterations` (or until `deadline`) and returns root-child sums.
fn search_tree<G, P, S, E>(
    root: &G,
    observer: Player,
    ctx: &Ctx<'_, '_, P, S, E>,
    iterations: u64,
    deadline: Option<Instant>,
    rng: &mut SmallRng,
) -> TreeResult<G::Action>
where
    G: Transposable,
    P: Policy<G>,
    S: Sampler<G>,
    E: Evaluator<G>,
{
    let cfg = ctx.cfg;
    let mut tree: Vec<Node<G::Action>> = vec![Node::new(None, observer)];
    let mut endgame = Endgame::<G>::new(cfg.objective);
    let mut legal = Vec::new();
    let mut available: Vec<usize> = Vec::new();
    let mut untried: Vec<G::Action> = Vec::new();
    let mut path: Vec<usize> = Vec::new();
    let mut done = 0u64;

    while done < iterations {
        if done.is_multiple_of(64) && deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        let mut s = ctx.sampler.sample(root, observer, rng);
        let mut node = 0usize;
        path.clear();

        // Selection and expansion.
        while let Some(to_move) = s.current_player() {
            s.legal_actions(&mut legal);
            available.clear();
            untried.clear();
            for &a in &legal {
                match tree[node]
                    .children
                    .iter()
                    .find(|&&c| tree[c].action == Some(a))
                {
                    Some(&c) => available.push(c),
                    None => untried.push(a),
                }
            }
            for &c in &available {
                tree[c].avail += 1;
            }
            if !untried.is_empty() {
                let a = untried[rng.gen_range(0..untried.len())];
                let child = tree.len();
                let mut n = Node::new(Some(a), to_move);
                n.avail = 1;
                tree.push(n);
                tree[node].children.push(child);
                s.apply(a);
                path.push(child);
                break;
            }
            let c = cfg.exploration;
            let best = *available
                .iter()
                .max_by(|&&x, &&y| ucb(&tree[x], c).total_cmp(&ucb(&tree[y], c)))
                .expect("a non-terminal state has a legal action");
            s.apply(tree[best].action.unwrap());
            path.push(best);
            node = best;
        }

        // Simulation (or evaluation).
        let value = evaluate_leaf(&mut s, ctx, &mut endgame, rng);

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
    (stats, done)
}

#[inline]
fn ucb<A>(n: &Node<A>, c: f64) -> f64 {
    let v = n.visits as f64;
    n.utility / v + c * ((n.avail as f64).ln() / v).sqrt()
}

/// The value of a simulation's last position: the result if the game is over or
/// solved exactly, else the leaf evaluator's estimate, mixed with a rollout.
fn evaluate_leaf<G, P, S, E>(
    s: &mut G,
    ctx: &Ctx<'_, '_, P, S, E>,
    endgame: &mut Endgame<G>,
    rng: &mut SmallRng,
) -> Estimate
where
    G: Transposable,
    P: Policy<G>,
    E: Evaluator<G>,
{
    let Some(leaf) = ctx.leaf else {
        return Estimate::from(&simulate(s, ctx.cfg, ctx.rollout, endgame, rng));
    };
    if let Some(o) = s.outcome() {
        return Estimate::from(&o);
    }
    let est = leaf.evaluator.evaluate(s);
    let w = leaf.rollout_weight;
    if w <= 0.0 {
        return est;
    }
    let played = Estimate::from(&simulate(s, ctx.cfg, ctx.rollout, endgame, rng));
    let mix = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter()
            .zip(b)
            .map(|(x, y)| w * x + (1.0 - w) * y)
            .collect()
    };
    Estimate {
        scores: mix(&played.scores, &est.scores),
        wins: mix(&played.wins, &est.wins),
    }
}

/// Finishes a determinized game: exactly if it is small enough, else by rollout.
pub(crate) fn simulate<G, P>(
    s: &mut G,
    cfg: &IsmctsConfig,
    rollout: &P,
    endgame: &mut Endgame<G>,
    rng: &mut SmallRng,
) -> Outcome
where
    G: Transposable,
    P: Policy<G>,
{
    if let Some(o) = s.outcome() {
        return o;
    }
    if s.endgame_size() <= cfg.endgame_threshold {
        if let Some(o) = endgame.solve(s, cfg.endgame_max_nodes) {
            return o;
        }
    }
    play_out(s, rollout, rng)
}
