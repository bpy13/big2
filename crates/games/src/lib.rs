//! Interfaces every game implements so that the solvers in `poker-search` (and the
//! RL code in Python) can work with any game, not only Big Two.
//!
//! The central trait is [`Game`]: a mutable game *state* that knows whose turn it
//! is, which actions are legal, how to apply one, and how the game was scored.
//! Card games hide information, so a game also implements
//! [`Game::determinize`]: given what one player can see, sample a full state
//! (all hidden cards filled in) that is consistent with it. That is the hook that
//! Monte Carlo search methods use to reason about hidden cards.

use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;

/// Index of a seat at the table, `0..num_players()`.
pub type Player = usize;

/// The outcome of a finished game, from every player's perspective.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Score per player under the game's scoring rule. Sums to zero for
    /// zero-sum games.
    pub scores: Vec<f64>,
    /// The single winner, if the game has one (in Big Two: the player who went out).
    pub winner: Option<Player>,
}

/// A game state that can be stepped forward.
///
/// States are cheap to clone; search algorithms clone them constantly.
pub trait Game: Clone + Send + Sync + Debug {
    /// An action a player can take. Actions must be *public*: once taken, every
    /// player sees which action it was. (This holds for Big Two plays and passes,
    /// and for poker bets; private information lives only in the state.)
    type Action: Copy + Eq + Hash + Debug + Send + Sync;

    /// Number of seats.
    fn num_players(&self) -> usize;

    /// Whose turn it is, or `None` if the game is over.
    fn current_player(&self) -> Option<Player>;

    /// Writes the legal actions for the current player into `out` (cleared first).
    /// Leaves `out` empty when the game is over.
    fn legal_actions(&self, out: &mut Vec<Self::Action>);

    /// Applies a legal action for the current player.
    ///
    /// Implementations may panic on illegal actions; callers that accept actions
    /// from users should validate them against [`Game::legal_actions`] first.
    fn apply(&mut self, action: Self::Action);

    /// The result, or `None` while the game is still running.
    fn outcome(&self) -> Option<Outcome>;

    /// The largest absolute score any player can get. Used to scale scores into
    /// `[-1, 1]` for search.
    fn max_abs_score(&self) -> f64;

    /// Samples a complete state that `observer` cannot tell apart from this one:
    /// the observer's own private information and all public information are kept,
    /// everything hidden from the observer is re-dealt at random (uniformly among
    /// consistent possibilities, unless the game documents otherwise).
    fn determinize<R: Rng + ?Sized>(&self, observer: Player, rng: &mut R) -> Self;

    /// `true` once the game is over.
    fn is_terminal(&self) -> bool {
        self.current_player().is_none()
    }

    /// Convenience wrapper around [`Game::legal_actions`].
    fn legal_actions_vec(&self) -> Vec<Self::Action> {
        let mut v = Vec::new();
        self.legal_actions(&mut v);
        v
    }
}

/// Something that picks an action in a state: a bot, a rollout policy inside a
/// search, or a learned network wrapped from Python.
pub trait Policy<G: Game>: Send + Sync {
    /// Picks one of the `legal` actions (never empty) for the current player.
    fn choose<R: Rng + ?Sized>(&self, state: &G, legal: &[G::Action], rng: &mut R) -> G::Action;
}

/// Picks uniformly at random among legal actions.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformRandom;

impl<G: Game> Policy<G> for UniformRandom {
    fn choose<R: Rng + ?Sized>(&self, _state: &G, legal: &[G::Action], rng: &mut R) -> G::Action {
        legal[rng.gen_range(0..legal.len())]
    }
}

/// Plays `state` to the end with `policy` for every seat and returns the outcome.
pub fn play_out<G: Game, P: Policy<G>, R: Rng + ?Sized>(
    state: &mut G,
    policy: &P,
    rng: &mut R,
) -> Outcome {
    let mut legal = Vec::new();
    while !state.is_terminal() {
        state.legal_actions(&mut legal);
        let a = policy.choose(state, &legal, rng);
        state.apply(a);
    }
    state.outcome().expect("terminal state has an outcome")
}
