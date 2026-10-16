//! Decision-time search for imperfect-information games.
//!
//! All solvers here work on any [`poker_games::Game`] and answer the same
//! question: *in this position, from this player's point of view, how good is each
//! legal action?* They return an [`Analysis`] with per-action statistics.
//!
//! * [`ismcts`] — single-observer information-set Monte Carlo tree search
//!   (Cowling, Powley & Whitehouse, 2012). The main solver.
//! * [`pimc`] — flat Monte Carlo over determinizations ("perfect-information
//!   Monte Carlo"). A simpler baseline.
//! * [`endgame`] — exact max^n search of a fully known position, used inside
//!   ISMCTS once few cards remain.
//!
//! Both Monte Carlo solvers reason about hidden cards the same way: they sample
//! full deals consistent with what the observer has seen
//! ([`poker_games::Game::determinize`]) and play them out. Their numbers are
//! **estimates** with sampling error, not game-theoretic values.

pub mod analysis;
pub mod endgame;
pub mod ismcts;
pub mod pimc;

pub use analysis::{Analysis, MoveStats, Objective};
pub use ismcts::{ismcts, IsmctsConfig};
pub use pimc::{pimc, PimcConfig};
