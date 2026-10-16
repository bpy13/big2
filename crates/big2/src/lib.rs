//! Big Two rules engine.
//!
//! The rules implemented here are specified in `docs/rules/big2.md`; section
//! numbers in comments (e.g. "rules §4.2") refer to that document.
//!
//! * [`card`] — card indices and hands as `u64` masks.
//! * [`rules`] — the [`RuleSet`] of table variants.
//! * [`combo`] — classifying cards into a [`Combo`] and comparing combos.
//! * [`movegen`] — generating legal plays.
//! * [`state`] — the [`State`] of a deal, implementing [`poker_games::Game`].
//! * [`bots`] — simple rule-based policies used as baselines and search rollouts.
//! * [`features`] — neural-network input encoding (Phase 3).
//!
//! ```
//! use poker_big2::{State, RuleSet};
//! use poker_games::{Game, UniformRandom, play_out};
//! use rand::SeedableRng;
//!
//! let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
//! let mut game = State::new_deal(RuleSet::default(), &mut rng);
//! let outcome = play_out(&mut game, &UniformRandom, &mut rng);
//! assert_eq!(outcome.scores.iter().sum::<f64>(), 0.0);
//! ```

pub mod bots;
pub mod card;
pub mod combo;
pub mod features;
pub mod movegen;
pub mod rules;
pub mod state;

pub use combo::{classify, Combo, Kind};
pub use rules::{FlushOrder, PassRule, RuleSet, Scoring, Straights};
pub use state::{Action, IllegalAction, State, Turn, HAND_SIZE, NUM_PLAYERS};
