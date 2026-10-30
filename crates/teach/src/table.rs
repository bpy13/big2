//! The move table: every legal action with its value, the input of reasons and grades.

use big2_big2::plan::min_plays;
use big2_big2::{Action, State, NUM_PLAYERS};
use big2_games::Game;
use big2_search::{Analysis, Forced};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sampled deals behind [`MoveTable::from_analysis`]'s expected plays-to-out.
pub const PLAYS_SAMPLES: usize = 32;

/// One legal action and its value, from the actor's point of view.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveRow {
    pub action: Action,
    /// Mean final score of the actor after this action, in points.
    pub mean: f64,
    /// Half-width of the 95 % interval of `mean`.
    pub ci95: f64,
    /// Share of deals in which the actor went out first.
    pub win: f64,
    /// Mean final score of every seat (absolute seat index) after this action.
    /// Set by the paired evaluator; `None` in a table built from a plain
    /// [`Analysis`], which values the actor only.
    pub seat_means: Option<[f64; NUM_PLAYERS]>,
}

/// Every legal action of the player to act, best first.
#[derive(Clone, Debug)]
pub struct MoveTable {
    /// The player to act (absolute seat).
    pub actor: usize,
    /// Every legal action, by `mean` descending (ties keep the engine's legal order).
    pub rows: Vec<MoveRow>,
    /// Sampled deals behind every row (the same deals for every row when paired).
    pub deals: u64,
    /// Expected fewest plays to go out per seat (absolute index), over the
    /// actor's sampled deals: exact for the actor, the sample mean for the others.
    pub plays_to_out: [f64; NUM_PLAYERS],
    /// Set when the exact endgame proved that the best row's winner is the same
    /// in every deal of the hidden cards.
    pub forced: Option<Forced>,
    /// Every seat's score per row per deal (rows in `rows` order), when paired.
    /// The grader's paired differences come from here.
    pub per_deal: Option<Vec<Vec<[f64; NUM_PLAYERS]>>>,
}

impl MoveTable {
    /// A table from any solver's analysis of `state`. Rows keep the analysis's
    /// order; `seat_means` and `per_deal` stay empty; `plays_to_out` is sampled
    /// from the actor's view with a fixed seed ([`PLAYS_SAMPLES`] deals).
    pub fn from_analysis(state: &State, analysis: &Analysis<Action>) -> MoveTable {
        let rows: Vec<MoveRow> = analysis
            .moves
            .iter()
            .map(|m| MoveRow {
                action: m.action,
                mean: m.mean_score,
                ci95: m.score_ci95,
                win: m.win_rate,
                seat_means: None,
            })
            .collect();
        MoveTable {
            actor: analysis.player,
            deals: analysis.moves.first().map_or(0, |m| m.visits),
            plays_to_out: expected_plays_to_out(state, analysis.player, PLAYS_SAMPLES, 0),
            rows,
            forced: None,
            per_deal: None,
        }
    }

    /// The best row (highest `mean`).
    ///
    /// # Panics
    /// Panics on an empty table (a finished deal has no table).
    pub fn best(&self) -> &MoveRow {
        &self.rows[0]
    }

    /// Index of `action` in `rows`.
    pub fn index_of(&self, action: Action) -> Option<usize> {
        self.rows.iter().position(|r| r.action == action)
    }

    /// The row of `action`.
    pub fn row(&self, action: Action) -> Option<&MoveRow> {
        self.index_of(action).map(|i| &self.rows[i])
    }

    /// The best row other than `action`'s: the runner-up when `action` is best,
    /// else the best row. `None` when `action` is the only legal action.
    pub fn best_other(&self, action: Action) -> Option<&MoveRow> {
        self.rows.iter().find(|r| r.action != action)
    }
}

/// Expected fewest plays to go out for every seat, seen from `observer`: exact
/// for the observer's own hand, the mean over `samples` deals of the hidden
/// cards (drawn from the observer's view, never the true hands) for the others.
pub fn expected_plays_to_out(
    state: &State,
    observer: usize,
    samples: usize,
    seed: u64,
) -> [f64; NUM_PLAYERS] {
    let rules = state.rules();
    let mut out = [0.0; NUM_PLAYERS];
    let samples = samples.max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..samples {
        let d = state.determinize(observer, &mut rng);
        for (p, o) in out.iter_mut().enumerate() {
            *o += min_plays(d.hand(p), rules) as f64;
        }
    }
    for o in &mut out {
        *o /= samples as f64;
    }
    out[observer] = min_plays(state.hand(observer), rules) as f64;
    out
}
