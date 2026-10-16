//! The result type shared by all solvers.

use poker_games::{Game, Outcome, Player};
use std::time::Duration;

/// What the searching player tries to maximize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Objective {
    /// Expected score under the game's scoring rule (Big Two: pairwise settlement).
    #[default]
    Score,
    /// Probability of being the winner (Big Two: going out first).
    Win,
}

impl Objective {
    /// The utility `player` gets from `outcome`, scaled to roughly `[-1, 1]`.
    pub fn utility<G: Game>(self, game: &G, outcome: &Outcome, player: Player) -> f64 {
        match self {
            Objective::Score => outcome.scores[player] / game.max_abs_score(),
            Objective::Win => (outcome.winner == Some(player)) as u8 as f64,
        }
    }
}

/// Running sums for one action. Kept as sums so results from parallel searches
/// can be merged by adding.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sums {
    pub n: u64,
    pub score: f64,
    pub score_sq: f64,
    pub wins: f64,
}

impl Sums {
    pub fn add(&mut self, outcome: &Outcome, player: Player) {
        let s = outcome.scores[player];
        self.n += 1;
        self.score += s;
        self.score_sq += s * s;
        self.wins += (outcome.winner == Some(player)) as u8 as f64;
    }

    pub fn merge(&mut self, other: &Sums) {
        self.n += other.n;
        self.score += other.score;
        self.score_sq += other.score_sq;
        self.wins += other.wins;
    }
}

/// Statistics for one legal action, from the searching player's point of view.
#[derive(Clone, Debug)]
pub struct MoveStats<A> {
    pub action: A,
    /// Simulated games that started with this action. In ISMCTS this is the
    /// visit count: the search spends more simulations on actions it rates higher.
    pub visits: u64,
    /// Mean final score after this action.
    pub mean_score: f64,
    /// Half-width of a 95% confidence interval for `mean_score`
    /// (`1.96 · stdev / √visits`).
    pub score_ci95: f64,
    /// Fraction of simulated games in which the player went out first.
    pub win_rate: f64,
    /// Half-width of a 95% confidence interval for `win_rate`.
    pub win_ci95: f64,
}

impl<A: Copy> MoveStats<A> {
    pub(crate) fn from_sums(action: A, s: &Sums) -> MoveStats<A> {
        let n = s.n.max(1) as f64;
        let mean = s.score / n;
        let var = (s.score_sq / n - mean * mean).max(0.0);
        let p = s.wins / n;
        MoveStats {
            action,
            visits: s.n,
            mean_score: mean,
            score_ci95: 1.96 * (var / n).sqrt(),
            win_rate: p,
            win_ci95: 1.96 * (p * (1.0 - p) / n).sqrt(),
        }
    }
}

/// A solver's verdict on a position.
#[derive(Clone, Debug)]
pub struct Analysis<A> {
    /// The player the analysis is for.
    pub player: Player,
    /// Every legal action, best first (see [`Analysis::best`]).
    pub moves: Vec<MoveStats<A>>,
    /// Total simulations run.
    pub simulations: u64,
    /// Wall-clock time spent.
    pub elapsed: Duration,
}

impl<A: Copy> Analysis<A> {
    /// The recommended action (the first entry of [`Analysis::moves`]).
    pub fn best(&self) -> A {
        self.moves[0].action
    }

    /// The `k` best actions.
    pub fn top(&self, k: usize) -> &[MoveStats<A>] {
        &self.moves[..k.min(self.moves.len())]
    }
}
